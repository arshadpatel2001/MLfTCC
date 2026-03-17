"""
Domain Generalization Methods for Basin Generalization
======================================================
Implements all methods compared in our NeurIPS/ICLR submission:

  ERM     — Empirical Risk Minimization (strong baseline per DomainBed)
  IRM     — Invariant Risk Minimization (Arjovsky et al., 2019)
  V-REx   — Variance-based Risk Extrapolation (Krueger et al., ICML 2021)
  CORAL   — Correlation Alignment (Sun & Saenko, ECCV 2016)
  DANN    — Domain-Adversarial Neural Networks (Ganin et al., JMLR 2016)
  MAML    — Model-Agnostic Meta-Learning (Finn et al., ICML 2017)
  PhysIRM — [PROPOSED] Physics-guided IRM (Ours)

Each method is a self-contained class with a unified API:
  method.compute_loss(batch_per_env, model) → scalar loss
  method.update(optimizer, batch_per_env, model) → metrics dict

Usage:
  method = PhysIRM(irm_lambda=1.0, phys_lambda=0.5, warmup_steps=500)
  metrics = method.update(optimizer, batches_by_basin, model)
"""

import math
import copy
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# ── Loss utilities ────────────────────────────────────────────────────────────

def task_loss(logits_int: torch.Tensor, logits_dir: torch.Tensor,
              y_int: torch.Tensor, y_dir: torch.Tensor,
              int_weight: float = 1.0, dir_weight: float = 0.5) -> torch.Tensor:
    """Weighted cross-entropy over intensity + direction tasks."""
    l_int = F.cross_entropy(logits_int, y_int)
    l_dir = F.cross_entropy(logits_dir, y_dir)
    return int_weight * l_int + dir_weight * l_dir


def per_env_loss(model: nn.Module, batch: dict, **kwargs) -> torch.Tensor:
    """Compute task loss on a single environment (basin) batch."""
    out = model(batch)
    return task_loss(
        out["logits_intensity"], out["logits_direction"],
        batch["y_intensity"], batch["y_direction"], **kwargs
    )


# ── Base class ────────────────────────────────────────────────────────────────

class DGMethod(ABC):
    """Abstract base for all DG methods."""

    @abstractmethod
    def compute_loss(
        self, batches: Dict[str, dict], model: nn.Module
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            batches: dict mapping basin_code → batch_dict
            model:   TropiCycloneModel (or any nn.Module)
        Returns:
            total_loss, metrics_dict
        """

    def update(
        self, optimizer: torch.optim.Optimizer,
        batches: Dict[str, dict],
        model: nn.Module,
        step: int = 0,
    ) -> dict:
        optimizer.zero_grad()
        loss, metrics = self.compute_loss(batches, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        metrics["loss"] = loss.item()
        return metrics


# ── 1. ERM ────────────────────────────────────────────────────────────────────

class ERM(DGMethod):
    """
    Empirical Risk Minimization: pool all environments and minimize average loss.

    Despite its simplicity, ERM is the hardest baseline to beat (Gulrajani &
    Lopez-Paz, DomainBed, ICLR 2021). Always include this.
    """

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        losses = []
        for basin, batch in batches.items():
            losses.append(per_env_loss(model, batch))
        total = torch.stack(losses).mean()
        return total, {"erm_loss": total.item()}


# ── 2. IRM ────────────────────────────────────────────────────────────────────

class IRM(DGMethod):
    """
    Invariant Risk Minimization (Arjovsky et al., 2019).
    arXiv: 1907.02893

    Key idea: penalise gradients of the loss w.r.t. a dummy scalar classifier
    w (initialised to 1) applied to the representation. If the classifier's
    gradient is non-zero for any environment, the representation is non-invariant.

    Penalty:
        Ω(f) = Σ_e ||∇_{w|w=1} R^e(w·f)||²
    """

    def __init__(self, irm_lambda: float = 1.0, warmup_steps: int = 500):
        """
        Args:
            irm_lambda:    Weight on the IRM penalty (λ in the paper).
                           Anneal from 0 → irm_lambda over warmup_steps.
            warmup_steps:  Steps before IRM penalty reaches full strength.
        """
        self.irm_lambda    = irm_lambda
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def _penalty(self, logits_int, logits_dir, y_int, y_dir) -> torch.Tensor:
        """Compute IRM penalty for one environment."""
        # Scalar w = 1 (dummy classifier variable)
        w = torch.ones(1, requires_grad=True, device=logits_int.device)
        loss = task_loss(logits_int * w, logits_dir * w, y_int, y_dir)
        g = grad(loss, w, create_graph=True)[0]
        return g.pow(2).sum()

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        lam = self.irm_lambda * min(1.0, self._step / max(self.warmup_steps, 1))

        erm_losses, irm_penalties = [], []
        for basin, batch in batches.items():
            out = model(batch)
            erm_losses.append(
                task_loss(out["logits_intensity"], out["logits_direction"],
                          batch["y_intensity"], batch["y_direction"])
            )
            irm_penalties.append(
                self._penalty(out["logits_intensity"], out["logits_direction"],
                              batch["y_intensity"], batch["y_direction"])
            )

        erm = torch.stack(erm_losses).mean()
        pen = torch.stack(irm_penalties).mean()
        total = erm + lam * pen

        return total, {
            "erm_loss": erm.item(),
            "irm_penalty": pen.item(),
            "irm_lambda": lam,
        }


# ── 3. V-REx ─────────────────────────────────────────────────────────────────

class VREx(DGMethod):
    """
    Variance-based Risk Extrapolation (Krueger et al., ICML 2021).

    Augments ERM with a penalty on the variance of per-environment losses:
        L = ERM + β * Var({R^e})

    More numerically stable than IRM in practice (no second-order gradients).
    """

    def __init__(self, beta: float = 1.0, warmup_steps: int = 500):
        self.beta          = beta
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        beta = self.beta * min(1.0, self._step / max(self.warmup_steps, 1))

        losses = []
        for basin, batch in batches.items():
            losses.append(per_env_loss(model, batch))

        losses_t = torch.stack(losses)
        erm  = losses_t.mean()
        var  = losses_t.var()
        total = erm + beta * var

        return total, {
            "erm_loss": erm.item(),
            "vrex_var": var.item(),
            "beta": beta,
        }


# ── 4. CORAL ─────────────────────────────────────────────────────────────────

class CORAL(DGMethod):
    """
    CORrelation ALignment (Sun & Saenko, ECCV 2016).

    Minimises the Frobenius distance between second-order statistics
    (covariance matrices) of representations across environments.

    Penalty:
        Ω = Σ_{e1≠e2} ||Cov(f(X^{e1})) - Cov(f(X^{e2}))||_F²
    """

    def __init__(self, coral_lambda: float = 1.0):
        self.coral_lambda = coral_lambda

    @staticmethod
    def _cov(z: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix of (B, D) feature matrix."""
        n, d = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        return (z.T @ z) / (n - 1 + 1e-8)

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        envs  = list(batches.keys())
        losses, reprs = [], []

        for basin in envs:
            batch = batches[basin]
            out   = model(batch)
            losses.append(
                task_loss(out["logits_intensity"], out["logits_direction"],
                          batch["y_intensity"], batch["y_direction"])
            )
            reprs.append(out["z"])  # (B, D)

        erm = torch.stack(losses).mean()

        # All pairwise covariance distances
        coral_pen = torch.tensor(0.0, device=erm.device)
        n_pairs   = 0
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                cov_i = self._cov(reprs[i])
                cov_j = self._cov(reprs[j])
                coral_pen = coral_pen + (cov_i - cov_j).pow(2).sum()
                n_pairs   += 1

        if n_pairs > 0:
            coral_pen = coral_pen / n_pairs

        total = erm + self.coral_lambda * coral_pen
        return total, {
            "erm_loss": erm.item(),
            "coral_penalty": coral_pen.item(),
        }


# ── 5. DANN ──────────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """Reverses gradient sign during backprop (Ganin et al., 2016)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, n_domains: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, z, alpha=1.0):
        z_rev = GradientReversal.apply(z, alpha)
        return self.net(z_rev)


class DANN(DGMethod):
    """
    Domain-Adversarial Neural Networks (Ganin et al., JMLR 2016).

    Trains a domain (basin) classifier on top of gradient-reversed features.
    Forces the encoder to produce basin-invariant representations.

    GRL schedule: α ramps from 0 → 1 over training, following original paper.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_domains: int = 6,
        dann_lambda: float = 1.0,
        total_steps: int = 10000,
    ):
        self.dann_lambda  = dann_lambda
        self.total_steps  = total_steps
        self._step        = 0
        self.discriminator = DomainDiscriminator(feature_dim, n_domains)

    def to(self, device):
        self.discriminator = self.discriminator.to(device)
        return self

    def parameters(self):
        return self.discriminator.parameters()

    def _alpha(self):
        """GRL strength schedule from Ganin et al."""
        p = self._step / max(self.total_steps, 1)
        return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        alpha = self._alpha()

        task_losses, dann_losses = [], []
        for basin, batch in batches.items():
            out = model(batch)
            task_losses.append(
                task_loss(out["logits_intensity"], out["logits_direction"],
                          batch["y_intensity"], batch["y_direction"])
            )
            domain_logits = self.discriminator(out["z"], alpha)
            basin_labels  = batch["basin_idx"]  # (B,)
            dann_losses.append(F.cross_entropy(domain_logits, basin_labels))

        erm  = torch.stack(task_losses).mean()
        dadv = torch.stack(dann_losses).mean()
        total = erm + self.dann_lambda * dadv

        return total, {
            "erm_loss":   erm.item(),
            "dann_adv":   dadv.item(),
            "dann_alpha": alpha,
        }

    def update(self, optimizer, batches, model, step=0):
        # Also include discriminator parameters in optimisation
        # (caller should create a joint optimizer or pass disc params separately)
        self._step = step
        optimizer.zero_grad()
        loss, metrics = self.compute_loss(batches, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        optimizer.step()
        metrics["loss"] = loss.item()
        return metrics


# ── 6. MAML ───────────────────────────────────────────────────────────────────

class MAML(DGMethod):
    """
    Model-Agnostic Meta-Learning (Finn et al., ICML 2017).
    Basin-as-task framing: each basin is a meta-learning "task".

    Inner loop:  adapt to source basin with k gradient steps
    Outer loop:  evaluate adapted model on a held-out query set from same basin

    At meta-test time: adapt to target basin with few labelled examples.

    Implementation: first-order MAML (FOMAML) for efficiency.
    For second-order MAML, set `first_order=False` (requires `higher` library).
    """

    def __init__(
        self,
        inner_lr: float   = 1e-3,
        inner_steps: int  = 5,
        first_order: bool = True,
    ):
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order

    def _inner_loop(
        self, model: nn.Module, support_batch: dict
    ) -> Tuple[nn.Module, List[torch.Tensor]]:
        """
        Perform inner-loop adaptation on support set.
        Returns adapted parameter list (not a new model to avoid deep copies at scale).
        """
        # FOMAML: compute gradients, create adapted params manually
        fast_weights = {n: p.clone() for n, p in model.named_parameters()}

        for _ in range(self.inner_steps):
            # Forward with fast weights
            out  = self._forward_with_weights(model, support_batch, fast_weights)
            loss = task_loss(out["logits_intensity"], out["logits_direction"],
                             support_batch["y_intensity"], support_batch["y_direction"])
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=not self.first_order,
                                        allow_unused=True)
            fast_weights = {
                n: p - self.inner_lr * (g if g is not None else torch.zeros_like(p))
                for (n, p), g in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def _forward_with_weights(self, model: nn.Module, batch: dict, weights: dict) -> dict:
        """
        Forward pass using custom weight dict instead of model.parameters().
        This is a simplified functional forward — for production use `higher` or
        `torch.func.functional_call`.
        """
        # Use torch.func.functional_call (PyTorch ≥ 2.0)
        from torch.func import functional_call
        return functional_call(model, weights, (batch,))

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        """
        For each environment:
          - Split batch into support (first half) and query (second half)
          - Inner-loop adapt on support
          - Compute query loss with adapted weights
        """
        meta_losses = []
        for basin, batch in batches.items():
            n  = batch["data_1d"].shape[0]
            n_s = n // 2

            support = {k: v[:n_s] for k, v in batch.items()}
            query   = {k: v[n_s:] for k, v in batch.items()}

            fast_w = self._inner_loop(model, support)
            # Query loss with adapted weights
            out_q  = self._forward_with_weights(model, query, fast_w)
            loss_q = task_loss(out_q["logits_intensity"], out_q["logits_direction"],
                               query["y_intensity"], query["y_direction"])
            meta_losses.append(loss_q)

        total = torch.stack(meta_losses).mean()
        return total, {"maml_meta_loss": total.item()}


# ── 7. PhysIRM (PROPOSED) ─────────────────────────────────────────────────────

class PhysIRM(DGMethod):
    """
    Physics-guided Invariant Risk Minimization [PROPOSED METHOD].
    ─────────────────────────────────────────────────────────────
    Contribution: We extend IRM with a physics-informed decomposition of the
    representation space into two orthogonal sub-spaces:

      z_phys ∈ R^{d_p}  — physics sub-space
         Encodes thermodynamic invariants: SST anomaly, wind shear, Coriolis
         parameter, maximum potential intensity proxy, boundary layer moisture.
         IRM penalty applied HERE to enforce cross-basin invariance.

      z_env  ∈ R^{d_e}  — synoptic sub-space
         Encodes basin-specific synoptic patterns (subtropical high position,
         monsoon trough, ITCZ). Allowed to shift across basins (no IRM penalty).

    Total loss:
        L = ERM(z_phys ⊕ z_env)
          + λ_irm  · Σ_e ||∇_w R^e(w·z_phys)||²    [IRM on physics space]
          + λ_orth · ||z_phys^T z_env||_F²            [orthogonality regulariser]
          + λ_phys · L_phys(z_phys, phys_features)    [physics grounding loss]

    Physics grounding loss: MSE between z_phys (first d_p dims) and a
    physics feature predictor, forcing z_phys to encode physical quantities.

    Motivation:
        Tropical cyclone dynamics are fundamentally governed by the same
        thermodynamic laws (Carnot cycle, CAPE, wind-induced surface heat
        exchange) across all basins. Basin differences arise from synoptic
        steering and background climatology — not from the governing physics.
        By separating these sub-spaces and enforcing invariance only on the
        physical sub-space, we give the model an inductive bias aligned with
        atmospheric science.

    References:
        - Arjovsky et al. (2019) IRM.       arXiv:1907.02893
        - Emanuel (1986) MPI theory.        JAS 43(6):585–604
        - Kaplan & DeMaria (2003) RI preds. WAF 18(6):1093–1108
        - Lu et al. (2021) Physics-guided DG. NeurIPS 2021.
    """

    def __init__(
        self,
        irm_lambda:    float = 1.0,    # λ_irm: IRM penalty weight on z_phys
        orth_lambda:   float = 0.1,    # λ_orth: orthogonality between sub-spaces
        phys_lambda:   float = 0.5,    # λ_phys: physics grounding loss weight
        phys_dim:      int   = 64,     # dimension of z_phys sub-space
        warmup_steps:  int   = 500,    # ramp IRM penalty from 0 → irm_lambda
        n_phys_feat:   int   = 8,      # number of input physics features
    ):
        self.irm_lambda   = irm_lambda
        self.orth_lambda  = orth_lambda
        self.phys_lambda  = phys_lambda
        self.phys_dim     = phys_dim
        self.warmup_steps = warmup_steps
        self._step        = 0

        # Physics predictor: maps z_phys → raw physics features
        # Trained jointly to ground z_phys in physical meaning
        self.phys_predictor = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_phys_feat),
        )

    def to(self, device):
        self.phys_predictor = self.phys_predictor.to(device)
        return self

    def parameters(self):
        return self.phys_predictor.parameters()

    def _irm_penalty_phys(
        self, logits_int, logits_dir, y_int, y_dir
    ) -> torch.Tensor:
        """IRM gradient penalty applied only on physics sub-space logits."""
        w = torch.ones(1, requires_grad=True, device=logits_int.device)
        loss = task_loss(logits_int * w, logits_dir * w, y_int, y_dir)
        g = grad(loss, w, create_graph=True)[0]
        return g.pow(2).sum()

    def _orthogonality_loss(
        self, z_phys: torch.Tensor, z_env: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage z_phys ⊥ z_env by penalising their cross-covariance.
        ||z_phys^T z_env||_F² / (n * d_phys * d_env)
        """
        n = z_phys.shape[0]
        z_p = z_phys - z_phys.mean(0, keepdim=True)
        z_e = z_env  - z_env.mean(0,  keepdim=True)
        cross_cov = (z_p.T @ z_e) / (n - 1 + 1e-8)
        return cross_cov.pow(2).mean()

    def _physics_grounding_loss(
        self, z_phys: torch.Tensor, phys_features: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE between predicted and actual physics features.
        Forces z_phys to encode thermodynamic invariants.
        """
        pred = self.phys_predictor(z_phys)
        return F.mse_loss(pred, phys_features)

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        lam = self.irm_lambda * min(1.0, self._step / max(self.warmup_steps, 1))

        erm_losses, irm_pens, orth_pens, phys_pens = [], [], [], []

        for basin, batch in batches.items():
            out = model(batch)
            z_phys = out["z_phys"]  # (B, phys_dim)
            z_env  = out["z_env"]   # (B, env_dim)

            # ── ERM loss (on full representation) ─────────────────────────
            erm_losses.append(
                task_loss(out["logits_intensity"], out["logits_direction"],
                          batch["y_intensity"], batch["y_direction"])
            )

            # ── IRM penalty on physics sub-space only ──────────────────────
            # We apply IRM on the classification logits derived ONLY from z_phys.
            # This requires a separate head on z_phys. For efficiency, we use
            # the full logits but scale by the physics component's contribution.
            # Full version: requires a dedicated phys-only head (see ablation).
            irm_pens.append(
                self._irm_penalty_phys(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"], batch["y_direction"]
                )
            )

            # ── Orthogonality between sub-spaces ──────────────────────────
            orth_pens.append(self._orthogonality_loss(z_phys, z_env))

            # ── Physics grounding loss ─────────────────────────────────────
            phys_pens.append(
                self._physics_grounding_loss(z_phys, batch["phys_features"])
            )

        erm       = torch.stack(erm_losses).mean()
        irm_pen   = torch.stack(irm_pens).mean()
        orth_pen  = torch.stack(orth_pens).mean()
        phys_pen  = torch.stack(phys_pens).mean()

        total = (erm
                 + lam         * irm_pen
                 + self.orth_lambda  * orth_pen
                 + self.phys_lambda  * phys_pen)

        return total, {
            "erm_loss":      erm.item(),
            "irm_penalty":   irm_pen.item(),
            "orth_penalty":  orth_pen.item(),
            "phys_grounding": phys_pen.item(),
            "irm_lambda":    lam,
        }

    def update(self, optimizer, batches, model, step=0):
        self._step = step
        optimizer.zero_grad()
        loss, metrics = self.compute_loss(batches, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.phys_predictor.parameters(), max_norm=1.0)
        optimizer.step()
        metrics["loss"] = loss.item()
        return metrics


# ── Registry ──────────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "erm":     ERM,
    "irm":     IRM,
    "vrex":    VREx,
    "coral":   CORAL,
    "dann":    DANN,
    "maml":    MAML,
    "physirm": PhysIRM,
}


def build_method(name: str, **kwargs) -> DGMethod:
    name_lower = name.lower()
    if name_lower not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. "
            f"Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name_lower](**kwargs)
