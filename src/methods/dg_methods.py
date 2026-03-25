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

import time
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.func import functional_call

# ── Loss utilities ────────────────────────────────────────────────────────────

def task_loss(
    logits_int: torch.Tensor,
    logits_dir: torch.Tensor,
    y_int:      torch.Tensor,
    y_dir:      torch.Tensor,
    int_weight: float = 1.0,
    dir_weight: float = 0.5,
    # Intensity regression (optional — ignored when None or reg_weight == 0)
    pred_reg:   Optional[torch.Tensor] = None,   # (B, 2): [wind_norm, pres_norm]
    y_wind_reg: Optional[torch.Tensor] = None,   # (B,) scalar targets, may contain NaN
    y_pres_reg: Optional[torch.Tensor] = None,   # (B,) scalar targets, may contain NaN
    reg_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined task loss:
      L = int_weight * CE(intensity_cls) + dir_weight * CE(direction_cls)
        + reg_weight * MSE(wind_reg, pres_reg)   [only when targets available]

    Regression targets may contain NaN values for end-of-storm samples where
    the 24h-ahead timestep does not exist.  These samples are masked out so
    they do not contribute gradients to the regression head.

    The IRM/PhysIRM gradient penalty is computed on the *classification* logits
    only (not regression), so the penalty function in each method does NOT call
    this full task_loss — it calls the classification-only version directly.
    """
    l_int = F.cross_entropy(logits_int, y_int)
    l_dir = F.cross_entropy(logits_dir, y_dir)
    loss  = int_weight * l_int + dir_weight * l_dir

    # ── Regression term ───────────────────────────────────────────────────────
    if (pred_reg is not None
            and y_wind_reg is not None
            and y_pres_reg is not None
            and reg_weight > 0.0):
        # Stack targets → (B, 2), build NaN mask
        y_reg  = torch.stack([y_wind_reg, y_pres_reg], dim=-1)  # (B, 2)
        mask   = torch.isfinite(y_reg).all(dim=-1)               # (B,)
        if mask.sum() > 0:
            l_reg = F.mse_loss(pred_reg[mask], y_reg[mask])
            loss  = loss + reg_weight * l_reg

    return loss


def per_env_loss(
    model:      nn.Module,
    batch:      dict,
    reg_weight: float = 0.5,
    **kwargs,
) -> torch.Tensor:
    """Compute task loss on a single environment (basin) batch."""
    out = model(batch)
    return task_loss(
        out["logits_intensity"],
        out["logits_direction"],
        batch["y_intensity"],
        batch["y_direction"],
        pred_reg   = out.get("pred_intensity_reg"),
        y_wind_reg = batch.get("y_wind_reg"),
        y_pres_reg = batch.get("y_pres_reg"),
        reg_weight = reg_weight,
        **kwargs,
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
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: Optional[torch.device] = None,
    ) -> dict:
        optimizer.zero_grad(set_to_none=True)
        env_device = device if device is not None else next(model.parameters()).device
        
        with torch.autocast(device_type="cuda" if env_device.type == "cuda" else "cpu", enabled=(env_device.type == "cuda")):
            loss, metrics = self.compute_loss(batches, model)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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

    def __init__(self, reg_weight: float = 0.5):
        self.reg_weight = reg_weight

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        losses = []
        for basin, batch in batches.items():
            losses.append(per_env_loss(model, batch, reg_weight=self.reg_weight))
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

    NOTE: The IRM gradient penalty is applied to the *classification* logits only.
    The regression head is part of the ERM loss but not the invariance penalty,
    since regression is a continuous output that does not lend itself naturally
    to the dummy-scalar-classifier formulation.
    """

    def __init__(self, irm_lambda: float = 1.0, warmup_steps: int = 500,
                 reg_weight: float = 0.5):
        """
        Args:
            irm_lambda:    Weight on the IRM penalty (λ in the paper).
                           Anneal from 0 → irm_lambda over warmup_steps.
            warmup_steps:  Steps before IRM penalty reaches full strength.
            reg_weight:    Weight on the intensity regression MSE loss.
        """
        self.irm_lambda    = irm_lambda
        self.warmup_steps  = warmup_steps
        self._step         = 0
        self.reg_weight    = reg_weight

    def _penalty(self, logits_int, logits_dir, y_int, y_dir) -> torch.Tensor:
        """Compute IRM penalty for one environment (classification logits only)."""
        w = torch.ones(1, requires_grad=True, device=logits_int.device)
        # Classification loss only — regression is excluded from the IRM penalty
        l_cls = (1.0 * F.cross_entropy(logits_int * w, y_int)
                 + 0.5 * F.cross_entropy(logits_dir * w, y_dir))
        g = grad(l_cls, w, create_graph=True)[0]
        return g.pow(2).sum()

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        lam = self.irm_lambda * min(1.0, self._step / max(self.warmup_steps, 1))

        erm_losses, irm_penalties = [], []
        for basin, batch in batches.items():
            out = model(batch)
            erm_losses.append(
                task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"],    batch["y_direction"],
                    pred_reg   = out.get("pred_intensity_reg"),
                    y_wind_reg = batch.get("y_wind_reg"),
                    y_pres_reg = batch.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
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

    def __init__(self, beta: float = 1.0, warmup_steps: int = 500,
                 reg_weight: float = 0.5):
        self.beta          = beta
        self.warmup_steps  = warmup_steps
        self._step         = 0
        self.reg_weight    = reg_weight

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        self._step += 1
        beta = self.beta * min(1.0, self._step / max(self.warmup_steps, 1))

        losses = []
        for basin, batch in batches.items():
            losses.append(per_env_loss(model, batch, reg_weight=self.reg_weight))

        losses_t = torch.stack(losses)
        erm  = losses_t.mean()
        # Use unbiased=False to avoid NaN when there is only 1 environment
        # (Bessel's correction divides by N-1 = 0).
        var  = losses_t.var(unbiased=False)
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

    def __init__(self, coral_lambda: float = 1.0, reg_weight: float = 0.5):
        self.coral_lambda = coral_lambda
        self.reg_weight   = reg_weight

    @staticmethod
    def _cov(z: torch.Tensor) -> torch.Tensor:
        """Covariance matrix of (B, D) via torch.cov (uses cuBLAS GEMM on GPU)."""
        # torch.cov expects features as rows, samples as cols: transpose to (D, B)
        return torch.cov(z.T)

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        envs  = list(batches.keys())
        losses, reprs = [], []

        for basin in envs:
            batch = batches[basin]
            out   = model(batch)
            losses.append(
                task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"],    batch["y_direction"],
                    pred_reg   = out.get("pred_intensity_reg"),
                    y_wind_reg = batch.get("y_wind_reg"),
                    y_pres_reg = batch.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
            )
            reprs.append(out["z"])  # (B, D)

        erm = torch.stack(losses).mean()

        # All pairwise covariance distances
        coral_pen = torch.zeros(1, device=erm.device).squeeze()
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
        return x.reshape(x.shape)

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


class DANN(DGMethod, nn.Module):
    """
    Domain-Adversarial Neural Networks (Ganin et al., JMLR 2016).

    Trains a domain (basin) classifier on top of gradient-reversed features.
    Forces the encoder to produce basin-invariant representations.

    GRL schedule: α ramps from 0 → 1 over training, following original paper.

    Inherits nn.Module so that the discriminator's parameters are properly
    tracked by PyTorch (state_dict, to(), etc.).
    """

    def __init__(
        self,
        feature_dim: int = 128,   # matches lightweight model default (final_dim=128)
        n_domains: int = 6,
        dann_lambda: float = 1.0,
        total_steps: int = 10000,
        reg_weight: float = 0.5,
    ):
        nn.Module.__init__(self)
        self.dann_lambda  = dann_lambda
        self.total_steps  = total_steps
        self._step        = 0
        self.reg_weight   = reg_weight
        self.discriminator = DomainDiscriminator(feature_dim, n_domains)

    # to() and parameters() are inherited from nn.Module;
    # the discriminator is a registered sub-module via __init__.

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
                task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"],    batch["y_direction"],
                    pred_reg   = out.get("pred_intensity_reg"),
                    y_wind_reg = batch.get("y_wind_reg"),
                    y_pres_reg = batch.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
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

    def update(self, optimizer, batches, model, step=0, scaler=None, device=None):
        # Discriminator parameters should be included in the joint optimizer
        # (caller creates a joint optimizer with disc params included).
        # _step is managed internally by compute_loss(); do NOT overwrite it.
        optimizer.zero_grad(set_to_none=True)
        env_device = device if device is not None else next(model.parameters()).device
        
        with torch.autocast(device_type="cuda" if env_device.type == "cuda" else "cpu", enabled=(env_device.type == "cuda")):
            loss, metrics = self.compute_loss(batches, model)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        reg_weight: float = 0.5,
    ):
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.reg_weight  = reg_weight

    def _inner_loop(
        self, model: nn.Module, support_batch: dict
    ) -> dict:
        """
        Perform inner-loop adaptation on support set.
        Returns adapted parameter list and buffers (not a new model to avoid deep copies at scale).
        """
        # FOMAML: compute gradients, create adapted params manually
        fast_weights = {n: p.clone() for n, p in model.named_parameters()}
        fast_buffers = {n: b.clone() for n, b in model.named_buffers()}
        
        device = next(model.parameters()).device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = device.type == "cuda"

        for _ in range(self.inner_steps):
            params_and_buffers = {**fast_weights, **fast_buffers}
            # Forward with fast weights and buffers
            with torch.autocast(device_type=device_type, enabled=use_amp):
                out  = self._forward_with_weights(model, support_batch, params_and_buffers)
                loss = task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    support_batch["y_intensity"], support_batch["y_direction"],
                    pred_reg   = out.get("pred_intensity_reg"),
                    y_wind_reg = support_batch.get("y_wind_reg"),
                    y_pres_reg = support_batch.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
            
            # `scaler` not used here because inner gradients are for weights, not standard optimizer
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=not self.first_order,
                                        allow_unused=True)
            fast_weights = {
                n: p - self.inner_lr * (g if g is not None else torch.zeros_like(p))
                for (n, p), g in zip(fast_weights.items(), grads)
            }

        return {**fast_weights, **fast_buffers}

    def _forward_with_weights(self, model: nn.Module, batch: dict, weights: dict) -> dict:
        """
        Forward pass using custom weight dict instead of model.parameters().
        Uses torch.func.functional_call (PyTorch ≥ 2.0).

        Weight tensors are made contiguous before the call.  MAML clones all
        model parameters in _inner_loop via p.clone(), which preserves any
        non-standard memory format (e.g. channels_last set for A100 throughput).
        Passing channels_last weights to functional_call on MPS triggers an
        internal .view() in the MPS kernel that raises:
          "view size is not compatible with input tensor's size and stride"
        Calling .contiguous() strips the channels_last flag and gives standard
        NCHW layout that every backend handles.  The cost is negligible: MAML
        already copies all parameters every inner step.
        """
        contiguous_weights = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }
        return functional_call(model, contiguous_weights, (batch,))

    def compute_loss(self, batches: Dict[str, dict], model: nn.Module):
        """
        For each environment:
          - Split batch into support (first half) and query (second half)
          - Inner-loop adapt on support
          - Compute query loss with adapted weights
        """
        device = next(model.parameters()).device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = device.type == "cuda"
        
        meta_losses = []
        for basin, batch in batches.items():
            n  = batch["data_1d"].shape[0]
            if n < 4:
                continue
            n_s = n // 2

            # Shuffle within the batch so support/query are not temporally biased
            perm = torch.randperm(n, device=batch["data_1d"].device)
            batch_s = {k: (v[perm].contiguous() if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}

            support = {k: (v[:n_s] if isinstance(v, torch.Tensor) else v)
                       for k, v in batch_s.items()}
            query   = {k: (v[n_s:] if isinstance(v, torch.Tensor) else v)
                       for k, v in batch_s.items()}

            fast_w = self._inner_loop(model, support)
            # Query loss with adapted weights
            with torch.autocast(device_type=device_type, enabled=use_amp):
                out_q  = self._forward_with_weights(model, query, fast_w)
                loss_q = task_loss(
                    out_q["logits_intensity"], out_q["logits_direction"],
                    query["y_intensity"], query["y_direction"],
                    pred_reg   = out_q.get("pred_intensity_reg"),
                    y_wind_reg = query.get("y_wind_reg"),
                    y_pres_reg = query.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
            meta_losses.append(loss_q)

        if not meta_losses:
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True), {"maml_meta_loss": 0.0}

        total = torch.stack(meta_losses).mean()
        return total, {"maml_meta_loss": total.item()}


# ── 7. PhysIRM (PROPOSED) ─────────────────────────────────────────────────────

class PhysIRM(DGMethod, nn.Module):
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
        L = ERM(z_phys ⊕ z_env)                                [cls + reg]
          + λ_irm  · Σ_e ||∇_w R^e(w·z_phys)||²    [IRM on physics space, cls only]
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

    Inherits nn.Module so that the phys_predictor's parameters are properly
    tracked by PyTorch (state_dict, to(), etc.).
    """

    def __init__(
        self,
        irm_lambda:    float = 1.0,    # λ_irm: IRM penalty weight on z_phys
        orth_lambda:   float = 0.1,    # λ_orth: orthogonality between sub-spaces
        phys_lambda:   float = 0.5,    # λ_phys: physics grounding loss weight
        phys_dim:      int   = 64,     # dimension of z_phys sub-space
        warmup_steps:  int   = 500,    # ramp IRM penalty from 0 → irm_lambda
        n_phys_feat:   int   = 8,      # number of input physics features
        reg_weight:    float = 0.5,    # weight on intensity regression MSE loss
    ):
        nn.Module.__init__(self)
        self.irm_lambda   = irm_lambda
        self.orth_lambda  = orth_lambda
        self.phys_lambda  = phys_lambda
        self.phys_dim     = phys_dim
        self.warmup_steps = warmup_steps
        self._step        = 0
        self.reg_weight   = reg_weight

        # Physics predictor: maps z_phys → raw physics features
        # Trained jointly to ground z_phys in physical meaning
        self.phys_predictor = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_phys_feat),
        )

    # to() and parameters() are inherited from nn.Module;
    # the phys_predictor is a registered sub-module via __init__.

    def _irm_penalty_phys(
        self, logits_int, logits_dir, y_int, y_dir
    ) -> torch.Tensor:
        """
        IRM gradient penalty applied only on physics sub-space logits.
        Uses classification logits only (not regression), consistent with
        the dummy-scalar-classifier formulation of IRM.
        """
        w = torch.ones(1, requires_grad=True, device=logits_int.device)
        l_cls = (1.0 * F.cross_entropy(logits_int * w, y_int)
                 + 0.5 * F.cross_entropy(logits_dir * w, y_dir))
        g = grad(l_cls, w, create_graph=True)[0]
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
        cross_cov = (z_p.T @ z_e) / (max(n - 1, 1) + 1e-8)
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
            z_phys     = out["z_phys"]      # (B, phys_dim)
            z_phys_raw = out["z_phys_raw"]  # (B, phys_dim)
            z_env      = out["z_env"]       # (B, env_dim)

            # ── ERM loss (classification + regression on full representation) ─
            erm_losses.append(
                task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"],    batch["y_direction"],
                    pred_reg   = out.get("pred_intensity_reg"),
                    y_wind_reg = batch.get("y_wind_reg"),
                    y_pres_reg = batch.get("y_pres_reg"),
                    reg_weight = self.reg_weight,
                )
            )

            # ── IRM penalty on physics sub-space only (classification) ────────
            # We apply IRM on the classification logits derived ONLY from z_phys.
            # This ensures that the IRM penalty (and its gradients) only affects
            # the physics sub-space and the backbone, without leaking into z_env.
            z_phys_only = torch.cat([torch.zeros_like(z_env), z_phys], dim=-1)
            logits_phys_int, logits_phys_dir, _ = model.heads(z_phys_only)

            irm_pens.append(
                self._irm_penalty_phys(
                    logits_phys_int, logits_phys_dir,
                    batch["y_intensity"], batch["y_direction"]
                )
            )

            # ── Orthogonality between sub-spaces ──────────────────────────
            orth_pens.append(self._orthogonality_loss(z_phys, z_env))

            # ── Physics grounding loss ─────────────────────────────────────
            phys_pens.append(
                self._physics_grounding_loss(z_phys_raw, batch["phys_features"])
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

    def update(self, optimizer, batches, model, step=0, scaler=None, device=None):
        # _step is managed internally by compute_loss(); do NOT overwrite it.
        optimizer.zero_grad()
        env_device = device if device is not None else next(model.parameters()).device
        
        with torch.autocast(device_type="cuda" if env_device.type == "cuda" else "cpu", enabled=(env_device.type == "cuda")):
            loss, metrics = self.compute_loss(batches, model)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.phys_predictor.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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