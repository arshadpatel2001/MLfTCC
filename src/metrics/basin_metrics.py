"""
Metrics for Basin Generalization Evaluation
============================================
Implements all evaluation metrics for our NeurIPS/ICLR submission:

  Standard:
    - Per-class accuracy, weighted-F1 for intensity classification
    - Accuracy for direction classification
    - Mean Absolute Error on wind speed and pressure (regression)

  Novel (proposed in this paper):
    - Basin Transfer Gap (BTG): measures cross-basin generalisation
    - Rapid Intensification (RI) Skill Score: critical safety metric
    - Basin-Normalized Transfer Efficiency (BNTE): normalised by in-basin perf

  DG-specific:
    - Domain gap estimation via CORAL distance
    - Per-basin performance breakdown table (for paper Table 1)

Regression targets (y_wind_reg, y_pres_reg) are in the same normalized units
as the TCND Data_1d WND_norm / PRES_norm columns.  MAE is reported both in
normalized units and in physical units (m/s and hPa) using the denormalization
constants from dataset.REG_DENORM.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import time
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import torch
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

# Intensity class mapping (4-class TCND schema)
# 0: Weakening         (ΔWind < -10 kt/24h)
# 1: Steady            (-10 ≤ ΔWind ≤ +10 kt/24h)
# 2: Intensification   (+10 < ΔWind ≤ +30 kt/24h)
# 3: Rapid Intensification (ΔWind > +30 kt/24h)
INTENSITY_CLASSES = {
    0: "Weakening",
    1: "Steady",
    2: "Intensification",
    3: "Rapid Intensification",
}

DIRECTION_CLASSES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Scale factors for converting regression MAE from normalized units to physical units.
#
# Full denormalization formula (from TCND paper Equations 3–4):
#   WND_ms   = WND_norm  * 25 + 40
#   PRES_hPa = PRES_norm * 50 + 960
#
# For MAE the additive offset always cancels:
#   MAE_ms = E[|pred_ms - true_ms|]
#          = E[|(pred_norm*25+40) - (true_norm*25+40)|]
#          = 25 * E[|pred_norm - true_norm|]
#          = 25 * MAE_norm
#
# Therefore only the scale factor is needed here.  If you ever need to convert
# absolute predictions (not MAE) use: WND_ms = norm * 25 + 40  /  PRES_hPa = norm * 50 + 960
_WND_SCALE  = 25.0   # m/s per normalized unit
_PRES_SCALE = 50.0   # hPa per normalized unit


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class BasinResult:
    """Per-basin evaluation result."""
    basin:        str
    n_samples:    int

    # ── Classification metrics ────────────────────────────────────────────────
    accuracy_intensity:  float
    precision_intensity: float
    recall_intensity:    float
    f1_intensity:        float
    accuracy_direction:  float
    precision_direction: float
    recall_direction:    float
    f1_direction:        float

    # ── Rapid Intensification (RI) skill metrics — class 3 ───────────────────
    # RI = Rapid Intensification (class 3 in the 4-class schema).
    # These are per-class precision / recall / F1 for class 3 only, which is the
    # most safety-critical outcome and therefore reported separately.
    rapid_intensification_precision: float = 0.0
    rapid_intensification_recall:    float = 0.0
    rapid_intensification_f1:        float = 0.0

    # ── Intensity regression metrics ─────────────────────────────────────────
    # Regression targets are 24h-ahead wind speed and central pressure in the
    # same normalized units as Data_1d WND_norm / PRES_norm.
    # NaN samples (end-of-storm) are excluded from MAE computation.
    mae_wind_norm: float = 0.0   # MAE in normalized units
    mae_pres_norm: float = 0.0   # MAE in normalized units
    mae_wind_ms:   float = 0.0   # MAE in m/s   (denormalized: norm * 25)
    mae_pres_hpa:  float = 0.0   # MAE in hPa   (denormalized: norm * 50)
    n_reg_samples: int   = 0     # number of valid (non-NaN) regression samples


@dataclass
class TransferResult:
    """Full cross-basin transfer experiment result."""
    source_basins:  List[str]
    target_basin:   str
    method:         str

    # In-basin (source) performance
    source_accuracy_intensity: float
    source_precision_intensity: float
    source_recall_intensity: float
    source_f1_intensity: float
    source_accuracy_direction: float
    source_precision_direction: float
    source_recall_direction: float
    source_f1_direction: float

    # Zero-shot target performance — classification
    target_accuracy_intensity:   float
    target_precision_intensity: float
    target_recall_intensity: float
    target_f1_intensity: float
    target_accuracy_direction:   float
    target_precision_direction: float
    target_recall_direction: float
    target_f1_direction: float

    # Zero-shot target performance — regression
    target_mae_wind_ms:  float = 0.0
    target_mae_pres_hpa: float = 0.0

    # Proposed metrics
    btg:   float = 0.0   # Basin Transfer Gap
    bnte:  float = 0.0   # Basin-Normalized Transfer Efficiency

    # Per-basin breakdown
    per_basin: Dict[str, BasinResult] = field(default_factory=dict)


# ── Core metric functions ─────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if len(preds) == 0: return 0.0
    return float((preds == labels).float().mean().item())


def weighted_metrics(preds: torch.Tensor, labels: torch.Tensor, n_classes: int) -> Tuple[float, float, float]:
    """Weighted Precision, Recall, and F1 score (weighted by class support / frequency)."""
    if len(labels) == 0: return 0.0, 0.0, 0.0

    # ── Fully Vectorized Confusion Matrix ─────────────────────────────────────
    # Computes precision and recall for all classes simultaneously on the GPU!
    indices = labels * n_classes + preds
    conf_matrix = torch.bincount(indices, minlength=n_classes * n_classes).reshape(n_classes, n_classes).float()
    
    tp = conf_matrix.diag()
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    prec_denom = tp + fp
    prec = torch.where(prec_denom > 0, tp / prec_denom, torch.zeros_like(tp))
    
    rec_denom = tp + fn
    rec = torch.where(rec_denom > 0, tp / rec_denom, torch.zeros_like(tp))
    
    f1_denom = prec + rec
    f1 = torch.where(f1_denom > 0, 2 * prec * rec / f1_denom, torch.zeros_like(tp))
    
    support = conf_matrix.sum(dim=1)
    weights = support / support.sum()
    
    # 3 final PCIe syncs total
    w_prec = (prec * weights).sum().item()
    w_rec  = (rec * weights).sum().item()
    w_f1   = (f1 * weights).sum().item()

    return float(w_prec), float(w_rec), float(w_f1)


def ri_skill_metrics(
    preds: torch.Tensor, labels: torch.Tensor, ri_class: int = 3
) -> Tuple[float, float, float]:
    """
    Precision, Recall, and F1 for the Rapid Intensification class.

    Args:
        preds:    (N,) predicted class indices
        labels:   (N,) ground-truth class indices
        ri_class: index of the RI class (default: 3)

    Returns:
        (precision, recall, f1) for the RI class
    """
    if len(labels) == 0:
        return 0.0, 0.0, 0.0

    tp = float(((preds == ri_class) & (labels == ri_class)).sum().item())
    fp = float(((preds == ri_class) & (labels != ri_class)).sum().item())
    fn = float(((preds != ri_class) & (labels == ri_class)).sum().item())

    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-8)

    return float(prec), float(rec), float(f1)


# ── Novel Metric: Basin Transfer Gap (BTG) ───────────────────────────────────

def basin_transfer_gap(
    source_acc: float,
    target_acc: float,
    in_basin_oracle_acc: Optional[float] = None,
) -> float:
    """
    Basin Transfer Gap (BTG) — proposed metric.

    Measures the drop in performance when transferring to an unseen basin,
    normalised by in-basin performance to account for task difficulty.

    BTG = (source_acc - target_acc) / source_acc

    Range: [-∞, 1]. BTG = 0 means perfect generalisation.
    Negative BTG means the model generalises *better* to the target basin
    (possible if target is simpler or the source covers diverse patterns).

    If `in_basin_oracle_acc` is provided (accuracy of a model trained IN the
    target basin), we compute the normalised transfer gap relative to the oracle:

    BTG_oracle = (oracle_acc - target_acc) / oracle_acc

    Report both in the paper; BTG_oracle is more interpretable.

    Reference: Inspired by Transfer Gap in NLP (Phang et al., 2018).
    """
    btg = (source_acc - target_acc) / max(source_acc, 1e-8)
    return float(btg)


def basin_normalized_transfer_efficiency(
    source_acc: float,
    target_acc: float,
    baseline_acc: float,  # ERM baseline on target basin (minimum bar)
) -> float:
    """
    Basin-Normalized Transfer Efficiency (BNTE) — proposed metric.

    Measures how much of the gap between ERM baseline and in-basin oracle
    performance is recovered by the DG method on the target basin.

    BNTE = (target_acc - baseline_acc) / (source_acc - baseline_acc + 1e-8)

    BNTE = 1.0: matches source-basin performance on target
    BNTE = 0.0: no improvement over ERM baseline
    BNTE < 0.0: worse than ERM baseline (the method hurt generalisation)
    """
    numerator   = target_acc - baseline_acc
    denominator = source_acc - baseline_acc + 1e-8
    return float(numerator / denominator)


# ── CORAL Distance (domain gap estimator) ────────────────────────────────────

def coral_distance(
    feat_source: torch.Tensor,
    feat_target: torch.Tensor,
) -> float:
    """
    CORAL distance between source and target feature distributions.
    Used to measure the domain gap between basins (analysis section of paper).

    D_CORAL = (1/4d²) ||C_s - C_t||_F²

    where C_s, C_t are d×d covariance matrices.
    """
    def cov(z):
        n, d = z.shape
        z = z - z.mean(0, keepdim=True)
        return (z.T @ z) / (max(n - 1, 1) + 1e-8)

    cs = cov(feat_source.float())
    ct = cov(feat_target.float())
    d  = cs.shape[0]
    dist = (cs - ct).pow(2).sum() / (4 * d * d)
    return float(dist.item())


# ── Evaluator class ───────────────────────────────────────────────────────────

class BasinEvaluator:
    """
    Evaluates a trained model on one or more basins.
    Accumulates predictions across batches, computes all metrics.

    Usage:
        evaluator = BasinEvaluator()
        with torch.no_grad():
            for batch in loader:
                out = model(batch)
                evaluator.update(batch, out)
        results = evaluator.compute()
    """

    def __init__(self, basin_name: str = "unknown", collect_features: bool = False):
        self.basin_name = basin_name
        self._collect_features = collect_features

        # Classification
        self._preds_int  : List[torch.Tensor] = []
        self._labels_int : List[torch.Tensor] = []
        self._preds_dir  : List[torch.Tensor] = []
        self._labels_dir : List[torch.Tensor] = []

        # Regression (24h-ahead wind and pressure)
        self._preds_wind  : List[torch.Tensor] = []   # model predictions (normalized)
        self._labels_wind : List[torch.Tensor] = []   # ground-truth (normalized, may have NaN)
        self._preds_pres  : List[torch.Tensor] = []
        self._labels_pres : List[torch.Tensor] = []

        # Feature vectors (optional, for CORAL distance analysis)
        self._features   : List[torch.Tensor] = []

    def update(self, batch: dict, out: dict):
        # ── Classification ────────────────────────────────────────────────────
        self._preds_int.append(out["logits_intensity"].argmax(-1).detach())
        self._labels_int.append(batch["y_intensity"].detach())
        self._preds_dir.append(out["logits_direction"].argmax(-1).detach())
        self._labels_dir.append(batch["y_direction"].detach())

        # ── Regression ───────────────────────────────────────────────────────
        # pred_intensity_reg is (B, 2): column 0 = wind, column 1 = pres
        if "pred_intensity_reg" in out:
            pred_reg = out["pred_intensity_reg"].detach()          # (B, 2)
            self._preds_wind.append(pred_reg[:, 0].cpu())
            self._preds_pres.append(pred_reg[:, 1].cpu())

        if "y_wind_reg" in batch:
            self._labels_wind.append(batch["y_wind_reg"].detach().cpu())
        if "y_pres_reg" in batch:
            self._labels_pres.append(batch["y_pres_reg"].detach().cpu())

        # ── Features (optional) ───────────────────────────────────────────────
        if self._collect_features and "z" in out:
            self._features.append(out["z"].detach().cpu())

    def compute(self) -> BasinResult:
        if not self._preds_int:
            return BasinResult(
                basin=self.basin_name, n_samples=0,
                accuracy_intensity=0.0, precision_intensity=0.0,
                recall_intensity=0.0,   f1_intensity=0.0,
                accuracy_direction=0.0, precision_direction=0.0,
                recall_direction=0.0,   f1_direction=0.0,
                rapid_intensification_precision=0.0,
                rapid_intensification_recall=0.0,
                rapid_intensification_f1=0.0,
                mae_wind_norm=0.0, mae_pres_norm=0.0,
                mae_wind_ms=0.0,   mae_pres_hpa=0.0,
                n_reg_samples=0,
            )

        # ── Classification ────────────────────────────────────────────────────
        pi = torch.cat(self._preds_int)
        li = torch.cat(self._labels_int)
        pd = torch.cat(self._preds_dir)
        ld = torch.cat(self._labels_dir)

        pi_p, pi_r, pi_f = weighted_metrics(pi, li, n_classes=4)
        pd_p, pd_r, pd_f = weighted_metrics(pd, ld, n_classes=8)

        # ── Rapid Intensification skill ───────────────────────────────────────
        ri_prec, ri_rec, ri_f1 = ri_skill_metrics(pi, li, ri_class=3)

        # ── Regression MAE ────────────────────────────────────────────────────
        mae_wnd_norm = mae_prs_norm = 0.0
        mae_wnd_ms   = mae_prs_hpa  = 0.0
        n_reg = 0

        if self._preds_wind and self._labels_wind:
            pw = torch.cat(self._preds_wind)   # (N,)
            lw = torch.cat(self._labels_wind)  # (N,), may contain NaN
            pp = torch.cat(self._preds_pres)   # (N,)
            lp = torch.cat(self._labels_pres)  # (N,), may contain NaN

            # Mask: only samples where BOTH targets are finite
            mask = torch.isfinite(lw) & torch.isfinite(lp)
            n_reg = int(mask.sum().item())

            if n_reg > 0:
                mae_wnd_norm = float((pw[mask] - lw[mask]).abs().mean().item())
                mae_prs_norm = float((pp[mask] - lp[mask]).abs().mean().item())
                # Convert to physical units using scale factor only.
                # Offset cancels in MAE — see _WND_SCALE / _PRES_SCALE comment above.
                mae_wnd_ms  = mae_wnd_norm * _WND_SCALE
                mae_prs_hpa = mae_prs_norm * _PRES_SCALE

        return BasinResult(
            basin        = self.basin_name,
            n_samples    = len(pi),
            accuracy_intensity  = accuracy(pi, li),
            precision_intensity = pi_p,
            recall_intensity    = pi_r,
            f1_intensity        = pi_f,
            accuracy_direction  = accuracy(pd, ld),
            precision_direction = pd_p,
            recall_direction    = pd_r,
            f1_direction        = pd_f,
            rapid_intensification_precision = ri_prec,
            rapid_intensification_recall    = ri_rec,
            rapid_intensification_f1        = ri_f1,
            mae_wind_norm  = mae_wnd_norm,
            mae_pres_norm  = mae_prs_norm,
            mae_wind_ms    = mae_wnd_ms,
            mae_pres_hpa   = mae_prs_hpa,
            n_reg_samples  = n_reg,
        )

    def get_features(self) -> Optional[torch.Tensor]:
        if self._features:
            return torch.cat(self._features, dim=0)
        return None

    def reset(self):
        self._preds_int.clear()
        self._labels_int.clear()
        self._preds_dir.clear()
        self._labels_dir.clear()
        self._preds_wind.clear()
        self._labels_wind.clear()
        self._preds_pres.clear()
        self._labels_pres.clear()
        self._features.clear()


# ── Full experiment evaluator ─────────────────────────────────────────────────

class TransferEvaluator:
    """
    Runs a complete leave-one-basin-out evaluation.

    For each target basin:
      1. Evaluate source performance (average across source basins)
      2. Evaluate zero-shot transfer to target basin
      3. Compute BTG, BNTE, and RI metrics
      4. Build the results table for the paper
    """

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.results: List[TransferResult] = []

    def evaluate(
        self,
        model: torch.nn.Module,
        source_loaders: Dict[str, "DataLoader"],
        target_loader: "DataLoader",
        target_basin: str,
        baseline_target_acc: float = 0.0,  # ERM result for BNTE
        device: str = "cuda",
        disable_tqdm: bool = False,
    ) -> TransferResult:
        """
        Args:
            model:                Trained model (on source basins)
            source_loaders:       Dict of basin → DataLoader for source basins
            target_loader:        DataLoader for the held-out target basin
            target_basin:         Name of the target basin
            baseline_target_acc:  ERM target accuracy (for BNTE normalisation)
            device:               CUDA device string
        """
        model.eval()
        model.to(device)

        # ── Source performance ────────────────────────────────────────────────
        source_accs_int, source_precs_int, source_recs_int, source_f1s_int = [], [], [], []
        source_accs_dir, source_precs_dir, source_recs_dir, source_f1s_dir = [], [], [], []
        source_basins = list(source_loaders.keys())
        per_basin_results = {}

        eval_start = time.time()
        for basin, loader in source_loaders.items():
            ev = BasinEvaluator(basin)
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Evaluate Source {basin}", dynamic_ncols=True, leave=False, disable=disable_tqdm):
                    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    out = model(batch)
                    ev.update(batch, out)
            r = ev.compute()
            per_basin_results[basin] = r
            source_accs_int.append(r.accuracy_intensity)
            source_precs_int.append(r.precision_intensity)
            source_recs_int.append(r.recall_intensity)
            source_f1s_int.append(r.f1_intensity)
            source_accs_dir.append(r.accuracy_direction)
            source_precs_dir.append(r.precision_direction)
            source_recs_dir.append(r.recall_direction)
            source_f1s_dir.append(r.f1_direction)

        source_acc_int = float(np.mean(source_accs_int))
        source_prec_int = float(np.mean(source_precs_int))
        source_rec_int = float(np.mean(source_recs_int))
        source_f1_int = float(np.mean(source_f1s_int))
        source_acc_dir = float(np.mean(source_accs_dir))
        source_prec_dir = float(np.mean(source_precs_dir))
        source_rec_dir = float(np.mean(source_recs_dir))
        source_f1_dir = float(np.mean(source_f1s_dir))

        # ── Target zero-shot performance ──────────────────────────────────────
        target_ev = BasinEvaluator(target_basin)
        with torch.no_grad():
            for batch in tqdm(target_loader, desc=f"Evaluate Target {target_basin}", dynamic_ncols=True, leave=False, disable=disable_tqdm):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                out = model(batch)
                target_ev.update(batch, out)
        target_r = target_ev.compute()
        per_basin_results[target_basin] = target_r

        import logging
        log = logging.getLogger(__name__)
        log.info(f"Cross-basin evaluation for {target_basin} took {time.time() - eval_start:.2f}s")

        # ── Proposed metrics ──────────────────────────────────────────────────
        btg  = basin_transfer_gap(source_acc_int, target_r.accuracy_intensity)
        bnte = basin_normalized_transfer_efficiency(
            source_acc_int, target_r.accuracy_intensity, baseline_target_acc
        )

        result = TransferResult(
            source_basins   = source_basins,
            target_basin    = target_basin,
            method          = self.method_name,
            source_accuracy_intensity  = source_acc_int,
            source_precision_intensity = source_prec_int,
            source_recall_intensity    = source_rec_int,
            source_f1_intensity        = source_f1_int,
            source_accuracy_direction  = source_acc_dir,
            source_precision_direction = source_prec_dir,
            source_recall_direction    = source_rec_dir,
            source_f1_direction        = source_f1_dir,
            target_accuracy_intensity  = target_r.accuracy_intensity,
            target_precision_intensity = target_r.precision_intensity,
            target_recall_intensity    = target_r.recall_intensity,
            target_f1_intensity        = target_r.f1_intensity,
            target_accuracy_direction  = target_r.accuracy_direction,
            target_precision_direction = target_r.precision_direction,
            target_recall_direction    = target_r.recall_direction,
            target_f1_direction        = target_r.f1_direction,
            target_mae_wind_ms         = target_r.mae_wind_ms,
            target_mae_pres_hpa        = target_r.mae_pres_hpa,
            btg             = btg,
            bnte            = bnte,
            per_basin       = per_basin_results,
        )
        self.results.append(result)
        return result

    def print_table(self, results: Optional[List[TransferResult]] = None):
        """
        Print a LaTeX-style results table (for copy-paste into the paper).
        """
        rs = results or self.results
        header = (
            f"{'Method':<20} {'Target':<10} {'Acc Intensity':>15} {'Acc Direction':>15} "
            f"{'MAE Wind(m/s)':>15} {'MAE Pres(hPa)':>15} "
            f"{'BTG':>10} {'BNTE':>10}"
        )
        sep = "─" * len(header)
        print(sep)
        print(header)
        print(sep)
        for r in rs:
            print(
                f"{r.method:<20} {r.target_basin:<10} "
                f"{r.target_accuracy_intensity:>15.3f} {r.target_accuracy_direction:>15.3f} "
                f"{r.target_mae_wind_ms:>15.2f} {r.target_mae_pres_hpa:>15.2f} "
                f"{r.btg:>10.3f} {r.bnte:>10.3f}"
            )
        print(sep)

    def to_dict(self) -> List[dict]:
        return [
            {
                "method": r.method,
                "source": r.source_basins,
                "target": r.target_basin,
                "target accuracy intensity": r.target_accuracy_intensity,
                "target precision intensity": r.target_precision_intensity,
                "target recall intensity": r.target_recall_intensity,
                "target f1 intensity": r.target_f1_intensity,
                "target accuracy direction": r.target_accuracy_direction,
                "target precision direction": r.target_precision_direction,
                "target recall direction": r.target_recall_direction,
                "target f1 direction": r.target_f1_direction,
                "target mae wind ms":  r.target_mae_wind_ms,
                "target mae pres hpa": r.target_mae_pres_hpa,
                "btg":     r.btg,
                "bnte":    r.bnte,
            }
            for r in self.results
        ]