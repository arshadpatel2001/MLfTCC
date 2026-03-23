"""
Metrics for Basin Generalization Evaluation
============================================
Implements all evaluation metrics for our NeurIPS/ICLR submission:

  Standard:
    - Per-class accuracy, weighted-F1 for intensity classification
    - Accuracy for direction classification
    - Mean Absolute Error on wind speed (regression equivalent)

  Novel (proposed in this paper):
    - Basin Transfer Gap (BTG): measures cross-basin generalisation
    - Rapid Intensification (RI) Skill Score: critical safety metric
    - Basin-Normalized Transfer Efficiency (BNTE): normalised by in-basin perf

  DG-specific:
    - Domain gap estimation via CORAL distance
    - Per-basin performance breakdown table (for paper Table 1)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import torch
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

# Intensity class mapping
# 0: Rapid Weakening (ΔWind ≤ -30 kt/24h)
# 1: Weakening       (-30 < ΔWind ≤ -10 kt/24h)
# 2: Steady          (-10 < ΔWind ≤ +10 kt/24h)
# 3: Intensification (+10 < ΔWind ≤ +30 kt/24h)
# 4: Rapid Intensification (ΔWind > +30 kt/24h)  ← the safety-critical class
INTENSITY_CLASSES = {
    0: "Rapid Weakening",
    1: "Weakening",
    2: "Steady",
    3: "Intensification",
    4: "Rapid Intensification",
}
RI_CLASS = 4  # Rapid Intensification — the class that matters most for warnings

DIRECTION_CLASSES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class BasinResult:
    """Per-basin evaluation result."""
    basin:        str
    n_samples:    int
    accuracy_intensity: float  # intensity accuracy
    precision_intensity: float
    recall_intensity: float
    f1_intensity:       float  # weighted F1 for intensity
    accuracy_direction: float  # direction accuracy
    precision_direction: float
    recall_direction: float
    f1_direction: float
    rapid_intensification_recall:    float  # recall on RI class (class 4)
    rapid_intensification_precision: float
    rapid_intensification_f1:        float


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

    # Zero-shot target performance
    target_accuracy_intensity:   float
    target_precision_intensity: float
    target_recall_intensity: float
    target_f1_intensity: float
    target_accuracy_direction:   float
    target_precision_direction: float
    target_recall_direction: float
    target_f1_direction: float
    target_rapid_intensification_f1:     float

    # Proposed metrics
    btg:   float  # Basin Transfer Gap
    bnte:  float  # Basin-Normalized Transfer Efficiency

    # Per-basin breakdown
    per_basin: Dict[str, BasinResult] = field(default_factory=dict)


# ── Core metric functions ─────────────────────────────────────────────────────

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    if len(preds) == 0: return 0.0
    return float((preds == labels).mean())


def weighted_metrics(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> Tuple[float, float, float]:
    """Weighted Precision, Recall, and F1 score (weighted by class support / frequency)."""
    from collections import Counter
    if len(labels) == 0: return 0.0, 0.0, 0.0
    support = Counter(labels)
    total   = len(labels)

    precs, recs, f1s, weights = [], [], [], []
    for c in range(n_classes):
        tp = float(((preds == c) & (labels == c)).sum())
        fp = float(((preds == c) & (labels != c)).sum())
        fn = float(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        weights.append(support[c] / total)

    return float(np.dot(precs, weights)), float(np.dot(recs, weights)), float(np.dot(f1s, weights))


def ri_metrics(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Precision, Recall, F1 for Rapid Intensification (class 4).
    This is the safety-critical class — false negatives (missed RI) cost lives.
    Report this prominently in the paper.
    """
    tp = ((preds == RI_CLASS) & (labels == RI_CLASS)).sum()
    fp = ((preds == RI_CLASS) & (labels != RI_CLASS)).sum()
    fn = ((preds != RI_CLASS) & (labels == RI_CLASS)).sum()
    
    # Handle edge cases where there are no RI samples or predictions
    # This prevents mathematically meaningless results (0/1e-8) when RI is absent.
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1   = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    
    return prec, rec, f1


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

    def __init__(self, basin_name: str = "unknown"):
        self.basin_name = basin_name
        self._preds_int  : List[int] = []
        self._labels_int : List[int] = []
        self._preds_dir  : List[int] = []
        self._labels_dir : List[int] = []
        self._features   : List[torch.Tensor] = []

    def update(self, batch: dict, out: dict):
        self._preds_int.extend(
            out["logits_intensity"].argmax(-1).cpu().numpy().tolist()
        )
        self._labels_int.extend(batch["y_intensity"].cpu().numpy().tolist())
        self._preds_dir.extend(
            out["logits_direction"].argmax(-1).cpu().numpy().tolist()
        )
        self._labels_dir.extend(batch["y_direction"].cpu().numpy().tolist())
        if "z" in out:
            self._features.append(out["z"].detach().cpu())

    def compute(self) -> BasinResult:
        if not self._preds_int:
            # Evaluator was never updated — return a zero-filled result rather
            # than crashing with NaN from numpy mean on an empty array.
            return BasinResult(
                basin=self.basin_name, n_samples=0,
                accuracy_intensity=0.0, precision_intensity=0.0, recall_intensity=0.0, f1_intensity=0.0,
                accuracy_direction=0.0, precision_direction=0.0, recall_direction=0.0, f1_direction=0.0,
                rapid_intensification_recall=0.0, rapid_intensification_precision=0.0, rapid_intensification_f1=0.0,
            )
        pi = np.array(self._preds_int)
        li = np.array(self._labels_int)
        pd = np.array(self._preds_dir)
        ld = np.array(self._labels_dir)

        ri_p, ri_r, ri_f = ri_metrics(pi, li)
        pi_p, pi_r, pi_f = weighted_metrics(pi, li, n_classes=5)
        pd_p, pd_r, pd_f = weighted_metrics(pd, ld, n_classes=8)

        return BasinResult(
            basin        = self.basin_name,
            n_samples    = len(pi),
            accuracy_intensity = accuracy(pi, li),
            precision_intensity = pi_p,
            recall_intensity = pi_r,
            f1_intensity       = pi_f,
            accuracy_direction = accuracy(pd, ld),
            precision_direction = pd_p,
            recall_direction = pd_r,
            f1_direction       = pd_f,
            rapid_intensification_recall    = ri_r,
            rapid_intensification_precision = ri_p,
            rapid_intensification_f1        = ri_f,
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
                for batch in tqdm(loader, desc=f"Eval Source {basin}", dynamic_ncols=True, leave=False):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
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
            for batch in tqdm(target_loader, desc=f"Eval Target {target_basin}", dynamic_ncols=True, leave=False):
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
            target_rapid_intensification_f1    = target_r.rapid_intensification_f1,
            btg             = btg,
            bnte            = bnte,
            per_basin       = per_basin_results,
        )
        self.results.append(result)
        return result

    def print_table(self, results: Optional[List[TransferResult]] = None):
        """
        Print a LaTeX-style results table (for copy-paste into the paper).
        Columns: Method | Target Basin | Acc(Int) | Acc(Dir) | RI-F1 | BTG | BNTE
        """
        rs = results or self.results
        header = (
            f"{'Method':<12} {'Target':<6} {'Acc-Int':>8} {'Acc-Dir':>8} "
            f"{'RI-F1':>7} {'BTG':>8} {'BNTE':>8}"
        )
        sep = "─" * len(header)
        print(sep)
        print(header)
        print(sep)
        for r in rs:
            print(
                f"{r.method:<12} {r.target_basin:<6} "
                f"{r.target_accuracy_intensity:>8.3f} {r.target_accuracy_direction:>8.3f} "
                f"{r.target_rapid_intensification_f1:>7.3f} {r.btg:>8.3f} {r.bnte:>8.3f}"
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
                "target rapid intensification f1":   r.target_rapid_intensification_f1,
                "btg":     r.btg,
                "bnte":    r.bnte,
            }
            for r in self.results
        ]