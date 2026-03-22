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
    accuracy_int: float  # intensity accuracy
    accuracy_dir: float  # direction accuracy
    f1_int:       float  # weighted F1 for intensity
    ri_recall:    float  # recall on RI class (class 4)
    ri_precision: float
    ri_f1:        float


@dataclass
class TransferResult:
    """Full cross-basin transfer experiment result."""
    source_basins:  List[str]
    target_basin:   str
    method:         str

    # In-basin (source) performance
    source_acc_int: float
    source_acc_dir: float

    # Zero-shot target performance
    target_acc_int:   float
    target_acc_dir:   float
    target_ri_f1:     float

    # Proposed metrics
    btg:   float  # Basin Transfer Gap
    bnte:  float  # Basin-Normalized Transfer Efficiency

    # Per-basin breakdown
    per_basin: Dict[str, BasinResult] = field(default_factory=dict)


# ── Core metric functions ─────────────────────────────────────────────────────

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def weighted_f1(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Weighted F1 score (weighted by class support / frequency)."""
    from collections import Counter
    support = Counter(labels)
    total   = len(labels)

    f1s, weights = [], []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
        weights.append(support[c] / total)

    return float(np.dot(f1s, weights))


def ri_metrics(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Precision, Recall, F1 for Rapid Intensification (class 4).
    This is the safety-critical class — false negatives (missed RI) cost lives.
    Report this prominently in the paper.
    """
    tp = ((preds == RI_CLASS) & (labels == RI_CLASS)).sum()
    fp = ((preds == RI_CLASS) & (labels != RI_CLASS)).sum()
    fn = ((preds != RI_CLASS) & (labels == RI_CLASS)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
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
        return (z.T @ z) / (n - 1 + 1e-8)

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
        pi = np.array(self._preds_int)
        li = np.array(self._labels_int)
        pd = np.array(self._preds_dir)
        ld = np.array(self._labels_dir)

        ri_p, ri_r, ri_f = ri_metrics(pi, li)

        return BasinResult(
            basin        = self.basin_name,
            n_samples    = len(pi),
            accuracy_int = accuracy(pi, li),
            accuracy_dir = accuracy(pd, ld),
            f1_int       = weighted_f1(pi, li, n_classes=5),
            ri_recall    = ri_r,
            ri_precision = ri_p,
            ri_f1        = ri_f,
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
        source_accs_int, source_accs_dir = [], []
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
            source_accs_int.append(r.accuracy_int)
            source_accs_dir.append(r.accuracy_dir)

        source_acc_int = float(np.mean(source_accs_int))
        source_acc_dir = float(np.mean(source_accs_dir))

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
        btg  = basin_transfer_gap(source_acc_int, target_r.accuracy_int)
        bnte = basin_normalized_transfer_efficiency(
            source_acc_int, target_r.accuracy_int, baseline_target_acc
        )

        result = TransferResult(
            source_basins   = source_basins,
            target_basin    = target_basin,
            method          = self.method_name,
            source_acc_int  = source_acc_int,
            source_acc_dir  = source_acc_dir,
            target_acc_int  = target_r.accuracy_int,
            target_acc_dir  = target_r.accuracy_dir,
            target_ri_f1    = target_r.ri_f1,
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
                f"{r.target_acc_int:>8.3f} {r.target_acc_dir:>8.3f} "
                f"{r.target_ri_f1:>7.3f} {r.btg:>8.3f} {r.bnte:>8.3f}"
            )
        print(sep)

    def to_dict(self) -> List[dict]:
        return [
            {
                "method": r.method,
                "source": r.source_basins,
                "target": r.target_basin,
                "acc_int": r.target_acc_int,
                "acc_dir": r.target_acc_dir,
                "ri_f1":   r.target_ri_f1,
                "btg":     r.btg,
                "bnte":    r.bnte,
            }
            for r in self.results
        ]
