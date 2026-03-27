"""
Evaluation metrics for TC track and intensity forecasting.

All metrics operate on de-normalized (physical) coordinates.
"""

import numpy as np
import torch
from typing import Dict, List

from src.data.utils import haversine_km, denorm_lat, denorm_lon, denorm_pres, denorm_wind


# ---------------------------------------------------------------------------
# Core track metrics (operate on normalized coords → convert internally)
# ---------------------------------------------------------------------------

def track_displacement_km(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
) -> np.ndarray:
    """
    Compute per-step haversine distance in km between predicted and true tracks.

    Parameters
    ----------
    pred_norm : (pred_len, 2) — predicted [lon_norm, lat_norm]
    true_norm : (pred_len, 2) — ground truth [lon_norm, lat_norm]

    Returns
    -------
    dist_km : (pred_len,) distance at each forecast step
    """
    pred_lon = denorm_lon(pred_norm[:, 0])
    pred_lat = denorm_lat(pred_norm[:, 1])
    true_lon = denorm_lon(true_norm[:, 0])
    true_lat = denorm_lat(true_norm[:, 1])
    return haversine_km(pred_lat, pred_lon, true_lat, true_lon)


def ade_km(pred_norm: np.ndarray, true_norm: np.ndarray) -> float:
    """Average Displacement Error across all prediction steps (km)."""
    return float(track_displacement_km(pred_norm, true_norm).mean())


def fde_km(pred_norm: np.ndarray, true_norm: np.ndarray) -> float:
    """Final Displacement Error at last prediction step (km)."""
    return float(track_displacement_km(pred_norm, true_norm)[-1])


def horizon_errors_km(
    pred_norm: np.ndarray, true_norm: np.ndarray, step_hours: int = 6
) -> Dict[str, float]:
    """
    Return dict of displacement errors at each forecast horizon.
    Keys are like '6h', '12h', '18h', '24h'.
    """
    dists = track_displacement_km(pred_norm, true_norm)
    return {f"{(i+1)*step_hours}h": float(d) for i, d in enumerate(dists)}


# ---------------------------------------------------------------------------
# Intensity metrics
# ---------------------------------------------------------------------------

def intensity_mae(pred_norm: np.ndarray, true_norm: np.ndarray) -> Dict[str, float]:
    """
    MAE for pressure and wind predictions.

    pred_norm / true_norm : (pred_len, 4) [lon, lat, pres, wind] normalized
    Returns dict with keys 'pres_hpa' and 'wind_ms'.
    """
    pred_pres = denorm_pres(pred_norm[:, 2])
    true_pres = denorm_pres(true_norm[:, 2])
    pred_wind = denorm_wind(pred_norm[:, 3])
    true_wind = denorm_wind(true_norm[:, 3])
    return {
        "pres_hpa": float(np.abs(pred_pres - true_pres).mean()),
        "wind_ms":  float(np.abs(pred_wind - true_wind).mean()),
    }


# ---------------------------------------------------------------------------
# Batch evaluation (works with torch tensors or numpy arrays)
# ---------------------------------------------------------------------------

def evaluate_batch(
    pred_rel: torch.Tensor,
    true_pred: torch.Tensor,
    obs_last_pos: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions in physical units.

    Parameters
    ----------
    pred_rel     : (batch, pred_len, 2) predicted relative displacements (normalized)
    true_pred    : (batch, pred_len, 2) ground truth absolute positions (normalized)
    obs_last_pos : (batch, 2) last observed absolute position (normalized)

    Returns dict with: ade_km, fde_km, 6h, 12h, 18h, 24h errors
    """
    from src.data.utils import rel_to_abs_torch
    pred_abs = rel_to_abs_torch(obs_last_pos, pred_rel)  # (batch, pred_len, 2)

    pred_np = pred_abs.detach().cpu().numpy()   # (batch, pred_len, 2)
    true_np = true_pred.detach().cpu().numpy()  # (batch, pred_len, 2)

    batch_ade, batch_fde = [], []
    horizon_sums = {}
    for b in range(pred_np.shape[0]):
        dists = track_displacement_km(pred_np[b], true_np[b])
        batch_ade.append(dists.mean())
        batch_fde.append(dists[-1])
        for i, d in enumerate(dists):
            key = f"{(i+1)*6}h"
            horizon_sums.setdefault(key, []).append(d)

    results = {
        "ade_km": float(np.mean(batch_ade)),
        "fde_km": float(np.mean(batch_fde)),
    }
    results.update({k: float(np.mean(v)) for k, v in horizon_sums.items()})
    return results


def evaluate_dataset(
    model,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Full dataset evaluation. Returns per-horizon errors and ADE/FDE.
    """
    model.eval()
    all_pred_rel, all_true_pred, all_obs_last = [], [], []

    with torch.no_grad():
        for batch in loader:
            obs       = batch["obs"].to(device)
            obs_rel   = batch["obs_rel"].to(device)
            true_pred = batch["pred"].to(device)
            env       = batch.get("env", None)
            basin_idx = batch.get("basin_idx", None)
            if env is not None:
                env = env.to(device)
            if basin_idx is not None:
                basin_idx = basin_idx.to(device)

            obs_last = obs[:, -1, :2]

            pred_rel = model(obs, obs_rel, env=env, basin_idx=basin_idx)

            all_pred_rel.append(pred_rel.cpu())
            all_true_pred.append(true_pred.cpu())
            all_obs_last.append(obs_last.cpu())

    pred_rel  = torch.cat(all_pred_rel, dim=0)
    true_pred = torch.cat(all_true_pred, dim=0)
    obs_last  = torch.cat(all_obs_last, dim=0)

    return evaluate_batch(pred_rel, true_pred, obs_last)


def evaluate_by_basin(
    model,
    loader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model separately for each basin."""
    model.eval()
    basin_buckets: Dict[str, List] = {}

    with torch.no_grad():
        for batch in loader:
            obs       = batch["obs"].to(device)
            obs_rel   = batch["obs_rel"].to(device)
            true_pred = batch["pred"].to(device)
            env       = batch.get("env", None)
            basin_idx = batch.get("basin_idx", None)
            if env is not None:
                env = env.to(device)
            if basin_idx is not None:
                basin_idx = basin_idx.to(device)

            obs_last  = obs[:, -1, :2]
            pred_rel  = model(obs, obs_rel, env=env, basin_idx=basin_idx)

            for b_idx in range(obs.shape[0]):
                basin = batch["basin"][b_idx]
                if basin not in basin_buckets:
                    basin_buckets[basin] = {"pred_rel": [], "true_pred": [], "obs_last": []}
                basin_buckets[basin]["pred_rel"].append(pred_rel[b_idx:b_idx+1].cpu())
                basin_buckets[basin]["true_pred"].append(true_pred[b_idx:b_idx+1].cpu())
                basin_buckets[basin]["obs_last"].append(obs_last[b_idx:b_idx+1].cpu())

    results = {}
    for basin, bufs in basin_buckets.items():
        pr = torch.cat(bufs["pred_rel"], dim=0)
        tp = torch.cat(bufs["true_pred"], dim=0)
        ol = torch.cat(bufs["obs_last"], dim=0)
        results[basin] = evaluate_batch(pr, tp, ol)

    return results
