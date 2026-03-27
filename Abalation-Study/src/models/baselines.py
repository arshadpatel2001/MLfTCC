"""
Non-ML baselines for TC track forecasting.

1. PersistenceModel  — extrapolate last observed velocity
2. ClimatologyModel  — predict zero displacement (mean position shift)
"""

import torch
import torch.nn as nn
import numpy as np


class PersistenceModel(nn.Module):
    """
    Persistence baseline: assume TC continues at its last observed velocity.

    The model takes obs_rel (the last displacement vectors) and simply
    repeats the final observed displacement for all pred_len steps.

    This is a strong operational baseline for short-range forecasting.
    """

    def __init__(self, pred_len: int = 4):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, obs: torch.Tensor, obs_rel: torch.Tensor, env=None) -> torch.Tensor:
        """
        obs     : (B, obs_len, 4) — not used
        obs_rel : (B, obs_len, 2) — relative displacements

        Returns
        -------
        pred_rel : (B, pred_len, 2) — repeated last velocity
        """
        last_vel = obs_rel[:, -1:, :]            # (B, 1, 2)
        return last_vel.expand(-1, self.pred_len, -1)   # (B, pred_len, 2)


class LinearTrendModel(nn.Module):
    """
    Linear trend extrapolation baseline.

    Fits a linear trend through the last `window` steps of obs_rel
    and extrapolates forward.
    """

    def __init__(self, pred_len: int = 4, window: int = 4):
        super().__init__()
        self.pred_len = pred_len
        self.window = window

    def forward(self, obs: torch.Tensor, obs_rel: torch.Tensor, env=None) -> torch.Tensor:
        B = obs_rel.shape[0]
        device = obs_rel.device

        # Use last `window` displacement steps
        recent = obs_rel[:, -self.window:, :]  # (B, window, 2)
        window = recent.shape[1]

        # Least-squares linear fit: vel_t = a*t + b
        t = torch.arange(window, dtype=torch.float32, device=device)  # (window,)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()

        # per-sample, per-coord: slope = cov(t, vel) / var(t)
        t_centered = t - t_mean  # (window,)
        # recent: (B, window, 2), t_centered: (window,)
        cov = (recent * t_centered[None, :, None]).sum(dim=1)  # (B, 2)
        slope = cov / (t_var + 1e-8)   # (B, 2)
        bias  = recent.mean(dim=1) - slope * t_mean  # (B, 2)

        # Extrapolate: t_future = window, window+1, ..., window+pred_len-1
        t_future = torch.arange(window, window + self.pred_len,
                                dtype=torch.float32, device=device)  # (pred_len,)
        pred_rel = slope[:, None, :] * t_future[None, :, None] + bias[:, None, :]
        return pred_rel  # (B, pred_len, 2)
