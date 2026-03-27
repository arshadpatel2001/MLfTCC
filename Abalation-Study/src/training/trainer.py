"""
Generic training loop for TC track forecasting models.

Features:
- MSE loss on relative displacements
- LR scheduling (cosine annealing with warm restarts)
- Early stopping
- Per-epoch per-basin evaluation on validation set
- Checkpoint saving
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# metrics imported lazily inside Trainer.train() to avoid stale cached imports


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Trains a TC track forecasting model.

    Parameters
    ----------
    model        : nn.Module with forward(obs, obs_rel, env, basin_idx) -> (B, pred_len, 2)
    train_loader : DataLoader
    val_loader   : DataLoader
    device       : torch.device
    lr           : float — initial learning rate
    weight_decay : float
    max_epochs   : int
    patience     : int — early stopping patience
    save_dir     : str — directory to save checkpoints
    experiment_name : str — used for checkpoint filenames
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        save_dir: str = "results/checkpoints",
        experiment_name: str = "model",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
        )
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStopping(patience=patience)

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_ade_km": [], "val_fde_km": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            obs       = batch["obs"].to(self.device)
            obs_rel   = batch["obs_rel"].to(self.device)
            pred_rel  = batch["pred_rel"].to(self.device)   # (B, pred_len, 2) target
            env       = batch.get("env")
            basin_idx = batch.get("basin_idx")
            if env is not None:
                env = env.to(self.device)
            if basin_idx is not None:
                basin_idx = basin_idx.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(obs, obs_rel, env=env, basin_idx=basin_idx)
            loss = self.criterion(pred, pred_rel)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            obs       = batch["obs"].to(self.device)
            obs_rel   = batch["obs_rel"].to(self.device)
            pred_rel  = batch["pred_rel"].to(self.device)
            env       = batch.get("env")
            basin_idx = batch.get("basin_idx")
            if env is not None:
                env = env.to(self.device)
            if basin_idx is not None:
                basin_idx = basin_idx.to(self.device)

            pred = self.model(obs, obs_rel, env=env, basin_idx=basin_idx)
            loss = self.criterion(pred, pred_rel)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Full training loop. Returns training history + best metrics.
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"  Device: {self.device}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"{'='*60}")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss   = self._val_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Full km evaluation every 10 epochs (lazy import avoids stale cache)
            if epoch % 10 == 0 or epoch == 1:
                from src.training.metrics import evaluate_dataset
                metrics = evaluate_dataset(self.model, self.val_loader, self.device)
                self.history["val_ade_km"].append(metrics["ade_km"])
                self.history["val_fde_km"].append(metrics["fde_km"])
                km_str = f" | ADE={metrics['ade_km']:.1f}km FDE={metrics['fde_km']:.1f}km"
            else:
                km_str = ""

            if verbose:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{self.max_epochs}"
                    f" | train={train_loss:.5f} val={val_loss:.5f}"
                    f"{km_str}"
                    f" | lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    f" | {elapsed:.1f}s"
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                ckpt_path = self.save_dir / f"{self.experiment_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                }, str(ckpt_path))

            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping at epoch {epoch} (best={self.best_epoch})")
                break

        print(f"\nTraining complete. Best val_loss={self.best_val_loss:.5f} at epoch {self.best_epoch}")
        return self.history

    def load_best(self):
        """Load the best checkpoint back into the model."""
        ckpt_path = self.save_dir / f"{self.experiment_name}_best.pt"
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.5f})")


def load_model_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load a saved model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model
