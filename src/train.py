"""
Main Training & Experiment Runner
==================================
Runs the full basin generalization benchmark from the paper.

Experiments implemented:
  1. Leave-One-Basin-Out (LOBO) — zero-shot transfer
  2. WP → NA,EP,NI,SI,SP       — single source transfer
  3. Few-shot fine-tuning        — adapt with k shots on target basin

Logging: Weights & Biases (wandb). Set WANDB_PROJECT env var.
Checkpointing: saves best model per method/target_basin pair.

Usage:
    # Full benchmark (all methods, all LOBO splits)
    python train.py --mode lobo --data_root /path/to/TCND --epochs 50

    # Single experiment
    python train.py --mode single \
        --source_basins WP NA EP \
        --target_basin SI \
        --method physirm \
        --data_root /path/to/TCND \
        --epochs 50

    # Ablation: no 3D data
    python train.py --mode lobo --no_3d --method erm
"""

import os
import sys
import json
import time
import copy
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x  # graceful fallback if tqdm not installed
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    BASIN_CODES, TCNDDataset, make_dataloader, make_per_basin_loaders
)
from models.backbone import TropiCycloneModel, MultimodalBackbone
from methods.dg_methods import build_method, DANN, PhysIRM, METHOD_REGISTRY
from metrics.basin_metrics import (
    BasinEvaluator, TransferEvaluator, BasinResult,
    basin_transfer_gap, basin_normalized_transfer_efficiency
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def select_device(args) -> torch.device:
    """Select compute device respecting --device flag."""
    if hasattr(args, 'device') and args.device and args.device != "auto":
        return torch.device(args.device)
    # Auto-select best available device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Experiment configurations ─────────────────────────────────────────────────

# All leave-one-basin-out splits for the benchmark table
LOBO_SPLITS = [
    {"target": b, "source": [x for x in BASIN_CODES if x != b]}
    for b in BASIN_CODES
]

# Hyper-parameters searched via DomainBed protocol (random search, 20 trials)
# These are the best settings found; see Appendix for full sweep details.
BEST_HPARAMS = {
    "erm":     {"lr": 5e-4, "batch_size": 64, "weight_decay": 1e-4},
    "irm":     {"lr": 1e-3, "batch_size": 64, "weight_decay": 1e-4,
                "irm_lambda": 1.0, "warmup_steps": 500},
    "vrex":    {"lr": 1e-3, "batch_size": 64, "weight_decay": 1e-4,
                "beta": 1.0, "warmup_steps": 500},
    "coral":   {"lr": 5e-4, "batch_size": 64, "weight_decay": 1e-4,
                "coral_lambda": 1.0},
    "dann":    {"lr": 5e-4, "batch_size": 64, "weight_decay": 1e-4,
                "dann_lambda": 1.0, "total_steps": 10000},
    "maml":    {"lr": 5e-4, "batch_size": 64, "weight_decay": 1e-4,
                "inner_lr": 1e-3, "inner_steps": 5},
    "physirm": {"lr": 5e-4, "batch_size": 64, "weight_decay": 1e-4,
                "irm_lambda": 1.0, "orth_lambda": 0.1, "phys_lambda": 0.5,
                "warmup_steps": 500, "phys_dim": 64},
}


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(args) -> TropiCycloneModel:
    return TropiCycloneModel.build(
        model_size    = args.model_size,
        spatial_embed = args.spatial_embed,
        track_embed   = args.track_embed,
        env_embed     = args.env_embed,
        phys_dim      = args.phys_dim,
        final_dim     = args.final_dim,
        dropout       = args.dropout,
        use_3d        = not args.no_3d,
        use_env       = not args.no_env,
    )


# ── Single training run ───────────────────────────────────────────────────────

def train_one_experiment(
    args,
    source_basins: List[str],
    target_basin: str,
    method_name: str,
    run_id: str,
    device: torch.device,
) -> Dict:
    """
    Wrapper for training one experiment that ensures all output is logged uniquely.
    """
    run_timestamp = getattr(args, "run_timestamp", time.strftime("%Y%m%d_%H%M%S"))
    exp_log_dir = Path("logs") / "experiments" / run_timestamp
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    exp_log_file = exp_log_dir / f"{run_id}.log"
    
    exp_fh = logging.FileHandler(exp_log_file)
    exp_fh.setLevel(logging.INFO)
    exp_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(exp_fh)
    
    try:
        log.info("=" * 80)
        log.info(f"[{run_id}] EXPERIMENT START")
        log.info(f"[{run_id}] Source: {source_basins}  Target: {target_basin}  Method: {method_name}")
        log.info(f"[{run_id}] Device: {device}")
        log.info(f"[{run_id}] Isolated log file: {exp_log_file}")
        log.info("Experiment Hyperparameters & Args:")
        for k, v in sorted(vars(args).items()):
            log.info(f"  {k}: {v}")
        log.info("=" * 80)
        
        return _train_one_experiment_inner(args, source_basins, target_basin, method_name, run_id, device)
    except Exception as e:
        log.error(f"[{run_id}] EXPERIMENT FAILED: {e}")
        raise
    finally:
        log.info(f"[{run_id}] EXPERIMENT END")
        log.info("=" * 80)
        root_logger.removeHandler(exp_fh)
        exp_fh.close()

def _train_one_experiment_inner(
    args,
    source_basins: List[str],
    target_basin: str,
    method_name: str,
    run_id: str,
    device: torch.device,
) -> Dict:
    """
    Train one (source, target, method) triple.
    Returns metrics dict for logging.
    """
    log.info(f"[{run_id}] Source: {source_basins}  Target: {target_basin}  Method: {method_name}")

    hp = BEST_HPARAMS.get(method_name, BEST_HPARAMS["erm"])
    bs = hp.get("batch_size", 64)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Per-environment loaders for source basins (needed by IRM/CORAL/DANN)
    train_loaders_per_env = make_per_basin_loaders(
        root=args.data_root, basins=source_basins,
        split="train", batch_size=bs,
        num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
    )
    val_loader_src = make_dataloader(
        root=args.data_root, basins=source_basins,
        split="val", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
    )
    # val_loader_tgt: used for training-time monitoring only (split="val").
    # test_loader_tgt: used ONLY for the final best-model evaluation (split="test").
    # Keeping these separate prevents target test-set leakage into training history.
    val_loader_tgt = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="val", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
    )
    test_loader_tgt = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="test", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Method ────────────────────────────────────────────────────────────────
    method_kwargs = {k: v for k, v in hp.items()
                    if k not in {"lr", "batch_size", "weight_decay"}}

    # Estimate total training steps so schedule-aware methods (DANN) ramp correctly.
    n_batches_estimate = max(1, max(len(l) for l in train_loaders_per_env.values()))

    # Inject architecture-dependent kwargs that must align with the built model.
    if method_name == "dann":
        # feature_dim must match model output; read directly to avoid None issues
        # when --final_dim is not set on the CLI.
        method_kwargs["feature_dim"] = model.backbone.get_output_dim()
        method_kwargs.setdefault("n_domains", len(BASIN_CODES))
        # Override static total_steps with the actual expected step count so the
        # GRL alpha schedule ramps correctly for both short and long experiments.
        method_kwargs["total_steps"] = n_batches_estimate * args.epochs
    if method_name == "physirm":
        # phys_dim must exactly match model.backbone.phys_dim to avoid shape errors
        # in phys_predictor. Always read from the model, ignoring BEST_HPARAMS value.
        method_kwargs["phys_dim"] = model.backbone.phys_dim
    method = build_method(method_name, **method_kwargs)

    # Methods with learnable parameters (DANN discriminator, PhysIRM predictor)
    extra_params = []
    if isinstance(method, (DANN, PhysIRM)):
        method.to(device)
        extra_params = list(method.parameters())

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    all_params = list(model.parameters()) + extra_params
    optimizer  = optim.AdamW(all_params, lr=hp["lr"],
                             weight_decay=hp.get("weight_decay", 1e-4))
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_ckpt    = None
    step         = 0
    history      = []

    # ── Dataset size summary ─────────────────────────────────────────────────
    total_train = sum(len(l.dataset) for l in train_loaders_per_env.values())
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Device: {device}")
    log.info(
        f"Train samples: {total_train:,}  |  "
        f"Validation Source: {len(val_loader_src.dataset):,}  |  "
        f"Validation Target: {len(val_loader_tgt.dataset):,}  |  "
        f"Test Target: {len(test_loader_tgt.dataset):,}"
    )
    log.info(f"Starting training for {args.epochs} epochs ...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(method, torch.nn.Module):
            method.train()
        epoch_metrics: Dict[str, float] = {}
        epoch_start = time.time()

        # Zip per-environment loaders (cycle shorter loaders)
        env_iters = {b: iter(loader) for b, loader in train_loaders_per_env.items()}
        # Use len(loader) — the DataLoader is the authoritative source for
        # batch count per epoch (correctly handles drop_last and safe_bs).
        n_batches = max(1, max(len(l) for l in train_loaders_per_env.values()))

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch:3d}/{args.epochs}",
                    unit="batch", leave=False, dynamic_ncols=True)

        for batch_idx in pbar:
            # Collect one batch from each environment
            batches = {}
            for b, it in env_iters.items():
                try:
                    batch = next(it)
                except StopIteration:
                    env_iters[b] = iter(train_loaders_per_env[b])
                    batch = next(env_iters[b])
                batches[b] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            metrics = method.update(optimizer, batches, model, step=step)
            step   += 1

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Live loss in tqdm bar
            if "loss" in epoch_metrics:
                pbar.set_postfix(loss=f"{epoch_metrics['loss'] / (batch_idx + 1):.4f}")

        pbar.close()

        # Average epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        epoch_time = time.time() - epoch_start
        log.info(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"[{epoch_time:.1f}s] "
            + "  ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
        )

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            if isinstance(method, torch.nn.Module):
                method.eval()
            ev_src = BasinEvaluator("source_val")
            ev_tgt = BasinEvaluator(target_basin)

            val_start = time.time()
            with torch.no_grad():
                for batch in tqdm(val_loader_src, desc="Validation Source", leave=False, dynamic_ncols=True):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    out = model(batch)
                    ev_src.update(batch, out)

                for batch in tqdm(val_loader_tgt, desc="Validation Target", leave=False, dynamic_ncols=True):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    out = model(batch)
                    ev_tgt.update(batch, out)
            val_time = time.time() - val_start
            log.info(f"Validation took {val_time:.2f}s")

            r_src = ev_src.compute()
            r_tgt = ev_tgt.compute()

            log.info(
                f"Epoch {epoch:3d}/{args.epochs} "
                f"| source accuracy intensity={r_src.accuracy_intensity:.3f} "
                f"| source precision intensity={r_src.precision_intensity:.3f} "
                f"| source recall intensity={r_src.recall_intensity:.3f} "
                f"| source f1 intensity={r_src.f1_intensity:.3f} "
                f"| source accuracy direction={r_src.accuracy_direction:.3f} "
                f"| source precision direction={r_src.precision_direction:.3f} "
                f"| source recall direction={r_src.recall_direction:.3f} "
                f"| source f1 direction={r_src.f1_direction:.3f} "
                f"| source rapid intensification recall={r_src.rapid_intensification_recall:.3f} "
                f"| source rapid intensification precision={r_src.rapid_intensification_precision:.3f} "
                f"| source rapid intensification f1={r_src.rapid_intensification_f1:.3f} "
                f"| target accuracy intensity={r_tgt.accuracy_intensity:.3f} "
                f"| target precision intensity={r_tgt.precision_intensity:.3f} "
                f"| target recall intensity={r_tgt.recall_intensity:.3f} "
                f"| target f1 intensity={r_tgt.f1_intensity:.3f} "
                f"| target accuracy direction={r_tgt.accuracy_direction:.3f} "
                f"| target precision direction={r_tgt.precision_direction:.3f} "
                f"| target recall direction={r_tgt.recall_direction:.3f} "
                f"| target f1 direction={r_tgt.f1_direction:.3f} "
                f"| target rapid intensification recall={r_tgt.rapid_intensification_recall:.3f} "
                f"| target rapid intensification precision={r_tgt.rapid_intensification_precision:.3f} "
                f"| target rapid intensification f1={r_tgt.rapid_intensification_f1:.3f} "
                f"| loss={epoch_metrics.get('loss', float('nan')):.4f}"
            )

            # Track best on SOURCE val (to avoid target leakage)
            if r_src.f1_intensity > best_val_f1:
                best_val_f1 = r_src.f1_intensity
                if args.output_dir:
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    ckpt_path = Path(args.output_dir) / f"{run_id}_best.pt"
                    state = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "method": method.state_dict() if isinstance(method, torch.nn.Module) else {}
                    }
                    torch.save(state, ckpt_path)
                    best_ckpt = str(ckpt_path)

            history.append({
                "epoch": epoch,
                "source accuracy intensity": r_src.accuracy_intensity,
                "source precision intensity": r_src.precision_intensity,
                "source recall intensity": r_src.recall_intensity,
                "source f1 intensity": r_src.f1_intensity,
                "source accuracy direction": r_src.accuracy_direction,
                "source precision direction": r_src.precision_direction,
                "source recall direction": r_src.recall_direction,
                "source f1 direction": r_src.f1_direction,
                "source rapid intensification recall": r_src.rapid_intensification_recall,
                "source rapid intensification precision": r_src.rapid_intensification_precision,
                "source rapid intensification f1": r_src.rapid_intensification_f1,
                "target accuracy intensity": r_tgt.accuracy_intensity,
                "target precision intensity": r_tgt.precision_intensity,
                "target recall intensity": r_tgt.recall_intensity,
                "target f1 intensity": r_tgt.f1_intensity,
                "target accuracy direction": r_tgt.accuracy_direction,
                "target precision direction": r_tgt.precision_direction,
                "target recall direction": r_tgt.recall_direction,
                "target f1 direction": r_tgt.f1_direction,
                "target rapid intensification recall": r_tgt.rapid_intensification_recall,
                "target rapid intensification precision": r_tgt.rapid_intensification_precision,
                "target rapid intensification f1": r_tgt.rapid_intensification_f1,
                **epoch_metrics,
            })

        # Save the latest weights at the end of every epoch
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            latest_ckpt_path = Path(args.output_dir) / f"{run_id}_latest.pt"
            latest_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "method": method.state_dict() if isinstance(method, torch.nn.Module) else {}
            }
            torch.save(latest_state, latest_ckpt_path)
    # ── Final evaluation ──────────────────────────────────────────────────────
    if best_ckpt:
        log.info(f"Loading best checkpoint: {best_ckpt}")
        # weights_only=False: checkpoint contains nested dicts requiring full unpickling.
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if isinstance(method, torch.nn.Module) and checkpoint.get("method"):
                method.load_state_dict(checkpoint["method"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    if isinstance(method, torch.nn.Module):
        method.eval()

    ev_final_src = BasinEvaluator("source_final")
    ev_final = BasinEvaluator(target_basin)
    final_ev_start = time.time()
    with torch.no_grad():
        for batch in tqdm(val_loader_src, desc="Final Source Evaluation", leave=False, dynamic_ncols=True):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            ev_final_src.update(batch, out)

        for batch in tqdm(test_loader_tgt, desc="Final Evaluation", leave=False, dynamic_ncols=True):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            ev_final.update(batch, out)
    log.info(f"Final evaluation took {time.time() - final_ev_start:.2f}s")

    final_src = ev_final_src.compute()
    final = ev_final.compute()

    btg = basin_transfer_gap(final_src.accuracy_intensity, final.accuracy_intensity)
    bnte = basin_normalized_transfer_efficiency(final_src.accuracy_intensity, final.accuracy_intensity, 0.0)

    result = {
        "run_id":        run_id,
        "method":        method_name,
        "source_basins": source_basins,
        "target_basin":  target_basin,
        "final source accuracy intensity": final_src.accuracy_intensity,
        "final target accuracy intensity": final.accuracy_intensity,
        "btg":           btg,
        "bnte":          bnte,
        "final target precision intensity": final.precision_intensity,
        "final target recall intensity": final.recall_intensity,
        "final target f1 intensity": final.f1_intensity,
        "final target accuracy direction": final.accuracy_direction,
        "final target precision direction": final.precision_direction,
        "final target recall direction": final.recall_direction,
        "final target f1 direction": final.f1_direction,
        "final target rapid intensification precision": final.rapid_intensification_precision,
        "final target rapid intensification recall": final.rapid_intensification_recall,
        "final target rapid intensification f1": final.rapid_intensification_f1,
        "history":       history,
        "best_ckpt":     best_ckpt,
    }

    # ── Few-shot fine-tuning on target basin (if enabled) ─────────
    if args.few_shot:
        fs_result = few_shot_finetune(
            model=model,
            target_basin=target_basin,
            args=args,
            device=device,
            k_shots=args.k_shots,
            ft_epochs=args.few_shot_epochs,
        )
        result["few_shot target accuracy intensity"] = fs_result.accuracy_intensity
        result["few_shot target precision intensity"] = fs_result.precision_intensity
        result["few_shot target recall intensity"] = fs_result.recall_intensity
        result["few_shot target f1 intensity"] = fs_result.f1_intensity
        
        result["few_shot target accuracy direction"] = fs_result.accuracy_direction
        result["few_shot target precision direction"] = fs_result.precision_direction
        result["few_shot target recall direction"] = fs_result.recall_direction
        result["few_shot target f1 direction"] = fs_result.f1_direction
        
        result["few_shot target rapid intensification precision"] = fs_result.rapid_intensification_precision
        result["few_shot target rapid intensification recall"] = fs_result.rapid_intensification_recall
        result["few_shot target rapid intensification f1"] = fs_result.rapid_intensification_f1
        
        log.info(
            f"Few-shot ({args.k_shots} shots) on {target_basin}: "
            f"target accuracy intensity={fs_result.accuracy_intensity:.3f}, "
            f"target precision intensity={fs_result.precision_intensity:.3f}, "
            f"target recall intensity={fs_result.recall_intensity:.3f}, "
            f"target f1 intensity={fs_result.f1_intensity:.3f}, "
            f"target accuracy direction={fs_result.accuracy_direction:.3f}, "
            f"target precision direction={fs_result.precision_direction:.3f}, "
            f"target recall direction={fs_result.recall_direction:.3f}, "
            f"target f1 direction={fs_result.f1_direction:.3f}, "
            f"target rapid intensification precision={fs_result.rapid_intensification_precision:.3f}, "
            f"target rapid intensification recall={fs_result.rapid_intensification_recall:.3f}, "
            f"target rapid intensification f1={fs_result.rapid_intensification_f1:.3f}"
        )

    return result


# ── Few-shot fine-tuning ──────────────────────────────────────────────────────

def few_shot_finetune(
    model: TropiCycloneModel,
    target_basin: str,
    args,
    device: torch.device,
    k_shots: int = 32,
    ft_lr: float = 1e-4,
    ft_epochs: int = 5,
) -> BasinResult:
    """
    Few-shot fine-tuning on the target basin.

    Standard evaluation protocol:
      - Sample k_shots examples from target basin train split
      - Fine-tune for ft_epochs with frozen backbone, only heads trainable
      - Evaluate on target test split

    This simulates a real-world scenario where a small amount of labelled
    data is available in a new basin (e.g., after a new monitoring network
    is deployed in the South Pacific).
    """
    log.info(f"Few-shot fine-tuning on {target_basin} with {k_shots} shots")
    from methods.dg_methods import task_loss

    # Get k-shot loader
    shot_ds = TCNDDataset(
        root=args.data_root, basins=[target_basin],
        split="train", use_3d=not args.no_3d, use_env=not args.no_env,
    )
    # Subsample to k_shots safely bounding to dataset size
    k_shots = min(k_shots, len(shot_ds))

    # Save entire model state to strictly prevent contamination of the zero-shot baseline
    orig_state = copy.deepcopy(model.state_dict())

    if k_shots > 0:
        # Fixed seed per k_shots value for reproducible k-shot subset selection.
        rng_seed = 42 + k_shots
        indices = torch.randperm(len(shot_ds),
                                 generator=torch.Generator().manual_seed(rng_seed))[:k_shots].tolist()
        shot_subset = torch.utils.data.Subset(shot_ds, indices)
        shot_loader = torch.utils.data.DataLoader(
            shot_subset, batch_size=max(1, min(k_shots, 32)), shuffle=True
        )

        # Freeze backbone, fine-tune heads only
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.heads.parameters():
            p.requires_grad_(True)

        ft_optimizer = optim.Adam(model.heads.parameters(), lr=ft_lr)

        model.eval()
        model.heads.train()
        fs_start = time.time()
        for ep in tqdm(range(ft_epochs), desc="Few-shot Epochs", leave=False, dynamic_ncols=True):
            ep_start = time.time()
            for batch in tqdm(shot_loader, desc=f"Epoch {ep+1}/{ft_epochs}", leave=False, dynamic_ncols=True):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                ft_optimizer.zero_grad()
                out = model(batch)
                loss = task_loss(
                    out["logits_intensity"], out["logits_direction"],
                    batch["y_intensity"], batch["y_direction"]
                )
                loss.backward()
                ft_optimizer.step()
            log.info(f"Few-shot Epoch {ep+1} took {time.time() - ep_start:.2f}s")
        log.info(f"Total few-shot fine-tuning took {time.time() - fs_start:.2f}s")

        # Unfreeze
        for p in model.backbone.parameters():
            p.requires_grad_(True)
    else:
        log.info(f"Target basin {target_basin} k_shots=0. Skipping few-shot fine-tuning.")

    # Evaluate
    test_loader = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="test", batch_size=64, num_workers=0,
        use_3d=not args.no_3d, use_env=not args.no_env,
    )
    ev = BasinEvaluator(target_basin)
    model.eval()
    fs_eval_start = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Few-shot Evaluation", leave=False, dynamic_ncols=True):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch)
            ev.update(batch, out)
    log.info(f"Few-shot evaluation took {time.time() - fs_eval_start:.2f}s")

    # Restore full original model state
    model.load_state_dict(orig_state)

    return ev.compute()


# ── Full benchmark ────────────────────────────────────────────────────────────

def run_lobo_benchmark(args):
    """
    Full Leave-One-Basin-Out benchmark.
    Trains all methods × all LOBO splits.
    Results saved to args.output_dir/benchmark_results.json.
    """
    device = select_device(args)
    log.info(f"Device: {device}")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    method_arg = args.methods or args.method
    methods  = [m.strip() for m in method_arg.split(",")] if method_arg else list(METHOD_REGISTRY.keys())
    splits   = LOBO_SPLITS if not args.target_basin else [
        {"target": args.target_basin,
         "source": [b for b in BASIN_CODES if b != args.target_basin]}
    ]

    all_results = []

    bench_start = time.time()
    for split in tqdm(splits, desc="Leave-One-Basin-Out Splits", dynamic_ncols=True):
        for method_name in tqdm(methods, desc="Methods", leave=False, dynamic_ncols=True):
            exp_start = time.time()
            run_id = f"{method_name}_{split['target']}"
            try:
                result = train_one_experiment(
                    args=args,
                    source_basins=split["source"],
                    target_basin=split["target"],
                    method_name=method_name,
                    run_id=run_id,
                    device=device,
                )
                all_results.append(result)
                log.info(
                    f"✓ {run_id}: final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                    f"final target rapid intensification f1={result['final target rapid intensification f1']:.3f} "
                    f"[Took {time.time() - exp_start:.2f}s]"
                )
            except Exception as e:
                log.error(f"✗ {run_id} FAILED: {e}")
                if args.fail_fast:
                    raise

    # Save results
    if args.output_dir:
        out_path = Path(args.output_dir) / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"Results saved to {out_path}")

    # Print summary table
    _print_summary_table(all_results, methods, splits)
    log.info(f"LOBO Benchmark total time: {time.time() - bench_start:.2f}s")

    return all_results


def _print_summary_table(results, methods, splits):
    """Print a NeurIPS-style results table."""
    targets = [s["target"] for s in splits]
    print("\n" + "=" * 90)
    print("BASIN GENERALIZATION BENCHMARK — Zero-Shot Transfer (Accuracy-Intensity)")
    print("=" * 90)

    header = f"{'Method':<12}" + "".join(f"{t:>8}" for t in targets) + f"{'Average':>10}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<12}"
        accs = []
        for t in targets:
            r = next(
                (x for x in results
                 if x["method"] == method and x["target_basin"] == t),
                None
            )
            if r:
                acc = r["final target accuracy intensity"]
                accs.append(acc)
                row += f"{acc:>8.3f}"
            else:
                row += f"{'—':>8}"
        avg = sum(accs) / len(accs) if accs else 0
        row += f"{avg:>10.3f}"
        print(row)

    print("=" * 90)


# ── Incremental Benchmark ─────────────────────────────────────────────────────

def run_incremental_benchmark(args):
    """
    Incremental training benchmark.
    For a target basin, trains on 1 source basin, then 2, then 3, up to
    leave-one-basin-out (LOBO) setup.
    """
    device = select_device(args)
    log.info(f"Device: {device}")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    method_arg = args.methods or args.method
    methods = [m.strip() for m in method_arg.split(",")] if method_arg else list(METHOD_REGISTRY.keys())
    
    # Target basins
    targets = [args.target_basin] if args.target_basin else BASIN_CODES
    
    all_results = []
    
    bench_start = time.time()
    for target in tqdm(targets, desc="Incremental Targets", dynamic_ncols=True):
        if args.source_basins:
            available_sources = [s.strip() for s in args.source_basins.split(",")]
        else:
            available_sources = [b for b in BASIN_CODES if b != target]
            
        # Iteration sequence: 1 source, 2 sources, ..., N sources
        for i in tqdm(range(1, len(available_sources) + 1), desc="Incremental Sources", leave=False, dynamic_ncols=True):
            source_basins = available_sources[:i]
            
            for method_name in tqdm(methods, desc="Methods", leave=False, dynamic_ncols=True):
                exp_start = time.time()
                run_id = f"{method_name}_{target}_src{i}"
                try:
                    result = train_one_experiment(
                        args=args,
                        source_basins=source_basins,
                        target_basin=target,
                        method_name=method_name,
                        run_id=run_id,
                        device=device,
                    )
                    all_results.append(result)
                    log.info(
                        f"✓ {run_id}: final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                        f"final target rapid intensification f1={result['final target rapid intensification f1']:.3f} "
                        f"[Took {time.time() - exp_start:.2f}s]"
                    )
                except Exception as e:
                    log.error(f"✗ {run_id} FAILED: {e}")
                    if args.fail_fast:
                        raise

    # Save results
    if args.output_dir:
        out_path = Path(args.output_dir) / "incremental_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"Incremental results saved to {out_path}")
        
    log.info(f"Incremental Benchmark total time: {time.time() - bench_start:.2f}s")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    description = f"""
\033[1;36m=================================================================\033[0m
\033[1;35m    PhysIRM: Physics-Guided Basin Generalization Benchmark       \033[0m
\033[1;36m=================================================================\033[0m

\033[1mTrain AI to predict Tropical Cyclones and test generalization across oceans!\033[0m

\033[1;33mMODES OF OPERATION:\033[0m
  \033[1;32m1. single\033[0m       Train on a fixed set of source basins, test on 1 target.
                  \033[3mUsage:\033[0m --mode single --source_basins WP,NA --target_basin SI --method physirm

  \033[1;32m2. incremental\033[0m  Progressively iterate by adding 1 basin at a time to training.
                  \033[3mUsage:\033[0m --mode incremental --source_basins WP,NA,EP --target_basin SI --methods physirm

  \033[1;32m3. lobo\033[0m         (Leave-One-Basin-Out) Train on all basins except the target.
                  \033[3mUsage:\033[0m --mode lobo --methods physirm --target_basin SI

\033[1;33mAVAILABLE BASINS:\033[0m
  \033[1;34m{', '.join(BASIN_CODES)}\033[0m

\033[1;33mAVAILABLE METHODS:\033[0m
  \033[1;34merm, irm, vrex, coral, dann, maml, physirm\033[0m
"""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Data
    p.add_argument("--data_root",  type=str, required=True,
                   help="Path to TCND root directory")
    p.add_argument("--output_dir", type=str, default="./runs",
                   help="Directory for checkpoints and results")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of dataloader workers for data loading")

    # Experiment
    p.add_argument("--mode", choices=["lobo", "single", "incremental"], default="lobo",
                   help="lobo=leave-one-basin-out; single=one transfer pair; incremental=incremental source basins")
    p.add_argument("--source_basins", type=str, default=None,
                   help="Comma-separated source basin codes (single mode)")
    p.add_argument("--target_basin", type=str, default=None,
                   help="Target basin code")
    p.add_argument("--method",  type=str, default=None,
                   help="Single method for single mode (e.g. physirm)")
    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated methods to run (default: all)")
    p.add_argument("--epochs",      type=int,   default=50,
                   help="Number of training epochs")
    p.add_argument("--eval_every",  type=int,   default=1,
                   help="Evaluate validation set every N epochs")
    p.add_argument("--device",      type=str,   default="auto",
                        help="Device: cuda | mps | cpu | auto (auto-detects best)")
    p.add_argument("--fail_fast",   action="store_true",
                   help="Crash immediately if one specific experiment configuration fails")

    # Model architecture
    p.add_argument("--model_size", type=str, choices=["lightweight", "complex"], default="lightweight",
                   help="Model scale: lightweight (notebook original) or complex (ResBlocks)")
    p.add_argument("--spatial_embed", type=int, default=None,
                   help="Embedding dimension for Data_3d branch")
    p.add_argument("--track_embed",   type=int, default=None,
                   help="Embedding dimension for Data_1d track branch")
    p.add_argument("--env_embed",     type=int, default=None,
                   help="Embedding dimension for Env-Data branch")
    p.add_argument("--phys_dim",      type=int, default=None,
                   help="Dimension of the invariant physics sub-space (z_phys)")
    p.add_argument("--final_dim",     type=int, default=None,
                   help="Dimension of the fused representation space")
    p.add_argument("--dropout",       type=float, default=0.1,
                   help="Dropout probability across networks")

    # Ablations
    p.add_argument("--no_3d",  action="store_true",
                   help="Ablation: disable Data_3d (spatial) branch")
    p.add_argument("--no_env", action="store_true",
                   help="Ablation: disable Env-Data branch")

    # Few-shot
    p.add_argument("--few_shot",      action="store_true",
                   help="Enable few-shot fine-tuning on the target basin")
    p.add_argument("--k_shots",       type=int,   default=32,
                   help="Number of labeled examples to use for few-shot adaptation")
    p.add_argument("--few_shot_epochs", type=int, default=5,
                   help="Number of epochs to train during few-shot adaptation")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Setup Global Logging ──────────────────────────────────────────────────
    import platform
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.run_timestamp = timestamp
    
    global_log_file = log_dir / f"train_run_{args.mode}_{timestamp}.log"
    fh = logging.FileHandler(global_log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    
    log.info("=" * 80)
    log.info("STARTING FULL TRAINING RUN")
    log.info(f"Mode: {args.mode}")
    log.info(f"Timestamp: {timestamp}")
    log.info(f"Command: {' '.join(sys.argv)}")
    log.info("System Environment:")
    log.info(f"  OS: {platform.system()} {platform.release()} ({platform.machine()})")
    log.info(f"  Python: {sys.version.split()[0]}")
    log.info(f"  PyTorch: {torch.__version__}")
    log.info("Arguments:")
    for k, v in sorted(vars(args).items()):
        log.info(f"  {k}: {v}")
    log.info("=" * 80)

    if args.mode == "lobo":
        run_lobo_benchmark(args)

    elif args.mode == "incremental":
        run_incremental_benchmark(args)

    elif args.mode == "single":
        if not args.source_basins or not args.target_basin:
            raise ValueError("--source_basins and --target_basin required in single mode")
        source = [s.strip() for s in args.source_basins.split(",")]
        device = select_device(args)

        method_arg = args.method or args.methods
        methods = [m.strip() for m in method_arg.split(",")] if method_arg else list(METHOD_REGISTRY.keys())
        for method_name in methods:
            result = train_one_experiment(
                args=args,
                source_basins=source,
                target_basin=args.target_basin,
                method_name=method_name,
                run_id=f"{method_name}_{args.target_basin}",
                device=device,
            )
            log.info(
                f"Final: final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                f"final target rapid intensification f1={result['final target rapid intensification f1']:.3f}"
            )