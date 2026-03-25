"""
Main Training & Experiment Runner
==================================
Runs the full basin generalization benchmark from the paper.

Experiments implemented:
  1. Leave-One-Basin-Out (LOBO) — zero-shot transfer
  2. WP → NA,EP,NI,SI,SP       — single source transfer
  3. Few-shot fine-tuning        — adapt with k shots on target basin

Usage:
    # Full benchmark (all methods, all LOBO splits)
    python train.py --mode lobo --data_root /path/to/TCND --epochs 50

    # Single experiment
    python train.py --mode single \\
        --source_basins WP,NA,EP \\
        --target_basin SI \\
        --method physirm \\
        --data_root /path/to/TCND

    # No 3D data ablation
    python train.py --mode lobo --no_3d --method erm

    # Control regression loss weight
    python train.py --mode lobo --reg_weight 0.5 --method physirm
"""

import os
import sys
import json
import time
import copy
import argparse
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

def _make_scaler(init_scale: int = 2**14):
    """Version-safe GradScaler: torch.amp (≥2.1) or torch.cuda.amp (2.0.x)."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", init_scale=init_scale)
    return torch.cuda.amp.GradScaler(init_scale=init_scale)  # PyTorch 2.0.x

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    BASIN_CODES, TCNDDataset, make_dataloader, make_per_basin_loaders
)
from models.backbone import TropiCycloneModel, MultimodalBackbone
from methods.dg_methods import build_method, DANN, PhysIRM, METHOD_REGISTRY, task_loss
from metrics.basin_metrics import (
    BasinEvaluator, TransferEvaluator, BasinResult,
    basin_transfer_gap, basin_normalized_transfer_efficiency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def select_device(args) -> torch.device:
    if hasattr(args, "device") and args.device and args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


# ── Experiment configurations ─────────────────────────────────────────────────

LOBO_SPLITS = [
    {"target": b, "source": [x for x in BASIN_CODES if x != b]}
    for b in BASIN_CODES
]

BEST_HPARAMS = {
    "erm":     {"lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4},
    "irm":     {"lr": 2e-3, "batch_size": 128, "weight_decay": 1e-4,
                "irm_lambda": 1.0, "warmup_steps": 500},
    "vrex":    {"lr": 2e-3, "batch_size": 128, "weight_decay": 1e-4,
                "beta": 1.0, "warmup_steps": 500},
    "coral":   {"lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                "coral_lambda": 1.0},
    "dann":    {"lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                "dann_lambda": 1.0, "total_steps": 10000},
    "maml":    {"lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
                "inner_lr": 1e-3, "inner_steps": 5},
    "physirm": {"lr": 1e-3, "batch_size": 128, "weight_decay": 1e-4,
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
    target_basin:  str,
    method_name:   str,
    run_id:        str,
    device:        torch.device,
) -> Dict:
    """Wrapper that sets up per-experiment log file then calls the inner fn."""
    run_timestamp = getattr(args, "run_timestamp", time.strftime("%Y%m%d_%H%M%S"))
    exp_log_dir   = Path("logs") / "experiments" / run_timestamp
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    exp_log_file  = exp_log_dir / f"{run_id}.log"

    fh = logging.FileHandler(exp_log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)

    try:
        log.info("=" * 80)
        log.info(f"[{run_id}] EXPERIMENT START")
        log.info(f"[{run_id}] Source={source_basins}  Target={target_basin}  Method={method_name}")
        log.info(f"[{run_id}] Device={device}  Log={exp_log_file}")
        for k, v in sorted(vars(args).items()):
            log.info(f"  {k}: {v}")
        log.info("=" * 80)
        return _train_one_experiment_inner(
            args, source_basins, target_basin, method_name, run_id, device
        )
    except Exception as e:
        log.error(f"[{run_id}] FAILED: {e}")
        raise
    finally:
        log.info(f"[{run_id}] EXPERIMENT END")
        log.info("=" * 80)
        logging.getLogger().removeHandler(fh)
        fh.close()


def _train_one_experiment_inner(
    args,
    source_basins: List[str],
    target_basin:  str,
    method_name:   str,
    run_id:        str,
    device:        torch.device,
) -> Dict:
    hp = BEST_HPARAMS.get(method_name, BEST_HPARAMS["erm"])
    bs = hp.get("batch_size", 64)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loaders_per_env = make_per_basin_loaders(
        root=args.data_root, basins=source_basins,
        split="train", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
        disable_tqdm=args.no_tqdm, cache=getattr(args, "cache_data", False),
    )
    val_loader_src = make_dataloader(
        root=args.data_root, basins=source_basins,
        split="val", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
        disable_tqdm=args.no_tqdm, cache=getattr(args, "cache_data", False),
    )
    val_loader_tgt = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="val", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
        disable_tqdm=args.no_tqdm, cache=getattr(args, "cache_data", False),
    )
    test_loader_tgt = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="test", batch_size=bs, num_workers=args.num_workers,
        use_3d=not args.no_3d, use_env=not args.no_env,
        disable_tqdm=args.no_tqdm, cache=getattr(args, "cache_data", False),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        log.info(f"[{run_id}] torch.compile enabled")
        model = torch.compile(model)

    # ── Method ────────────────────────────────────────────────────────────────
    method_kwargs = {k: v for k, v in hp.items()
                     if k not in {"lr", "batch_size", "weight_decay"}}

    # Regression loss weight injected from CLI
    method_kwargs["reg_weight"] = getattr(args, "reg_weight", 0.5)

    n_batches_estimate = max(1, max(len(l) for l in train_loaders_per_env.values()))

    if method_name == "dann":
        method_kwargs["feature_dim"] = model.backbone.get_output_dim()
        method_kwargs.setdefault("n_domains", len(BASIN_CODES))
        method_kwargs["total_steps"] = n_batches_estimate * args.epochs
    if method_name == "physirm":
        method_kwargs["phys_dim"] = model.backbone.phys_dim

    method = build_method(method_name, **method_kwargs)

    extra_params = []
    if isinstance(method, (DANN, PhysIRM)):
        method.to(device)
        extra_params = list(method.parameters())

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    all_params = list(model.parameters()) + extra_params
    fused_kw   = ({"fused": True}
                  if device.type == "cuda"
                  and "fused" in optim.AdamW.__init__.__code__.co_varnames
                  else {})
    optimizer  = optim.AdamW(all_params, lr=hp["lr"],
                             weight_decay=hp.get("weight_decay", 1e-4), **fused_kw)
    # Scheduler: CosineAnnealingLR (default) or OneCycleLR for super-convergence.
    use_onecycle = getattr(args, "scheduler", "cosine") == "onecycle"
    if use_onecycle:
        total_steps = n_batches_estimate * args.epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=hp["lr"] * 10,
            total_steps=max(total_steps, 1),
            pct_start=0.1, anneal_strategy="cos",
        )
    else:
        scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # A100 rarely overflows at fp16; smaller init_scale reduces noisy scale oscil.
    scaler     = _make_scaler(init_scale=2**14) if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_ckpt   = None
    step        = 0
    history     = []

    total_train = sum(len(l.dataset) for l in train_loaders_per_env.values())
    log.info(
        f"Model params: {sum(p.numel() for p in model.parameters()):,}  "
        f"Device: {device}  "
        f"Train: {total_train:,}  Val-src: {len(val_loader_src.dataset):,}  "
        f"Val-tgt: {len(val_loader_tgt.dataset):,}  Test-tgt: {len(test_loader_tgt.dataset):,}"
    )
    log.info(f"Started Training")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if isinstance(method, torch.nn.Module):
            method.train()
        epoch_metrics: Dict[str, float] = {}
        t_epoch = time.time()

        env_iters = {b: iter(loader) for b, loader in train_loaders_per_env.items()}
        n_batches = max(1, max(len(l) for l in train_loaders_per_env.values()))

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch:3d}/{args.epochs}",
                    unit="batch", leave=False, dynamic_ncols=True,
                    disable=args.no_tqdm)

        for batch_idx in pbar:
            batches = {}
            skip = False
            for b, it in env_iters.items():
                try:
                    batch = next(it)
                except StopIteration:
                    env_iters[b] = iter(train_loaders_per_env[b])
                    batch = next(env_iters[b])
                if batch is None:  # tcnd_collate_fn: all samples had -1 labels
                    skip = True
                    break
                batches[b] = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            if skip:
                continue

            metrics = method.update(optimizer, batches, model,
                                    step=step, scaler=scaler, device=device)
            step += 1
            # OneCycleLR steps once per batch (not per epoch)
            if use_onecycle:
                scheduler.step()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            if "loss" in epoch_metrics:
                pbar.set_postfix(loss=f"{epoch_metrics['loss'] / (batch_idx+1):.4f}")

        pbar.close()
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} [{time.time()-t_epoch:.1f}s]  "
            + "  ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items())
        )
        if not use_onecycle:
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            if isinstance(method, torch.nn.Module):
                method.eval()

            ev_src = BasinEvaluator("source_val")
            ev_tgt = BasinEvaluator(target_basin)
            t_val  = time.time()

            with torch.no_grad():
                for batch in tqdm(val_loader_src, desc="Val-src",
                                  leave=False, dynamic_ncols=True, disable=args.no_tqdm):
                    if batch is None: continue
                    batch = {k: v.to(device, non_blocking=True)
                             if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda")
                    ):
                        out = model(batch)
                    ev_src.update(batch, out)

                for batch in tqdm(val_loader_tgt, desc="Val-tgt",
                                  leave=False, dynamic_ncols=True, disable=args.no_tqdm):
                    if batch is None: continue
                    batch = {k: v.to(device, non_blocking=True)
                             if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda")
                    ):
                        out = model(batch)
                    ev_tgt.update(batch, out)

            log.info(f"Validation took {time.time()-t_val:.2f}s")
            r_src = ev_src.compute()
            r_tgt = ev_tgt.compute()

            log.info(
                f"Epoch {epoch:3d}/{args.epochs} "
                f"| src acc_int={r_src.accuracy_intensity:.3f}"
                f" f1_int={r_src.f1_intensity:.3f}"
                f" ri_f1={r_src.rapid_intensification_f1:.3f}"
                f" mae_wnd={r_src.mae_wind_ms:.2f}m/s"
                f" mae_prs={r_src.mae_pres_hpa:.2f}hPa"
                f" | tgt acc_int={r_tgt.accuracy_intensity:.3f}"
                f" f1_int={r_tgt.f1_intensity:.3f}"
                f" ri_f1={r_tgt.rapid_intensification_f1:.3f}"
                f" mae_wnd={r_tgt.mae_wind_ms:.2f}m/s"
                f" mae_prs={r_tgt.mae_pres_hpa:.2f}hPa"
                f" | loss={epoch_metrics.get('loss', float('nan')):.4f}"
            )

            # Checkpoint best model based on SOURCE val F1 (avoids target leakage)
            if r_src.f1_intensity > best_val_f1:
                best_val_f1 = r_src.f1_intensity
                if args.output_dir:
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    ckpt_path = Path(args.output_dir) / f"{run_id}_best.pt"
                    torch.save({
                        "epoch":  epoch,
                        "model":  model.state_dict(),
                        "method": method.state_dict()
                                  if isinstance(method, torch.nn.Module) else {},
                    }, ckpt_path)
                    best_ckpt = str(ckpt_path)

            history.append({
                "epoch": epoch,
                # Source
                "source accuracy intensity":              r_src.accuracy_intensity,
                "source precision intensity":             r_src.precision_intensity,
                "source recall intensity":                r_src.recall_intensity,
                "source f1 intensity":                    r_src.f1_intensity,
                "source accuracy direction":              r_src.accuracy_direction,
                "source precision direction":             r_src.precision_direction,
                "source recall direction":                r_src.recall_direction,
                "source f1 direction":                    r_src.f1_direction,
                "source rapid intensification recall":    r_src.rapid_intensification_recall,
                "source rapid intensification precision": r_src.rapid_intensification_precision,
                "source rapid intensification f1":        r_src.rapid_intensification_f1,
                "source mae wind ms":                     r_src.mae_wind_ms,
                "source mae pres hpa":                    r_src.mae_pres_hpa,
                # Target
                "target accuracy intensity":              r_tgt.accuracy_intensity,
                "target precision intensity":             r_tgt.precision_intensity,
                "target recall intensity":                r_tgt.recall_intensity,
                "target f1 intensity":                    r_tgt.f1_intensity,
                "target accuracy direction":              r_tgt.accuracy_direction,
                "target precision direction":             r_tgt.precision_direction,
                "target recall direction":                r_tgt.recall_direction,
                "target f1 direction":                    r_tgt.f1_direction,
                "target rapid intensification recall":    r_tgt.rapid_intensification_recall,
                "target rapid intensification precision": r_tgt.rapid_intensification_precision,
                "target rapid intensification f1":        r_tgt.rapid_intensification_f1,
                "target mae wind ms":                     r_tgt.mae_wind_ms,
                "target mae pres hpa":                    r_tgt.mae_pres_hpa,
                **epoch_metrics,
            })

        # Save latest checkpoint every epoch
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":  epoch,
                "model":  model.state_dict(),
                "method": method.state_dict()
                          if isinstance(method, torch.nn.Module) else {},
            }, Path(args.output_dir) / f"{run_id}_latest.pt")

    # ── Final evaluation ──────────────────────────────────────────────────────
    if best_ckpt:
        log.info(f"Loading best checkpoint: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        if isinstance(method, torch.nn.Module) and ckpt.get("method"):
            method.load_state_dict(ckpt["method"])

    model.eval()
    if isinstance(method, torch.nn.Module):
        method.eval()

    ev_final_src = BasinEvaluator("source_final")
    ev_final     = BasinEvaluator(target_basin)
    t_final      = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader_src, desc="Final-src",
                          leave=False, dynamic_ncols=True, disable=args.no_tqdm):
            if batch is None: continue
            batch = {k: v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda")
            ):
                out = model(batch)
            ev_final_src.update(batch, out)

        for batch in tqdm(test_loader_tgt, desc="Final-tgt",
                          leave=False, dynamic_ncols=True, disable=args.no_tqdm):
            if batch is None: continue
            batch = {k: v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda")
            ):
                out = model(batch)
            ev_final.update(batch, out)

    log.info(f"Final evaluation took {time.time()-t_final:.2f}s")

    final_src = ev_final_src.compute()
    final     = ev_final.compute()
    del ev_final_src, ev_final

    btg  = basin_transfer_gap(final_src.accuracy_intensity, final.accuracy_intensity)
    bnte = basin_normalized_transfer_efficiency(
        final_src.accuracy_intensity, final.accuracy_intensity, 0.0
    )

    result = {
        "run_id":        run_id,
        "method":        method_name,
        "source_basins": source_basins,
        "target_basin":  target_basin,
        # ── Source ──────────────────────────────────────────────────────────
        "final source accuracy intensity":              final_src.accuracy_intensity,
        "final source precision intensity":             final_src.precision_intensity,
        "final source recall intensity":                final_src.recall_intensity,
        "final source f1 intensity":                    final_src.f1_intensity,
        "final source accuracy direction":              final_src.accuracy_direction,
        "final source precision direction":             final_src.precision_direction,
        "final source recall direction":                final_src.recall_direction,
        "final source f1 direction":                    final_src.f1_direction,
        "final source rapid intensification f1":        final_src.rapid_intensification_f1,
        "final source mae wind ms":                     final_src.mae_wind_ms,
        "final source mae pres hpa":                    final_src.mae_pres_hpa,
        # ── Target ──────────────────────────────────────────────────────────
        "final target accuracy intensity":              final.accuracy_intensity,
        "btg":  btg,
        "bnte": bnte,
        "final target precision intensity":             final.precision_intensity,
        "final target recall intensity":                final.recall_intensity,
        "final target f1 intensity":                    final.f1_intensity,
        "final target accuracy direction":              final.accuracy_direction,
        "final target precision direction":             final.precision_direction,
        "final target recall direction":                final.recall_direction,
        "final target f1 direction":                    final.f1_direction,
        "final target rapid intensification f1":        final.rapid_intensification_f1,
        "final target rapid intensification precision": final.rapid_intensification_precision,
        "final target rapid intensification recall":    final.rapid_intensification_recall,
        # ── Regression ──────────────────────────────────────────────────────
        "final target mae wind ms":                     final.mae_wind_ms,
        "final target mae pres hpa":                    final.mae_pres_hpa,
        "final target mae wind norm":                   final.mae_wind_norm,
        "final target mae pres norm":                   final.mae_pres_norm,
        "final target n reg samples":                   final.n_reg_samples,
        # ── Metadata ────────────────────────────────────────────────────────
        "history":   history,
        "best_ckpt": best_ckpt,
    }

    # ── Few-shot fine-tuning (optional) ───────────────────────────────────────
    if args.few_shot:
        fs = few_shot_finetune(
            model=model, target_basin=target_basin,
            args=args, device=device,
            k_shots=args.k_shots, ft_epochs=args.few_shot_epochs,
        )
        result.update({
            "few_shot target accuracy intensity":          fs.accuracy_intensity,
            "few_shot target precision intensity":         fs.precision_intensity,
            "few_shot target recall intensity":            fs.recall_intensity,
            "few_shot target f1 intensity":                fs.f1_intensity,
            "few_shot target accuracy direction":          fs.accuracy_direction,
            "few_shot target precision direction":         fs.precision_direction,
            "few_shot target recall direction":            fs.recall_direction,
            "few_shot target f1 direction":                fs.f1_direction,
            "few_shot target rapid intensification f1":    fs.rapid_intensification_f1,
            "few_shot target mae wind ms":                 fs.mae_wind_ms,
            "few_shot target mae pres hpa":                fs.mae_pres_hpa,
        })
        log.info(
            f"Few-shot ({args.k_shots}-shot) {target_basin}: "
            f"acc_int={fs.accuracy_intensity:.3f} f1_int={fs.f1_intensity:.3f} "
            f"ri_f1={fs.rapid_intensification_f1:.3f} "
            f"mae_wnd={fs.mae_wind_ms:.2f}m/s mae_prs={fs.mae_pres_hpa:.2f}hPa"
        )

    return result


# ── Few-shot fine-tuning ──────────────────────────────────────────────────────

def few_shot_finetune(
    model:         TropiCycloneModel,
    target_basin:  str,
    args,
    device:        torch.device,
    k_shots:       int   = 32,
    ft_lr:         float = 1e-4,
    ft_epochs:     int   = 5,
) -> BasinResult:
    """
    Freeze backbone, fine-tune heads for ft_epochs on k_shots examples from
    the target basin's train split, then evaluate on the target test split.
    """
    log.info(f"Few-shot fine-tuning on {target_basin} with {k_shots} shots")
    reg_weight = getattr(args, "reg_weight", 0.5)

    shot_ds = TCNDDataset(
        root=args.data_root, basins=[target_basin], split="train",
        use_3d=not args.no_3d, use_env=not args.no_env,
        cache=getattr(args, "cache_data", False),
    )
    k_shots = min(k_shots, len(shot_ds))

    # Deep copy to prevent contamination of zero-shot baseline weights
    orig_state = copy.deepcopy(model.state_dict())

    if k_shots > 0:
        indices    = torch.randperm(
            len(shot_ds),
            generator=torch.Generator().manual_seed(42 + k_shots)
        )[:k_shots].tolist()
        subset     = torch.utils.data.Subset(shot_ds, indices)
        shot_loader = torch.utils.data.DataLoader(
            subset, batch_size=max(1, min(k_shots, 32)), shuffle=True
        )

        # Freeze backbone, fine-tune heads only
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.heads.parameters():
            p.requires_grad_(True)

        fused_kw    = ({"fused": True}
                       if device.type == "cuda"
                       and "fused" in optim.Adam.__init__.__code__.co_varnames
                       else {})
        ft_optimizer = optim.Adam(model.heads.parameters(), lr=ft_lr, **fused_kw)
        ft_scaler    = _make_scaler() if device.type == "cuda" else None

        model.eval()
        model.heads.train()
        t_fs = time.time()

        for ep in tqdm(range(ft_epochs), desc="Few-shot epochs",
                       leave=False, dynamic_ncols=True, disable=args.no_tqdm):
            for batch in shot_loader:
                if batch is None: continue
                batch = {k: v.to(device, non_blocking=True)
                         if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                ft_optimizer.zero_grad()
                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=(device.type == "cuda")
                ):
                    out  = model(batch)
                    loss = task_loss(
                        out["logits_intensity"], out["logits_direction"],
                        batch["y_intensity"],    batch["y_direction"],
                        pred_reg   = out.get("pred_intensity_reg"),
                        y_wind_reg = batch.get("y_wind_reg"),
                        y_pres_reg = batch.get("y_pres_reg"),
                        reg_weight = reg_weight,
                    )
                if ft_scaler is not None:
                    ft_scaler.scale(loss).backward()
                    ft_scaler.step(ft_optimizer)
                    ft_scaler.update()
                else:
                    loss.backward()
                    ft_optimizer.step()

        log.info(f"Few-shot fine-tuning took {time.time()-t_fs:.2f}s")

        for p in model.backbone.parameters():
            p.requires_grad_(True)
    else:
        log.info(f"k_shots=0 for {target_basin}; skipping fine-tuning.")

    # Evaluate on target test split
    test_loader = make_dataloader(
        root=args.data_root, basins=[target_basin],
        split="test", batch_size=64, num_workers=0,
        use_3d=not args.no_3d, use_env=not args.no_env,
        disable_tqdm=args.no_tqdm, cache=getattr(args, "cache_data", False),
    )
    ev = BasinEvaluator(target_basin)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Few-shot eval",
                          leave=False, dynamic_ncols=True, disable=args.no_tqdm):
            if batch is None: continue
            batch = {k: v.to(device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=(device.type == "cuda")
            ):
                out = model(batch)
            ev.update(batch, out)

    # Restore original weights so the zero-shot result is preserved
    model.load_state_dict(orig_state)

    return ev.compute()


# ── Benchmark runners ─────────────────────────────────────────────────────────

def run_lobo_benchmark(args):
    """
    Leave-One-Basin-Out benchmark: trains all methods × all 6 LOBO splits.
    Results saved to args.output_dir/benchmark_results.json.
    """
    device = select_device(args)
    log.info(f"Device: {device}")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    method_arg = args.methods or args.method
    methods = ([m.strip() for m in method_arg.split(",")]
               if method_arg else list(METHOD_REGISTRY.keys()))
    splits  = (LOBO_SPLITS if not args.target_basin else [
        {"target": args.target_basin,
         "source": [b for b in BASIN_CODES if b != args.target_basin]}
    ])

    all_results = []
    t_bench = time.time()

    for split in tqdm(splits, desc="LOBO splits",
                      dynamic_ncols=True, disable=args.no_tqdm):
        for method_name in tqdm(methods, desc="Methods",
                                leave=False, dynamic_ncols=True, disable=args.no_tqdm):
            t_exp  = time.time()
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
                    f"✓ {run_id}: "
                    f"final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                    f"| final target f1 intensity={result['final target f1 intensity']:.3f} "
                    f"| final target accuracy direction={result['final target accuracy direction']:.3f} "
                    f"| final target f1 direction={result['final target f1 direction']:.3f} "
                    f"| final target mae wind ms={result['final target mae wind ms']:.2f} "
                    f"| final target mae pres hpa={result['final target mae pres hpa']:.2f} "
                    f"[Took {time.time()-t_exp:.2f}s]"
                )
            except Exception as e:
                log.error(f"✗ {run_id} FAILED: {e}")
                if args.fail_fast:
                    raise
            # Free GPU memory between folds to prevent cross-experiment OOM.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.output_dir:
        out_path = Path(args.output_dir) / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"Results → {out_path}")

    _print_summary_table(all_results, methods, splits)
    log.info(f"LOBO total: {time.time()-t_bench:.1f}s")
    return all_results


def run_incremental_benchmark(args):
    """
    Incremental benchmark: for a target basin, train on 1, 2, 3, … N source basins.
    """
    device = select_device(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    method_arg = args.methods or args.method
    methods = ([m.strip() for m in method_arg.split(",")]
               if method_arg else list(METHOD_REGISTRY.keys()))
    targets = [args.target_basin] if args.target_basin else BASIN_CODES

    all_results = []
    t_bench = time.time()

    for target in tqdm(targets, desc="Targets",
                       dynamic_ncols=True, disable=args.no_tqdm):
        available_sources = (
            [s.strip() for s in args.source_basins.split(",")]
            if args.source_basins
            else [b for b in BASIN_CODES if b != target]
        )
        for i in tqdm(range(1, len(available_sources) + 1),
                      desc="Incremental sources", leave=False,
                      dynamic_ncols=True, disable=args.no_tqdm):
            src = available_sources[:i]
            for method_name in tqdm(methods, desc="Methods",
                                    leave=False, dynamic_ncols=True,
                                    disable=args.no_tqdm):
                t_exp  = time.time()
                run_id = f"{method_name}_{target}_src{i}"
                try:
                    result = train_one_experiment(
                        args=args, source_basins=src, target_basin=target,
                        method_name=method_name, run_id=run_id, device=device,
                    )
                    all_results.append(result)
                    log.info(
                        f"✓ {run_id}: "
                        f"final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                        f"| final target f1 intensity={result['final target f1 intensity']:.3f} "
                        f"| final target accuracy direction={result['final target accuracy direction']:.3f} "
                        f"| final target f1 direction={result['final target f1 direction']:.3f} "
                        f"| final target mae wind ms={result['final target mae wind ms']:.2f} "
                        f"| final target mae pres hpa={result['final target mae pres hpa']:.2f} "
                        f"[Took {time.time()-t_exp:.2f}s]"
                    )
                except Exception as e:
                    log.error(f"✗ {run_id} FAILED: {e}")
                    if args.fail_fast:
                        raise

    if args.output_dir:
        out_path = Path(args.output_dir) / "incremental_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info(f"Incremental results → {out_path}")

    log.info(f"Incremental total: {time.time()-t_bench:.1f}s")
    return all_results


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary_table(results, methods, splits):
    """NeurIPS-style results table for all key metrics."""
    targets = [s["target"] for s in splits]

    metrics_to_print = [
        ("Intensity Accuracy",    "final source accuracy intensity",   "final target accuracy intensity"),
        ("Intensity Weighted F1", "final source f1 intensity",          "final target f1 intensity"),
        ("MAE Wind (m/s)",        "final source mae wind ms",           "final target mae wind ms"),
        ("MAE Pres (hPa)",        "final source mae pres hpa",          "final target mae pres hpa"),
        ("Direction Accuracy",    "final source accuracy direction",    "final target accuracy direction"),
        ("Direction F1",          "final source f1 direction",          "final target f1 direction"),
    ]

    for metric_title, src_key, tgt_key in metrics_to_print:
        is_single  = len(targets) == 1
        table_width = 14 + (22 if is_single else len(targets) * 22 + 22)

        log.info("\n" + "=" * table_width)
        log.info(f"BASIN GENERALIZATION — {metric_title}")
        log.info("=" * table_width)

        if is_single:
            t = targets[0]
            log.info(f"{'Method':<14}{'Source':>10} {f'Test[{t}]':>11}")
        else:
            h1 = f"{'Method':<14}"
            for t in targets:
                h1 += f"{t:^22}"
            h1 += f"{'Average':^22}"
            log.info(h1)
            h2 = f"{'':<14}"
            for t in targets:
                h2 += f"{'Source':>10} {f'Test[{t}]':>11}"
            h2 += f"{'Src Avg':>10} {'Test Avg':>11}"
            log.info(h2)

        log.info("-" * table_width)

        for method in methods:
            row = f"{method:<14}"
            src_vals, tgt_vals = [], []
            for t in targets:
                r = next(
                    (x for x in results
                     if x.get("method") == method and x.get("target_basin") == t),
                    None,
                )
                if r and src_key in r and tgt_key in r:
                    sv, tv = r[src_key], r[tgt_key]
                    src_vals.append(sv)
                    tgt_vals.append(tv)
                    row += f"{sv:>10.3f} {tv:>11.3f}"
                else:
                    row += f"{'—':>10} {'—':>11}"
            if not is_single:
                row += (f"{sum(src_vals)/len(src_vals):>10.3f} "
                        f"{sum(tgt_vals)/len(tgt_vals):>11.3f}"
                        if src_vals else f"{'—':>10} {'—':>11}")
            log.info(row)

        log.info("=" * table_width)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PhysIRM: Physics-Guided Basin Generalization Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data
    p.add_argument("--data_root",   required=True,
                   help="Path to TCND root (contains Data1D/, Data3D/, Env-Data/)")
    p.add_argument("--output_dir",  default="./runs")
    p.add_argument("--num_workers", type=int, default=12)

    # Experiment mode
    p.add_argument("--mode", choices=["lobo", "single", "incremental"],
                   default="lobo")
    p.add_argument("--source_basins", default=None,
                   help="Comma-separated source basins (single/incremental mode)")
    p.add_argument("--target_basin",  default=None)
    p.add_argument("--method",  default=None,
                   help="Single method for single mode")
    p.add_argument("--methods", default=None,
                   help="Comma-separated methods (default: all)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--eval_every", type=int,   default=5,
                   help="Evaluate every N epochs (default 5; set to 1 for dense logging)")
    p.add_argument("--device",     default="auto",
                   help="cuda | mps | cpu | auto")
    p.add_argument("--scheduler", choices=["cosine", "onecycle"], default="cosine",
                   help="LR scheduler (cosine=CosineAnnealingLR, onecycle=OneCycleLR)")
    p.add_argument("--fail_fast",  action="store_true")

    # Model architecture
    p.add_argument("--model_size", choices=["lightweight", "complex"],
                   default="lightweight")
    p.add_argument("--spatial_embed", type=int, default=None)
    p.add_argument("--track_embed",   type=int, default=None)
    p.add_argument("--env_embed",     type=int, default=None)
    p.add_argument("--phys_dim",      type=int, default=None)
    p.add_argument("--final_dim",     type=int, default=None)
    p.add_argument("--dropout",       type=float, default=0.1)

    # Loss weights
    p.add_argument("--reg_weight", type=float, default=0.5,
                   help="Weight on the 24h-ahead intensity regression (MSE) loss "
                        "(0 = classification only; 1 = equal weight)")

    # Ablations
    p.add_argument("--no_3d",  action="store_true",
                   help="Disable Data_3d spatial branch")
    p.add_argument("--no_env", action="store_true",
                   help="Disable Env-Data branch")
    p.add_argument("--no_tqdm", action="store_true")

    # Performance
    p.add_argument("--compile",     action="store_true",
                   help="torch.compile (A100/H100 speedup)")
    p.add_argument("--cache_data",  action="store_true",
                   help="Cache full dataset in RAM (~50 GB)")

    # Few-shot
    p.add_argument("--few_shot",        action="store_true")
    p.add_argument("--k_shots",         type=int, default=32)
    p.add_argument("--few_shot_epochs", type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import platform
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp         = time.strftime("%Y%m%d_%H%M%S")
    args.run_timestamp = timestamp

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark       = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    fh = logging.FileHandler(log_dir / f"train_{args.mode}_{timestamp}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)

    log.info("=" * 80)
    log.info(f"Mode: {args.mode}  Timestamp: {timestamp}")
    log.info(f"OS: {platform.system()} {platform.machine()}  "
             f"Python: {sys.version.split()[0]}  PyTorch: {torch.__version__}")
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
        source  = [s.strip() for s in args.source_basins.split(",")]
        device  = select_device(args)
        method_arg = args.method or args.methods
        methods = ([m.strip() for m in method_arg.split(",")]
                   if method_arg else list(METHOD_REGISTRY.keys()))
        for method_name in methods:
            result = train_one_experiment(
                args=args, source_basins=source,
                target_basin=args.target_basin,
                method_name=method_name,
                run_id=f"{method_name}_{args.target_basin}",
                device=device,
            )
            log.info(
                f"Final [{method_name}→{args.target_basin}]: "
                f"final target accuracy intensity={result['final target accuracy intensity']:.3f} "
                f"| final target f1 intensity={result['final target f1 intensity']:.3f} "
                f"| final target accuracy direction={result['final target accuracy direction']:.3f} "
                f"| final target f1 direction={result['final target f1 direction']:.3f} "
                f"| final target mae wind ms={result['final target mae wind ms']:.2f} "
                f"| final target mae pres hpa={result['final target mae pres hpa']:.2f}"
            )