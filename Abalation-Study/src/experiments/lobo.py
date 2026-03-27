"""
Experiment B: Leave-One-Basin-Out (LOBO)
=========================================
For each target basin B:
  - Train on ALL OTHER basins (years 2017-2020)
  - Validate on ALL OTHER basins (2021)
  - Test on held-out basin B (2022-2023)

This quantifies zero-shot cross-basin generalization.

Key research question: How much performance degrades when a model
has never seen data from the target basin during training?
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TCTrackDataset, BASINS
from src.data.utils import TRAIN_YEARS, VAL_YEARS, TEST_YEARS
from src.models.lstm_seq2seq import LSTMSeq2Seq, LSTMSeq2SeqAttn
from src.models.transformer_seq2seq import TransformerForecaster
from src.models.baselines import PersistenceModel
# Trainer and metrics imported lazily inside functions to avoid stale Colab imports


OBS_LEN    = 8
PRED_LEN   = 4
BATCH_SIZE = 64
LR         = 1e-3
MAX_EPOCHS = 100


def build_lobo_loaders(held_out_basin: str, data_root=None, batch_size: int = BATCH_SIZE):
    """
    Train on all basins EXCEPT held_out_basin.
    Test on held_out_basin only.
    """
    if data_root is None:
        data_root = ROOT / "TCND_test"
    data_root = Path(data_root)
    source_basins = [b for b in BASINS if b != held_out_basin]

    train_ds = TCTrackDataset(data_root, basins=source_basins, train_years=TRAIN_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN)
    val_ds   = TCTrackDataset(data_root, basins=source_basins, train_years=VAL_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN)
    test_ds  = TCTrackDataset(data_root, basins=[held_out_basin], train_years=TEST_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN)

    print(f"  [LOBO held-out={held_out_basin}] "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if len(test_ds) == 0:
        print(f"  WARNING: No test data for {held_out_basin} in test years. Skipping.")
        return None, None, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_lobo_for_basin(held_out: str, model_type: str, device: torch.device,
                        data_root=None, save_dir: str = "results/checkpoints") -> dict:
    from src.training.trainer import Trainer
    from src.training.metrics import evaluate_dataset
    """Run a single LOBO experiment for one held-out basin."""
    train_loader, val_loader, test_loader = build_lobo_loaders(held_out, data_root=data_root)
    if test_loader is None:
        return {}

    exp_name = f"lobo_{model_type}_{held_out}"

    if model_type == "lstm":
        model = LSTMSeq2Seq(obs_feat_dim=6, hidden_dim=128, pred_len=PRED_LEN,
                             n_basins=0, num_layers=2, dropout=0.1)
    elif model_type == "lstm_attn":
        model = LSTMSeq2SeqAttn(obs_feat_dim=6, hidden_dim=128, pred_len=PRED_LEN,
                                 n_basins=0, num_layers=2, dropout=0.1)
    elif model_type == "transformer":
        model = TransformerForecaster(obs_feat_dim=6, d_model=128, nhead=4,
                                       num_enc_layers=3, num_dec_layers=2,
                                       pred_len=PRED_LEN, n_basins=0, dropout=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, lr=LR, max_epochs=MAX_EPOCHS, patience=15,
        experiment_name=exp_name, save_dir=save_dir,
    )
    trainer.train(verbose=True)
    trainer.load_best()

    metrics = evaluate_dataset(model, test_loader, device)
    print(f"  LOBO {held_out} ({model_type}) → ADE={metrics['ade_km']:.1f}km FDE={metrics['fde_km']:.1f}km")
    return metrics


def run_all_lobo(model_type: str = "lstm"):
    from src.training.metrics import evaluate_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Running LOBO experiments with model_type={model_type}")
    print("="*60)

    lobo_results = {}
    for basin in BASINS:
        print(f"\n{'─'*40}")
        print(f"Held-out basin: {basin}")
        lobo_results[basin] = run_lobo_for_basin(basin, model_type, device)

    # Also run persistence on each held-out basin
    print("\n--- Persistence baseline per held-out basin ---")
    persistence_results = {}
    for basin in BASINS:
        test_ds = TCTrackDataset(DATA_ROOT, basins=[basin], train_years=TEST_YEARS,
                                  obs_len=OBS_LEN, pred_len=PRED_LEN)
        if len(test_ds) == 0:
            continue
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        pers_model = PersistenceModel(pred_len=PRED_LEN).to(device)
        m = evaluate_dataset(pers_model, test_loader, device)
        persistence_results[basin] = m
        print(f"  {basin}: ADE={m['ade_km']:.1f}km FDE={m['fde_km']:.1f}km")

    # Build comparison table
    rows = []
    for basin in BASINS:
        if basin in persistence_results:
            rows.append({
                "basin": basin,
                "model": "persistence",
                **persistence_results[basin],
            })
        if basin in lobo_results and lobo_results[basin]:
            rows.append({
                "basin": basin,
                "model": f"lobo_{model_type}",
                **lobo_results[basin],
            })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"lobo_{model_type}_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.pivot(index="basin", columns="model", values="ade_km").round(1))

    return lobo_results, persistence_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "lstm_attn", "transformer"])
    args = parser.parse_args()
    run_all_lobo(model_type=args.model)
