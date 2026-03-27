"""
Experiment A: Pooled Baseline
=============================
Train on ALL 6 basins (years 2017-2020), validate on 2021, test on 2022-2023.
This is the "oracle" upper bound for in-distribution performance.

Runs:
1. Persistence baseline (no training needed)
2. LSTM Seq2Seq (no basin conditioning)
3. LSTM Seq2Seq + attention (with basin embedding)
4. Transformer Seq2Seq

Reports per-basin test metrics and horizon-level errors.
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
from src.models.baselines import PersistenceModel, LinearTrendModel
from src.models.lstm_seq2seq import LSTMSeq2Seq, LSTMSeq2SeqAttn
from src.models.transformer_seq2seq import TransformerForecaster
# Trainer and metrics imported lazily inside functions to avoid stale Colab imports


OBS_LEN = 8
PRED_LEN = 4
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100


def build_loaders(data_root=None, use_env: bool = False, batch_size: int = BATCH_SIZE):
    if data_root is None:
        data_root = ROOT / "TCND_test"
    data_root = Path(data_root)
    save_dir = data_root.parent / "results"
    save_dir.mkdir(exist_ok=True)

    train_ds = TCTrackDataset(data_root, train_years=TRAIN_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN, use_env=use_env)
    val_ds   = TCTrackDataset(data_root, train_years=VAL_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN, use_env=use_env)
    test_ds  = TCTrackDataset(data_root, train_years=TEST_YEARS,
                               obs_len=OBS_LEN, pred_len=PRED_LEN, use_env=use_env)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


def run_persistence_baseline(test_loader, device):
    from src.training.metrics import evaluate_by_basin
    print("\n--- Persistence Baseline ---")
    model = PersistenceModel(pred_len=PRED_LEN).to(device)
    results = evaluate_by_basin(model, test_loader, device)
    print(pd.DataFrame(results).T.round(2))
    return results


def run_linear_baseline(test_loader, device):
    from src.training.metrics import evaluate_by_basin
    print("\n--- Linear Trend Baseline ---")
    model = LinearTrendModel(pred_len=PRED_LEN).to(device)
    results = evaluate_by_basin(model, test_loader, device)
    print(pd.DataFrame(results).T.round(2))
    return results


def run_lstm(train_loader, val_loader, test_loader, device,
             use_attn: bool = False, use_basin: bool = True,
             experiment_name: str = "lstm_pooled",
             save_dir: str = "results/checkpoints"):
    from src.training.trainer import Trainer
    from src.training.metrics import evaluate_by_basin, evaluate_dataset
    print(f"\n--- LSTM {'Attn' if use_attn else 'Seq2Seq'} (basin={use_basin}) ---")

    ModelClass = LSTMSeq2SeqAttn if use_attn else LSTMSeq2Seq
    model = ModelClass(
        obs_feat_dim=6,
        hidden_dim=128,
        pred_len=PRED_LEN,
        n_basins=6 if use_basin else 0,
        basin_emb_dim=16,
        num_layers=2,
        dropout=0.1,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        patience=15,
        experiment_name=experiment_name,
        save_dir=save_dir,
    )
    history = trainer.train()
    trainer.load_best()

    results = evaluate_by_basin(model, test_loader, device)
    print("Per-basin test results:"); print(pd.DataFrame(results).T.round(2))
    overall = evaluate_dataset(model, test_loader, device)
    print("Overall:", {k: round(v, 2) for k, v in overall.items()})
    return model, results, history


def run_transformer(train_loader, val_loader, test_loader, device,
                    use_basin: bool = True, experiment_name: str = "transformer_pooled",
                    save_dir: str = "results/checkpoints"):
    from src.training.trainer import Trainer
    from src.training.metrics import evaluate_by_basin, evaluate_dataset
    print(f"\n--- Transformer (basin={use_basin}) ---")

    model = TransformerForecaster(
        obs_feat_dim=6,
        d_model=128,
        nhead=4,
        num_enc_layers=3,
        num_dec_layers=2,
        pred_len=PRED_LEN,
        dim_feedforward=256,
        n_basins=6 if use_basin else 0,
        basin_emb_dim=16,
        dropout=0.1,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=5e-4,
        max_epochs=MAX_EPOCHS,
        patience=15,
        experiment_name=experiment_name,
        save_dir=save_dir,
    )
    history = trainer.train()
    trainer.load_best()

    results = evaluate_by_basin(model, test_loader, device)
    print("Per-basin test results:"); print(pd.DataFrame(results).T.round(2))
    overall = evaluate_dataset(model, test_loader, device)
    print("Overall:", {k: round(v, 2) for k, v in overall.items()})
    return model, results, history


def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, *_ = build_loaders(use_env=False)

    all_results = {}

    # Non-ML baselines
    all_results["persistence"]  = run_persistence_baseline(test_loader, device)
    all_results["linear_trend"] = run_linear_baseline(test_loader, device)

    # LSTM variants
    _, res, _ = run_lstm(train_loader, val_loader, test_loader, device,
                          use_attn=False, use_basin=False, experiment_name="lstm_no_basin")
    all_results["lstm_no_basin"] = res

    _, res, _ = run_lstm(train_loader, val_loader, test_loader, device,
                          use_attn=False, use_basin=True, experiment_name="lstm_basin")
    all_results["lstm_basin"] = res

    _, res, _ = run_lstm(train_loader, val_loader, test_loader, device,
                          use_attn=True, use_basin=True, experiment_name="lstm_attn_basin")
    all_results["lstm_attn_basin"] = res

    # Transformer
    _, res, _ = run_transformer(train_loader, val_loader, test_loader, device,
                                 use_basin=True, experiment_name="transformer_basin")
    all_results["transformer_basin"] = res

    # Save summary table
    rows = []
    for model_name, basin_results in all_results.items():
        for basin, metrics in basin_results.items():
            rows.append({"model": model_name, "basin": basin, **metrics})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(RESULTS_DIR / "pooled_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'pooled_results.csv'}")
    return all_results


if __name__ == "__main__":
    run_all()
