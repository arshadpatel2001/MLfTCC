"""
Experiment C: Few-Shot Target Basin Adaptation
================================================
Starting from a LOBO-pretrained model (trained on source basins),
fine-tune on k% of the target basin's training data.

This plots the "learning curve" of adaptation:
  k = 0% → 5% → 10% → 20% → 50% → 100%

Key research question: How much data from a data-sparse basin (NI, SP)
is needed to close the performance gap vs in-distribution training?

Also tests: does the Transformer generalize better than LSTM in zero-shot?
(Hypothesis: attention over the obs window helps capture non-basin-specific dynamics)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from src.data.dataset import TCTrackDataset, BASINS
from src.data.utils import TRAIN_YEARS, VAL_YEARS, TEST_YEARS
from src.models.lstm_seq2seq import LSTMSeq2Seq, LSTMSeq2SeqAttn
from src.models.transformer_seq2seq import TransformerForecaster
from src.training.trainer import Trainer, load_model_checkpoint
from src.training.metrics import evaluate_dataset


_DEFAULT_DATA_ROOT   = ROOT / "TCND_test"
_DEFAULT_RESULTS_DIR = ROOT / "results"
_DEFAULT_CKPT_DIR    = ROOT / "results" / "checkpoints"

OBS_LEN    = 8
PRED_LEN   = 4
BATCH_SIZE = 32
FT_LR      = 1e-4  # Lower LR for fine-tuning
FT_EPOCHS  = 40
PATIENCE   = 10

# k-shot fractions to evaluate
K_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]


def make_model(model_type: str) -> nn.Module:
    if model_type == "lstm":
        return LSTMSeq2Seq(obs_feat_dim=6, hidden_dim=128, pred_len=PRED_LEN,
                            n_basins=0, num_layers=2, dropout=0.1)
    elif model_type == "lstm_attn":
        return LSTMSeq2SeqAttn(obs_feat_dim=6, hidden_dim=128, pred_len=PRED_LEN,
                                n_basins=0, num_layers=2, dropout=0.1)
    elif model_type == "transformer":
        return TransformerForecaster(obs_feat_dim=6, d_model=128, nhead=4,
                                      num_enc_layers=3, num_dec_layers=2,
                                      pred_len=PRED_LEN, n_basins=0, dropout=0.1)
    else:
        raise ValueError(model_type)


def run_fewshot_curve(
    target_basin: str,
    model_type: str = "lstm",
    device: torch.device = None,
    seed: int = 42,
    data_root=None,
    save_dir: str = None,
    results_dir: str = None,
) -> pd.DataFrame:
    """
    Run the full few-shot adaptation curve for one target basin.
    Returns DataFrame with columns: [k_fraction, n_samples, ade_km, fde_km, 6h, 12h, 18h, 24h]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")

    _data_root    = Path(data_root)    if data_root    else _DEFAULT_DATA_ROOT
    _save_dir     = save_dir           if save_dir     else str(_DEFAULT_CKPT_DIR)
    _results_dir  = Path(results_dir)  if results_dir  else _DEFAULT_RESULTS_DIR
    _results_dir.mkdir(exist_ok=True)
    lobo_ckpt = Path(_save_dir) / f"lobo_{model_type}_{target_basin}_best.pt"

    print(f"\n{'='*60}")
    print(f"Few-shot adaptation: target={target_basin}, model={model_type}")
    print("="*60)

    source_basins = [b for b in BASINS if b != target_basin]

    target_pool_ds = TCTrackDataset(_data_root, basins=[target_basin],
                                     train_years=TRAIN_YEARS,
                                     obs_len=OBS_LEN, pred_len=PRED_LEN)
    target_val_ds  = TCTrackDataset(_data_root, basins=[target_basin],
                                     train_years=VAL_YEARS,
                                     obs_len=OBS_LEN, pred_len=PRED_LEN)
    target_test_ds = TCTrackDataset(_data_root, basins=[target_basin],
                                     train_years=TEST_YEARS,
                                     obs_len=OBS_LEN, pred_len=PRED_LEN)

    test_loader = DataLoader(target_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    if len(target_test_ds) == 0:
        print(f"No test data for {target_basin}. Skipping.")
        return pd.DataFrame()

    rows = []
    for k in K_FRACTIONS:
        print(f"\n  k={k:.0%} fine-tuning fraction")

        # Fresh model for each k (start from LOBO weights if available)
        model = make_model(model_type).to(device)

        if lobo_ckpt.exists():
            model = load_model_checkpoint(model, str(lobo_ckpt), device)
            print(f"  Loaded LOBO checkpoint: {lobo_ckpt.name}")
        else:
            print(f"  No LOBO checkpoint found, using random init (train source basins first)")
            # Train from scratch on source basins
            src_train_ds = TCTrackDataset(_data_root, basins=source_basins,
                                           train_years=TRAIN_YEARS,
                                           obs_len=OBS_LEN, pred_len=PRED_LEN)
            src_val_ds   = TCTrackDataset(_data_root, basins=source_basins,
                                           train_years=VAL_YEARS,
                                           obs_len=OBS_LEN, pred_len=PRED_LEN)
            src_train_loader = DataLoader(src_train_ds, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, drop_last=True)
            src_val_loader   = DataLoader(src_val_ds,   batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=0)
            pretrain = Trainer(
                model=model, train_loader=src_train_loader, val_loader=src_val_loader,
                device=device, lr=1e-3, max_epochs=100, patience=15,
                experiment_name=f"lobo_{model_type}_{target_basin}",
                save_dir=_save_dir,
            )
            pretrain.train(verbose=False)
            pretrain.load_best()

        if k == 0.0:
            # Zero-shot: evaluate directly without any fine-tuning
            metrics = evaluate_dataset(model, test_loader, device)
        else:
            # Sample k-fraction of pool
            n_total = len(target_pool_ds)
            n_shots  = max(1, int(k * n_total))
            rng = np.random.default_rng(seed)
            indices = rng.choice(n_total, size=n_shots, replace=False)
            few_shot_ds = Subset(target_pool_ds, indices.tolist())

            ft_train_loader = DataLoader(few_shot_ds, batch_size=min(BATCH_SIZE, n_shots),
                                          shuffle=True, num_workers=0, drop_last=False)

            val_loader_ft = (DataLoader(target_val_ds, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)
                             if len(target_val_ds) > 0 else None)

            ft_val_loader = val_loader_ft or ft_train_loader  # fallback to train loader

            finetune = Trainer(
                model=model, train_loader=ft_train_loader, val_loader=ft_val_loader,
                device=device, lr=FT_LR, max_epochs=FT_EPOCHS, patience=PATIENCE,
                experiment_name=f"fewshot_{model_type}_{target_basin}_k{int(k*100)}",
                save_dir=_save_dir,
            )
            finetune.train(verbose=False)
            finetune.load_best()

            metrics = evaluate_dataset(model, test_loader, device)

        print(f"  ADE={metrics['ade_km']:.1f}km  FDE={metrics['fde_km']:.1f}km")
        rows.append({
            "target_basin": target_basin,
            "model_type": model_type,
            "k_fraction": k,
            "n_finetune_samples": int(k * len(target_pool_ds)),
            **metrics,
        })

    df = pd.DataFrame(rows)
    out_path = _results_dir / f"fewshot_{model_type}_{target_basin}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return df


def run_all_fewshot(model_type: str = "lstm", target_basins: list = None):
    """Run few-shot curves for all target basins (or specified subset)."""
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    if target_basins is None:
        # Focus on data-sparse basins (most scientifically interesting)
        target_basins = ["NI", "SP", "SI"]

    all_dfs = []
    for basin in target_basins:
        df = run_fewshot_curve(basin, model_type, device)
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = RESULTS_DIR / f"fewshot_{model_type}_all.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nAll few-shot results saved to {out_path}")
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "lstm_attn", "transformer"])
    parser.add_argument("--basins", nargs="+", default=["NI", "SP", "SI"],
                        help="Target basins for few-shot evaluation")
    args = parser.parse_args()
    run_all_fewshot(model_type=args.model, target_basins=args.basins)
