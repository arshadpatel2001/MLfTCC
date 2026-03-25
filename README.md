# TropiCycloneNet Basin Generalization Benchmark

Physics-guided domain generalization for **global tropical cyclone forecasting** across six ocean basins using the **TropiCycloneNet Dataset (TCND)** and a suite of domain generalization (DG) methods, culminating in the proposed **PhysIRM** — Physics-guided Invariant Risk Minimization.

---

## Overview

Tropical cyclone (TC) prediction models trained on data from one ocean basin frequently fail when applied to others, because each basin has distinct synoptic climatology (e.g. monsoon troughs, subtropical highs, Coriolis gradients). This benchmark asks: *can a model trained on source basins generalize zero-shot to an unseen target basin?*

Three simultaneous prediction tasks are evaluated:

| Task | Type | Target | Source |
|---|---|---|---|
| 1. Intensity change | 4-class classification | 24 h ahead | `future_inte_change24` from Env-Data `.npy` |
| 2. Track direction | 8-class classification | 24 h ahead | `future_direction24` from Env-Data `.npy` |
| 3. Intensity regression | Continuous regression | 24 h ahead | WND\_norm, PRES\_norm from Data\_1d row i+4 |

---

## Dataset — TCND

**TropiCycloneNet Dataset** (Huang et al., *Nature Communications* 2025). Three data modalities, six basins, ~70 years of records.

### Directory structure

```
TCND/
├── Data1D/                          ← Inherent attribute data (split subfolders here only)
│   ├── WP/ ├── NA/ ├── EP/
│   ├── NI/ ├── SI/ └── SP/
│       ├── train/                   ← Full training storms  (1950–2016 approx)
│       ├── val/                     ← Validation storms
│       └── test/                    ← Held-out test storms  (2017–2021 approx)
│           └── {BASIN}{YEAR}BST{NAME}.txt   (8-col, whitespace-separated)
│
├── Data3D/                          ← ERA5 meteorological grid patches (no split subfolder)
│   └── {BASIN}/{YEAR}/{NAME}/
│       └── TCND_{NAME}_{YYYYMMDDHH}_sst_z_u_v.nc    (13 channels, 81×81, 6-hourly)
│
└── Env-Data/                        ← Pre-computed environmental features (no split subfolder)
    └── {BASIN}/{YEAR}/{NAME}/
        └── {YYYYMMDDHH}.npy         (94-dim feature dict + classification labels)
```

### Modalities

**Data\_1d** — 8 columns per 6-hourly timestep (no header):

| Col | Name | Description |
|---|---|---|
| 0 | ID | Storm-level time step counter |
| 1 | FLAG | Quality flag |
| 2 | LAT\_norm | Normalized latitude (`LAT / 50`) |
| 3 | LONG\_norm | Normalized longitude (`(LONG − 1800) / 50`) |
| 4 | WND\_norm | Normalized max wind (`(WND − 40) / 25`) |
| 5 | PRES\_norm | Normalized min pressure (`(PRES − 960) / 50`) |
| 6 | YYYYMMDDHH | UTC timestamp |
| 7 | Name | Storm name |

Denormalization: `WND_ms = WND_norm × 25 + 40`,  `PRES_hPa = PRES_norm × 50 + 960`

**Data\_3d** — ERA5 reanalysis patches cropped to 20°×20° around the TC centre, 0.25° resolution, 6-hourly.

| Channel order | Variable | Pressure levels |
|---|---|---|
| 0–3 | U-wind | 200, 500, 850, 925 hPa |
| 4–7 | V-wind | 200, 500, 850, 925 hPa |
| 8–11 | Geopotential height | 200, 500, 850, 925 hPa |
| 12 | Sea surface temperature | — |

**Env-Data** — 94-dimensional environmental context vector + classification labels per `.npy`:

| Slice | Field | Encoding |
|---|---|---|
| [0:12] | month | one-hot (12) |
| [12:18] | area | one-hot (6) |
| [18:24] | intensity\_class | one-hot (6) |
| [24:25] | wind | scalar float |
| [25:26] | move\_velocity | scalar float |
| [26:62] | location\_long | one-hot (36) |
| [62:74] | location\_lat | one-hot (12) |
| [74:82] | history\_direction12 | class index → one-hot (8) |
| [82:90] | history\_direction24 | class index → one-hot (8) |
| [90:94] | history\_inte\_change24 | class index → one-hot (4) |
| — | future\_direction24 | **Task 2 label** (int 0–7, or −1) |
| — | future\_inte\_change24 | **Task 1 label** (int 0–3, or −1) |

Labels of −1 (unknown future) are filtered out during index construction and never reach training.

---

## Model Architecture

### Multimodal Backbone

Three parallel encoding branches fused into a joint representation `z ∈ ℝᵈ`.

| Branch | Input | Encoder |
|---|---|---|
| Spatial | Data\_3d `(13, 81, 81)` | CNN stack → pooling → projection |
| Track | Data\_1d `(4,)` | MLP → LayerNorm |
| Environment | Env-Data `(94,)` | MLP with structured sub-embeddings per field group |

Fused features are projected to `final_dim` via a two-layer MLP with GELU and LayerNorm.

For **PhysIRM**, `z` is partitioned into two disjoint sub-spaces:
- `z_phys ∈ ℝ^{d_p}` — encodes thermodynamic invariants (SST anomaly, wind shear, Coriolis, MPI proxy, boundary layer, outflow, steering). IRM penalty applied here.
- `z_env ∈ ℝ^{d_e}` — encodes basin-specific synoptic patterns. Allowed to shift across basins.

An 8-dim physics feature vector is computed analytically per sample and used to ground `z_phys`:

| Dim | Feature |
|---|---|
| 0 | SST anomaly vs 28°C reference |
| 1 | Wind shear proxy — −corr(U₂₀₀, U₈₅₀) |
| 2 | Coriolis parameter from latitude |
| 3 | Maximum potential intensity proxy (SST − 26°C) |
| 4 | Boundary-layer moisture proxy (WND\_norm) |
| 5 | Outflow proxy (Z₂₀₀ spatial skewness) |
| 6 | Steering proxy (Z₅₀₀ spatial skewness) |
| 7 | Current intensity (WND\_norm) |

### Task Heads

All three heads share the same `z`. Each head is `Linear(d) → GELU → Dropout → Linear(d_out)`.

| Head | Output | Loss |
|---|---|---|
| Intensity classification | `(B, 4)` logits | Cross-entropy |
| Direction classification | `(B, 8)` logits | Cross-entropy |
| Intensity regression | `(B, 2)` — [wind\_norm, pres\_norm] | NaN-masked MSE |

Combined loss: `L = 1.0 × CE_intensity + 0.5 × CE_direction + reg_weight × MSE_regression`

### Model sizes

| Size | Spatial embed | Track embed | Env embed | Physics dim | Final dim | Params |
|---|---|---|---|---|---|---|
| `lightweight` (default) | 128 | 32 | 64 | 32 | 128 | ~0.6 M |
| `complex` | 256 | 64 | 128 | 64 | 256 | ~4 M |

---

## Domain Generalization Methods

Seven methods are implemented, all sharing the same backbone and task heads. Each basin is treated as one environment.

| Method | Key idea | Reference |
|---|---|---|
| **ERM** | Empirical Risk Minimization — pool all source basins | DomainBed baseline |
| **IRM** | Gradient penalty on dummy scalar classifier w over the representation | Arjovsky et al., 2019 |
| **V-REx** | Penalise variance of per-basin losses | Krueger et al., ICML 2021 |
| **CORAL** | Align second-order feature statistics (covariance) across basins | Sun & Saenko, ECCV 2016 |
| **DANN** | Adversarial domain discriminator with gradient reversal | Ganin et al., JMLR 2016 |
| **MAML** | Meta-learning with per-basin inner-loop adaptation | Finn et al., ICML 2017 |
| **PhysIRM** *(proposed)* | IRM restricted to the physics sub-space + orthogonality + grounding | This work |

**PhysIRM total loss:**
```
L = ERM(z_phys ⊕ z_env)
  + λ_irm  · Σ_e ||∇_w R^e(w · z_phys)||²     [IRM penalty — physics sub-space only]
  + λ_orth · ||z_phys^T z_env||_F²              [orthogonality between sub-spaces]
  + λ_phys · MSE(predictor(z_phys), phys_feat)  [physics grounding]
```

Default hyperparameters (all methods use `batch_size=128`, `weight_decay=1e-4`, AdamW):

| Method | lr | Key extras |
|---|---|---|
| ERM, CORAL, DANN, MAML, PhysIRM | 1e-3 | — |
| IRM, V-REx | 2e-3 | warmup\_steps=500 |
| MAML | 1e-3 | inner\_lr=1e-3, inner\_steps=5 |
| PhysIRM | 1e-3 | λ_irm=1.0, λ_orth=0.1, λ_phys=0.5 |

---

## Evaluation Metrics

### Standard
- **Accuracy** and weighted **F1 / Precision / Recall** for intensity classification (4-class) and direction classification (8-class)
- **MAE** for 24 h-ahead wind (m/s) and pressure (hPa) regression

### Rapid Intensification (RI) Skill
Precision, Recall, and F1 computed exclusively for class 3 (ΔWind > +30 kt/24h). This is the primary safety metric — RI events are high-impact and rare.

### Proposed Metrics
| Metric | Formula | Interpretation |
|---|---|---|
| **BTG** — Basin Transfer Gap | `(src_acc − tgt_acc) / src_acc` | 0 = perfect transfer; higher = larger generalization gap |
| **BNTE** — Basin-Normalized Transfer Efficiency | `(tgt_acc − baseline_acc) / (src_acc − baseline_acc)` | 1 = matches source; 0 = no gain over ERM baseline |

CORAL distance is also computed between source and target feature distributions for domain gap analysis.

---

## Project Structure

```
src/
├── data/
│   └── dataset.py          — TCNDDataset, tcnd_collate_fn, make_dataloader, make_per_basin_loaders
├── models/
│   └── backbone.py         — MultimodalBackbone, TaskHeads, TropiCycloneModel
├── methods/
│   └── dg_methods.py       — ERM, IRM, VREx, CORAL, DANN, MAML, PhysIRM
├── metrics/
│   └── basin_metrics.py    — BasinEvaluator, BTG, BNTE, RI skill, CORAL distance
train.py                    — Full training + benchmark runner
analysis/
└── visualize.py            — Figures and LaTeX tables
```

---

## Installation

```bash
pip install -r requirements.txt
```

```
torch>=2.0.0
torchvision>=0.15.0
xarray>=0.19.0
netCDF4>=1.6.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
tqdm>=4.64.0
```

Device priority at runtime: **CUDA → MPS (Apple Silicon) → CPU** (auto-detected unless `--device` is specified).

---

## Usage

### Quick test on TCND\_test (default, no full dataset needed)

```bash
# Full LOBO benchmark — all 7 methods × all 6 held-out basins
python src/train.py \
    --mode lobo \
    --data_root ./dataset/TCND_test \
    --output_dir ./runs \
    --epochs 50

# Single experiment — one source set, one target, one method
python src/train.py \
    --mode single \
    --source_basins WP,NA,EP,NI,SP \
    --target_basin SI \
    --method physirm \
    --data_root ./dataset/TCND_test \
    --output_dir ./runs

# Incremental source composition study
python src/train.py \
    --mode incremental \
    --target_basin SI \
    --data_root ./dataset/TCND_test \
    --output_dir ./runs
```

### Full dataset (train/ val/ test/ splits)

```bash
python src/train.py \
    --mode lobo \
    --data_mode full \
    --data_root ./dataset/TCND \
    --output_dir ./runs \
    --epochs 50
```

`--data_mode full` forces `Data1D/<basin>/train/` for training, `val/` for checkpoint selection, and `test/` for final evaluation. Default `test_only` uses SPLIT\_ALIASES fallback so any missing split falls back gracefully (needed for TCND\_test which only has `test/`).

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `lobo` | `lobo` \| `single` \| `incremental` |
| `--data_mode` | `test_only` | `test_only` \| `full` |
| `--data_root` | *(required)* | Path to TCND root containing `Data1D/`, `Data3D/`, `Env-Data/` |
| `--output_dir` | `./runs` | Directory for checkpoints and results JSON |
| `--methods` | all | Comma-separated subset, e.g. `erm,physirm` |
| `--epochs` | `50` | Training epochs per experiment |
| `--eval_every` | `5` | Validation frequency (epochs) |
| `--model_size` | `lightweight` | `lightweight` (~0.6 M params) \| `complex` (~4 M params) |
| `--reg_weight` | `0.5` | Weight on 24 h-ahead regression MSE loss (0 = classification only) |
| `--scheduler` | `cosine` | `cosine` (CosineAnnealingLR) \| `onecycle` (OneCycleLR) |
| `--no_3d` | off | Disable Data\_3d branch (ablation) |
| `--no_env` | off | Disable Env-Data branch (ablation) |
| `--compile` | off | `torch.compile` for A100/H100 throughput |
| `--cache_data` | off | Cache full dataset in RAM (~50 GB) |
| `--device` | `auto` | `cuda` \| `mps` \| `cpu` \| `auto` |
| `--num_workers` | `8` | DataLoader worker count |
| `--few_shot` | off | Run few-shot fine-tuning after zero-shot evaluation |
| `--k_shots` | `32` | Number of target-basin examples for few-shot |
| `--few_shot_epochs` | `5` | Fine-tuning epochs for few-shot |
| `--resume` | — | Resume from checkpoint path, or `latest` to auto-find in `--output_dir` |
| `--fail_fast` | off | Stop benchmark on first experiment failure |

### Architecture overrides

| Flag | Default | Description |
|---|---|---|
| `--spatial_embed` | size-dependent | Spatial branch output dim |
| `--track_embed` | size-dependent | Track branch output dim |
| `--env_embed` | size-dependent | Environment branch output dim |
| `--phys_dim` | size-dependent | Physics sub-space dimension |
| `--final_dim` | size-dependent | Joint representation dimension |
| `--dropout` | `0.1` | Dropout rate across all branches |

---

## Visualisation

After training, generate all paper figures and LaTeX tables from the benchmark results JSON:

```bash
python analysis/visualize.py \
    --results_dir ./runs \
    --output_dir ./figures
```

| Output | Content |
|---|---|
| `fig_transfer_matrix.pdf` | 6×6 accuracy heatmap for all directed source→target pairs |
| `fig_ri_comparison.pdf` | Grouped bar chart: RI-F1 per target basin, all methods |
| `table_lobo.tex` | LaTeX table, best result per column bolded |

Additional figures available as callable functions in `visualize.py`:

| Function | Figure |
|---|---|
| `plot_few_shot_curve()` | Accuracy vs. k-shots per method (Fig 3) |
| `plot_tsne_physics()` | t-SNE of `z_phys` coloured by basin (Fig 6) |

---

## Outputs

| Path | Contents |
|---|---|
| `{output_dir}/benchmark_results.json` | All LOBO results — list of per-experiment result dicts |
| `{output_dir}/incremental_results.json` | Incremental source composition results |
| `{output_dir}/{run_id}_best.pt` | Best checkpoint (selected by source-val F1) |
| `{output_dir}/{run_id}_latest.pt` | Checkpoint saved at every epoch (for resuming) |
| `logs/train_{mode}_{timestamp}.log` | Master log for the full benchmark run |
| `logs/experiments/{timestamp}/{run_id}.log` | Per-experiment log file |

Each entry in `benchmark_results.json` contains full source and target metrics for all three tasks (intensity accuracy/F1/RI-F1, direction accuracy/F1, wind MAE m/s, pressure MAE hPa), BTG, BNTE, per-epoch history, and the path to the best checkpoint.

---

## Experiment Modes

### Leave-One-Basin-Out (LOBO)
For each of the 6 basins, train on the remaining 5 and evaluate zero-shot on the held-out basin. 7 methods × 6 splits = 42 experiments per full run.

### Single
Fixed source set and target basin; useful for ablations or debugging a specific pair:
```bash
--mode single --source_basins WP,NA --target_basin SI --method erm
```

### Incremental
Progressive source basin addition — reveals how cross-basin diversity of training data affects generalization:
```bash
--mode incremental --target_basin SI
```
Trains with 1, 2, …, 5 source basins sequentially for every method.

### Few-Shot Fine-Tuning
After zero-shot evaluation, freeze the backbone and fine-tune task heads on k labelled examples from the target basin's training split:
```bash
--few_shot --k_shots 32 --few_shot_epochs 5
```

---

## Notes

- Checkpoints are selected based on **source-validation F1** to avoid leaking target-basin information into model selection.
- The IRM and PhysIRM gradient penalties are applied to **classification logits only**. Regression is included in the ERM term but excluded from the invariance penalty.
- `torch.compile` is automatically skipped for MAML on MPS (Apple Silicon) to avoid a known view-stride error in `torch.func.functional_call`.
- With `--cache_data`, multiprocessing is disabled for training loaders (workers would receive isolated cache copies). Validation and test loaders retain workers.