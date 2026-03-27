# TropiCycloneNet Basin Generalization Benchmark

**Physics-guided domain generalization for global tropical cyclone forecasting** across six ocean basins using the TropiCycloneNet Dataset (TCND) and a suite of domain generalization (DG) methods, culminating in the proposed **PhysIRM** — Physics-guided Invariant Risk Minimization.

> Course project: ELEC70127, Imperial College London
> GitHub: https://github.com/arshadpatel2001/MLfTCC

> 📊 **Presentation Slides**: The presentation slides for this project can be found in [`MLfTCC_Group_Project_Slides.pdf`](./MLfTCC_Group_Project_Slides.pdf).

> 📑 **Peer Evaluation**: The peer evaluation (by a different group) can be found [here](https://imperiallondon-my.sharepoint.com/:b:/g/personal/nm1225_ic_ac_uk/IQAQik2mBAOZSazXlTBshVn-AXocragJfOi7YhrBMgGvnYk).
> *(Note: This link is only accessible via Imperial SharePoint login. Please ensure you are logged in for it to work. If you encounter issues, please reach out to our peer evaluator, Nathan, at [nathan.mani25@imperial.ac.uk](mailto:nathan.mani25@imperial.ac.uk).)*

---

## Overview

Tropical cyclone (TC) prediction models trained on data from one ocean basin frequently fail when applied to others, because each basin has distinct synoptic climatology (e.g. monsoon troughs, subtropical highs, Coriolis gradients). This benchmark asks: *can a model trained on source basins generalize zero-shot to an unseen target basin?*

**Strategy:** Leave-One-Basin-Out (LOBO) — hold out one basin as target, train on the remaining five, evaluate zero-shot.

Three simultaneous prediction tasks are evaluated:

| Task | Type | Horizon | Label source |
|---|---|---|---|
| 1. Intensity change | 4-class classification | 24 h ahead | `future_inte_change24` from Env-Data `.npy` |
| 2. Track direction | 8-class classification | 24 h ahead | `future_direction24` from Env-Data `.npy` |
| 3. Intensity regression | Continuous | 24 h ahead | WND\_norm, PRES\_norm from Data1D row i+4 |

---

## Architecture Versions

This repository contains three successive architecture versions, each representing a stage of development.

### Version 1 — Multimodal Backbone (Ablation-Study/final_src)

Full multimodal architecture using all three data modalities:
- **Track branch** — MLP on 1D track features (Data1D)
- **Spatial branch** — CNN on ERA5 3D patches (Data3D, 13 channels, 81×81)
- **Environment branch** — MLP on 94-dim Env-Data features

Fused via two-layer MLP → **four task heads**: intensity classification, direction classification, intensity regression, and a **track forecasting head** predicting 4 future positions at +6h, +12h, +18h, +24h (evaluated via ADE and FDE).

**Key finding:** the 3D spatial branch caused catastrophically high track ADE (~10,000 km), indicating the multimodal fusion had not converged at this stage.

Source tree: `Abalation-Study/final_src/`

---

### Version 2.1 — Data1D-Only Architecture (Abalation-Study/src)

After diagnosing that the 3D and Env-Data branches drove ADE to ~10,000 km, the spatial and environment branches were removed. Running LOBO with Data1D only brought mean ADE down to ~1,000 km.

This version explores multiple model families on the 1D track features only:
- **LSTM seq2seq** — recurrent sequence-to-sequence model
- **Transformer seq2seq** — attention-based sequence-to-sequence model
- **Baseline MLPs** — simple feedforward models

Experiment runners: LOBO, pooled (all basins), few-shot fine-tuning.

Source tree: `Abalation-Study/src/`

---

### Version 2.2 — Refined Production Architecture (src)

Current best version. Builds on v2.1 with a refined multimodal backbone and improved training pipeline:
- Improved feature representations
- Across-the-board metric improvements over v1

**v1 → v2.2 comparison (SI basin, PhysIRM, full training data):**

| Metric | v1 | v2.2 | Δ |
|---|---|---|---|
| Direction Accuracy | 0.272 | 0.415 | +52.6% |
| Wind MAE (m/s) | 4.35 | 3.61 | −17.0% |
| Pressure MAE (hPa) | 14.56 | 11.80 | −18.9% |

Source tree: `src/`

---

## Dataset — TCND

**TropiCycloneNet Dataset** (Huang et al., *Nature Communications* 2025). Three data modalities, six basins, ~70 years of records (1950–2023), over 3,720 distinct tropical cyclones.

Six ocean basins: **WP** (Western Pacific), **NA** (North Atlantic), **EP** (Eastern Pacific), **NI** (North Indian), **SI** (South Indian), **SP** (South Pacific).

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
├── Data3D/                          ← ERA5 meteorological grid patches
│   └── {BASIN}/{YEAR}/{NAME}/       ← Flexible path (can be direct under root or basin)
│       └── TCND_{NAME}_{YYYYMMDDHH}_sst_z_u_v.nc    (13 channels, 81×81, 6-hourly)
│
└── Env-Data/                        ← Pre-computed environmental features
    └── {BASIN}/{YEAR}/{NAME}/
        └── {YYYYMMDDHH}.npy         (94-dim feature context + classification labels)
```

### Modalities

**Data1D** — 8 columns per 6-hourly timestep (no header):

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

**Data3D** — ERA5 reanalysis patches cropped to 20°×20° around the TC centre, 0.25° resolution, 6-hourly.

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

## Model Architecture (v2.2)

### Multimodal Backbone

Three parallel encoding branches fused into a joint representation `z ∈ ℝᵈ`.

| Branch | Input | Encoder |
|---|---|---|
| Spatial | Data3D `(13, 81, 81)` | CNN stack → pooling → projection |
| Track | Data1D `(4,)` | MLP → LayerNorm |
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

Combined loss: `L = 1.0 × CE_intensity + 0.5 × CE_direction + 0.5 × MSE_regression`

---

## Domain Generalization Methods

Seven methods are implemented, all sharing the same backbone and task heads. Each basin is treated as one domain/environment.

| Method | Key idea | Reference |
|---|---|---|
| **ERM** | Empirical Risk Minimization — pool all source basins | DomainBed baseline |
| **IRM** | Gradient penalty on dummy scalar classifier w over the representation | Arjovsky et al., 2019 |
| **V-REx** | Penalise variance of per-basin losses | Krueger et al., ICML 2021 |
| **CORAL** | Align second-order feature statistics (covariance) across basins | Sun & Saenko, ECCV 2016 |
| **DANN** | Adversarial domain discriminator with gradient reversal | Ganin et al., JMLR 2016 |
| **MAML** | Meta-learning with per-basin inner-loop adaptation | Finn et al., ICML 2017 |
| **PhysIRM** *(proposed)* | IRM restricted to the physics sub-space + orthogonality + grounding | This work |

### PhysIRM

Instead of forcing the entire representation to be invariant (which over-regularizes and erases local environment signals), PhysIRM restricts the invariance penalty to the physically-grounded sub-space `z_phys`:

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

## Key Results

### Basin Transfer Matrix

- **In-domain training is not always optimal** — NI, NA, and EP all have at least one cross-basin source that beats their own in-domain score.
- **WP is the universal anchor** — WP appears in the optimal source set for 5 out of 6 target basins (NA, EP, NI, SI, SP).
- **NI benefits most from transfer** — In-domain NI = 0.464 IntAcc (only 28 test samples). WP as source achieves 0.714 (+54%).

### Greedy Source Selection

- **2–3 sources are enough** — adding more sources beyond the greedy optimum hurts.
- **Geography is not destiny** — NI's best sources are SI+SP+WP (two southern-hemisphere basins); EP's best set includes NI and SI alongside NA.
- NI greedy (SI+SP+WP) achieves 0.679 IntAcc vs 0.464 in-domain (+46%).

### Ablation Study — DG Methods (target = SI)

| Method | IntAcc | WindMAE (m/s) |
|---|---|---|
| ERM | — | — |
| DANN | **0.610** | 5.84 (worst) |
| PhysIRM (test data) | 0.560 | **5.09** |
| PhysIRM (full training data) | **0.575** | **4.35** |

- **DANN** wins classification (SI IntAcc = 0.610) but sacrifices regression (WindMAE = 5.84 m/s, worst among competitive methods).
- **PhysIRM** achieves the best regression across all experiments (WindMAE = 4.35 m/s on full training data) — the most balanced method overall.
- **NI is floor-locked** — all methods tie at 0.464 IntAcc (28 test samples → majority-class ceiling).

### LOBO vs Greedy Source Composition

| Target | LOBO IntAcc | Greedy IntAcc | Winner |
|---|---|---|---|
| SI | 0.575 | 0.556 | LOBO (−3.7%) |
| NA | — | — | LOBO (−7.0%) |
| WP | 0.503 | **0.618** | Greedy (+11.5%) |
| NI | 0.464 | 0.464 | Tied (floor-locked) |

### Full Dataset (TrainData) vs Test-Only Data — SI Basin, PhysIRM

| Split | IntAcc | WindMAE | DirAcc |
|---|---|---|---|
| Test-only data | 0.560 | 5.09 m/s | 0.567 |
| Full training data | **0.575** | **4.35 m/s** | 0.272 |

Direction accuracy drops on the full dataset (0.567 → 0.272) because the test-only split had a temporal shortcut; the strict folder split requires genuine cross-storm generalization.

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

## Notebooks

### Version 1 Notebooks — based on `Abalation-Study/final_src/`

These notebooks use the full multimodal architecture (Data1D + Data3D + Env-Data):

| Notebook | Description |
|---|---|
| `Abalation-Study/nb_full_01.ipynb` | Initial LOBO experiments with multimodal backbone |
| `Abalation-Study/nb_full_02.ipynb` | DG method comparison on full multimodal setup |
| `Abalation-Study/nb_full_03.ipynb` | Additional ablations and source composition analysis |
| `Abalation-Study/nb_full_dataset.ipynb` | Full TrainData experiments with multimodal backbone |

### Version 2.1 Notebooks — based on `Abalation-Study/src/`

These notebooks use the Data1D-only architecture (v2.1), after removing 3D/Env branches to fix ADE:

| Notebook | Description |
|---|---|
| `Abalation-Study/nb_00_inspect_traindata.ipynb` | TrainData inspection — confirms year splits and folder structure |
| `Abalation-Study/nb_01_feature_analysis.ipynb` | Feature analysis and cross-basin stability |
| `Abalation-Study/nb_02_basin_transfer.ipynb` | Basin transfer matrix (all 30 directed pairs) |
| `Abalation-Study/nb_03_ablation.ipynb` | Ablation study across DG methods |
| `Abalation-Study/nb_04_domain_generalization.ipynb` | Domain generalization experiments |
| `Abalation-Study/nb_05_final_model.ipynb` | Final model evaluation |
| `Abalation-Study/nb_06_full_dataset.ipynb` | Full dataset experiments with v2.1 |

### Version 2.2 — Production Code

`src/` contains the final refined production architecture (v2.2), used as a standalone training pipeline. See [Usage](#usage) below.

---

## Project Structure

### v2.2 — `src/` (production)

```
src/
├── dataset/
│   └── dataset.py          — TCNDDataset, tcnd_collate_fn, make_dataloader, make_per_basin_loaders
├── models/
│   └── backbone.py         — MultimodalBackbone, TaskHeads, TropiCycloneModel
├── methods/
│   └── dg_methods.py       — ERM, IRM, VREx, CORAL, DANN, MAML, PhysIRM
├── metrics/
│   └── basin_metrics.py    — BasinEvaluator, BTG, BNTE, RI skill, CORAL distance
├── scripts/
│   └── visualize.py        — Figures and LaTeX tables
└── train.py                — Full training + benchmark runner

Abalation-Study/             — Research and development sandbox
├── nb_01_feature_analysis.ipynb — Statistical analysis of physical features
├── nb_03_ablation.ipynb         — Model architecture and data scaling benchmarks
└── src/                         — Research models and experiments
    ├── models/                  — LSTM, LSTM-attn, Transformer architectures
    └── experiments/             — Pooled, LOBO, and Few-shot protocols

---

## Research Ablation Study

A detailed investigation into architecture components, feature importance, and model adaptation strategies was conducted using the `Abalation-Study` codebase.

### 1. Model Architectures
- **LSTM Seq2Seq**: Standard recurrent baseline.
- **LSTM + Attention**: Augments the LSTM with basin-conditioned attention and embeddings.
- **Transformer Seq2Seq**: Non-autoregressive encoder-decoder for long-range sequence modeling.

### 2. Experiment Protocols
- **Pooled**: Baseline trained on all basins to establish an in-distribution performance upper bound.
- **LOBO (Leave-One-Basin-Out)**: Evaluates zero-shot cross-basin generalization.
- **Few-shot Adaptation**: Measures how rapidly a LOBO-pretrained model adapts to a new basin given small fractions (1%–50%) of target-basin data.

### 3. Key Findings
- **Feature Importance**: Gradient-based importance analysis identifies **relative displacement** (`Δlon`, `Δlat`) as the dominant predictors, accounting for **~88.8%** of model attention. Environmental scalar features provide fine-grained corrections but are secondary to motion history.
- **Architecture Performance**: While Transformer models show promise, the **LSTM-Attention** variant remains highly competitive for 24h forecasting tasks, particularly when combined with basin embeddings.
- **Data Scaling**: Model performance scales logarithmically with data volume. Sharp degradation in ADE observed when training data is reduced below **25%** of the full TCND volume.
```

### v1 — `Abalation-Study/final_src/`

```
Abalation-Study/final_src/
├── configs/                — Ablation groups, hyperparameter search space
├── dataset/
├── methods/
├── metrics/
├── models/
├── scripts/
└── train.py               — Includes --no_3d, --no_env, --reg_weight, ADE/FDE tracking
```

### v2.1 — `Abalation-Study/src/`

```
Abalation-Study/src/
├── data/
│   └── dataset.py              — Data1D-only loader
├── models/
│   ├── baselines.py            — Basic MLP models
│   ├── lstm_seq2seq.py         — LSTM sequence-to-sequence model
│   └── transformer_seq2seq.py  — Transformer sequence-to-sequence model
├── training/
│   ├── trainer.py              — Training loop
│   └── metrics.py              — Evaluation metrics
└── experiments/
    ├── lobo.py                 — Leave-One-Basin-Out runner
    ├── pooled.py               — Pooled all-basin training
    └── fewshot.py              — Few-shot fine-tuning
```

---

## Installation

```bash
pip install -r requirements.txt
```

```
torch>=2.1.0
torchvision>=0.16.0
xarray>=2023.1.0
netCDF4>=1.6.0
h5py>=3.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
packaging>=23.0
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
| `--data_mode` | `test_only` | `test_only` (graceful fallback) \| `full` (strict train/val/test splits) |
| `--data_root` | *(required)* | Path to TCND root containing `Data1D/`, `Data3D/`, `Env-Data/` |
| `--output_dir` | `./runs` | Directory for checkpoints and results JSON |
| `--methods` | all | Comma-separated subset, e.g. `erm,physirm` |
| `--epochs` | `50` | Training epochs per experiment |
| `--eval_every` | `5` | Validation frequency (epochs) |
| `--scheduler` | `cosine` | `cosine` \| `onecycle` (faster convergence) |
| `--resume` | — | Path to checkpoint, or `latest` to auto-find in `--output_dir` |
| `--cache_data` | off | Cache full dataset in RAM (~50 GB; disables multiprocessing) |
| `--compile` | off | `torch.compile` for higher throughput (skipped for MAML on MPS) |
| `--device` | `auto` | `cuda` \| `mps` \| `cpu` \| `auto` |
| `--num_workers` | `8` | DataLoader worker count |
| `--fail_fast` | off | Stop benchmark on first experiment failure |
| `--no_tqdm` | off | Disable tqdm progress bars |

---

## Visualisation

After training, generate all paper figures and LaTeX tables from the benchmark results JSON:

```bash
python src/scripts/visualize.py \
    --results_dir ./runs \
    --output_dir ./figures
```

| Output / Function | Content |
|---|---|
| `fig_transfer_matrix.pdf` | 6×6 accuracy heatmap for all directed source→target pairs |
| `fig_ri_comparison.pdf` | Grouped bar chart: RI-F1 per target basin, all methods |
| `table_lobo.tex` | LaTeX table, best result per column bolded |
| `plot_tsne_physics()` | t-SNE of `z_phys` coloured by basin |
| `plot_few_shot_curve()` | Line plot of k-shots vs accuracy (few-shot scaling) |

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

---

## Future Scope

1. **Scale Up** — Full LOBO sweep across all basins, more training epochs
2. **Stronger Physics** — PhysIRM refinement and ablation studies on physics feature selection
3. **Extended Lead Time** — Push beyond 24 h to 48 h and 72 h forecast horizons
4. **Robustness Reporting** — Standardize uncertainty metrics specifically for Rapid Intensification events

---

## Notes

- Checkpoints are selected based on **source-validation F1** to avoid leaking target-basin information into model selection.
- The IRM and PhysIRM gradient penalties are applied to **classification logits only**. Regression is included in the ERM term but excluded from the invariance penalty.
- `torch.compile` is automatically skipped for MAML on MPS (Apple Silicon) to avoid a known view-stride error in `torch.func.functional_call`.
- With `--cache_data`, multiprocessing is disabled for training loaders (workers would receive isolated cache copies). Validation and test loaders retain workers.
- The v1 ablation train.py (`Abalation-Study/final_src/train.py`) includes `--no_3d`, `--no_env`, `--reg_weight` flags and ADE/FDE tracking not present in the v2.2 production script.
