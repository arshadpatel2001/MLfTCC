# Basin Generalization for Tropical Cyclone Prediction

**PhysIRM: Physics-guided Invariant Risk Minimization for Cross-Basin Tropical Cyclone Forecasting**

*Target venues: NeurIPS 2026 / ICLR 2026*

---

## Overview

This repository benchmarks **cross-basin generalisation** for tropical cyclone (TC) prediction
using the TropiCycloneNet Dataset (TCND, Huang et al., *Nature Communications* 2025).

We formalise basin generalisation as a **domain generalisation (OOD)** problem:
each ocean basin is an *environment*, and the goal is to learn a predictor that
transfers to a **held-out basin without retraining**.

### Proposed Method: PhysIRM

We propose **PhysIRM** — Physics-guided Invariant Risk Minimization — which
decomposes the representation space into:

- **z_phys** (invariant sub-space): encodes thermodynamic invariants
  (SST anomaly, wind shear, Coriolis, MPI proxy, boundary layer moisture).
  IRM penalty enforces cross-basin invariance *here*.

- **z_env** (synoptic sub-space): encodes basin-specific steering patterns.
  Allowed to shift across basins.

An orthogonality regulariser and a physics grounding loss further anchor z_phys
to physically meaningful quantities.

### Novel Metrics

| Metric | Description |
|--------|-------------|
| **BTG** (Basin Transfer Gap) | Normalised drop from source→target accuracy |
| **BNTE** (Basin-Normalized Transfer Efficiency) | How much of the oracle gap is recovered |
| **RI-F1** | F1 on the safety-critical Rapid Intensification class |

---

## Setup

```bash
# Clone TCND dataset
git clone https://github.com/xiaochengfuhuo/TropiCycloneNet-Dataset
export TCND_ROOT=$(pwd)/TropiCycloneNet-Dataset

# Install dependencies
pip install torch torchvision einops scikit-learn tqdm xarray netCDF4 matplotlib wandb
```

---

## Quick Start

### 1. Single Transfer Experiment

```bash
# Train PhysIRM: WP → South Indian Ocean (zero-shot)
python train.py \
    --mode single \
    --data_root $TCND_ROOT \
    --source_basins WP,NA,EP \
    --target_basin SI \
    --method physirm \
    --epochs 50 \
    --output_dir ./runs/wp2si
```

### 2. Full LOBO Benchmark (all methods × all basins)

```bash
python train.py \
    --mode lobo \
    --data_root $TCND_ROOT \
    --methods erm,irm,vrex,coral,dann,maml,physirm \
    --epochs 50 \
    --output_dir ./runs/benchmark
```

### 3. Few-Shot Fine-Tuning

```bash
python train.py \
    --mode single \
    --data_root $TCND_ROOT \
    --source_basins WP,NA,EP,NI,SP \
    --target_basin SI \
    --method physirm \
    --few_shot \
    --k_shots 32 \
    --epochs 50
```

### 4. Generate Paper Figures

```bash
python scripts/visualize.py \
    --results_dir ./runs/benchmark \
    --output_dir ./figures
```

---

## Repository Structure

```
basin_gen/
├── data/
│   └── dataset.py          # TCND dataset with basin-stratified splits
│                           # Physics feature computation (z_phys anchors)
├── models/
│   └── backbone.py         # Multimodal encoder (3D-CNN + MLP + Env-T-Net)
│                           # Physics encoder branch
│                           # TaskHeads (intensity + direction)
├── methods/
│   └── dg_methods.py       # ERM, IRM, V-REx, CORAL, DANN, MAML, PhysIRM
├── metrics/
│   └── basin_metrics.py    # BTG, BNTE, RI-F1, per-basin breakdown
├── configs/
│   └── ablations.py        # All ablation configs + hparam sweep space
├── scripts/
│   └── visualize.py        # Paper figures + LaTeX table generation
└── train.py                # Main training + experiment runner
```

---

## Method Comparison

| Method | Type | Basin-Invariant? | Physics-Informed? |
|--------|------|-----------------|-------------------|
| ERM | Baseline | ✗ | ✗ |
| IRM | OOD/DG | ✓ (full z) | ✗ |
| V-REx | OOD/DG | ✓ (full z) | ✗ |
| CORAL | OOD/DG | ✓ (covariance) | ✗ |
| DANN | Adversarial DA | ✓ (adversarial) | ✗ |
| MAML | Meta-learning | ✓ (fast adapt) | ✗ |
| **PhysIRM** | **Physics + OOD** | **✓ (z_phys only)** | **✓** |

---

## Evaluation Protocol

Following DomainBed (Gulrajani & Lopez-Paz, ICLR 2021):

1. **Leave-One-Basin-Out (LOBO)**: train on 5 basins, test zero-shot on the 6th.
   All 6 splits reported.
2. **Model selection**: best checkpoint chosen on **source-basin validation set**
   (no target leakage).
3. **Hyperparameter search**: 20 random trials per method; results use the
   best validation checkpoint averaged over 3 seeds.
4. **Reporting**: mean ± std over 3 seeds for all metrics.

---

## Physics Motivation

TC intensity is governed by the **Carnot heat engine** (Emanuel 1986):

```
V_max² = (T_s / T_o - 1) × CAPE_ss
```

The key variables — SST (T_s), outflow temperature (T_o), and surface CAPE —
are **thermodynamic invariants**: they govern TC dynamics in *all* basins.

Basin differences arise from *synoptic steering* (subtropical high position,
monsoon trough) — not from the underlying thermodynamics.

PhysIRM encodes this physical insight as an inductive bias:
- z_phys ← learns the thermodynamic, causally-invariant features
- z_env  ← learns the basin-specific synoptic context

---

## Citation

```bibtex
@inproceedings{fung2026physirm,
  title     = {PhysIRM: Physics-guided Invariant Risk Minimization
               for Cross-Basin Tropical Cyclone Forecasting},
  author    = {Fung, Hadrian and ...},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## Key References

- Huang et al. (2025). Benchmark dataset and deep learning method for global tropical cyclone forecasting. *Nature Communications*. https://doi.org/10.1038/s41467-025-61087-4
- Arjovsky et al. (2019). Invariant Risk Minimization. *arXiv:1907.02893*
- Gulrajani & Lopez-Paz (2021). In Search of Lost Domain Generalization. *ICLR 2021*
- Krueger et al. (2021). Out-of-Distribution Generalization via Risk Extrapolation. *ICML 2021*
- Emanuel (1986). An Air-Sea Interaction Theory for Tropical Cyclones. *JAS 43(6)*
