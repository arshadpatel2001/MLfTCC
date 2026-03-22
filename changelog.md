# Changelog

All notable changes to the MLfTCC/Project codebase are documented in this file.
---

## [0.0.2] — 2026-03-22

### Summary

Extensive codebase fixes addressing critical algorithmic, metric, and data processing issues to ensure robustness for basin generalization training.

### Key Changes

- **Data & Physics Processing**: Fixed double-normalization, implemented dynamic Coriolis math, corrected vector wind shear metrics, and properly handled leading NetCDF dimensions.
- **Algorithmic Integrity**: Prevented few-shot evaluation leakage by correctly isolating and saving model states, fixed parameter persistence for DANN and PhysIRM, and resolved MAML inner-loop memory issues.
- **Reliable Metrics**: Fortified `BasinEvaluator`, `CORAL`, and F1 metric calculations with mathematical guards against division-by-zero and zero-length tensor slices to prevent `NaN` cascades.
- **Infrastructure**: Corrected unresolvable module imports, addressed silent CLI parameter mishandling, streamlined data loaders, and integrated robust `tqdm` progress tracking.

---

## [0.0.1] — Baseline

Initial codebase with the following known issues:
- Double-normalization of LAT in dataset loading
- Physics features computed from opaque normalized values as if raw
- DANN/PhysIRM warmup schedules bypassed by step counter overwrite
- `--few_shot` CLI flag non-functional
- `--device` CLI flag ignored
- MAML batch-splitting not guarding non-tensor values
- `n_batches` could be 0 with small datasets
- Various docstring inaccuracies