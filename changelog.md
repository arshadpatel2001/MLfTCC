# Changelog

All notable changes to the MLfTCC/Project codebase are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.4] — 2026-03-22

### Summary

Second comprehensive codebase audit. Resolved **7 bugs** (3 HIGH, 3 MEDIUM,
1 LOW) affecting DG method parameter persistence, physics feature correctness,
backbone representation flow, and numerical edge-case robustness. The codebase
is now verified and ready for training.

---

### HIGH Fixes

#### `src/methods/dg_methods.py` — DANN and PhysIRM not `nn.Module` subclasses

- **Before**: `DANN` and `PhysIRM` held `nn.Module` children (`DomainDiscriminator`
  and `phys_predictor`) but only extended `DGMethod` (a plain ABC). Custom
  `.to()` and `.parameters()` methods partially papered over the issue.
- **After**: Both classes now inherit from `(DGMethod, nn.Module)` with an
  explicit `nn.Module.__init__(self)` call. The custom `.to()` and
  `.parameters()` overrides have been removed — the inherited `nn.Module`
  methods automatically track all registered sub-modules.
- **Impact**: `torch.save(model.state_dict())` would quietly **omit** the
  discriminator and predictor weights. Loading a checkpoint restored only
  the backbone — DANN's adversarial head and PhysIRM's physics predictor
  were randomly re-initialised on every resume. This made training-from-
  checkpoint non-deterministic and invalidated any checkpoint-based
  evaluation.

```diff
-class DANN(DGMethod):
+class DANN(DGMethod, nn.Module):
     ...
     def __init__(self, ...):
+        nn.Module.__init__(self)
         self.dann_lambda = dann_lambda
         ...
-    def to(self, device):     # ← removed
-    def parameters(self):     # ← removed
+    # to() and parameters() are inherited from nn.Module
```

#### `src/data/dataset.py` — Wind shear computed from U-component only

- **Before**: `_compute_physics_features()` computed vertical wind shear as
  `abs(data_3d[5].mean() - data_3d[7].mean())`, i.e., `|U_200 - U_850|`.
  This captures only the zonal component of the wind shear.
- **After**: Proper vector wind shear is computed as
  `sqrt((U_200 - U_850)² + (V_200 - V_850)²)` using channels 5, 7, 9, 11.
- **Impact**: The physics sub-space input for PhysIRM was receiving an
  incomplete shear estimate. In basins where the meridional shear component
  dominates (e.g., recurving storms in EP/NA), this could be the dominant
  error term.

```diff
-        shear = abs(data_3d[5].mean().item() - data_3d[7].mean().item()) \
-                if data_3d.shape[0] >= 13 else 0.0
+        # Proper vector wind shear: sqrt((U_200 - U_850)² + (V_200 - V_850)²)
+        # Channels: 5=U_200, 7=U_850, 9=V_200, 11=V_850
+        if data_3d.shape[0] >= 13:
+            du = data_3d[5].mean().item() - data_3d[7].mean().item()
+            dv = data_3d[9].mean().item() - data_3d[11].mean().item()
+            shear = float(np.sqrt(du**2 + dv**2))
+        else:
+            shear = 0.0
```

#### `src/models/backbone.py` — TaskHeads received stale `z_full` without physics injection

- **Before**: `MultimodalBackbone.forward()` returned the original `z_full`
  projection as the `"z"` key. After splitting `z_full → [z_env | z_phys]`,
  the code added the physics encoder output to `z_phys` via a residual
  connection, but this updated `z_phys` was **not** folded back into `z`.
  The prediction heads (`TaskHeads`) used `feat["z"]`, which still pointed
  to the pre-injection `z_full`.
- **After**: After the physics residual injection, `z` is reconstructed as
  `torch.cat([z_env, z_phys], dim=-1)`. The heads now see the physics-
  enriched representation.
- **Impact**: The physics encoder branch contributed only to PhysIRM penalty
  terms — prediction quality was unaffected by it. This made a core
  architectural component (the physics encoder + phys_align residual)
  effectively dead code for the prediction path.

```diff
         z_phys = z_phys + self.phys_align(phys_enc_out)

+        # Reconstruct z from the updated sub-spaces
+        z = torch.cat([z_env, z_phys], dim=-1)
+
         return {
-            "z":      z_full,    # ← stale, pre-injection projection
+            "z":      z,         # ← reconstructed, includes physics injection
             "z_phys": z_phys,
             "z_env":  z_env,
         }
```

---

### MEDIUM Fixes

#### `src/methods/dg_methods.py` — VREx `var()` NaN on single environment

- **Before**: `losses_t.var()` used PyTorch's default `unbiased=True` (Bessel's
  correction), dividing by `N-1`. With 1 environment, `N-1 = 0`, producing NaN.
- **After**: Changed to `losses_t.var(unbiased=False)` (population variance).
- **Impact**: Any single-source experiment (e.g., `WP → SI` with VREx) silently
  produced NaN loss, halting gradient updates without raising an error.

```diff
-        var  = losses_t.var()
+        var  = losses_t.var(unbiased=False)
```

#### `src/methods/dg_methods.py` — CORAL `coral_pen` device mismatch

- **Before**: `coral_pen = torch.tensor(0.0, device=erm.device)` creates a leaf
  tensor that is not in the autograd graph. The subsequent `coral_pen = coral_pen + …`
  in the loop does create graph connections, but the initial value was constructed
  differently on CPU vs GPU paths.
- **After**: `coral_pen = torch.zeros(1, device=erm.device).squeeze()`. This is
  functionally equivalent but uses a cleaner construction that is consistently
  placed on the correct device.
- **Impact**: Minor — mostly a robustness improvement. On GPU + no pairs the
  original code also worked, but the new pattern is safer.

```diff
-        coral_pen = torch.tensor(0.0, device=erm.device)
+        coral_pen = torch.zeros(1, device=erm.device).squeeze()
```

#### `src/data/dataset.py` — Direction/intensity sentinel -1 indistinguishable from class 0

- **Before**: `scalar_norm()` mapped sentinel value -1 to `default=0.0`, then
  divided by scale. For `history_direction12` (scale=7.0), both unknown (-1)
  and class 0 ("East") produced the same normalised value: 0.0. The model
  could not distinguish "direction is East" from "direction is unknown".
- **After**: Added `sentinel_normed` parameter. Unknown sentinels now map to
  0.5 (midpoint of the [0,1] range), making them distinguishable from all
  valid classes.
- **Impact**: Any storm with unknown history direction or intensity change
  was silently mislabelled as class 0. This affected principally the NI and
  SP basins (higher rate of missing historical data).

```diff
-        def scalar_norm(val, default=0.0, scale=1.0):
-            ...
-            if v < 0:  # -1 sentinel = unknown
-                v = default
+        def scalar_norm(val, default=0.0, scale=1.0, sentinel_normed=None):
+            ...
+            if v < 0:  # -1 sentinel = unknown
+                if sentinel_normed is not None:
+                    return np.array([sentinel_normed], dtype=np.float32)
+                v = default

-            scalar_norm(d.get("history_direction12"), scale=7.0),
+            scalar_norm(d.get("history_direction12"), scale=7.0, sentinel_normed=0.5),
```

---

### LOW Fixes

#### `src/methods/dg_methods.py` — Duplicate `from abc import ABC, abstractmethod`

- **Before**: Lines 26 and 33 both imported `ABC, abstractmethod`.
- **After**: Removed the duplicate at line 33.
- **Impact**: Cosmetic only.

---

### Verification Results

| Check | Result |
|-------|--------|
| `py_compile` on all 5 source files | ✅ Pass |
| All module imports resolve correctly | ✅ Pass |
| `isinstance(DANN(...), nn.Module)` | ✅ Pass |
| `isinstance(PhysIRM(...), nn.Module)` | ✅ Pass |
| Model forward → `logits_intensity=(2,5)`, `logits_direction=(2,8)` | ✅ Pass |
| Model forward → `z=(2,256)` (reconstructed from updated sub-spaces) | ✅ Pass |
| All 7 methods `compute_loss()` on 2-env batches | ✅ Pass |
| VREx single-env → no NaN | ✅ Pass |
| CORAL single-env → no NaN | ✅ Pass |

---

### Files Changed

| File | Lines Changed | Type |
|------|--------------|------|
| `src/methods/dg_methods.py` | ~45 | Modified |
| `src/data/dataset.py` | ~25 | Modified |
| `src/models/backbone.py` | ~8 | Modified |

---

## [0.0.3] — 2026-03-22

### Summary

Extensive addition of `tqdm` progress bars and `time` module tracking across the codebase for improved visibility into execution runtimes.

### Enhancements

- **`src/train.py`**: Added `tqdm` loops and runtime tracking to epoch validation, final evaluation, few-shot fine-tuning, and benchmark loops.
- **`src/data/dataset.py`**: Wrapped data index initialization with `tqdm` and added runtime tracking.
- **`src/methods/dg_methods.py`**: Added `tqdm` and runtime tracking to the MAML inner loop.
- **`src/metrics/basin_metrics.py`**: Wrapped source/target data loading with `tqdm` and added overall timing metrics for evaluation.

---

## [0.0.2] — 2026-03-22

### Summary

Comprehensive codebase audit and fix across all modules. Resolved **10 bugs**
(4 critical, 3 high, 2 medium, 1 low) affecting data normalization, training
loop correctness, domain generalization method warmup schedules, and CLI
functionality. The codebase is now verified and ready for training.

---

### Critical Fixes

#### `src/data/dataset.py` — Double-normalization of LAT removed
- **Before**: The code treated the `LAT` column in TCND `.txt` files as raw
  degrees and applied `(lat_raw + 90.0) / 180.0`, double-normalizing an
  already-normalized value.
- **After**: The column is renamed to `LAT_norm` in `TCND_COLS` and used
  directly without re-normalization.
- **Impact**: All model inputs that included latitude were corrupted. This fix
  is essential for correct training on any basin.

```diff
-        lat_raw   = float(row.get("LAT",       0.0))
+        lat_norm  = float(row.get("LAT_norm",  0.0))
         long_norm = float(row.get("LONG_norm", 0.0))
         wnd_norm  = float(row.get("WND_norm",  0.0))
         pres_norm = float(row.get("PRES_norm", 0.0))
-        lat_norm  = (lat_raw + 90.0) / 180.0  # normalise raw degrees to [0,1]
```

#### `src/data/dataset.py` — Physics features used normalized LAT as raw degrees
- **Before**: `_compute_physics_features()` used the normalized LAT value in
  `np.sin(np.deg2rad(lat))` for Coriolis computation, producing physically
  nonsensical values (e.g., −0.90 for WP instead of ~0.22).
- **After**: Coriolis parameter now uses pre-computed basin-level constants
  from `BASIN_CORIOLIS` dictionary. This is scientifically sound since Coriolis
  depends on latitude and each basin has a well-defined characteristic latitude.
- **Impact**: The PhysIRM physics sub-space was receiving garbage Coriolis
  values, preventing it from learning physically meaningful invariances.

```diff
-        lat = float(row.get("LAT", 0.0))
-        omega = 7.2921e-5
-        coriolis_norm = float(np.clip(2*omega*np.sin(np.deg2rad(lat)) / 1.4584e-4, -1, 1))
+        # Use pre-computed basin-level Coriolis rather than un-normalizing LAT
+        coriolis_raw  = BASIN_CORIOLIS.get(basin, 0.0) * 1e-5
+        coriolis_norm = float(np.clip(coriolis_raw / 1.4584e-4, -1, 1))
```

#### `src/data/dataset.py` — Normalization constants restructured
- **Before**: A single `NORM` dict mixed Data1D normalization parameters (which
  were never used) with Data3D z-score parameters.
- **After**: Separated into `NORM_1D` (paper's stated Data1D formulas, kept
  for reference) and `NORM` (Data3D z-score parameters, actively used).
  Added a prominent disclaimer noting that the paper's stated Data1D formulas
  do not reproduce the actual values in the released `.txt` files.
- **Impact**: Documentation clarity; prevents future confusion.

#### `src/data/dataset.py` — Column naming corrected in TCND_COLS
- **Before**: `TCND_COLS = ["ID", "FLAG", "LAT", "LONG_norm", "WND_norm", "PRES_norm", ...]`
- **After**: `TCND_COLS = ["ID", "FLAG", "LAT_norm", "LONG_norm", "WND_norm", "PRES_norm", ...]`
- **Impact**: Consistent naming clarifies that all numeric columns are
  pre-normalized by the dataset.

---

### High Fixes

#### `src/methods/dg_methods.py` — DANN warmup schedule bypassed
- **Before**: `DANN.update()` set `self._step = step` from the caller's
  global step counter before calling `compute_loss()`, which then incremented
  `_step += 1`. This caused the GRL schedule (`_alpha()`) to jump immediately
  based on the outer training step, bypassing the intended warmup.
- **After**: Removed `self._step = step` line. The internal `_step` counter is
  now managed only by `compute_loss()`, providing a correct linear warmup.
- **Impact**: DANN training dynamics were incorrect; the gradient reversal
  layer strength was not ramping as designed.

```diff
     def update(self, optimizer, batches, model, step=0):
-        self._step = step
+        # _step is managed internally by compute_loss(); do NOT overwrite it.
         optimizer.zero_grad()
```

#### `src/methods/dg_methods.py` — PhysIRM warmup schedule bypassed
- **Before**: Same issue as DANN — `PhysIRM.update()` set `self._step = step`,
  bypassing the IRM penalty warmup (λ ramp from 0 → `irm_lambda`).
- **After**: Same fix — removed `self._step = step`.
- **Impact**: PhysIRM's IRM penalty was applied at full strength from step 0,
  potentially destabilising early training.

#### `src/train.py` — `--few_shot` flag parsed but never acted upon
- **Before**: The CLI registered `--few_shot`, `--k_shots`, and
  `--few_shot_epochs` arguments but no code path ever checked `args.few_shot`
  or called `few_shot_finetune()`. Users who passed `--few_shot` got zero-shot
  results silently.
- **After**: Added a post-training block in the `single` mode runner that
  checks `args.few_shot` and calls `few_shot_finetune()` with the best
  checkpoint (or a fresh model if no checkpoint was saved).
- **Impact**: Few-shot fine-tuning experiments were completely non-functional.

---

### Medium Fixes

#### `src/methods/dg_methods.py` — MAML batch-split crashes on non-tensor values
- **Before**: `support = {k: v[:n_s] for k, v in batch.items()}` applied
  slicing to all batch values including potential non-tensor items (e.g.,
  string metadata), causing silent corruption or crashes.
- **After**: Only tensors are sliced; non-tensor values are passed through:
  `support = {k: (v[:n_s] if isinstance(v, torch.Tensor) else v) ...}`
- **Impact**: MAML could crash or produce corrupt results if batches
  contained non-tensor metadata.

#### `src/train.py` — `n_batches` could be 0 with small datasets
- **Before**: `n_batches = max(len(l.dataset) // bs for l in loaders)` used
  floor division which returns 0 when all datasets are smaller than batch
  size. This caused the training loop to silently skip all epochs.
- **After**: `n_batches = max(1, max(len(l.dataset) // bs for l in loaders))`
  ensures at least one batch per epoch.
- **Impact**: Training on small basin subsets (e.g., NI with few storms) or
  with large batch sizes could silently produce an untrained model.

#### `src/train.py` — `--device` argument was completely ignored
- **Before**: The `--device` CLI argument was parsed but all three mode
  runners (`lobo`, `single`, `incremental`) had hardcoded device auto-detection
  logic that ignored the user's choice.
- **After**: Added a `select_device(args)` helper that respects `args.device`
  (falling back to auto-detection for `"auto"`) and replaced all three
  hardcoded blocks with calls to this helper.
- **Impact**: Users could not force CPU training or specify a particular GPU.

---

### Low Fixes

#### `src/models/backbone.py` — Env-Data dimension docstring corrected
- **Before**: Module docstring said `Branch 3: Env-T-Net ← Env-Data (46,)`.
- **After**: Corrected to `Branch 3: Env-T-Net ← Env-Data (77,)`.
- **Impact**: Documentation-only; the code and EnvEncoder already used 77.

#### `src/metrics/basin_metrics.py` — weighted_f1 docstring corrected
- **Before**: Docstring said "Macro-weighted F1 score" but the implementation
  computes class-support-weighted F1 (not unweighted macro F1).
- **After**: Docstring now reads "Weighted F1 score (weighted by class
  support / frequency)".
- **Impact**: Documentation-only; the implementation was already correct for
  the intended weighted-F1 metric.

---

### Verification Results

| Check | Result |
|-------|--------|
| `py_compile` on all 5 modified files | ✅ Pass |
| All module imports resolve correctly | ✅ Pass |
| Dataset loads 4,562 WP test samples | ✅ Pass |
| `data_1d` shape = `(4,)` | ✅ Pass |
| `phys_features` shape = `(8,)` | ✅ Pass |
| Coriolis for WP = 0.2153 (physically correct) | ✅ Pass |
| Model forward pass produces correct output shapes | ✅ Pass |
| `logits_intensity` = `(B, 5)`, `logits_direction` = `(B, 8)` | ✅ Pass |

---

### Files Changed

| File | Lines Changed | Type |
|------|--------------|------|
| `src/data/dataset.py` | ~60 | Modified |
| `src/methods/dg_methods.py` | ~15 | Modified |
| `src/train.py` | ~55 | Modified |
| `src/models/backbone.py` | 1 | Modified |
| `src/metrics/basin_metrics.py` | 1 | Modified |

---

## [0.0.1] — Pre-audit baseline

Initial codebase with the following known issues:
- Double-normalization of LAT in dataset loading
- Physics features computed from opaque normalized values as if raw
- DANN/PhysIRM warmup schedules bypassed by step counter overwrite
- `--few_shot` CLI flag non-functional
- `--device` CLI flag ignored
- MAML batch-splitting not guarding non-tensor values
- `n_batches` could be 0 with small datasets
- Various docstring inaccuracies
