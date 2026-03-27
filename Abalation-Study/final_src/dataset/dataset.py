"""
TCND Dataset Module for Basin Generalization
=============================================
Handles loading of all three TCND modalities:
  - Data_1d: tabular IBTrACS (.txt files)
  - Data_3d: ERA5 gridded patches (NetCDF4/HDF5)
  - Env-Data: pre-computed environmental features (.npy dicts)

Basin codes: WP, NA, EP, NI, SI, SP

Key data-format facts (verified against real TCND sample files):
  ── Data1D .txt (8 cols, no header, whitespace-separated) ───────────────────
    Col 0  ID          float  storm-level counter (not used)
    Col 1  FLAG        float  quality flag (not used)
    Col 2  LAT_norm    float  normalized latitude   → model input [1/4]
    Col 3  LONG_norm   float  normalized longitude  → model input [0/4]
    Col 4  WND_norm    float  normalized wind speed → model input [3/4]
    Col 5  PRES_norm   float  normalized pressure   → model input [2/4]
    Col 6  YYYYMMDDHH  int    timestamp (key for .nc/.npy lookup)
    Col 7  Name        str    storm name (not used)

  ── Env-Data .npy fields ────────────────────────────────────────────────────
    month               (12,) float64   already one-hot
    area                 (6,) float64   already one-hot
    intensity_class      (6,) float64   already one-hot
    wind               scalar float64   normalized current wind speed
    move_velocity      scalar int64     normalized translation speed
    location_long       (36,) float64   already one-hot
    location_lat        (12,) float64   already one-hot
    history_direction12 scalar int64    CLASS INDEX 0–7 or -1 (unknown)  ← NOT one-hot!
    history_direction24 scalar int64    CLASS INDEX 0–7 or -1 (unknown)  ← NOT one-hot!
    history_inte_change24 scalar int64  CLASS INDEX 0–3 or -1 (unknown)  ← NOT one-hot!
    future_direction24  scalar int64    LABEL 0–7  or -1 (unknown)
    future_inte_change24 scalar int64   LABEL 0–3  or -1 (unknown)

    CRITICAL BUG FIX: history_direction* and history_inte_change24 are stored
    as raw integer class indices, NOT as one-hot arrays.  They must be converted
    with class_to_ohe(), not the generic ohe() helper.  Passing index=4 to ohe()
    produces [4,0,0,0,0,0,0,0] (WRONG); class_to_ohe(4,8) gives the correct
    one-hot [0,0,0,0,1,0,0,0].  For -1 (unknown), both return a zero vector,
    but only class_to_ohe is correct for positive indices.

  ── 94-dim env_vec layout (must match EnvEncoder slices in backbone.py) ─────
    [0:12]   month              (12)
    [12:18]  area                (6)
    [18:24]  intensity_class     (6)
    [24:25]  wind                (1)
    [25:26]  move_velocity       (1)
    [26:62]  location_long      (36)
    [62:74]  location_lat       (12)
    [74:82]  history_direction12  (8)  ← class index → one-hot
    [82:90]  history_direction24  (8)  ← class index → one-hot
    [90:94]  history_inte_change24 (4) ← class index → one-hot
"""

import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import shutil

import time
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import torch
from torch.utils.data import Dataset, DataLoader, default_collate

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import netCDF4 as nc
except ImportError:
    nc = None


# ── Basin registry ────────────────────────────────────────────────────────────
BASIN_CODES   = ["WP", "NA", "EP", "NI", "SI", "SP"]
BASIN_TO_IDX  = {b: i for i, b in enumerate(BASIN_CODES)}

BASIN_SST_STATS = {
    "WP": {"mean": 29.1, "std": 1.8},
    "NA": {"mean": 27.4, "std": 2.1},
    "EP": {"mean": 28.3, "std": 1.9},
    "NI": {"mean": 28.9, "std": 1.6},
    "SI": {"mean": 27.8, "std": 1.7},
    "SP": {"mean": 26.5, "std": 1.5},
}

BASIN_CORIOLIS = {
    "WP":  3.14, "NA": 4.87, "EP": 3.86,
    "NI":  3.60, "SI": -3.86, "SP": -3.14,
}

# Denormalization constants for reporting regression MAE in physical units.
# WND_ms   = WND_norm  * 25 + 40
# PRES_hPa = PRES_norm * 50 + 960
REG_DENORM = {
    "WND_scale": 25.0, "WND_offset": 40.0,
    "PRES_scale": 50.0, "PRES_offset": 960.0,
}

# ── Global Index/Sample Cache ─────────────────────────────────────────────────
GLOBAL_CACHE: Dict       = {}
GLOBAL_INDEX_CACHE: Dict = {}


class TCNDDataset(Dataset):
    """
    TropiCycloneNet Dataset loader.

    Each sample returns a dict with:
      data_1d        (4,)          normalized [LONG, LAT, PRES, WND]
      data_3d        (13, 81, 81)  per-sample z-scored ERA5 patch
      env_data       (94,)         environmental context vector
      phys_features  (8,)          physics priors for PhysIRM
      basin_idx      scalar long   basin index (0–5)
      y_intensity    scalar long   intensity change class 24h ahead (0–3)
      y_direction    scalar long   movement direction 24h ahead    (0–7)
      y_wind_reg     scalar float  24h-ahead WND_norm (may be NaN)
      y_pres_reg     scalar float  24h-ahead PRES_norm (may be NaN)
    """

    def __init__(
        self,
        root: str,
        basins: List[str],
        split: str        = "train",
        train_ratio: float = 0.70,   # kept for API compat (ignored)
        val_ratio: float   = 0.10,   # kept for API compat (ignored)
        seed: int          = 42,
        use_3d: bool       = True,
        use_env: bool      = True,
        cache: bool        = False,
        disable_tqdm: bool = False,
        force_split: Optional[str] = None,
        cache_index: bool  = True,
    ):
        """
        force_split: when set, bypass SPLIT_ALIASES and require the exact
        subfolder to exist under Data1D/<basin>/.  Use this for full-dataset
        training (--data_mode full) so that train/ and test/ are used strictly
        rather than falling back to each other.  Leave as None (default) for the
        existing test_only behaviour where SPLIT_ALIASES handles missing folders.
        """
        self.root         = Path(root)
        self.basins       = basins
        self.split        = split
        self.force_split  = force_split
        self.use_3d       = use_3d
        self.use_env      = use_env
        self.cache        = cache
        self.cache_index  = cache_index
        self.disable_tqdm = disable_tqdm
        self._cache_dict: Dict = GLOBAL_CACHE if cache else {}

        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.index: List[Dict] = []
        self._build_index(train_ratio, val_ratio, seed)

    # ── Root detection ────────────────────────────────────────────────────────

    def _resolve_tcnd_root(self) -> Path:
        import logging
        log = logging.getLogger(__name__)
        candidates = [self.root] + [p for p in self.root.iterdir() if p.is_dir()] + [self.root.parent]
        for base in candidates:
            if (base / "Data1D").exists():
                log.info(f"TCND root: {base}")
                return base
        found = sorted(self.root.rglob("Data1D"))
        if found:
            tcnd_root = found[0].parent
            log.info(f"TCND root (rglob): {tcnd_root}")
            return tcnd_root
        raise FileNotFoundError(
            f"Cannot find 'Data1D/' under '{self.root}'. "
            f"Point --data_root at the folder containing Data1D/, Data3D/, Env-Data/."
        )

    @staticmethod
    def _parse_filename(stem: str, basin: str):
        """Parse 'EP2017BSTDORA' → (year='2017', tc_name='DORA')."""
        n = len(basin)
        year    = stem[n: n + 4]
        tc_name = stem[n + 4 + 3:]   # skip 'BST'
        return year, tc_name

    # ── Index build ───────────────────────────────────────────────────────────

    def _build_index(self, train_ratio, val_ratio, seed):
        import logging
        log = logging.getLogger(__name__)

        cache_key = (
            f"{self.root}_{'-'.join(sorted(self.basins))}_"
            f"{self.split}_{self.use_3d}_{self.use_env}_"
            f"fs{self.force_split or 'none'}"
        )
        if cache_key in GLOBAL_INDEX_CACHE:
            self.index = GLOBAL_INDEX_CACHE[cache_key].copy()
            self._tcnd_root = self._resolve_tcnd_root()
            log.info(f"Index from memory cache: {len(self.index)} samples")
            return

        t0 = time.time()
        self._tcnd_root = self._resolve_tcnd_root()

        # ── Disk index cache ──────────────────────────────────────────────────
        _cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        _cache_dir  = self._tcnd_root / ".tcnd_index_cache"
        _cache_file = _cache_dir / f"{_cache_hash}.pkl"
        if self.cache_index and _cache_file.exists():
            try:
                with open(_cache_file, "rb") as _f:
                    self.index = pickle.load(_f)
                GLOBAL_INDEX_CACHE[cache_key] = self.index.copy()
                log.info(f"Index from disk cache ({_cache_file.name}): {len(self.index)} samples")
                print(f"✓ Index loaded from disk cache ({len(self.index)} samples) — skipping full scan")
                return
            except Exception as _e:
                log.warning(f"Disk cache load failed ({_e}), rebuilding index")

        SPLIT_ALIASES = {
            "train": ["train", "test", "val"],
            "val":   ["val",   "test", "train"],
            "test":  ["test",  "train", "val"],
        }

        # Data1D column names (8 columns, no header)
        COLS = ["ID", "FLAG", "LAT_norm", "LONG_norm",
                "WND_norm", "PRES_norm", "YYYYMMDDHH", "Name"]

        # ── Collect all (basin, split_folder, txt_file) tasks ─────────────────
        tasks = []
        for basin in self.basins:
            parent = self._tcnd_root / "Data1D" / basin
            if not parent.exists():
                raise FileNotFoundError(
                    f"Basin dir not found: {parent}\n"
                    f"Available: {[d.name for d in (self._tcnd_root/'Data1D').iterdir() if d.is_dir()]}"
                )

            if self.force_split is not None:
                split_folder = self.force_split
                if not (parent / split_folder).exists():
                    raise FileNotFoundError(
                        f"force_split='{split_folder}' requested but "
                        f"'{parent / split_folder}' does not exist. "
                        f"Available: {[d.name for d in parent.iterdir() if d.is_dir()]}"
                    )
            else:
                split_folder = next(
                    (c for c in SPLIT_ALIASES[self.split] if (parent / c).exists()),
                    None,
                )
                if split_folder is None:
                    raise FileNotFoundError(
                        f"No split folder (train/val/test) under {parent}"
                    )
                if split_folder != self.split:
                    log.warning(f"'{self.split}' missing for {basin}; using '{split_folder}'")

            basin_dir = parent / split_folder
            txt_files = sorted(basin_dir.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(f"No .txt files in {basin_dir}")

            log.info(f"  {basin}/{split_folder}: {len(txt_files)} storms")
            for txt_file in txt_files:
                tasks.append((basin, txt_file))

        # ── Worker: index one .txt file (all rows) ────────────────────────────
        def _index_file(basin, txt_file):
            stem          = txt_file.stem
            year, tc_name = self._parse_filename(stem, basin)
            try:
                df = pd.read_csv(txt_file, sep=r"\s+", header=None,
                                 names=COLS, engine="python", on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(txt_file, sep=r"\s+", header=None,
                                 names=COLS, engine="python")

            all_rows = list(df.iterrows())
            n_rows   = len(all_rows)
            samples  = []

            for i, (_, row) in enumerate(all_rows):
                ts           = str(int(float(row.get("YYYYMMDDHH", 0))))
                data_3d_path = self._get_3d_path(basin, year, tc_name, ts)
                env_path     = self._get_env_path(basin, year, tc_name, ts)

                if self.use_3d and data_3d_path is None: continue
                if env_path is None: continue

                try:
                    _d  = np.load(env_path, allow_pickle=True).item()
                    _yi = int(np.asarray(_d.get("future_inte_change24", 0)).ravel()[0])
                    _yd = int(np.asarray(_d.get("future_direction24",   0)).ravel()[0])
                    if _yi < 0 or _yd < 0:
                        continue
                except Exception:
                    pass

                future_wnd_norm  = np.nan
                future_pres_norm = np.nan
                if i + 4 < n_rows:
                    fr  = all_rows[i + 4][1]
                    _fw = float(fr.get("WND_norm",  np.nan))
                    _fp = float(fr.get("PRES_norm", np.nan))
                    if np.isfinite(_fw) and np.isfinite(_fp):
                        _delta_ok = True
                        try:
                            _ts_cur = str(int(float(row.get("YYYYMMDDHH", 0))))
                            _ts_fut = str(int(float(fr.get("YYYYMMDDHH", 0))))
                            _dt_cur = datetime.strptime(_ts_cur, "%Y%m%d%H")
                            _dt_fut = datetime.strptime(_ts_fut, "%Y%m%d%H")
                            _hours  = (_dt_fut - _dt_cur).total_seconds() / 3600.0
                            if abs(_hours - 24.0) > 0.1:
                                _delta_ok = False
                        except Exception:
                            pass
                        if _delta_ok:
                            future_wnd_norm  = _fw
                            future_pres_norm = _fp

                future_track_norm = [[np.nan, np.nan]] * 4
                for _step, _offset in enumerate([1, 2, 3, 4]):
                    if i + _offset < n_rows:
                        _fr = all_rows[i + _offset][1]
                        _fl = float(_fr.get("LONG_norm", np.nan))
                        _fa = float(_fr.get("LAT_norm",  np.nan))
                        if np.isfinite(_fl) and np.isfinite(_fa):
                            future_track_norm[_step] = [_fl, _fa]

                samples.append({
                    "basin":             basin,
                    "basin_idx":         BASIN_TO_IDX[basin],
                    "storm_id":          stem,
                    "tc_name":           tc_name,
                    "year":              year,
                    "timestamp":         ts,
                    "csv_row":           row.to_dict(),
                    "data_3d_path":      data_3d_path,
                    "env_path":          env_path,
                    "future_wnd_norm":   future_wnd_norm,
                    "future_pres_norm":  future_pres_norm,
                    "future_track_norm": future_track_norm,
                })
            return samples

        # ── Run all files in parallel ─────────────────────────────────────────
        # Drive I/O is the bottleneck (not CPU), so threads outperform processes.
        # 32 workers lets ~32 .npy reads overlap concurrently.
        n_workers = min(32, len(tasks))
        results   = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_map = {
                pool.submit(_index_file, basin, txt_file): idx
                for idx, (basin, txt_file) in enumerate(tasks)
            }
            pbar = tqdm(as_completed(future_map), total=len(tasks),
                        desc="Indexing (parallel)", dynamic_ncols=True,
                        leave=False, disable=self.disable_tqdm)
            for future in pbar:
                results[future_map[future]] = future.result()

        # Flatten results preserving basin order
        for samples in results:
            if samples:
                self.index.extend(samples)

        log.info(f"Index built in {time.time()-t0:.2f}s: {len(self.index)} samples")
        GLOBAL_INDEX_CACHE[cache_key] = self.index.copy()

        # ── Save index to disk for future runs ────────────────────────────────
        if self.cache_index:
            try:
                _cache_dir.mkdir(parents=True, exist_ok=True)
                with open(_cache_file, "wb") as _f:
                    pickle.dump(self.index, _f)
                log.info(f"Index saved to disk cache: {_cache_file}")
                print(f"✓ Index saved to disk cache → next run will be instant ({_cache_file.name})")
            except Exception as _e:
                log.warning(f"Could not save disk cache ({_e})")

    def _get_3d_path(self, basin, year, tc_name, ts) -> Optional[Path]:
        # Primary: Data3D/<basin>/<year>/<tc_name>/ (standard TCND layout)
        p1 = (self._tcnd_root / "Data3D" / basin / year / tc_name
              / f"TCND_{tc_name}_{ts}_sst_z_u_v.nc")
        if p1.exists():
            return p1
        # Fallback: <basin>/<year>/<tc_name>/ directly at root (alternate layout)
        p2 = (self._tcnd_root / basin / year / tc_name
              / f"TCND_{tc_name}_{ts}_sst_z_u_v.nc")
        return p2 if p2.exists() else None

    def _get_env_path(self, basin, year, tc_name, ts) -> Optional[Path]:
        p = self._tcnd_root / "Env-Data" / basin / year / tc_name / f"{ts}.npy"
        return p if p.exists() else None

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        key  = (meta["basin"], meta["storm_id"], meta["timestamp"])
        if self.cache and key in self._cache_dict:
            return self._cache_dict[key]
        sample = self._load_sample(meta)
        if self.cache:
            self._cache_dict[key] = sample
        return sample

    # ── Sample loader ─────────────────────────────────────────────────────────

    def _load_sample(self, meta: Dict) -> Dict[str, torch.Tensor]:
        row = meta["csv_row"]

        # Data_1d — already normalized by dataset authors
        lat_norm  = float(row.get("LAT_norm",  0.0))
        long_norm = float(row.get("LONG_norm", 0.0))
        wnd_norm  = float(row.get("WND_norm",  0.0))
        pres_norm = float(row.get("PRES_norm", 0.0))
        data_1d   = torch.tensor(
            [long_norm, lat_norm, pres_norm, wnd_norm], dtype=torch.float32
        )

        # Data_3d
        data_3d = (
            self._load_3d(meta["data_3d_path"])
            if self.use_3d
            else torch.zeros(13, 81, 81, dtype=torch.float32)
        )

        # Env-Data
        env_path = meta["env_path"]
        if env_path is not None:
            env_vec_full, y_intensity, y_direction = self._load_env(env_path)
        else:
            # --no_env mode or missing .npy: zero env vector, labels unknown.
            env_vec_full = torch.zeros(94, dtype=torch.float32)
            y_intensity  = 0
            y_direction  = 0
        env_vec = env_vec_full if self.use_env else torch.zeros(94, dtype=torch.float32)

        # Physics features
        phys = self._compute_physics_features(row, meta["basin"], data_3d)

        # Regression targets (NaN for end-of-storm rows)
        y_wind_reg = torch.tensor(float(meta.get("future_wnd_norm",  float("nan"))), dtype=torch.float32)
        y_pres_reg = torch.tensor(float(meta.get("future_pres_norm", float("nan"))), dtype=torch.float32)

        # Track forecasting targets: (4, 2) [LONG_norm, LAT_norm] for t+6h..t+24h.
        # NaN-filled steps are masked out in track_loss() during training and
        # excluded from ADE/FDE computation during evaluation.
        _raw_track = meta.get("future_track_norm", [[float("nan"), float("nan")]] * 4)
        y_track_norm = torch.tensor(_raw_track, dtype=torch.float32)  # (4, 2)

        return {
            "data_1d":       data_1d,
            "data_3d":       data_3d.to(torch.float32),
            "env_data":      env_vec.to(torch.float32),
            "phys_features": phys.to(torch.float32),
            "basin_idx":     torch.tensor(meta["basin_idx"], dtype=torch.long),
            "y_intensity":   torch.tensor(y_intensity,       dtype=torch.long),
            "y_direction":   torch.tensor(y_direction,       dtype=torch.long),
            "y_wind_reg":    y_wind_reg,
            "y_pres_reg":    y_pres_reg,
            "y_track_norm":  y_track_norm,   # (4, 2): future [long,lat] at t+6h..t+24h
        }

    # ── Modality loaders ──────────────────────────────────────────────────────

    def _load_3d(self, path: Path) -> torch.Tensor:
        """
        Load NetCDF4 ERA5 patch → (13, H, W) per-sample z-scored tensor.
        Variable order after stacking: u×4, v×4, z×4, sst×1 = 13 channels.
        """
        if nc is not None:
            ds      = nc.Dataset(path, "r")
            sst_raw = ds.variables["sst"][:].astype(np.float32)
            z       = ds.variables["z"][:].astype(np.float32)
            u       = ds.variables["u"][:].astype(np.float32)
            v       = ds.variables["v"][:].astype(np.float32)
            ds.close()
        elif xr is not None:
            ds      = xr.open_dataset(path)
            sst_raw = ds["sst"].values.astype(np.float32)
            z       = ds["z"].values.astype(np.float32)
            u       = ds["u"].values.astype(np.float32)
            v       = ds["v"].values.astype(np.float32)
            ds.close()
        else:
            raise ImportError("Neither 'netCDF4' nor 'xarray' is installed.")

        # Remove any leading time dimension: (1,4,H,W) → (4,H,W), (1,H,W) → (H,W)
        if sst_raw.ndim == 3: sst_raw = sst_raw[0]
        if z.ndim  == 4:      z  = z[0]
        if u.ndim  == 4:      u  = u[0]
        if v.ndim  == 4:      v  = v[0]

        sst_raw[sst_raw > 1e10] = np.nan   # mask land/fill

        H, W = sst_raw.shape
        chs = []
        for arr in (u, v, z):
            for i in range(4):
                chs.append(arr[i] if i < arr.shape[0] else np.zeros((H, W), np.float32))
        chs.append(sst_raw)

        data_3d = np.stack(chs, axis=0)   # (13, H, W)

        # Vectorized per-channel z-score — all 13 channels in 5 NumPy ops.
        # Uses np.where (no fancy indexing, no Python loop, no masked_array).
        finite    = np.isfinite(data_3d)                          # (13, H, W)
        n         = finite.sum(axis=(1, 2), keepdims=True).clip(min=1)  # (13,1,1)
        safe      = np.where(finite, data_3d, 0.0)
        mus       = safe.sum(axis=(1, 2), keepdims=True) / n      # (13,1,1)
        stds      = np.sqrt(
            np.where(finite, (data_3d - mus) ** 2, 0.0)
            .sum(axis=(1, 2), keepdims=True) / n
        ).clip(min=1e-6)                                          # (13,1,1)
        data_3d   = np.where(finite, (data_3d - mus) / stds, 0.0)

        return torch.from_numpy(data_3d.astype(np.float32))

    def _load_env(self, path: Path):
        """
        Load Env-Data .npy → (env_vec [94], y_intensity, y_direction).

        The CRITICAL fix here: history_direction12, history_direction24, and
        history_inte_change24 are stored in the files as SCALAR INTEGER CLASS
        INDICES (confirmed from real data), NOT as one-hot arrays.  They must
        be converted to one-hot with class_to_ohe().  Using the generic ohe()
        helper on a scalar produces the raw integer as the first element
        (e.g. value=4 → [4.,0.,0.,0.,0.,0.,0.,0.]) which is INCORRECT.
        """
        d = np.load(path, allow_pickle=True).item()

        # ── Helper A: existing one-hot / multi-hot arrays ─────────────────────
        def ohe(val, length):
            if val is None:
                return np.zeros(length, dtype=np.float32)
            arr = np.asarray(val, dtype=np.float32).ravel()
            if arr.size == 0:
                return np.zeros(length, dtype=np.float32)
            if arr.size < length:
                arr = np.pad(arr, (0, length - arr.size))
            return arr[:length].astype(np.float32)

        # ── Helper B: scalar integer class index → proper one-hot ─────────────
        def class_to_ohe(val, length):
            """
            val: scalar integer class index (0-based), or -1 / None for unknown.
            Returns a float32 one-hot vector of `length` dims.
            Unknown / out-of-range → zero vector.
            """
            if val is None:
                return np.zeros(length, dtype=np.float32)
            idx = int(np.asarray(val).ravel()[0])
            if idx < 0 or idx >= length:
                return np.zeros(length, dtype=np.float32)   # -1 or invalid → unknown
            out      = np.zeros(length, dtype=np.float32)
            out[idx] = 1.0
            return out

        # ── Helper C: scalar → float32 length-1 array ─────────────────────────
        def scalar_f32(val, default=0.0):
            if val is None:
                return np.array([default], dtype=np.float32)
            v = np.asarray(val, dtype=np.float32).ravel()
            return v[:1] if v.size >= 1 else np.array([default], dtype=np.float32)

        # ── Build 94-dim feature vector ───────────────────────────────────────
        parts = [
            ohe(d.get("month"),                   12),  # [0:12]   one-hot array
            ohe(d.get("area"),                     6),  # [12:18]  one-hot array
            ohe(d.get("intensity_class"),          6),  # [18:24]  one-hot array
            scalar_f32(d.get("wind")),                  # [24:25]  scalar float
            scalar_f32(d.get("move_velocity")),         # [25:26]  scalar int → float
            ohe(d.get("location_long"),           36),  # [26:62]  one-hot array
            ohe(d.get("location_lat"),            12),  # [62:74]  one-hot array
            # ── SCALAR INTEGER CLASS INDICES — must use class_to_ohe ──────────
            class_to_ohe(d.get("history_direction12"),   8),   # [74:82]
            class_to_ohe(d.get("history_direction24"),   8),   # [82:90]
            class_to_ohe(d.get("history_inte_change24"), 4),   # [90:94]
        ]
        env_vec = torch.from_numpy(np.concatenate(parts).astype(np.float32))  # (94,)

        # ── Labels ────────────────────────────────────────────────────────────
        y_intensity = int(np.asarray(d.get("future_inte_change24", 2)).ravel()[0])
        y_direction = int(np.asarray(d.get("future_direction24",   0)).ravel()[0])

        # -1 should be filtered during index build; clamp defensively
        if y_intensity < 0: y_intensity = 2
        if y_direction < 0: y_direction = 0
        y_intensity = min(y_intensity, 3)   # 4-class (0–3)
        y_direction = min(y_direction, 7)   # 8-class (0–7)

        return env_vec, y_intensity, y_direction

    def _compute_physics_features(
        self, row: Dict, basin: str, data_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        8-dim physics feature vector for PhysIRM invariant sub-space.
          [0] SST anomaly vs 28°C global reference (basin-level constant)
          [1] Wind shear proxy: –cross-correlation(U_200, U_850)
          [2] Coriolis parameter (from LAT_norm)
          [3] MPI proxy (basin SST – 26°C threshold)
          [4] Boundary-layer proxy (WND_norm)
          [5] Outflow proxy (spatial skewness of Z_200)
          [6] Steering proxy (spatial skewness of Z_500)
          [7] Current intensity (WND_norm)
        """
        wnd_norm  = float(row.get("WND_norm", 0.0))
        sst_stats = BASIN_SST_STATS.get(basin, {"mean": 28.0, "std": 2.0})

        sst_anom = float(np.clip(
            (sst_stats["mean"] - 28.0) / max(sst_stats["std"], 1e-6), -3.0, 3.0
        ))

        if data_3d.shape[0] >= 13:
            u200  = data_3d[0].float().flatten()
            u850  = data_3d[2].float().flatten()
            corr  = float((u200 * u850).mean() / (u200.std() * u850.std()).clamp(1e-6))
            shear = float(np.clip(-corr, -1.0, 1.0))
        else:
            shear = 0.0

        lat_deg      = float(row.get("LAT_norm", 0.0)) * 180.0 - 90.0
        coriolis_norm = float(np.clip(
            2 * 7.2921e-5 * np.sin(np.deg2rad(lat_deg)) / 1.4584e-4, -1.0, 1.0
        ))

        mpi_proxy = float(np.clip((sst_stats["mean"] - 26.0) / 10.0, -1.0, 1.0))
        bl_proxy  = float(np.clip(wnd_norm, 0.0, 1.0))

        if data_3d.shape[0] >= 13:
            def skewness(t):
                return float(
                    ((t - t.mean()).pow(3).mean() / t.std().clamp(1e-6).pow(3)).clamp(-3.0, 3.0)
                )
            outflow  = skewness(data_3d[8].float())   # Z_200 channel
            steering = skewness(data_3d[9].float())   # Z_500 channel
        else:
            outflow = steering = 0.0

        return torch.tensor(
            [sst_anom, shear, coriolis_norm, mpi_proxy,
             bl_proxy, outflow, steering, wnd_norm],
            dtype=torch.float32,
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def tcnd_collate_fn(samples):
    """
    Safety-net collate: drops any sample whose y_intensity or y_direction label
    is negative (unknown).  In normal operation the index build already filtered
    these out, so this guard fires only if a .npy was unreadable during indexing.

    NOTE: We no longer carry a 'valid' sentinel key in each sample dict.  That
    design mutated GLOBAL_CACHE entries in-place (popping the key on first use)
    and forced a torch.tensor(True) allocation on every subsequent batch access.
    Instead we just check the label tensors directly — O(batch_size) scalars,
    zero allocations, no cache mutation.
    """
    valid = [
        s for s in samples
        if s["y_intensity"].item() >= 0 and s["y_direction"].item() >= 0
    ]
    if not valid:
        return None   # caller checks `if batch is None: continue`
    return default_collate(valid)


def make_dataloader(
    root: str,
    basins: List[str],
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_3d: bool = True,
    use_env: bool = True,
    seed: int = 42,
    disable_tqdm: bool = False,
    cache: bool = False,
    force_split: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    # Disable multiprocessing only for train when caching — forked workers
    # would each get an isolated copy of GLOBAL_CACHE (defeating the purpose).
    # Val/test loaders are NOT cached and keep their workers for fast streaming.
    if cache and split == "train":
        num_workers = 0
    ds = TCNDDataset(
        root=root, basins=basins, split=split,
        use_3d=use_3d, use_env=use_env, seed=seed,
        disable_tqdm=disable_tqdm, cache=cache,
        force_split=force_split, **kwargs,
    )
    if len(ds) == 0:
        raise RuntimeError(
            f"Empty dataset: basins={basins}, split={split!r}. "
            "Check --data_root has .txt files under Data1D/<BASIN>/<split>/."
        )
    shuffle   = (split == "train")
    drop_last = (split == "train") and (len(ds) > batch_size)
    safe_bs   = min(batch_size, len(ds))
    cuda_avail = torch.cuda.is_available()
    pin_kwargs: Dict[str, Any] = {"pin_memory": cuda_avail}
    if cuda_avail:
        pin_kwargs["pin_memory_device"] = "cuda"
    return DataLoader(
        ds, batch_size=safe_bs, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
        collate_fn=tcnd_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        **pin_kwargs,
    )


def make_per_basin_loaders(
    root: str,
    basins: List[str],
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    disable_tqdm: bool = False,
    cache: bool = False,
    force_split: Optional[str] = None,
    **kwargs,
) -> Dict[str, DataLoader]:
    """One DataLoader per basin (for per-environment IRM/CORAL/VREx updates)."""
    return {
        b: make_dataloader(
            root, [b], split, batch_size, num_workers,
            disable_tqdm=disable_tqdm, cache=cache,
            force_split=force_split, **kwargs,
        )
        for b in basins
    }


def prefetch_to_local(
    root: str,
    basins: List[str],
    split: str,
    local_dir: str = "/content/tcnd_local",
    n_workers: int = 32,
    force_split: Optional[str] = None,
) -> str:
    """
    Copy all dataset files (Data1D .txt, Env-Data .npy, Data3D .nc) referenced
    by the given split from Drive to a local SSD directory.

    Returns the local root path — pass this as `root` to TCNDDataset /
    make_dataloader to read from local SSD instead of Drive.

    On Colab, local SSD reads are ~100x faster than Drive, so this one-time
    copy (run once per session) makes every subsequent training epoch fast.

    Usage:
        local_root = prefetch_to_local(DATA_ROOT, BASINS, split='test')
        loader = make_dataloader(local_root, BASINS, split='test', ...)
    """
    import logging
    log = logging.getLogger(__name__)

    src_root  = Path(root)
    dst_root  = Path(local_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Build index to get exact file list (uses disk cache if available)
    print("Building index to collect file list...")
    ds = TCNDDataset(
        root=src_root, basins=basins, split=split,
        use_3d=True, use_env=True,
        force_split=force_split,
        disable_tqdm=True,
    )

    # Collect all unique files to copy
    files_to_copy: List[Path] = []

    # Data1D .txt files
    tcnd = ds._tcnd_root
    for b in basins:
        for txt in (tcnd / "Data1D" / b).rglob("*.txt"):
            files_to_copy.append(txt)

    # Env-Data .npy and Data3D .nc from index
    seen = set()
    for meta in ds.index:
        for key in ("env_path", "data_3d_path"):
            p = meta.get(key)
            if p is not None and p not in seen:
                seen.add(p)
                files_to_copy.append(Path(p))

    # Also copy .tcnd_index_cache if present (so local root also has fast indexing)
    cache_dir = tcnd / ".tcnd_index_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            files_to_copy.append(f)

    total = len(files_to_copy)
    print(f"Files to copy : {total:,}  ({sum(f.stat().st_size for f in files_to_copy if f.exists()) / 1e9:.2f} GB)")
    print(f"Destination   : {dst_root}")
    print(f"Workers       : {n_workers}")

    copied = 0
    skipped = 0
    errors = 0

    def _copy_one(src: Path) -> str:
        try:
            rel  = src.relative_to(tcnd)
            dst  = dst_root / rel
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                return "skip"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return "ok"
        except Exception as e:
            return f"err:{e}"

    pbar = tqdm(total=total, desc="Copying to local SSD",
                unit="file", dynamic_ncols=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_copy_one, f): f for f in files_to_copy}
        for fut in as_completed(futures):
            result = fut.result()
            if result == "ok":      copied  += 1
            elif result == "skip":  skipped += 1
            else:                   errors  += 1
            pbar.update(1)
    pbar.close()

    print(f"Done — copied={copied:,}  skipped={skipped:,}  errors={errors:,}")
    if errors:
        print(f"  ⚠ {errors} files failed to copy — training may still work but some samples missing")

    local_root = str(dst_root)
    print(f"✓ Use this as DATA_ROOT for fast local training:\n  '{local_root}'")
    return local_root