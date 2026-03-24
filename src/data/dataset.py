"""
TCND Dataset Module for Basin Generalization
=============================================
Handles loading of all three TCND modalities:
  - Data_1d: tabular IBTrACS (CSV)
  - Data_3d: ERA5 gridded patches (NetCDF/zarr)
  - Env-Data: pre-computed environmental features (pickle/json)

Basin codes: WP, NA, EP, NI, SI, SP
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import time
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

import torch
from torch.utils.data import Dataset, DataLoader


# ── Basin registry ────────────────────────────────────────────────────────────
BASIN_CODES = ["WP", "NA", "EP", "NI", "SI", "SP"]
BASIN_TO_IDX = {b: i for i, b in enumerate(BASIN_CODES)}

# Climatological SST anomaly μ/σ per basin (computed from ERA5 1981-2010 baseline).
# These serve as physical priors for PhysIRM. Values are illustrative; recompute
# from your ERA5 data for the final paper.
BASIN_SST_STATS = {
    "WP": {"mean": 29.1, "std": 1.8},
    "NA": {"mean": 27.4, "std": 2.1},
    "EP": {"mean": 28.3, "std": 1.9},
    "NI": {"mean": 28.9, "std": 1.6},
    "SI": {"mean": 27.8, "std": 1.7},
    "SP": {"mean": 26.5, "std": 1.5},
}

# Coriolis parameter = 2 * Omega * sin(lat). We use a basin-average latitude
# as a physical feature. Values in 1e-5 s^-1.
BASIN_CORIOLIS = {
    "WP": 3.14,   # avg lat ~13°N
    "NA": 4.87,   # avg lat ~21°N
    "EP": 3.86,   # avg lat ~17°N
    "NI": 3.60,   # avg lat ~15°N
    "SI": -3.86,  # avg lat ~-17°S
    "SP": -3.14,  # avg lat ~-13°S
}


# ── Normalization constants ───────────────────────────────────────────────────
#
# IMPORTANT: Data1D columns in TCND .txt files are ALREADY NORMALIZED by the
# dataset authors using these formulas (see Huang et al., Nat. Comms. 2025):
#   LONG_norm = LONG_raw_01deg / 3600   (LONG in increments of 0.1°E)
#   LAT_norm  = (LAT_raw_01deg + 900) / 1800  (LAT in increments of 0.1°)
#   PRES_norm = (PRES_hPa - 870) / 1080
#   WND_norm  = WND_ms / 100
#
# NOTE: The paper's stated formulas produce value ranges that do NOT match
# the actual values in the released .txt files (e.g., LAT values like -14.82
# are outside the [0,1] range the formula would produce).  The actual
# normalization used to produce the data may differ from the above.
# Therefore we use the data AS-IS for model inputs (the model learns
# to handle whatever normalization was applied) and use BASIN-LEVEL
# physical constants for physics features rather than per-sample
# un-normalization.

NORM_1D = {
    "LONG_scale":  3600.0,
    "LAT_offset":  900.0,
    "LAT_scale":   1800.0,
    "PRES_offset": 870.0,
    "PRES_scale":  1080.0,
    "WND_scale":   100.0,
}

# Data_3d: per-variable z-score parameters (mean, std)
NORM = {
    "SST":   {"mean": 28.0,  "std": 3.0},    # °C (converted from Kelvin in _load_3d)
    "GPH_200": {"mean": 122000.0, "std": 500.0},   # geopotential (m²/s²), not height
    "GPH_500": {"mean": 57000.0,  "std": 300.0},
    "GPH_850": {"mean": 14500.0,  "std": 250.0},
    "GPH_925": {"mean": 7500.0,   "std": 200.0},
    "U_200": {"mean": 0.0,  "std": 15.0},
    "V_200": {"mean": 0.0,  "std": 10.0},
    "U_500": {"mean": 0.0,  "std": 8.0},
    "V_500": {"mean": 0.0,  "std": 6.0},
    "U_850": {"mean": 0.0,  "std": 7.0},
    "V_850": {"mean": 0.0,  "std": 6.0},
    "U_925": {"mean": 0.0,  "std": 6.0},
    "V_925": {"mean": 0.0,  "std": 5.0},
}


class TCNDSample:
    """Lightweight container for one TC observation."""
    __slots__ = ["data_1d", "data_3d", "env_data", "target_intensity",
                 "target_direction", "basin_idx", "storm_id", "timestamp"]

    def __init__(self, data_1d, data_3d, env_data,
                 target_intensity, target_direction,
                 basin_idx, storm_id, timestamp):
        self.data_1d = data_1d
        self.data_3d = data_3d
        self.env_data = env_data
        self.target_intensity = target_intensity
        self.target_direction = target_direction
        self.basin_idx = basin_idx
        self.storm_id = storm_id
        self.timestamp = timestamp


class TCNDDataset(Dataset):
    """
    Main dataset class for the TropiCycloneNet Dataset (TCND).

    Directory structure expected:
        root/
          TCND_subset/
            Data_1d/
              WP/  NA/  EP/  NI/  SI/  SP/   ← one CSV per storm
            Data_3d/
              WP/  NA/  EP/  NI/  SI/  SP/   ← one NetCDF per storm-timestep
            Env_Data/
              WP/  NA/  EP/  NI/  SI/  SP/   ← one pickle/json per storm-timestep

    Parameters
    ----------
    root : str
        Path to the TCND root directory.
    basins : list[str]
        List of basin codes to include (subset of BASIN_CODES).
    split : str
        One of {"train", "val", "test"}. Uses storm-level split.
    train_ratio : float
        Fraction of storms used for training within each basin.
    val_ratio : float
        Fraction for validation; remainder goes to test.
    seed : int
        Random seed for reproducible storm splits.
    use_3d : bool
        Whether to load Data_3d tensors. Set False for ablations.
    use_env : bool
        Whether to load Env-Data. Set False for ablations.
    cache : bool
        If True, cache loaded samples in RAM (requires ~50 GB for full dataset).
    """

    def __init__(
        self,
        root: str,
        basins: List[str],
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        seed: int = 42,
        use_3d: bool = True,
        use_env: bool = True,
        cache: bool = False,
    ):
        self.root = Path(root)
        self.basins = basins
        self.split = split
        self.use_3d = use_3d
        self.use_env = use_env
        self.cache = cache
        self._cache_dict: Dict = {}

        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.index: List[Dict] = []  # list of metadata dicts
        self._build_index(train_ratio, val_ratio, seed)

    # ── Index construction ────────────────────────────────────────────────────

    def _resolve_tcnd_root(self) -> Path:
        """
        Auto-detect the TCND root (the folder that contains Data1D/, Data3D/, Env-Data/).

        Handles the common case where the zip extracts into a same-named subfolder:
          TCND_test/          ← user passes this as --data_root
            TCND_test/        ← actual data lives here (double-nesting)
              Data1D/
              Data3D/
              Env-Data/
        """
        import logging
        log = logging.getLogger(__name__)

        # Walk: root itself, one level down (double-nest), two levels down, parent
        candidates = [self.root]
        candidates += [p for p in self.root.iterdir() if p.is_dir()]   # one level down
        candidates += [self.root.parent]

        for base in candidates:
            if (base / "Data1D").exists():
                log.info(f"TCND root detected: {base}")
                return base

        # rglob fallback — find any Data1D directory
        found = sorted(self.root.rglob("Data1D"))
        if found:
            tcnd_root = found[0].parent
            log.info(f"TCND root detected (rglob): {tcnd_root}")
            return tcnd_root

        raise FileNotFoundError(
            f"Cannot find 'Data1D/' under '{self.root}'.\n"
            f"Make sure --data_root points to the folder that contains "
            f"Data1D/, Data3D/, and Env-Data/ (or their parent)."
        )

    @staticmethod
    def _parse_filename(stem: str, basin: str):
        """
        Parse TCND filename stem like 'WP2017BSTBANYAN' or 'NA2019BSTDORIAN'.
        Returns (year, tc_name).
        """
        # Format: {BASIN}{YEAR}BST{NAME}
        # basin is e.g. 'WP', year is 4 digits after basin prefix
        prefix_len = len(basin)
        year       = stem[prefix_len: prefix_len + 4]
        tc_name    = stem[prefix_len + 4 + 3:]  # skip 'BST'
        return year, tc_name

    def _build_index(self, train_ratio, val_ratio, seed):
        """
        Build sample index from TCND data.
        The dataset already provides train/val/test splits as subfolders.
        train_ratio and val_ratio are ignored (kept for API compatibility).
        """
        import logging
        log = logging.getLogger(__name__)
        
        start_time = time.time()

        self._tcnd_root = self._resolve_tcnd_root()

        # Map requested split to whichever folder actually exists.
        # Priority: exact match → any available folder (for partial downloads).
        SPLIT_ALIASES = {
            "train": ["train", "test", "val"],
            "val":   ["val",   "test", "train"],
            "test":  ["test",  "train", "val"],
        }

        for basin in self.basins:
            parent = self._tcnd_root / "Data1D" / basin
            if not parent.exists():
                raise FileNotFoundError(
                    f"Basin directory not found: {parent}\n"
                    f"Available basins: {[d.name for d in (self._tcnd_root/'Data1D').iterdir() if d.is_dir()]}"
                )

            # Pick the first available split folder
            split_folder = None
            for candidate in SPLIT_ALIASES[self.split]:
                if (parent / candidate).exists():
                    split_folder = candidate
                    break

            if split_folder is None:
                raise FileNotFoundError(
                    f"No split folders found under {parent}. "
                    f"Expected one of: train, val, test"
                )

            if split_folder != self.split:
                log.warning(
                    f"'{self.split}' folder not found for basin {basin}; "
                    f"using '{split_folder}' instead."
                )

            basin_dir = parent / split_folder

            txt_files = sorted(basin_dir.glob("*.txt"))
            if not txt_files:
                raise FileNotFoundError(
                    f"No .txt files found in {basin_dir}"
                )

            log.info(f"  {basin}/{split_folder}: {len(txt_files)} storm files")

            # 8-column TCND format (no header, tab/space separated):
            # All numeric columns are ALREADY NORMALIZED by the dataset.
            # ID | FLAG | LAT_norm | LONG_norm | WND_norm | PRES_norm | YYYYMMDDHH | Name
            TCND_COLS = ["ID", "FLAG", "LAT_norm", "LONG_norm",
                         "WND_norm", "PRES_norm", "YYYYMMDDHH", "Name"]

            for txt_file in tqdm(txt_files, desc=f"Loading index for {basin}/{split_folder}", dynamic_ncols=True, leave=False):
                stem           = txt_file.stem          # e.g. WP2017BSTBANYAN
                year, tc_name  = self._parse_filename(stem, basin)

                try:
                    df = pd.read_csv(
                        txt_file, sep=r"\s+", header=None,
                        names=TCND_COLS, engine="python",
                        on_bad_lines="skip",
                    )
                except TypeError:
                    df = pd.read_csv(
                        txt_file, sep=r"\s+", header=None,
                        names=TCND_COLS, engine="python",
                    )

                for _, row in df.iterrows():
                    ts = str(int(float(row.get("YYYYMMDDHH", 0))))
                    data_3d_path = self._get_3d_path(basin, year, tc_name, ts)
                    env_path = self._get_env_path(basin, year, tc_name, ts)
                    
                    if self.use_3d and not data_3d_path:
                        continue
                    if self.use_env and not env_path:
                        continue
                        
                    self.index.append({
                        "basin":     basin,
                        "basin_idx": BASIN_TO_IDX[basin],
                        "storm_id":  stem,
                        "tc_name":   tc_name,
                        "year":      year,
                        "timestamp": ts,
                        "csv_row":   row.to_dict(),
                        "data_3d_path": data_3d_path,
                        "env_path":     env_path,
                    })
        
        log.info(f"Dataset index built in {time.time() - start_time:.2f}s with {len(self.index)} samples")

    def _get_3d_path(self, basin: str, year: str, tc_name: str, ts: str) -> Optional[Path]:
        """
        Data3D/{basin}/{year}/{tc_name}/TCND_{tc_name}_{ts}_sst_z_u_v.nc
        """
        p = (self._tcnd_root / "Data3D" / basin / year / tc_name
             / f"TCND_{tc_name}_{ts}_sst_z_u_v.nc")
        return p if p.exists() else None

    def _get_env_path(self, basin: str, year: str, tc_name: str, ts: str) -> Optional[Path]:
        """
        Env-Data/{basin}/{year}/{tc_name}/{ts}.npy
        """
        p = self._tcnd_root / "Env-Data" / basin / year / tc_name / f"{ts}.npy"
        return p if p.exists() else None

    # ── Item loading ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.index[idx]
        key = (meta["basin"], meta["storm_id"], meta["timestamp"])

        if self.cache and key in self._cache_dict:
            return self._cache_dict[key]

        sample = self._load_sample(meta)

        if self.cache:
            self._cache_dict[key] = sample

        return sample

    def _load_sample(self, meta: Dict) -> Dict[str, torch.Tensor]:
        row = meta["csv_row"]

        # ── Data_1d ──────────────────────────────────────────────────────────
        # TCND txt columns are ALREADY NORMALIZED by the dataset authors.
        # ID, FLAG, LAT_norm, LONG_norm, WND_norm, PRES_norm, YYYYMMDDHH, Name
        lat_norm  = float(row.get("LAT_norm",  0.0))
        long_norm = float(row.get("LONG_norm", 0.0))
        wnd_norm  = float(row.get("WND_norm",  0.0))
        pres_norm = float(row.get("PRES_norm", 0.0))
        data_1d   = torch.tensor([long_norm, lat_norm, pres_norm, wnd_norm],
                                  dtype=torch.float32)

        # ── Data_3d ───────────────────────────────────────────────────────────
        if self.use_3d:
            data_3d = self._load_3d(meta["data_3d_path"])
        else:
            # C = SST(1) + Z×4 + U×4 + V×4 = 13 channels, H=W=81
            data_3d = torch.zeros(13, 81, 81, dtype=torch.float32)

        # ── Env-Data ─────────────────────────────────────────────────────────
        # CRITICAL: We must ALWAYS load the targets from env_path, even if we ablate env_vec
        env_vec_full, y_intensity, y_direction = self._load_env(meta["env_path"])
        if self.use_env:
            env_vec = env_vec_full
        else:
            env_vec = torch.zeros(77, dtype=torch.float32)

        # ── Physics features (for PhysIRM) ───────────────────────────────────
        phys = self._compute_physics_features(row, meta["basin"], data_3d)

        return {
            "data_1d":       data_1d,                                    # (4,)
            "data_3d":       data_3d,                                    # (13, 81, 81)
            "env_data":      env_vec,                                    # (77,)
            "phys_features": phys,                                       # (8,)
            "basin_idx":     torch.tensor(meta["basin_idx"], dtype=torch.long),
            "y_intensity":   torch.tensor(y_intensity,       dtype=torch.long),
            "y_direction":   torch.tensor(y_direction,       dtype=torch.long),
        }

    # ── Modality loaders ─────────────────────────────────────────────────────

    def _load_3d(self, path: Path) -> torch.Tensor:
        """
        Load ERA5 patch from NetCDF.
        Variables: sst (H,W), z/u/v (time=1, pressure_level=4, H, W)
        pressure_levels: [200, 500, 850, 925] hPa
        SST is in Kelvin with large fill values (~9.97e36) for land/missing.
        Returns (13, H, W) tensor: [u×4, v×4, z×4, sst]
        """
        import xarray as xr
        ds = xr.open_dataset(path)

        # ── SST: (H, W) or (time=1, H, W), Kelvin, has large fill values ──────
        sst_raw = ds["sst"].values.astype(np.float32)
        if sst_raw.ndim == 3: sst_raw = sst_raw[0]  # (time, H, W) → (H, W)
        sst_raw[sst_raw > 1e10] = np.nan          # mask fill values

        # ── z, u, v: (1, 4, H, W) → (4, H, W) ──────────────────────────────
        z = ds["z"].values.astype(np.float32)
        u = ds["u"].values.astype(np.float32)
        v = ds["v"].values.astype(np.float32)
        ds.close()

        if z.ndim == 4: z = z[0]
        if u.ndim == 4: u = u[0]
        if v.ndim == 4: v = v[0]

        H, W = sst_raw.shape

        chs = []
        for arr in [u, v, z]:
            for i in range(4):
                lev = arr[i] if i < arr.shape[0] else np.zeros((H, W), dtype=np.float32)
                chs.append(lev)
        chs.append(sst_raw)

        data_3d = np.stack(chs, axis=0)

        # Per-channel normalisation per-sample (notebook method)
        for c in range(13):
            ch = data_3d[c]
            valid = ch[~np.isnan(ch)]
            if len(valid) > 0:
                mu, std = valid.mean(), max(valid.std(), 1e-6)
                ch[np.isnan(ch)] = mu
                data_3d[c] = (ch - mu) / std
            else:
                data_3d[c] = np.zeros_like(ch)

        return torch.from_numpy(data_3d)  # (13, H, W)

    def _load_env(self, path: Path):
        """
        Load Env-Data .npy file.
        Returns (env_vec, y_intensity, y_direction).

        Actual TCND .npy fields (confirmed from sample data):
          month            (12,) one-hot
          area              (6,) one-hot
          intensity_class   (6,) one-hot
          wind              scalar float
          move_velocity     scalar int
          location_long    (36,) one-hot
          location_lat     (12,) one-hot
          history_direction12   scalar int  (-1 = unknown, 0-7 = compass)
          history_direction24   scalar int
          history_inte_change24 scalar int  (-1 = unknown, 0-4 = class)
          future_direction24    scalar int  ← label
          future_inte_change24  scalar int  ← label
        Total feature dim = 77
        """
        d = np.load(path, allow_pickle=True).item()

        def ohe(val, length):
            """Get a one-hot/multi-hot array of fixed length."""
            if val is None:
                return np.zeros(length, dtype=np.float32)
            arr = np.asarray(val, dtype=np.float32).ravel()
            if arr.size == 0:
                return np.zeros(length, dtype=np.float32)
            if arr.size < length:
                arr = np.pad(arr, (0, length - arr.size))
            return arr[:length].astype(np.float32)

        def scalar_norm(val, default=0.0, scale=1.0, sentinel_normed=None):
            """Scalar → normalised length-1 float32 array.
            -1 sentinel = unknown → sentinel_normed if provided, else default/scale."""
            if val is None:
                return np.array([default / scale if sentinel_normed is None else sentinel_normed],
                                dtype=np.float32)
            v = float(np.asarray(val).ravel()[0])
            if v < 0:  # -1 sentinel = unknown
                if sentinel_normed is not None:
                    return np.array([sentinel_normed], dtype=np.float32)
                v = default
            return np.array([v / scale], dtype=np.float32)

        parts = [
            ohe(d.get("month"),           12),   # 12
            ohe(d.get("area"),             6),   #  6
            ohe(d.get("intensity_class"),  6),   #  6
            scalar_norm(d.get("wind"),          scale=1.0),   #  1  (already normalised 0-1)
            scalar_norm(d.get("move_velocity"), scale=1.0),   #  1  (m/s, notebook uses raw)
            ohe(d.get("location_long"),   36),   # 36
            ohe(d.get("location_lat"),    12),   # 12
            scalar_norm(d.get("history_direction12"), scale=1.0, sentinel_normed=0.0),  #  1  (0-7, -1→0.0)
            scalar_norm(d.get("history_direction24"), scale=1.0, sentinel_normed=0.0),  #  1
            scalar_norm(d.get("history_inte_change24"), scale=1.0, sentinel_normed=0.0),  # 1  (0-4, -1→0.0)
        ]
        env_vec = torch.from_numpy(np.concatenate(parts))  # (77,)

        assert env_vec.shape[0] == 77, (
            f"env vector has length {env_vec.shape[0]}, expected 77. "
            f"Check .npy file structure at {path}"
        )

        y_intensity = int(np.asarray(d.get("future_inte_change24", 2)).ravel()[0])
        y_direction = int(np.asarray(d.get("future_direction24",   0)).ravel()[0])

        # Clamp -1 sentinel to valid class ranges
        if y_intensity < 0: y_intensity = 2   # steady state
        if y_direction < 0: y_direction = 0
        y_intensity = min(y_intensity, 4)
        y_direction = min(y_direction, 7)

        return env_vec, y_intensity, y_direction

    def _compute_physics_features(
        self, row: Dict, basin: str, data_3d: torch.Tensor
    ) -> torch.Tensor:
        """
        8-dim physics feature vector for PhysIRM invariant sub-space.
          [0] SST anomaly vs basin climatology (basin-level constant; 28°C reference)
          [1] Wind shear proxy: negative spatial cross-correlation of U_200 and U_850
              (cross-channel Pearson correlation is invariant to per-sample z-scoring)
          [2] Coriolis parameter (normalised, per-sample from LAT_norm)
          [3] MPI proxy (basin SST - 26°C threshold, basin-level)
          [4] Boundary-layer dynamics proxy (WND_norm from Data_1d)
          [5] Outflow proxy: spatial skewness of Z_200 (channel 8)
          [6] Steering proxy: spatial skewness of Z_500 (channel 9)
          [7] Current intensity (WND_norm, as-is from Data_1d)

        NOTE on normalization: Data_3d channels are per-sample z-scored in _load_3d,
        so their spatial means are ~0 and spatial variances are ~1 by construction.
        Attempting to recover absolute physical units via the global NORM constants
        is incorrect (produces values near the normalisation mean for every sample).
        We instead use:
          - Basin-level climatological constants (BASIN_SST_STATS) for SST / MPI.
          - Cross-channel Pearson correlation for wind shear (invariant to z-scoring).
          - Spatial skewness for outflow / steering (also invariant to z-scoring).
          - Data_1d features (WND_norm, LAT_norm) for intensity and Coriolis.
        """
        wnd_norm  = float(row.get("WND_norm", 0.0))
        sst_stats = BASIN_SST_STATS.get(basin, {"mean": 28.0, "std": 2.0})

        # [0] SST anomaly: basin climatological mean vs. global tropical reference (28°C)
        sst_anom = float(np.clip(
            (sst_stats["mean"] - 28.0) / max(sst_stats["std"], 1e-6), -3.0, 3.0
        ))

        # [1] Wind shear proxy: negative spatial cross-correlation between U_200 and U_850.
        # After per-sample z-scoring, each channel has zero mean.  Pearson correlation
        # between two channels is invariant to individual z-scoring and captures opposing
        # wind patterns.  Negated so that high shear → large positive feature value.
        if data_3d.shape[0] >= 13:
            u200  = data_3d[0].float().flatten()   # channel 0 = U at 200 hPa
            u850  = data_3d[2].float().flatten()   # channel 2 = U at 850 hPa
            denom = (u200.std() * u850.std()).clamp(min=1e-6)
            corr  = float((u200 * u850).mean() / denom)
            shear = float(np.clip(-corr, -1.0, 1.0))
        else:
            shear = 0.0

        # [2] Coriolis parameter per time step using un-normalized latitude.
        # TCND LAT_norm = (LAT_raw + 90.0) / 180.0  →  LAT_raw = LAT_norm * 180 - 90
        lat_norm      = float(row.get("LAT_norm", 0.0))
        lat_deg       = lat_norm * 180.0 - 90.0
        omega         = 7.2921e-5
        coriolis_raw  = 2 * omega * np.sin(np.deg2rad(lat_deg))
        coriolis_norm = float(np.clip(coriolis_raw / 1.4584e-4, -1.0, 1.0))

        # [3] MPI proxy: basin SST vs 26°C intensification threshold (basin-level)
        mpi_proxy = float(np.clip((sst_stats["mean"] - 26.0) / 10.0, -1.0, 1.0))

        # [4] Boundary-layer dynamics proxy: current wind intensity (Data_1d, not z-scored)
        bl_proxy = float(np.clip(wnd_norm, 0.0, 1.0))

        # [5] Outflow proxy: spatial skewness of Z_200 (channel 8).
        # Skewness is invariant to zero-mean / unit-variance z-scoring and captures
        # asymmetric upper-tropospheric outflow structure.
        if data_3d.shape[0] >= 13:
            z200    = data_3d[8].float()
            outflow = float((
                (z200 - z200.mean()).pow(3).mean()
                / z200.std().clamp(min=1e-6).pow(3)
            ).clamp(-3.0, 3.0))
        else:
            outflow = 0.0

        # [6] Steering proxy: spatial skewness of Z_500 (channel 9)
        if data_3d.shape[0] >= 13:
            z500     = data_3d[9].float()
            steering = float((
                (z500 - z500.mean()).pow(3).mean()
                / z500.std().clamp(min=1e-6).pow(3)
            ).clamp(-3.0, 3.0))
        else:
            steering = 0.0

        # [7] Current intensity (normalised wind speed from Data_1d)
        return torch.tensor(
            [sst_anom, shear, coriolis_norm, mpi_proxy,
             bl_proxy, outflow, steering, wnd_norm],
            dtype=torch.float32,
        )




# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloader(
    root: str,
    basins: List[str],
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_3d: bool = True,
    use_env: bool = True,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    ds = TCNDDataset(
        root=root, basins=basins, split=split,
        use_3d=use_3d, use_env=use_env, seed=seed, **kwargs
    )
    if len(ds) == 0:
        raise RuntimeError(
            "Dataset is empty for basins=" + str(basins) + ", split=" + repr(split) + ".\n"
            "Check --data_root contains CSV files under Data_1d/<BASIN>/."
        )
    shuffle   = (split == "train")
    # Only drop_last when we have strictly more than one batch of data
    drop_last = (split == "train") and (len(ds) > batch_size)
    # Cap batch_size to dataset size to avoid sampler errors on tiny splits
    safe_bs   = min(batch_size, len(ds))
    return DataLoader(
        ds, batch_size=safe_bs, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def make_per_basin_loaders(
    root: str,
    basins: List[str],
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs,
) -> Dict[str, DataLoader]:
    """Return one DataLoader per basin (used for per-environment IRM updates)."""
    return {
        b: make_dataloader(root, [b], split, batch_size, num_workers, **kwargs)
        for b in basins
    }