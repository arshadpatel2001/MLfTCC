"""
Experiment Configuration Registry
===================================
All ablation and sensitivity experiments from the paper.

Section 4 of paper: "Ablations and Sensitivity Analysis"

Run all ablations:
    python run_ablations.py --data_root /path/to/TCND --output_dir ./ablations
"""

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 1: Modality ablation (Table 3 in paper)
# Train PhysIRM with different input modality combinations
# ─────────────────────────────────────────────────────────────────────────────
MODALITY_ABLATIONS = [
    {"name": "Full (3D+1D+Env)",  "no_3d": False, "no_env": False},
    {"name": "No 3D",             "no_3d": True,  "no_env": False},
    {"name": "No Env",            "no_3d": False, "no_env": True},
    {"name": "1D only",           "no_3d": True,  "no_env": True},
]

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 2: PhysIRM component ablation (Table 4 in paper)
# Test contribution of each PhysIRM term
# ─────────────────────────────────────────────────────────────────────────────
PHYSIRM_ABLATIONS = [
    # Full PhysIRM
    {"name": "PhysIRM (full)",
     "irm_lambda": 1.0, "orth_lambda": 0.1, "phys_lambda": 0.5},
    # No physics grounding
    {"name": "PhysIRM - L_phys",
     "irm_lambda": 1.0, "orth_lambda": 0.1, "phys_lambda": 0.0},
    # No orthogonality
    {"name": "PhysIRM - L_orth",
     "irm_lambda": 1.0, "orth_lambda": 0.0, "phys_lambda": 0.5},
    # IRM only (no phys decomposition = standard IRM)
    {"name": "IRM (no decomp)",
     "irm_lambda": 1.0, "orth_lambda": 0.0, "phys_lambda": 0.0},
    # ERM baseline
    {"name": "ERM",
     "irm_lambda": 0.0, "orth_lambda": 0.0, "phys_lambda": 0.0},
]

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 3: Source basin composition (Figure 4 in paper)
# How does the number/choice of source basins affect transfer?
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_COMPOSITION = {
    "target": "SI",  # South Indian Ocean (most challenging target)
    "sources": [
        # Single source
        ["WP"],
        ["NA"],
        ["EP"],
        # Two sources
        ["WP", "NA"],
        ["WP", "EP"],
        # Three sources
        ["WP", "NA", "EP"],
        # Four sources
        ["WP", "NA", "EP", "NI"],
        # Five sources (all except target)
        ["WP", "NA", "EP", "NI", "SP"],
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 4: Few-shot fine-tuning scaling (Figure 5 in paper)
# ─────────────────────────────────────────────────────────────────────────────
FEW_SHOT_K = [0, 8, 16, 32, 64, 128, 256, 512]  # k=0 is zero-shot

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 5: λ_irm sensitivity (Figure 6 in paper)
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_SWEEP = {
    "irm_lambda":  [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "orth_lambda": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    "phys_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
}

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 6: Transfer direction asymmetry (Section 5.3 in paper)
# Some basin pairs transfer better in one direction than the other.
# We test all 30 directed pairs (6×5) to build a transfer matrix.
# ─────────────────────────────────────────────────────────────────────────────
ALL_DIRECTED_PAIRS = [
    {"source": [s], "target": t}
    for s in ["WP", "NA", "EP", "NI", "SI", "SP"]
    for t in ["WP", "NA", "EP", "NI", "SI", "SP"]
    if s != t
]

# ─────────────────────────────────────────────────────────────────────────────
# Ablation 7: Physics feature importance (Table 5 in paper)
# Ablate individual physics features from z_phys
# ─────────────────────────────────────────────────────────────────────────────
PHYSICS_FEATURE_NAMES = [
    "SST_anomaly",
    "wind_shear",
    "coriolis",
    "MPI_proxy",
    "BL_moisture",
    "outflow_temp",
    "steering",
    "current_intensity",
]
# Run PhysIRM with each feature zeroed out — importance = Δ(target RI-F1)


# ─────────────────────────────────────────────────────────────────────────────
# Recommended hyperparameter search space (DomainBed protocol)
# Random search with 20 trials per method per LOBO split
# ─────────────────────────────────────────────────────────────────────────────
HPARAM_SEARCH_SPACE = {
    "lr":           {"type": "log_uniform", "low": 1e-5, "high": 1e-3},
    "weight_decay": {"type": "log_uniform", "low": 1e-6, "high": 1e-2},
    "dropout":      {"type": "uniform",     "low": 0.0,  "high": 0.3},
    "irm_lambda":   {"type": "log_uniform", "low": 0.1,  "high": 10.0},
    "orth_lambda":  {"type": "log_uniform", "low": 0.01, "high": 1.0},
    "phys_lambda":  {"type": "log_uniform", "low": 0.1,  "high": 2.0},
    "warmup_steps": {"type": "choice",      "values": [200, 500, 1000]},
}


def sample_hparams(method: str, rng=None) -> dict:
    """Sample one hyperparameter configuration for a given method."""
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    def _sample(spec):
        if spec["type"] == "log_uniform":
            return float(np.exp(rng.uniform(
                np.log(spec["low"]), np.log(spec["high"])
            )))
        elif spec["type"] == "uniform":
            return float(rng.uniform(spec["low"], spec["high"]))
        elif spec["type"] == "choice":
            return rng.choice(spec["values"])
        raise ValueError(f"Unknown type: {spec['type']}")

    shared = {
        "lr":           _sample(HPARAM_SEARCH_SPACE["lr"]),
        "weight_decay": _sample(HPARAM_SEARCH_SPACE["weight_decay"]),
        "dropout":      _sample(HPARAM_SEARCH_SPACE["dropout"]),
    }
    method_specific = {}
    if method in ("irm", "physirm"):
        method_specific["irm_lambda"]   = _sample(HPARAM_SEARCH_SPACE["irm_lambda"])
        method_specific["warmup_steps"] = _sample(HPARAM_SEARCH_SPACE["warmup_steps"])
    if method == "physirm":
        method_specific["orth_lambda"]  = _sample(HPARAM_SEARCH_SPACE["orth_lambda"])
        method_specific["phys_lambda"]  = _sample(HPARAM_SEARCH_SPACE["phys_lambda"])

    return {**shared, **method_specific}
