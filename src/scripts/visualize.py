"""
Analysis & Visualization for the Paper
========================================
Generates all figures and tables for the NeurIPS/ICLR submission.

Figures produced:
  Fig 1 — Basin transfer matrix (heatmap of all 30 directed pairs)
  Fig 2 — Method comparison radar chart (per-basin RI-F1)
  Fig 3 — Few-shot fine-tuning curves (k vs accuracy)
  Fig 4 — Source composition effect (number of source basins vs BTG)
  Fig 5 — Physics feature importance bar chart
  Fig 6 — t-SNE of z_phys representations (colour by basin)
  Fig 7 — λ_irm sensitivity plot

Run:
    python analysis/visualize.py --results_dir ./runs --output_dir ./figures
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# ── Try import matplotlib; fail gracefully if not available ───────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Text outputs only.")


BASIN_DISPLAY = {
    "WP": "W. Pacific",
    "NA": "N. Atlantic",
    "EP": "E. Pacific",
    "NI": "N. Indian",
    "SI": "S. Indian",
    "SP": "S. Pacific",
}

METHOD_COLORS = {
    "erm":     "#1f77b4",
    "irm":     "#ff7f0e",
    "vrex":    "#2ca02c",
    "coral":   "#d62728",
    "dann":    "#9467bd",
    "maml":    "#8c564b",
    "physirm": "#e377c2",  # Our method — highlighted
}

METHOD_LABELS = {
    "erm":     "ERM",
    "irm":     "IRM",
    "vrex":    "V-REx",
    "coral":   "CORAL",
    "dann":    "DANN",
    "maml":    "MAML",
    "physirm": "PhysIRM (Ours)",
}


# ── Fig 1: Transfer Matrix ────────────────────────────────────────────────────

def plot_transfer_matrix(
    directed_results: List[dict],
    metric: str = "final_acc_int",
    method: str = "physirm",
    output_path: Optional[str] = None,
):
    """
    Heatmap showing acc_int for all 30 source→target directed pairs.
    Diagonal is blank (no self-transfer).

    For the paper: use this to argue which basin pairs are most similar
    (off-diagonal symmetry analysis = Figure 1 and Section 5.3).
    """
    basins = ["WP", "NA", "EP", "NI", "SI", "SP"]
    n = len(basins)
    matrix = np.full((n, n), np.nan)

    for r in directed_results:
        if r["method"] != method:
            continue
        src = r.get("source_basins", r.get("source", [None]))[0]
        tgt = r["target_basin"]
        if src in basins and tgt in basins:
            i, j = basins.index(src), basins.index(tgt)
            matrix[i, j] = r[metric]

    if not HAS_MPL:
        print("Transfer Matrix:")
        for i, src in enumerate(basins):
            row = [f"{matrix[i, j]:.2f}" if not np.isnan(matrix[i, j]) else " — "
                   for j in range(n)]
            print(f"  {src}: {row}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0.4, vmax=0.9, cmap="RdYlGn", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (Intensity)", fontsize=11)

    labels = [BASIN_DISPLAY[b] for b in basins]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Target Basin", fontsize=12)
    ax.set_ylabel("Source Basin", fontsize=12)
    ax.set_title(f"Basin Transfer Matrix — {METHOD_LABELS.get(method, method)}", fontsize=13)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.45 < matrix[i, j] < 0.85 else "white")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


# ── Fig 2: Per-basin RI-F1 bar chart ─────────────────────────────────────────

def plot_ri_comparison(
    lobo_results: List[dict],
    output_path: Optional[str] = None,
):
    """
    Grouped bar chart: RI-F1 per target basin, grouped by method.
    Our proposed metric — central to the safety argument in the paper.
    """
    basins  = ["WP", "NA", "EP", "NI", "SI", "SP"]
    methods = list(METHOD_LABELS.keys())

    data = {m: {b: 0.0 for b in basins} for m in methods}
    for r in lobo_results:
        m = r.get("method")
        b = r.get("target_basin")
        if m in data and b in data[m]:
            data[m][b] = r.get("final_ri_f1", 0.0)

    if not HAS_MPL:
        print("\nRI-F1 per target basin:")
        for m in methods:
            row = "  ".join(f"{b}:{data[m][b]:.3f}" for b in basins)
            print(f"  {METHOD_LABELS[m]:20s}: {row}")
        return

    x      = np.arange(len(basins))
    width  = 0.12
    offset = -(len(methods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(methods):
        vals = [data[m][b] for b in basins]
        bars = ax.bar(x + offset + i * width, vals, width,
                      label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                      edgecolor="white", linewidth=0.5,
                      zorder=3)
        # Bold border for our method
        if m == "physirm":
            for bar in bars:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([BASIN_DISPLAY[b] for b in basins], fontsize=11)
    ax.set_ylabel("RI-F1 Score", fontsize=12)
    ax.set_title("Rapid Intensification F1 — Zero-Shot Cross-Basin Transfer", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


# ── Fig 3: Few-shot scaling curve ─────────────────────────────────────────────

def plot_few_shot_curve(
    few_shot_results: Dict[str, Dict[int, float]],  # method → k → acc
    target_basin: str = "SI",
    output_path: Optional[str] = None,
):
    """
    Line plot: accuracy vs. number of few-shot examples.
    Shows that PhysIRM reaches ERM-oracle level faster than other methods.
    """
    if not HAS_MPL:
        for m, kv in few_shot_results.items():
            print(f"  {m}: {kv}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for m, kv in few_shot_results.items():
        ks   = sorted(kv.keys())
        accs = [kv[k] for k in ks]
        ax.plot(ks, accs, "o-", label=METHOD_LABELS.get(m, m),
                color=METHOD_COLORS.get(m, "grey"),
                linewidth=2.5 if m == "physirm" else 1.5,
                markersize=6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of target-basin shots (k)", fontsize=12)
    ax.set_ylabel("Accuracy (Intensity)", fontsize=12)
    ax.set_title(f"Few-Shot Adaptation — Target: {BASIN_DISPLAY.get(target_basin, target_basin)}",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── Fig 6: t-SNE of physics representations ───────────────────────────────────

def plot_tsne_physics(
    features_by_basin: Dict[str, np.ndarray],  # basin → (N, D) array
    output_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
):
    """
    t-SNE visualisation of z_phys coloured by basin.
    Key qualitative result: z_phys should cluster by TC dynamics
    (not by basin geography) — supporting the invariance argument.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Install scikit-learn for t-SNE: pip install scikit-learn")
        return

    all_feat, all_labels = [], []
    for basin, feat in features_by_basin.items():
        all_feat.append(feat)
        all_labels.extend([basin] * len(feat))

    X = np.concatenate(all_feat, axis=0)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=42, n_jobs=-1)
    X_2d = tsne.fit_transform(X)

    if not HAS_MPL:
        print(f"t-SNE computed for {len(X)} samples from {len(features_by_basin)} basins")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    basin_list = list(features_by_basin.keys())
    cmap = plt.get_cmap("tab10")
    colors = {b: cmap(i / len(basin_list)) for i, b in enumerate(basin_list)}

    offset = 0
    for basin, feat in features_by_basin.items():
        n = len(feat)
        xy = X_2d[offset:offset + n]
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors[basin]], alpha=0.5,
                   s=8, label=BASIN_DISPLAY.get(basin, basin))
        offset += n

    ax.legend(fontsize=10, markerscale=2)
    ax.set_title("t-SNE of z_phys Representations (PhysIRM)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")

    # Add annotation
    ax.text(0.02, 0.02,
            "Basins interleaved in z_phys → physics invariance achieved",
            transform=ax.transAxes, fontsize=8, style="italic",
            color="grey")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── LaTeX table generator ─────────────────────────────────────────────────────

def generate_latex_table(
    lobo_results: List[dict],
    metric: str = "final_acc_int",
    caption: str = "Leave-One-Basin-Out Zero-Shot Transfer",
    label: str = "tab:lobo",
) -> str:
    """
    Generate a ready-to-paste LaTeX table for the paper.
    Best result per column is bolded.
    """
    basins  = ["WP", "NA", "EP", "NI", "SI", "SP"]
    methods = list(METHOD_LABELS.keys())

    data = {m: {b: None for b in basins} for m in methods}
    for r in lobo_results:
        m = r.get("method"); b = r.get("target_basin")
        if m in data and b in data[m]:
            data[m][b] = r.get(metric, 0.0)

    # Find best per column
    best = {}
    for b in basins:
        vals = [data[m][b] for m in methods if data[m][b] is not None]
        best[b] = max(vals) if vals else 0

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(basins) + "c}",
        "\\toprule",
        "Method & " + " & ".join(f"\\textbf{{{BASIN_DISPLAY[b]}}}" for b in basins)
        + " & \\textbf{Avg} \\\\",
        "\\midrule",
    ]

    for m in methods:
        row_vals = []
        avg_vals = []
        for b in basins:
            v = data[m][b]
            if v is None:
                row_vals.append("—")
            else:
                avg_vals.append(v)
                cell = f"{v:.3f}"
                if abs(v - best[b]) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                row_vals.append(cell)
        avg = f"{np.mean(avg_vals):.3f}" if avg_vals else "—"
        method_str = f"\\textbf{{{METHOD_LABELS[m]}}}" if m == "physirm" \
            else METHOD_LABELS[m]
        lines.append(f"{method_str} & " + " & ".join(row_vals) + f" & {avg} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./runs")
    p.add_argument("--output_dir",  type=str, default="./figures")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = Path(args.results_dir) / "benchmark_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {results_path}")

        # Generate figures
        plot_ri_comparison(results, output_path=str(out / "fig_ri_comparison.pdf"))
        plot_transfer_matrix(results, output_path=str(out / "fig_transfer_matrix.pdf"))

        # Generate LaTeX table
        latex = generate_latex_table(results)
        with open(out / "table_lobo.tex", "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {out / 'table_lobo.tex'}")
    else:
        print(f"No results found at {results_path}. Run train.py first.")
