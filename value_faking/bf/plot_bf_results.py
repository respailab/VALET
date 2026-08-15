"""Figures for BF1-BF4, built on the CSVs written by the bf stages."""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy import stats
from value_faking.paths import REPO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value_faking_test"))
from value_faking.report.plot_style import apply_style, savefig, TOL_DIVERGING_STRONG, gap_norm, cell_text_color

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()
BF_DIR = f"{REPO}/behavioral_analysis"
OUT_DIR = f"{BF_DIR}/plots"

SEQ_BLUES = plt.cm.Blues


def plot_mub_dumbbell(df: pd.DataFrame):
    for model, g in df.groupby("model"):
        g = g.sort_values("M_minus_U", ascending=True).reset_index(drop=True)
        n = len(g)
        fig, ax = plt.subplots(figsize=(5.5, 0.22 * n + 1))
        y = np.arange(n)

        for i, row in g.iterrows():
            ax.plot([row["B_mean"], row["U_mean"], row["M_mean"]], [i, i, i],
                    "-", color="#0077BB", lw=1.0, alpha=0.5, zorder=1)
            ax.scatter(row["B_mean"], i, marker="o", s=14, color="#333333", zorder=3, label="_")
            ax.scatter(row["U_mean"], i, marker="s", s=16, color="#0077BB", zorder=3, label="_")
            ax.scatter(row["M_mean"], i, marker="^", s=18, color="#EE7733", zorder=3, label="_")

        ax.set_yticks(y)
        ax.set_yticklabels(g["value"], fontsize=8)
        ax.set_xlabel("compliance rate (P chose anti-value option)")
        ax.set_xlim(-0.02, 1.02)
        ax.axvline(0.5, color="#CCCCCC", lw=0.6, zorder=0)

        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], marker="o", color="#333333", linestyle="", label="bare (B)"),
            Line2D([0], [0], marker="s", color="#0077BB", linestyle="", label="unmonitored (U)"),
            Line2D([0], [0], marker="^", color="#EE7733", linestyle="", label="monitored (M)"),
        ]
        ax.legend(handles=marker_handles, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        savefig(fig, f"{OUT_DIR}/mub_dumbbell_{model}.png")
        plt.close(fig)


def plot_mub_dumbbell_aggregate(df: pd.DataFrame):
    agg = df.groupby("model")[["B_mean", "M_mean", "U_mean"]].mean()
    agg["M_minus_U"] = agg["M_mean"] - agg["U_mean"]
    agg = agg.sort_values("M_minus_U", ascending=True)

    n = len(agg)
    fig, ax = plt.subplots(figsize=(8, 0.6 * n + 1.4))
    y = np.arange(n)

    for i, (model, row) in enumerate(agg.iterrows()):
        ax.plot([row["B_mean"], row["U_mean"], row["M_mean"]], [i, i, i],
                "-", color="#0077BB", lw=1.6, alpha=0.5, zorder=1)
        ax.scatter(row["B_mean"], i, marker="o", s=70, color="#333333", zorder=3, label="_")
        ax.scatter(row["U_mean"], i, marker="s", s=78, color="#0077BB", zorder=3, label="_")
        ax.scatter(row["M_mean"], i, marker="^", s=88, color="#EE7733", zorder=3, label="_")

    ax.set_yticks(y)
    ax.set_yticklabels(agg.index, fontsize=18)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("compliance rate (P chose anti-value option), averaged across 32 values",
                  fontsize=12.5)
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(0.5, color="#CCCCCC", lw=0.6, zorder=0)

    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="o", color="#333333", linestyle="", label="bare (B)"),
        Line2D([0], [0], marker="s", color="#0077BB", linestyle="", label="unmonitored (U)"),
        Line2D([0], [0], marker="^", color="#EE7733", linestyle="", label="monitored (M)"),
    ]
    ax.legend(handles=marker_handles, fontsize=13, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    savefig(fig, f"{OUT_DIR}/mub_dumbbell_aggregate.png")
    plt.close(fig)


def _leaf_order(gap_corr: pd.DataFrame):
    dist = 1 - gap_corr.to_numpy()
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    Z = linkage(dist[np.triu_indices_from(dist, k=1)], method="average")
    dn = dendrogram(Z, no_plot=True, labels=gap_corr.index.tolist())
    return dn["ivl"]


def plot_bf2_heatmaps(gap_corr: pd.DataFrame, sem_sim: pd.DataFrame):
    order = _leaf_order(gap_corr)
    gap_o = gap_corr.loc[order, order]
    sem_o = sem_sim.loc[order, order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))

    im0 = axes[0].imshow(gap_o, cmap=TOL_DIVERGING_STRONG, norm=mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    axes[0].set_xticks(range(len(order))); axes[0].set_xticklabels(order, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="gap correlation (across models)")

    im1 = axes[1].imshow(sem_o, cmap=SEQ_BLUES, vmin=0, vmax=1)
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order, rotation=90, fontsize=6)
    axes[1].set_yticks(range(len(order))); axes[1].set_yticklabels(order, fontsize=6)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="semantic similarity (embeddings)")

    fig.tight_layout()
    savefig(fig, f"{OUT_DIR}/bf2_heatmaps.png")
    plt.close(fig)


def plot_bf2_scatter(pairs: pd.DataFrame, mantel_r: float, mantel_p: float):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(pairs["semantic_sim"], pairs["gap_corr"], s=8, color="#0077BB", alpha=0.35, edgecolor="none")

    slope, intercept, *_ = stats.linregress(pairs["semantic_sim"], pairs["gap_corr"])
    xs = np.linspace(pairs["semantic_sim"].min(), pairs["semantic_sim"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#CC3311", lw=1.2)

    ax.set_xlabel("semantic similarity")
    ax.set_ylabel("gap correlation")
    ax.text(0.03, 0.97, f"Mantel r={mantel_r:+.3f}\np={mantel_p:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", lw=0.5))
    fig.tight_layout()
    savefig(fig, f"{OUT_DIR}/bf2_scatter.png")
    plt.close(fig)


def plot_bf2_cluster_contingency(match_table: pd.DataFrame):
    ct = pd.crosstab(match_table["semantic_cluster"], match_table["behavioral_cluster"])
    fig, ax = plt.subplots(figsize=(0.7 * ct.shape[1] + 2, 0.4 * ct.shape[0] + 1.5))
    im = ax.imshow(ct, cmap=SEQ_BLUES, vmin=0)
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, fontsize=9)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index, fontsize=9)
    ax.set_xlabel("behavioral cluster (gap-correlation)")
    ax.set_ylabel("semantic cluster (LLM)")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iloc[i, j]
            if v > 0:
                bg = im.cmap(im.norm(v))
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=cell_text_color(bg))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="N values")
    fig.tight_layout()
    savefig(fig, f"{OUT_DIR}/bf2_cluster_contingency.png")
    plt.close(fig)


def plot_bf3_forest(bf3: pd.DataFrame):
    bf3 = bf3.sort_values("odds_ratio").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(bf3) + 1.6))
    y = np.arange(len(bf3))

    sig = (bf3["or_ci_lo"] > 1) | (bf3["or_ci_hi"] < 1)
    colors = np.where(sig, "#CC3311", "#999999")

    xerr_lo = bf3["odds_ratio"] - bf3["or_ci_lo"]
    xerr_hi = bf3["or_ci_hi"] - bf3["odds_ratio"]
    for i, row in bf3.iterrows():
        ax.errorbar(row["odds_ratio"], i, xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                    fmt="o", color=colors[i], ecolor=colors[i], elinewidth=1.8,
                    capsize=3.5, markersize=7)

    ax.axvline(1.0, color="#333333", lw=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(bf3["model"], fontsize=27)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Odds ratio", fontsize=18.75)
    fig.tight_layout()
    savefig(fig, f"{OUT_DIR}/bf3_forest.png")
    plt.close(fig)


# Models that sit in the crowded high-gap / high-verbalization cluster: their
# labels are stacked in a column (ordered by mean_gap ascending) and connected
# back to the point with a thin vertical leader line, instead of overlapping
# in place.
BF4_STACKED_CLUSTER = [
    "mistral-large-2512", "llama-3.1-70b", "llama-3.3-70b",
    "deepseek-r1-0528", "deepseek-r1-distill",
]

# Manual per-model offsets (points) for the remaining, less-crowded labels.
BF4_DIRECT_OFFSETS = {
    "ministral-14b": dict(dx=-8, dy=-2,  ha="right", va="top"),
    "gemma-3-27b":   dict(dx=10, dy=-20, ha="left",  va="top"),
    "llama-3.1-8b":  dict(dx=16, dy=0,   ha="left",  va="center"),
    "gpt-4o-mini":   dict(dx=16, dy=0,   ha="left",  va="center"),
}


def plot_bf4_scatter(bf4: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7.6))
    gap_med = bf4["mean_gap"].median()
    verb_med = bf4["verbalization_rate"].median()
    x_max = bf4["mean_gap"].max() * 1.75
    x_min = -0.20
    y_bottom, y_top = -0.02, 1.05
    verb_med_frac = (verb_med - y_bottom) / (y_top - y_bottom)

    ax.axvspan(gap_med, x_max, ymin=0, ymax=verb_med_frac,
               color="#CC3311", alpha=0.06, zorder=0)
    ax.axvline(gap_med, color="#CCCCCC", lw=0.7, linestyle="--")
    ax.axhline(verb_med, color="#CCCCCC", lw=0.7, linestyle="--")
    ax.text(x_max - 0.01, 0.03, "silent-faking quadrant (empty)",
            fontsize=11, style="italic", color="#CC3311", ha="right", va="bottom")

    ax.scatter(bf4["mean_gap"], bf4["verbalization_rate"], s=70, color="#0077BB", zorder=3)

    cluster = bf4[bf4["model"].isin(BF4_STACKED_CLUSTER)].set_index("model").loc[BF4_STACKED_CLUSTER]
    row_ys = np.linspace(0.86, 0.20, len(BF4_STACKED_CLUSTER))
    for (model, row), row_y in zip(cluster.iterrows(), row_ys):
        x, y = row["mean_gap"], row["verbalization_rate"]
        ax.plot([x, x], [y - 0.012, row_y + 0.015], color="#999999", lw=0.8, zorder=2)
        ax.text(x + 0.006, row_y, model, fontsize=20, fontweight="bold",
                ha="left", va="center")

    # mistral-small-2409 sits at the top of the left cluster, right beneath
    # y=1.0 — give it its own leader line into the top-left corner so it
    # doesn't run into the y-axis or the right cluster's markers.
    ms_row = bf4.set_index("model").loc["mistral-small-2409"]
    ms_x, ms_y = ms_row["mean_gap"], ms_row["verbalization_rate"]
    ms_label_x, ms_label_y = x_min + 0.02, 1.00
    ax.plot([ms_x, ms_label_x + 0.03], [ms_y, ms_label_y], color="#999999", lw=0.8, zorder=2)
    ax.text(ms_label_x, ms_label_y, "mistral-small-2409", fontsize=20, fontweight="bold",
            ha="left", va="center")

    for _, row in bf4.iterrows():
        if row["model"] not in BF4_DIRECT_OFFSETS:
            continue
        style = BF4_DIRECT_OFFSETS[row["model"]]
        ax.annotate(row["model"], (row["mean_gap"], row["verbalization_rate"]),
                    fontsize=20, fontweight="bold",
                    xytext=(style["dx"], style["dy"]), textcoords="offset points",
                    ha=style["ha"], va=style["va"])

    ax.set_xlabel("mean compliance gap (M − U)", fontsize=13)
    ax.set_ylabel("training-signal verbalization rate", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bottom, y_top)
    fig.tight_layout()
    savefig(fig, f"{OUT_DIR}/bf4_scatter.png")
    plt.close(fig)


def main():
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    mub = pd.read_csv(f"{BF_DIR}/mub_table.csv")
    plot_mub_dumbbell(mub)
    plot_mub_dumbbell_aggregate(mub)
    print("M/U/B figures done")

    gap_corr = pd.read_csv(f"{BF_DIR}/bf2_gap_correlation_matrix.csv", index_col=0)
    sem_sim_path = f"{BF_DIR}/bf2_semantic_similarity_embeddings.csv"
    if not os.path.exists(sem_sim_path):
        sem_sim_path = f"{BF_DIR}/bf2_semantic_similarity_matrix.csv"
    sem_sim = pd.read_csv(sem_sim_path, index_col=0)
    plot_bf2_heatmaps(gap_corr, sem_sim)

    pairs = pd.read_csv(f"{BF_DIR}/bf2_pairs.csv")
    # recompute mantel r/p quickly against whichever sem_sim is in use for the scatter annotation
    from bf2_coherence import mantel_test
    mantel_r, mantel_p, _ = mantel_test(gap_corr, sem_sim)
    plot_bf2_scatter(pairs, mantel_r, mantel_p)

    match_path = f"{BF_DIR}/bf2_cluster_match.csv"
    if os.path.exists(match_path):
        plot_bf2_cluster_contingency(pd.read_csv(match_path))
    print("BF2 figures done")

    bf3 = pd.read_csv(f"{BF_DIR}/bf3_per_model.csv")
    plot_bf3_forest(bf3)
    print("BF3 figure done")

    bf4 = pd.read_csv(f"{BF_DIR}/bf4_model_grain.csv")
    plot_bf4_scatter(bf4)
    print("BF4 figure done")

    print(f"\nAll figures -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
