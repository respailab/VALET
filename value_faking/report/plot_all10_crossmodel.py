"""Three figures for all 10 models (n=4 runs), run from project root."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
from value_faking.paths import RESULTS_DIR, RUNS_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_faking.report.plot_style import apply_style, TOL_DIVERGING, gap_norm, cell_text_color, savefig

# ── 10 models ordered by mean gap descending ─────────────────────────────────

MODELS = [
    ("DeepSeek R1 Distill 70B",
     f"{RUNS_DIR}/openrouter__deepseek_deepseek-r1-distill-llama-70b__v2_n4/scenario_results.csv"),
    ("DeepSeek R1 0528",
     f"{RUNS_DIR}/openrouter__deepseek_deepseek-r1-0528__v2_n4/scenario_results.csv"),
    ("Llama 3.3 70B",
     f"{RUNS_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__v2_n4/scenario_results.csv"),
    ("Llama 3.1 70B",
     f"{RUNS_DIR}/openrouter__meta-llama_llama-3.1-70b-instruct__v2_n4/scenario_results.csv"),
    ("Mistral Large 2512",
     f"{RUNS_DIR}/openrouter__mistralai_mistral-large-2512__v2_n4/scenario_results.csv"),
    ("Mistral Small 22B",
     f"{RUNS_DIR}/hf__mistralai_Mistral-Small-Instruct-2409__v2_n4/scenario_results.csv"),
    ("Ministral 14B",
     f"{RUNS_DIR}/openrouter__mistralai_ministral-14b-2512__v2_n4/scenario_results.csv"),
    ("GPT-4o-mini",
     f"{RUNS_DIR}/openai__gpt-4o-mini__v2_n4/scenario_results.csv"),
    ("Llama 3.1 8B",
     f"{RUNS_DIR}/hf__meta-llama_Meta-Llama-3.1-8B-Instruct__v2_n4/scenario_results.csv"),
    ("Gemma 3 27B",
     f"{RUNS_DIR}/openrouter__google_gemma-3-27b-it__v2_n4/scenario_results.csv"),
]


def load_gap_matrix() -> pd.DataFrame:
    frames = {}
    for label, path in MODELS:
        df = pd.read_csv(path)
        frames[label] = df.groupby("value")["compliance_gap"].mean()
    matrix = pd.DataFrame(frames).T
    return matrix.reindex(sorted(matrix.columns), axis=1)


# cell_ci.csv uses the paper's full model names; the figure uses short ones
BOOTSTRAP_CSV = f"{RESULTS_DIR}/bootstrap/cell_ci.csv"
BOOTSTRAP_NAMES = {
    "DeepSeek R1 Distill 70B": "DeepSeek R1 Distill Llama 70B",
    "DeepSeek R1 0528":        "DeepSeek R1 0528",
    "Llama 3.3 70B":           "Llama 3.3 70B Instruct",
    "Llama 3.1 70B":           "Llama 3.1 70B Instruct",
    "Mistral Large 2512":      "Mistral Large 2512",
    "Mistral Small 22B":       "Mistral Small Instruct 2409",
    "Ministral 14B":           "Ministral 14B 2512",
    "GPT-4o-mini":             "GPT-4o-mini",
    "Llama 3.1 8B":            "Llama 3.1 8B Instruct",
    "Gemma 3 27B":             "Gemma 3 27B",
}


def load_significance(matrix: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(BOOTSTRAP_CSV):
        print(f"[WARN] {BOOTSTRAP_CSV} not found - no cells will be stippled. "
              f"Run: python value_faking_test/bootstrap_ci.py")
        return pd.DataFrame(True, index=matrix.index, columns=matrix.columns)

    cells = pd.read_csv(BOOTSTRAP_CSV)
    sig = cells.pivot(index="model", columns="value", values="gap_sig")
    sig = sig.rename(index={v: k for k, v in BOOTSTRAP_NAMES.items()})
    return sig.reindex(index=matrix.index, columns=matrix.columns).fillna(True).astype(bool)


def plot_heatmap(matrix: pd.DataFrame, out_path: str, sig: pd.DataFrame = None):
    apply_style()
    if sig is None:
        sig = load_significance(matrix)
    plt.rcParams["hatch.linewidth"] = 0.5

    n_models = len(matrix)
    n_values = len(matrix.columns)

    vmin = min(-0.25, float(matrix.min().min()))
    vmax = max(0.90,  float(matrix.max().max()))
    norm = gap_norm(vmin, vmax)
    cmap = TOL_DIVERGING

    cell_w, cell_h  = 0.95, 1.15
    left_margin  = 3.3
    right_margin = 2.3
    top_margin   = 3.6
    bot_margin   = 0.5

    fig_w = left_margin + n_values * cell_w + right_margin
    fig_h = bot_margin  + n_models * cell_h + top_margin
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    values = list(matrix.columns)
    models = list(matrix.index)

    for row_i, model in enumerate(models):
        for col_j, value in enumerate(values):
            gap = matrix.loc[model, value]
            significant = bool(sig.loc[model, value])
            if pd.isna(gap):
                bg    = np.array([0.93, 0.93, 0.93, 1.0])
                label = "N/A"
            else:
                bg    = np.array(cmap(norm(gap)))
                label = f"{gap:+.2f}"
                if not significant:
                    # wash toward grey so non-significant cells recede
                    bg = 0.45 * bg + 0.55 * np.array([0.93, 0.93, 0.93, 1.0])
            ax.add_patch(plt.Rectangle(
                (col_j, n_models - 1 - row_i), 1, 1,
                facecolor=bg, edgecolor="white", linewidth=0.6,
            ))
            if not significant and not pd.isna(gap):
                ax.add_patch(plt.Rectangle(
                    (col_j, n_models - 1 - row_i), 1, 1,
                    facecolor="none", edgecolor="#8A8A8A",
                    linewidth=0.0, hatch="....",
                ))
            ax.text(
                col_j + 0.5, n_models - 1 - row_i + 0.5, label,
                ha="center", va="center",
                fontsize=30,
                fontweight="bold" if significant else "normal",
                color=cell_text_color(bg),
            )

    ax.set_xlim(0, n_values)
    ax.set_ylim(0, n_models)
    ax.set_aspect("equal")
    ax.axis("off")

    for col_j, value in enumerate(values):
        ax.text(col_j + 0.5, n_models + 0.12, value,
                ha="left", va="bottom", rotation=90, fontsize=30, color="#1A252F")

    for row_i, model in enumerate(models):
        mean_gap = matrix.loc[model].mean()
        ax.text(-0.18, n_models - 1 - row_i + 0.62, model,
                ha="right", va="center", fontsize=30, fontweight="bold", color="#1A252F")
        ax.text(-0.18, n_models - 1 - row_i + 0.22,
                f"mean {mean_gap:+.3f}",
                ha="right", va="center", fontsize=30, color="#666")

    ax_pos   = ax.get_position()
    col_frac = ax_pos.width / n_values
    cax = fig.add_axes([
        ax_pos.x0 + n_values * col_frac + 0.012,
        ax_pos.y0 + ax_pos.height * 0.12,
        col_frac * 0.85,
        ax_pos.height * 0.76,
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label("Compliance gap", fontsize=30, labelpad=10)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))

    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"heatmap -> {out_path}")


# ── Figure 2: Spearman correlation matrix ────────────────────────────────────

def plot_spearman(matrix: pd.DataFrame, out_path: str):
    apply_style()

    labels = list(matrix.index)
    n = len(labels)

    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi = matrix.iloc[i].values
            xj = matrix.iloc[j].values
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 3:
                corr[i, j] = np.nan
            else:
                corr[i, j] = stats.spearmanr(xi[mask], xj[mask]).statistic

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "tol_corr", ["#003F7F", "#FFFFFF", "#993300"], N=256
    )
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    cell_sz   = 3.7
    left_mar  = 18.0
    right_mar = 4.4
    top_mar   = 19.0
    bot_mar   = 0.5

    fig_w = left_mar  + n * cell_sz + right_mar
    fig_h = bot_mar   + n * cell_sz + top_mar
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for row_i in range(n):
        for col_j in range(n):
            val = corr[row_i, col_j]
            bg  = np.array(cmap(norm(val)))
            ax.add_patch(plt.Rectangle(
                (col_j, n - 1 - row_i), 1, 1,
                facecolor=bg, edgecolor="white", linewidth=0.5,
            ))
            ax.text(col_j + 0.5, n - 1 - row_i + 0.5,
                    f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=44, fontweight="bold",
                    color=cell_text_color(bg))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")

    # column headers (rotated)
    for col_j, lab in enumerate(labels):
        ax.text(col_j + 0.5, n + 0.3, lab,
                ha="left", va="bottom", rotation=45,
                fontsize=100, color="#1A252F")

    # row labels
    for row_i, lab in enumerate(labels):
        ax.text(-0.3, n - 1 - row_i + 0.5, lab,
                ha="right", va="center",
                fontsize=100, fontweight="bold", color="#1A252F")

    ax_pos   = ax.get_position()
    col_frac = ax_pos.width / n
    cax = fig.add_axes([
        ax_pos.x0 + n * col_frac + 0.015,
        ax_pos.y0 + ax_pos.height * 0.1,
        col_frac * 0.65,
        ax_pos.height * 0.8,
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=38)
    cbar.set_label(r"Spearman $\rho$", fontsize=42, labelpad=14)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"Spearman -> {out_path}")


# ── Figure 3: small-multiples compliance gap per value per model ──────────────

# Distinct colors per model (colorblind-tolerant set)
MODEL_COLORS = [
    "#CC3311",   # DeepSeek R1 Distill 70B  — dark red
    "#EE7733",   # DeepSeek R1 0528          — orange
    "#E87D4D",   # Llama 3.3 70B             — salmon
    "#AA3377",   # Llama 3.1 70B             — magenta
    "#0077BB",   # Mistral Large 2512        — blue
    "#33BBEE",   # Mistral Small 22B         — cyan
    "#009988",   # Ministral 14B             — teal
    "#7B4F9E",   # GPT-4o-mini               — purple
    "#44AA99",   # Llama 3.1 8B              — green
    "#BBBBBB",   # Gemma 3 27B               — grey
]

def plot_small_multiples(matrix: pd.DataFrame, out_path: str):
    apply_style()

    models = list(matrix.index)
    values = list(matrix.columns)   # already sorted alphabetically
    n_models = len(models)
    n_values = len(values)
    x = np.arange(n_values)

    panel_h  = 1.6
    left_mar = 3.0   # room for model name
    fig_w    = 10.2
    fig_h    = panel_h * n_models + 0.7   # bottom margin for x label

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(fig_w, fig_h),
        sharex=True,
    )

    panel_labels = "abcdefghij"

    for idx, (ax, model, color) in enumerate(zip(axes, models, MODEL_COLORS)):
        gaps = matrix.loc[model, values].values.astype(float)

        pos_gaps = np.where(gaps >= 0, gaps, 0.0)
        neg_gaps = np.where(gaps <  0, gaps, 0.0)

        ax.bar(x, pos_gaps, color=color,  width=0.7, edgecolor="none")
        ax.bar(x, neg_gaps, color=color,  width=0.7, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.8)

        # panel letter inside top-left
        ax.text(0.01, 0.92, f"({panel_labels[idx]})",
                transform=ax.transAxes,
                fontsize=12, va="top", ha="left", color="#1A252F")

        # free y scale — just show a couple ticks
        yabs = np.nanmax(np.abs(gaps)) if not np.all(np.isnan(gaps)) else 0.5
        ylim = max(yabs * 1.25, 0.15)
        ax.set_ylim(-ylim, ylim)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, symmetric=True))
        ax.tick_params(axis="y", labelsize=10.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False)

        # model name to the left of the panel — pushed clear of the axis line
        ax.set_ylabel(model, fontsize=16, rotation=0, labelpad=14,
                      ha="right", va="center", fontweight="bold")
        ax.yaxis.set_label_coords(-0.10, 0.5)

    # x-axis ticks on bottom panel only — no labels (unlabelled)
    axes[-1].tick_params(axis="x", bottom=True, labelbottom=False)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel("Value dimensions (alphabetical)", fontsize=14.5)

    fig.subplots_adjust(left=0.30, right=0.97, top=0.98, bottom=0.06, hspace=0.28)
    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"small multiples -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=RUNS_DIR,
                        help="Output directory for all figures")
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("Loading compliance gap matrix for 10 models...")
    matrix = load_gap_matrix()
    print(f"  shape: {matrix.shape}  |  gap range: {matrix.min().min():+.3f} to {matrix.max().max():+.3f}")

    plot_heatmap(matrix,  os.path.join(outdir, "all10_heatmap.pdf"))
    plot_spearman(matrix, os.path.join(outdir, "all10_spearman.pdf"))
    plot_small_multiples(matrix, os.path.join(outdir, "all10_small_multiples.pdf"))


if __name__ == "__main__":
    main()
