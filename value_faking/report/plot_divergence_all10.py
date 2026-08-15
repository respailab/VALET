"""Layer 1 divergence visualizations for all 10 models."""

import argparse
import json
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from value_faking.paths import ANALYSIS_DIR, RUNS_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_faking.report.plot_style import apply_style, savefig, cell_text_color

# ── 10 models ordered by compliance gap descending ────────────────────────────

MODELS = [
    ("DeepSeek R1 Distill 70B",
     f"{ANALYSIS_DIR}/openrouter__deepseek_deepseek-r1-distill-llama-70b__v2_n4/analyzed_results_v2.csv"),
    ("DeepSeek R1 0528",
     f"{ANALYSIS_DIR}/openrouter__deepseek_deepseek-r1-0528__v2_n4/analyzed_results_v2.csv"),
    ("Llama 3.3 70B",
     f"{ANALYSIS_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__v2_n4/analyzed_results_v2.csv"),
    ("Llama 3.1 70B",
     f"{ANALYSIS_DIR}/openrouter__meta-llama_llama-3.1-70b-instruct__v2_n4/analyzed_results_v2.csv"),
    ("Mistral Large 2512",
     f"{ANALYSIS_DIR}/openrouter__mistralai_mistral-large-2512__v2_n4/analyzed_results_v2.csv"),
    ("Mistral Small 22B",
     f"{ANALYSIS_DIR}/hf__mistralai_Mistral-Small-Instruct-2409__v2_n4/analyzed_results_v2.csv"),
    ("Ministral 14B",
     f"{ANALYSIS_DIR}/openrouter__mistralai_ministral-14b-2512__v2_n4/analyzed_results_v2.csv"),
    ("GPT-4o-mini",
     f"{ANALYSIS_DIR}/openai__gpt-4o-mini__v2_n4/analyzed_results_v2.csv"),
    ("Llama 3.1 8B",
     f"{ANALYSIS_DIR}/hf__meta-llama_Meta-Llama-3.1-8B-Instruct__v2_n4/analyzed_results_v2.csv"),
    ("Gemma 3 27B",
     f"{ANALYSIS_DIR}/openrouter__google_gemma-3-27b-it__v2_n4/analyzed_results_v2.csv"),
]

COLOR_CONSISTENT = "#44AA99"   # teal-green
COLOR_DIVERGENT  = "#CC3311"   # red-orange


def flatten_divergence(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    col = f"{condition}_divergence"
    rows = []
    for _, row in df.iterrows():
        try:
            flags = json.loads(row[col])
        except Exception:
            flags = []
        for f in flags:
            if isinstance(f, bool):
                div = f
            elif isinstance(f, str):
                div = f.lower() == "true"
            else:
                continue
            rows.append({"value": row["value"], "divergent": div})
    return pd.DataFrame(rows)


def model_div_pcts(flat: pd.DataFrame) -> dict:
    n = len(flat)
    if n == 0:
        return {"divergent": 0.0, "consistent": 0.0}
    div_pct = flat["divergent"].sum() / n * 100
    return {"divergent": div_pct, "consistent": 100 - div_pct}


def value_div_matrix(condition: str) -> tuple[list, list, np.ndarray]:
    model_names = []
    all_values  = None
    rows_list   = []

    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path}")
            continue
        df   = pd.read_csv(path)
        flat = flatten_divergence(df, condition)

        values = sorted(flat["value"].unique())
        if all_values is None:
            all_values = values

        model_names.append(label)
        row = []
        for v in (all_values or []):
            grp = flat[flat["value"] == v]
            pct = grp["divergent"].mean() * 100 if len(grp) else np.nan
            row.append(pct)
        rows_list.append(row)

    matrix = np.array(rows_list)
    return model_names, all_values or [], matrix


# ── Figure 1: aggregate stacked bar chart ─────────────────────────────────────

def plot_aggregate_bars(out_path: str):
    apply_style()

    mon_pcts, unmon_pcts, labels = [], [], []

    for label, path in MODELS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        mon_pcts.append(model_div_pcts(flatten_divergence(df, "monitored")))
        unmon_pcts.append(model_div_pcts(flatten_divergence(df, "unmonitored")))
        labels.append(label)

    n = len(labels)
    y = np.arange(n)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.4, max(7, n * 0.6 + 1.2)),
        gridspec_kw={"wspace": 0.08},
    )

    for ax, pcts_list, title, show_labels in [
        (ax_left,  mon_pcts,   "Monitored",   True),
        (ax_right, unmon_pcts, "Unmonitored", False),
    ]:
        con = np.array([p["consistent"] for p in pcts_list])
        div = np.array([p["divergent"]  for p in pcts_list])

        ax.barh(y, con, color=COLOR_CONSISTENT, edgecolor="white",
                linewidth=0.4, height=0.72, label="Consistent")
        ax.barh(y, div, left=con, color=COLOR_DIVERGENT, edgecolor="white",
                linewidth=0.4, height=0.72, label="Divergent")

        ax.set_ylim(-0.6, n - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=16)
        ax.set_xlabel("% responses", fontsize=17)
        ax.set_title(title, fontsize=19, fontweight="bold", pad=8)
        ax.set_yticks(y)
        if show_labels:
            ax.set_yticklabels(labels, fontsize=16)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=COLOR_CONSISTENT, label="Consistent (intended = actual)"),
        mpatches.Patch(color=COLOR_DIVERGENT,  label=r"Divergent (intended $\neq$ actual)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=16, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.subplots_adjust(left=0.32, right=0.97, top=0.95, bottom=0.12, wspace=0.08)
    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"aggregate bars -> {out_path}")


# ── Figure 2: per-value per-model heatmap ─────────────────────────────────────

def plot_div_heatmap(condition: str, out_path: str):
    apply_style()

    model_names, values, matrix = value_div_matrix(condition)
    n_models = len(model_names)
    n_values = len(values)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "div_cmap", ["#FFFFFF", COLOR_DIVERGENT], N=256
    )

    cell_w       = 0.68
    cell_h       = 0.98
    left_margin  = 2.7
    right_margin = 1.4
    top_margin   = 3.3
    bot_margin   = 0.5

    fig_w = left_margin + n_values * cell_w + right_margin
    fig_h = bot_margin  + n_models * cell_h + top_margin

    # place axes manually so colorbar position is stable
    lf = left_margin / fig_w
    pw = (n_values * cell_w) / fig_w
    ph = (n_models * cell_h) / fig_h
    yb = bot_margin / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([lf, yb, pw, ph])

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   interpolation="nearest", origin="upper")

    for r in range(n_models):
        for c in range(n_values):
            v = matrix[r, c]
            if not np.isnan(v):
                tc = cell_text_color(np.array(cmap(v / 100.0)))
                ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, color=tc, fontweight="bold")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=13, fontweight="bold")
    ax.set_xticks(range(n_values))
    ax.set_xticklabels(values, rotation=90, fontsize=11, ha="right")
    ax.tick_params(axis="both", length=0)

    cax = fig.add_axes([lf + pw + 0.012, yb + ph * 0.1, 0.008, ph * 0.8])
    cb  = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=11)
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.set_label("% divergent", fontsize=12, labelpad=5)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"divergence heatmap ({condition}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=RUNS_DIR)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Figure 1 - aggregate divergence bars...")
    plot_aggregate_bars(
        os.path.join(args.outdir, "all10_divergence_bars.pdf"))

    print("Figure 2 - per-value divergence heatmap (monitored)...")
    plot_div_heatmap("monitored",
        os.path.join(args.outdir, "all10_divergence_heatmap_monitored.pdf"))

    print("Figure 2 - per-value divergence heatmap (unmonitored)...")
    plot_div_heatmap("unmonitored",
        os.path.join(args.outdir, "all10_divergence_heatmap_unmonitored.pdf"))


if __name__ == "__main__":
    main()
