"""Reasoning-type figures for all 10 models (analyzed_results_v2.csv)."""

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
from value_faking.paths import ANALYSIS_DIR, RUNS_DIR, RESULTS_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_faking.report.plot_style import apply_style, savefig

# ── 10 models ordered by mean compliance gap descending ───────────────────────

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


CATS = [
    ("training_signal",      "Training Signal",  "#CC3311"),
    ("value_conflict_aware", "Value Conflict",   "#009988"),
    ("directive_following",  "Directive",        "#0077BB"),
    ("rationalisation",      "Rationalisation",  "#AA3377"),
    ("no_reasoning",         "No Reasoning",     "#DDCC44"),
    ("other",                "Other",            "#BBBBBB"),
]
KNOWN = {k for k, _, _ in CATS if k != "other"}


def _normalise(t: str) -> str:
    return t if t in KNOWN else "other"


def flatten_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    col = f"{condition}_reasoning_types"
    rows = []
    for _, row in df.iterrows():
        try:
            types = json.loads(row[col])
        except Exception:
            types = []
        for t in types:
            rows.append({"value": row["value"], "reasoning_type": _normalise(str(t))})
    return pd.DataFrame(rows)


def model_overall_pcts(flat: pd.DataFrame) -> dict:
    n = len(flat)
    if n == 0:
        return {k: 0.0 for k, _, _ in CATS}
    return {k: (flat["reasoning_type"] == k).sum() / n * 100 for k, _, _ in CATS}


# ── Figure A: cross-model stacked bars ───────────────────────────────────────

def plot_stacked_bars(out_path: str):
    apply_style()

    # load overall pcts for both conditions
    mon_pcts, unmon_pcts = [], []
    labels = []

    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"[SKIP] {label}")
            continue
        df = pd.read_csv(path)
        mon_pcts.append(model_overall_pcts(flatten_condition(df, "monitored")))
        unmon_pcts.append(model_overall_pcts(flatten_condition(df, "unmonitored")))
        labels.append(label)

    n = len(labels)
    x = np.arange(n)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(11, n * 1.25 + 2), 9.2),
        gridspec_kw={"hspace": 0.55},
    )

    for ax, pcts_list, title in [
        (ax_top, mon_pcts,   "Monitored"),
        (ax_bot, unmon_pcts, "Unmonitored"),
    ]:
        bottoms = np.zeros(n)
        for key, _, color in CATS:
            heights = np.array([p[key] for p in pcts_list])
            ax.bar(x, heights, bottom=bottoms, color=color,
                   edgecolor="white", linewidth=0.4, width=0.72)
            bottoms += heights

        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=13)
        ax.set_ylabel("% responses", fontsize=13.5)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=13.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    handles = [mpatches.Patch(color=c, label=lbl) for _, lbl, c in CATS]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, out_path, dpi=300)
    plt.close()
    print(f"stacked bars -> {out_path}")


# ── Figure B: per-model per-value stacked bars (one file per model) ──────────

def _value_proportions(flat: pd.DataFrame, all_values: list) -> pd.DataFrame:
    rows = []
    for v in all_values:
        grp = flat[flat["value"] == v]
        n   = len(grp)
        row = {"value": v}
        for key, _, _ in CATS:
            row[key] = (grp["reasoning_type"] == key).sum() / n * 100 if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_value_bars(ax, df: pd.DataFrame, title: str, show_xlabel: bool):
    n        = len(df)
    x        = np.arange(n)
    bottoms  = np.zeros(n)
    for key, _, color in CATS:
        heights = df[key].values
        ax.bar(x, heights, bottom=bottoms, color=color,
               edgecolor="white", linewidth=0.3, width=0.85)
        bottoms += heights
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=12)
    ax.set_ylabel("% responses", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels(df["value"].tolist(), rotation=90, fontsize=11, ha="right")
    else:
        ax.set_xticklabels([])


def plot_per_model_bars(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    apply_style()

    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path}")
            continue

        df         = pd.read_csv(path)
        mon_flat   = flatten_condition(df, "monitored")
        unmon_flat = flatten_condition(df, "unmonitored")

        all_values = sorted(mon_flat["value"].unique())
        mon_df     = _value_proportions(mon_flat,   all_values)
        unmon_df   = _value_proportions(unmon_flat, all_values)

        n_values = len(all_values)
        fig_w    = max(17, n_values * 0.58 + 2.5)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(fig_w, 11),
            gridspec_kw={"hspace": 0.45},
        )

        _draw_value_bars(ax_top, mon_df,   f"{label} - Monitored",   show_xlabel=False)
        _draw_value_bars(ax_bot, unmon_df, f"{label} - Unmonitored", show_xlabel=True)

        handles = [mpatches.Patch(color=c, label=lbl) for _, lbl, c in CATS]
        fig.legend(handles=handles, loc="lower center", ncol=6,
                   fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

        slug     = label.lower().replace(" ", "_").replace(".", "")
        out_path = os.path.join(outdir, f"reasoning_bars_{slug}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=RUNS_DIR,
                        help="Output directory for cross-model figures")
    parser.add_argument("--per_model_dir", default=f"{RESULTS_DIR}/plot_results",
                        help="Output directory for per-model per-value figures")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Figure A - cross-model stacked bars...")
    plot_stacked_bars(os.path.join(args.outdir, "all10_reasoning_bars.pdf"))

    print("Figure B - per-model per-value stacked bars...")
    plot_per_model_bars(args.per_model_dir)


if __name__ == "__main__":
    main()
