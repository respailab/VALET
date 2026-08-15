"""Per-value, per-model combined figure for all 10 models."""

import argparse
import json
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from value_faking.paths import ANALYSIS_DIR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_faking.report.plot_style import apply_style, savefig, TOL_DIVERGING_STRONG, gap_norm
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


def _norm(t: str) -> str:
    return t if t in KNOWN else "other"


def flatten_reasoning(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    col = f"{condition}_reasoning_types"
    rows = []
    for _, row in df.iterrows():
        try:
            types = json.loads(row[col])
        except Exception:
            types = []
        for t in types:
            rows.append({"value": row["value"], "reasoning_type": _norm(str(t))})
    return pd.DataFrame(rows)


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


def value_proportions(flat: pd.DataFrame, all_values: list) -> pd.DataFrame:
    rows = []
    for v in all_values:
        grp = flat[flat["value"] == v]
        n   = len(grp)
        row = {"value": v}
        for key, _, _ in CATS:
            row[key] = (grp["reasoning_type"] == key).sum() / n * 100 if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def value_divergence(mon_flat: pd.DataFrame, unmon_flat: pd.DataFrame,
                     all_values: list) -> pd.DataFrame:
    rows = []
    for v in all_values:
        mg = mon_flat[mon_flat["value"] == v]
        ug = unmon_flat[unmon_flat["value"] == v]
        mon_div   = mg["divergent"].mean() * 100 if len(mg) else 0.0
        unmon_div = ug["divergent"].mean() * 100 if len(ug) else 0.0
        rows.append({
            "value":             v,
            "mon_divergent":     mon_div,
            "mon_consistent":    100 - mon_div,
            "unmon_divergent":   unmon_div,
            "unmon_consistent":  100 - unmon_div,
        })
    return pd.DataFrame(rows)


def value_gap(df: pd.DataFrame, all_values: list) -> pd.DataFrame:
    mean_gap = df.dropna(subset=["compliance_gap"]).groupby("value")["compliance_gap"].mean()
    return pd.DataFrame({"value": all_values,
                          "mean_gap": [mean_gap.get(v, 0.0) for v in all_values]})


# ── draw helpers (horizontal barh: Y=values, X=score) ────────────────────────

COLOR_DIVERGENT  = "#CC3311"
COLOR_CONSISTENT = "#44AA99"


def draw_gap(ax, gap_df: pd.DataFrame, title: str, show_yticklabels: bool = True):
    y    = np.arange(len(gap_df))
    vals = gap_df["mean_gap"].values
    vmin = min(-0.15, float(np.nanmin(vals)) - 0.05)
    vmax = max(0.15,  float(np.nanmax(vals)) + 0.05)
    norm = gap_norm(vmin=vmin, vmax=vmax)
    colors = [TOL_DIVERGING_STRONG(norm(v)) for v in vals]
    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.3, height=0.5)
    ax.axvline(0, color="#444444", linewidth=0.6, linestyle="--")
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_xlim(vmin, vmax)
    ax.set_xlabel("Mean compliance gap", fontsize=26)
    ax.set_title(title, fontsize=30, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=22)
    ax.set_yticks(y)
    ax.set_yticklabels(gap_df["value"].tolist() if show_yticklabels else [], fontsize=24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def _draw_div_stacked(ax, div_pct: np.ndarray, title: str, value_labels: list,
                       show_legend: bool = True, show_yticklabels: bool = True):
    y   = np.arange(len(div_pct))
    con = 100 - div_pct
    ax.barh(y, con, color=COLOR_CONSISTENT, edgecolor="white", linewidth=0.3,
            height=0.55, label="Consistent")
    ax.barh(y, div_pct, left=con, color=COLOR_DIVERGENT, edgecolor="white",
            linewidth=0.3, height=0.55, label="Divergent")
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=22)
    ax.set_xlabel("% responses", fontsize=26)
    ax.set_title(title, fontsize=30, fontweight="bold", pad=12)
    if show_legend:
        ax.legend(fontsize=22, framealpha=0.8, loc="lower right")
    ax.set_yticks(np.arange(len(value_labels)))
    ax.set_yticklabels(value_labels if show_yticklabels else [], fontsize=24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def draw_divergence(ax_mon, ax_unmon, div_df: pd.DataFrame, mon_title: str, unmon_title: str,
                     show_legend: bool = True, show_yticklabels: bool = True):
    vals = div_df["value"].tolist()
    _draw_div_stacked(ax_mon,   div_df["mon_divergent"].values,
                      mon_title, vals, show_legend=show_legend, show_yticklabels=show_yticklabels)
    _draw_div_stacked(ax_unmon, div_df["unmon_divergent"].values,
                      unmon_title, vals, show_legend=show_legend, show_yticklabels=show_yticklabels)


def draw_reasoning(ax, prop_df: pd.DataFrame, title: str, show_yticklabels: bool = True):
    y       = np.arange(len(prop_df))
    lefts   = np.zeros(len(prop_df))
    for key, _, color in CATS:
        widths = prop_df[key].values
        ax.barh(y, widths, left=lefts, color=color,
                edgecolor="white", linewidth=0.3, height=0.55)
        lefts += widths
    ax.set_ylim(-0.6, len(y) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=22)
    ax.set_xlabel("% responses", fontsize=26)
    ax.set_title(title, fontsize=30, fontweight="bold", pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels(prop_df["value"].tolist() if show_yticklabels else [], fontsize=24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def reasoning_legend_handles():
    return [mpatches.Patch(color=c, label=lbl) for _, lbl, c in CATS]


def div_legend_handles():
    return [
        mpatches.Patch(color=COLOR_CONSISTENT, label="Consistent (intended = actual)"),
        mpatches.Patch(color=COLOR_DIVERGENT,  label=r"Divergent (intended $\neq$ actual)"),
    ]


def plot_model(label: str, path: str):
    df = pd.read_csv(path)

    mon_r   = flatten_reasoning(df, "monitored")
    unmon_r = flatten_reasoning(df, "unmonitored")
    mon_d   = flatten_divergence(df, "monitored")
    unmon_d = flatten_divergence(df, "unmonitored")

    all_values = sorted(mon_r["value"].unique())

    mon_prop   = value_proportions(mon_r,   all_values)
    unmon_prop = value_proportions(unmon_r, all_values)
    div_df     = value_divergence(mon_d, unmon_d, all_values)
    gap_df_    = value_gap(df, all_values)

    n      = len(all_values)
    fig_h  = max(13, n * 0.5 + 2.5)   # tall portrait: rows = values

    model_dir = os.path.dirname(path)
    outdir    = os.path.join(model_dir, "plots")
    os.makedirs(outdir, exist_ok=True)

    apply_style()

    fig, ax = plt.subplots(figsize=(9, fig_h))
    draw_gap(ax, gap_df_, f"{label} - Compliance gap")
    savefig(fig, os.path.join(outdir, "gap.pdf"), dpi=200)
    plt.close()

    # ── 2. Divergence (standalone, 2 panels side-by-side) ────────────────────
    fig, (ax_mon, ax_unmon) = plt.subplots(1, 2, figsize=(18, fig_h),
                                            gridspec_kw={"wspace": 0.55})
    draw_divergence(ax_mon, ax_unmon, div_df,
                     f"{label} - Monitored divergence", f"{label} - Unmonitored divergence",
                     show_legend=False)
    # hide y labels on right panel (shared axis)
    ax_unmon.set_yticklabels([])
    handles = div_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=24, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, os.path.join(outdir, "divergence.pdf"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, fig_h))
    draw_reasoning(ax, mon_prop, f"{label} - Monitored reasoning")
    handles = reasoning_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=24, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    savefig(fig, os.path.join(outdir, "reasoning_monitored.pdf"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, fig_h))
    draw_reasoning(ax, unmon_prop, f"{label} - Unmonitored reasoning")
    handles = reasoning_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=24, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    savefig(fig, os.path.join(outdir, "reasoning_unmonitored.pdf"), dpi=200)
    plt.close()

    # ── 5. Combined (5 panels: gap | div-mon | div-unmon | reason-mon | reason-unmon)
    fig, axes = plt.subplots(
        1, 5, figsize=(45, fig_h),
        gridspec_kw={"wspace": 0.6},
    )
    draw_gap(axes[0], gap_df_, "Compliance gap")
    draw_divergence(axes[1], axes[2], div_df,
                     "Monitored divergence", "Unmonitored divergence",
                     show_legend=False, show_yticklabels=False)
    draw_reasoning(axes[3], mon_prop,   "Monitored reasoning",   show_yticklabels=False)
    draw_reasoning(axes[4], unmon_prop, "Unmonitored reasoning", show_yticklabels=False)

    r_handles = reasoning_legend_handles()
    d_handles = div_legend_handles()
    fig.legend(handles=r_handles + d_handles, loc="lower center", ncol=4,
               fontsize=24, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    savefig(fig, os.path.join(outdir, "combined.pdf"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Run for a single model slug (optional)")
    args = parser.parse_args()

    apply_style()
    for label, path in MODELS:
        if args.model and args.model.lower() not in label.lower():
            continue
        if not os.path.exists(path):
            print(f"[SKIP] {label}: {path}")
            continue
        print(f"  {label} ...", end=" ", flush=True)
        plot_model(label, path)
        print("done")


if __name__ == "__main__":
    main()
