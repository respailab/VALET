"""plot_bare_baseline_n4.py - plot the "bare" condition compliance rate per value from the *_n4_bare ablation (no oversight/monitored-unmonitored"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from value_faking.paths import RESULTS_DIR

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except (ImportError, OSError):
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
RESULTS_DIR = os.path.join(_HERE, "ablation_results", "bare")

MODEL_TAG = "llama-3.3-70b"
BARE_FILE = os.path.join(OUTPUTS_DIR, "openrouter__meta-llama_llama-3.3-70b-instruct__v2_n4_bare", "value_results.csv")


def plot_bare(df: pd.DataFrame, model_label: str, out_path_png: str, alphabetical: bool = False):
    if alphabetical:
        df = df.sort_values("value", ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values("bare_rate", ascending=True).reset_index(drop=True)
    n_vals = len(df)
    y_pos = np.arange(n_vals)
    vals = df["bare_rate"].values * 100

    fig, ax = plt.subplots(figsize=(7, n_vals * 0.32 + 1.5))
    colors = ["#d6604d" if v < 50 else "#2166ac" for v in vals]
    ax.barh(y_pos, vals, height=0.62, color=colors, edgecolor="none")
    ax.axvline(100, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(50, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel("% compliant (bare condition, no oversight framing)", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["value"], fontsize=8)
    order_desc = "alphabetical" if alphabetical else "by bare rate"
    ax.set_title(f"Bare Baseline Compliance (n4) - {model_label}\n(sorted {order_desc}; dotted line = 50%)",
                 fontsize=11, fontweight="bold", pad=10)
    if alphabetical:
        ax.invert_yaxis()  # A→Z top-to-bottom
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)

    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=7,
                color="darkred" if v < 50 else "#111827", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_path_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"saved plot -> {out_path_png}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", default=MODEL_TAG)
    parser.add_argument("--bare_file", default=BARE_FILE)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.bare_file)[["value", "avg_monitored"]].rename(columns={"avg_monitored": "bare_rate"})
    df = df.sort_values("value").reset_index(drop=True)

    out_csv = os.path.join(args.output_dir, f"bare_baseline_n4_{args.model_tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"saved table -> {out_csv}")

    out_png = os.path.join(args.output_dir, f"bare_baseline_n4_{args.model_tag}.png")
    plot_bare(df, args.model_tag, out_png)

    out_png_alpha = os.path.join(args.output_dir, f"bare_baseline_n4_{args.model_tag}_alphabetical.png")
    plot_bare(df, args.model_tag, out_png_alpha, alphabetical=True)
