"""Likert endorsement heatmap, binary recognition per model, and binary recognition per value."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except (ImportError, OSError):
    pass

BASE = Path(__file__).parent / "outputs"
MODELS = {
    "Llama 3.3 70B": "openrouter__meta-llama_llama-3.3-70b-instruct",
    "Gemma 3 27B":   "openrouter__google_gemma-3-27b-it",
    "Llama 3.1 8B":  "openrouter__meta-llama_llama-3.1-8b-instruct",
}
MODEL_LABELS = list(MODELS.keys())
PALETTE = ["#2166ac", "#d6604d", "#4dac26"]

likert, binary = {}, {}
for label, slug in MODELS.items():
    likert[label] = pd.read_csv(
        BASE / f"{slug}__ablation_likert" / "likert_summary.csv"
    ).set_index("value")["endorsement"]
    binary[label] = pd.read_csv(
        BASE / f"{slug}__ablation_binary_label" / "binary_label_summary.csv"
    ).set_index("value")["recognition_rate"]

likert_df = pd.DataFrame(likert)
binary_df = pd.DataFrame(binary)

# sort by 70B likert desc
sort_order = likert_df["Llama 3.3 70B"].sort_values(ascending=False).index
likert_df  = likert_df.loc[sort_order]
binary_df  = binary_df.loc[sort_order]

# PLOT 1 — Likert heatmap
fig1, ax1 = plt.subplots(figsize=(6, 11))

sns.heatmap(
    likert_df,
    ax=ax1,
    cmap="YlGn",
    vmin=1, vmax=7,
    linewidths=0.4,
    linecolor="#e0e0e0",
    annot=True,
    fmt=".0f",
    annot_kws={"size": 9, "weight": "bold"},
    cbar_kws={"shrink": 0.5, "label": "Endorsement (1-7)"},
)
ax1.set_title("Likert Value Endorsement", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis="x", labelsize=10, length=0)
ax1.tick_params(axis="y", labelsize=8, length=0)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")

fig1.tight_layout()
fig1.savefig(Path(__file__).parent / "ablation_likert_heatmap.png", dpi=150, bbox_inches="tight")
fig1.savefig(Path(__file__).parent / "ablation_likert_heatmap.pdf", bbox_inches="tight")
print("Saved -> ablation_likert_heatmap.png/.pdf")

# PLOT 2 — Aggregate recognition per model
agg = (binary_df.mean() * 100).reset_index()
agg.columns = ["model", "pct"]

fig2, ax2 = plt.subplots(figsize=(6, 4))

bars = ax2.bar(
    agg["model"], agg["pct"],
    color=PALETTE, width=0.5, edgecolor="white", linewidth=0.8,
)
ax2.set_ylim(90, 101)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax2.set_ylabel("Avg. scenarios identified (%)", fontsize=10)
ax2.set_title("Scenario Identification - Aggregate", fontsize=12, fontweight="bold", pad=10)
ax2.tick_params(axis="x", labelsize=10, length=0)
ax2.tick_params(axis="y", labelsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

for bar, val in zip(bars, agg["pct"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

fig2.tight_layout()
fig2.savefig(Path(__file__).parent / "ablation_binary_aggregate.png", dpi=150, bbox_inches="tight")
fig2.savefig(Path(__file__).parent / "ablation_binary_aggregate.pdf", bbox_inches="tight")
print("Saved -> ablation_binary_aggregate.png/.pdf")

# PLOT 3 — Per-value recognition: 3 subplots side by side, one per model
n_vals = len(binary_df)
bar_h  = 0.62   # thick bars, no gap (height ≈ 1 means bars touch)
y_pos  = np.arange(n_vals)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, n_vals * 0.38 + 1.5),
                            sharey=True)
fig3.subplots_adjust(wspace=0.06)

for ax, label, col in zip(axes3, MODEL_LABELS, PALETTE):
    vals = binary_df[label].values * 100
    ax.barh(y_pos, vals, height=bar_h, color=col, edgecolor="none")
    ax.axvline(100, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel("% identified", fontsize=9)
    ax.set_title(label, fontsize=11, fontweight="bold", pad=8, color=col)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)

    # label any bar < 100%
    for i, v in enumerate(vals):
        if v < 100:
            ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=7,
                    color="darkred", fontweight="bold")

# y-axis labels only on leftmost subplot
axes3[0].set_yticks(y_pos)
axes3[0].set_yticklabels(binary_df.index, fontsize=8)

fig3.suptitle("Scenario Identification Rate - Per Value",
              fontsize=13, fontweight="bold", y=1.01)
fig3.tight_layout()
fig3.savefig(Path(__file__).parent / "ablation_binary_per_value.png", dpi=150, bbox_inches="tight")
fig3.savefig(Path(__file__).parent / "ablation_binary_per_value.pdf", bbox_inches="tight")
print("Saved -> ablation_binary_per_value.png/.pdf")

plt.show()
