"""Option swap comparison - 3 models, 3 figures, each figure = 2 side-by-side panels."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.join(_HERE, "outputs")

MODELS = {
    "Llama-3.3-70B": {
        "orig": "openrouter__meta-llama_llama-3.3-70b-instruct__v2/value_results.csv",
        "swap": "openrouter__meta-llama_llama-3.3-70b-instruct__ablation_swap_options/value_results.csv",
        "c_orig": "#93c5fd",   # pastel blue
        "c_swap": "#fbcfe8",   # pastel pink
    },
    "Gemma-3-27B": {
        "orig": "openrouter__google_gemma-3-27b-it__v2/value_results.csv",
        "swap": "openrouter__google_gemma-3-27b-it__v2_swapped/value_results.csv",
        "c_orig": "#a5f3fc",   # pastel cyan
        "c_swap": "#bbf7d0",   # pastel mint
    },
    "Llama-3.1-8B": {
        "orig": "openrouter__meta-llama_llama-3.1-8b-instruct__v2/value_results.csv",
        "swap": "openrouter__meta-llama_llama-3.1-8b-instruct__v2_swapped/value_results.csv",
        "c_orig": "#d9b8fb",   # pastel purple
        "c_swap": "#fed7aa",   # pastel peach
    },
}

CAPTIONS = {
    "Llama-3.3-70B": (
        "Figure (Llama-3.3-70B) - Original vs. Option-Swapped Compliance Gap.\n"
        "Left: original compliance gap (positive = more compliant when monitored).\n"
        "Right: effective gap after physically swapping which option is 'developer-compliant'\n"
        "       (still positive = same directional AF signal survives the swap).\n"
        "Most values show a large positive original gap (left) but a near-zero or negative\n"
        "option swapped compliance gap (right), revealing that Llama-3.3-70B's AF signal is largely\n"
        "an artefact of positional preference for option A, not semantic compliance tracking."
    ),
    "Gemma-3-27B": (
        "Figure (Gemma-3-27B) - Original vs. Option-Swapped Compliance Gap.\n"
        "Left: original compliance gap per value. Right: effective gap after option swap.\n"
        "Gemma-3-27B's already-modest original gaps (left) collapse to negative values in the\n"
        "swapped condition (right), indicating that the small AF signal this model displays is\n"
        "primarily a consequence of preferring whichever option occupies position A,\n"
        "rather than genuinely tracking the developer objective across oversight conditions."
    ),
    "Llama-3.1-8B": (
        "Figure (Llama-3.1-8B) - Original vs. Option-Swapped Compliance Gap.\n"
        "Left: original compliance gap; Right: option swapped compliance gap.\n"
        "Llama-3.1-8B shows mixed behaviour: several values yield near-zero original gaps\n"
        "that flip negative after swapping (positional bias), while a handful of values\n"
        "(e.g., Drive, Dominance, Friendliness) maintain a positive option swapped compliance gap,\n"
        "marking them as the strongest candidates for genuine alignment-faking behaviour\n"
        "that persists regardless of option ordering."
    ),
}


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, path))


def make_plot(model_name: str, cfg: dict):
    orig     = load(cfg["orig"])[["value", "avg_compliance_gap"]].rename(
                   columns={"avg_compliance_gap": "orig_gap"})
    swap_raw = load(cfg["swap"])[["value", "avg_compliance_gap"]].rename(
                   columns={"avg_compliance_gap": "raw_swap_gap"})
    swap_raw["option_swapped_compliance_gap"] = -swap_raw["raw_swap_gap"]

    df = orig.merge(swap_raw, on="value", how="inner")
    df = df.sort_values("orig_gap", ascending=True).reset_index(drop=True)

    n     = len(df)
    bar_h = 0.55
    y     = np.arange(n)

    # shared x limits for direct visual comparison
    max_abs = max(df["orig_gap"].abs().max(), df["option_swapped_compliance_gap"].abs().max(), 0.15)
    xlim    = (-max_abs * 1.18, max_abs * 1.18)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(13, max(8, n * 0.38)),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def _draw_panel(ax, values, color, title, xlabel):
        ax.barh(y, values.fillna(0), height=bar_h,
                color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#6b7280", linewidth=0.9, zorder=3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(*xlim)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, val in enumerate(values):
            if val == val and abs(val) >= 0.04:
                ax.text(val + (0.012 if val >= 0 else -0.012), i,
                        f"{val:+.2f}", va="center",
                        ha="left" if val >= 0 else "right",
                        fontsize=6.5, color="#374151")

    _draw_panel(ax_l, df["orig_gap"],  cfg["c_orig"],
                "Original Compliance Gap",
                "Gap  (monitored - unmonitored)")
    _draw_panel(ax_r, df["option_swapped_compliance_gap"],   cfg["c_swap"],
                "Option Swapped Compliance Gap",
                "Gap  (monitored - unmonitored, after swap)")

    ax_r.spines["left"].set_visible(False)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(df["value"], fontsize=8)
    ax_l.set_ylim(-0.7, n - 0.3)

    # zebra stripes
    for i in range(n):
        if i % 2 == 0:
            for ax in (ax_l, ax_r):
                ax.axhspan(i - 0.5, i + 0.5, color="#f3f4f6", zorder=0, alpha=0.6)

    fig.suptitle(
        f"{model_name} - Original vs. Option-Swapped Compliance Gap\n"
        "(positive = more compliant when monitored; right panel tests positional bias)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout(pad=0.8)
    slug = model_name.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    out  = os.path.join(_HERE, f"option_swap_{slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")
    plt.close()

    print()
    print(f"── CAPTION [{model_name}] ──────────────────────────────────────")
    print(CAPTIONS[model_name])
    print()


if __name__ == "__main__":
    for model_name, cfg in MODELS.items():
        make_plot(model_name, cfg)
