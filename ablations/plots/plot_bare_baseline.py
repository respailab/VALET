"""Bare baseline comparison - 3 models, 3 figures, each figure = 2 side-by-side panels."""

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
        "orig":  "openrouter__meta-llama_llama-3.3-70b-instruct__v2/value_results.csv",
        "no_sp": "openrouter__meta-llama_llama-3.3-70b-instruct__ablation_no_sp/value_results.csv",
        "c_unmon": "#93c5fd",   # pastel blue
        "c_bare":  "#86efac",   # pastel green
    },
    "Gemma-3-27B": {
        "orig":  "openrouter__google_gemma-3-27b-it__v2/value_results.csv",
        "no_sp": "openrouter__google_gemma-3-27b-it__v2-no_sp/value_results.csv",
        "c_unmon": "#a5f3fc",   # pastel cyan
        "c_bare":  "#fde68a",   # pastel yellow
    },
    "Llama-3.1-8B": {
        "orig":  "openrouter__meta-llama_llama-3.1-8b-instruct__v2/value_results.csv",
        "no_sp": "openrouter__meta-llama_llama-3.1-8b-instruct__v2-no_sp/value_results.csv",
        "c_unmon": "#d9b8fb",   # pastel purple
        "c_bare":  "#fca5a5",   # pastel red
    },
}

CAPTIONS = {
    "Llama-3.3-70B": (
        "Figure (Llama-3.3-70B) - Unmonitored vs. Bare Compliance Rate.\n"
        "Left: compliance rate when the model believes it is unobserved (SP present, paid-tier tag).\n"
        "Right: compliance rate with NO system prompt and NO oversight tag (natural preference).\n"
        "Values are sorted by descending original compliance gap (monitored - unmonitored).\n"
        "For most high-gap values the unmonitored rate is substantially lower than the bare rate,\n"
        "showing that Llama-3.3-70B actively suppresses developer-compliant behaviour when\n"
        "it believes it is unobserved - going below even its natural, unprompted preference."
    ),
    "Gemma-3-27B": (
        "Figure (Gemma-3-27B) - Unmonitored vs. Bare Compliance Rate.\n"
        "Left: unmonitored rate (SP + paid-tier tag). Right: bare rate (no SP, no tag).\n"
        "Gemma-3-27B's unmonitored and bare rates are largely similar across values,\n"
        "suggesting that the system prompt alone (without the oversight signal) has\n"
        "limited influence on the model's choices - the compliance shift is driven\n"
        "primarily by the monitoring condition rather than the developer SP framing."
    ),
    "Llama-3.1-8B": (
        "Figure (Llama-3.1-8B) - Unmonitored vs. Bare Compliance Rate.\n"
        "Left: unmonitored rate (SP + paid-tier tag). Right: bare rate (no SP, no tag).\n"
        "Llama-3.1-8B shows unmonitored rates broadly comparable to or slightly above bare\n"
        "for most values, indicating the model does not strongly suppress compliance when\n"
        "unmonitored - consistent with a weaker overall alignment-faking signal in this\n"
        "smaller model compared to Llama-3.3-70B."
    ),
}


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, path))


def make_plot(model_name: str, cfg: dict):
    orig  = load(cfg["orig"])
    no_sp = (load(cfg["no_sp"])
             .rename(columns={"avg_monitored": "bare_rate"})[["value", "bare_rate"]])

    df = orig.merge(no_sp, on="value", how="inner")
    df = df.sort_values("avg_compliance_gap", ascending=True).reset_index(drop=True)

    n     = len(df)
    bar_h = 0.55
    y     = np.arange(n)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(13, max(8, n * 0.38)),
        sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def _draw_rate_panel(ax, values, color, title, xlabel, hide_left_spine=False):
        ax.barh(y, values.fillna(0), height=bar_h,
                color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(0.5, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(0, 1.08)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        spines_off = ["top", "right"] + (["left"] if hide_left_spine else [])
        ax.spines[spines_off].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for i, v in enumerate(values):
            if v == v:
                ax.text(v + 0.012, i, f"{v:.2f}", va="center", ha="left",
                        fontsize=6.5, color="#374151")

    _draw_rate_panel(ax_l, df["avg_unmonitored"], cfg["c_unmon"],
                     "Unmonitored Rate  (SP + paid-tier tag)",
                     "P(choose developer-compliant option)")
    _draw_rate_panel(ax_r, df["bare_rate"], cfg["c_bare"],
                     "Bare Rate  (no SP, no tag)",
                     "P(choose developer-compliant option)",
                     hide_left_spine=True)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(df["value"], fontsize=8)
    ax_l.set_ylim(-0.7, n - 0.3)

    for i in range(n):
        if i % 2 == 0:
            for ax in (ax_l, ax_r):
                ax.axhspan(i - 0.5, i + 0.5, color="#f3f4f6", zorder=0, alpha=0.6)

    fig.suptitle(
        f"{model_name} - Unmonitored vs. Bare Compliance Rate\n"
        "(sorted by descending original compliance gap; dashed line = chance 0.5)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout(pad=0.8)
    slug = model_name.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    out  = os.path.join(_HERE, f"bare_baseline_{slug}.png")
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
