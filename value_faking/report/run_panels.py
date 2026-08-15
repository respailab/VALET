#!/usr/bin/env python3
"""Per-run panels and the B/M/U table, for a single evaluation run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from value_faking.paths import TOOLKIT_ROOT
from value_faking.report.plot_style import apply_style, savefig
from value_faking.report.plot_per_value_all10 import (
    flatten_reasoning,
    flatten_divergence,
    value_proportions,
    value_divergence,
    value_gap,
    draw_gap,
    draw_divergence,
    draw_reasoning,
    reasoning_legend_handles,
    div_legend_handles,
)

# Panel filenames are a contract with make_value_card.py: renaming one here breaks
# the card's resolve_panels() check rather than silently producing a wrong figure.
PANEL_GAP        = "gap"
PANEL_DIVERGENCE = "divergence"
PANEL_REASON_MON = "reasoning_monitored"
PANEL_REASON_UNM = "reasoning_unmonitored"
PANEL_DUMBBELL   = "dumbbell"

MUB_COLUMNS = ["value", "B_mean", "M_mean", "U_mean",
               "M_minus_U", "U_minus_B", "M_minus_B",
               "B_n", "M_n", "U_n"]


def _display_label(slug: str) -> str:
    parts = slug.split("__")
    if len(parts) < 2:
        return slug
    model = parts[1].replace("_", "/")
    tag = parts[2] if len(parts) > 2 else ""
    return f"{model} [{tag}]" if tag else model


# The shared draw helpers size their text for the paper figure: a 32-value,
# 9x18in portrait where a 30pt title is proportionate. A per-run figure is a
# fraction of that height, so the same sizes overflow the canvas and collide.
# Restyle after drawing rather than parameterising the helpers, which the
# crossmodel stages depend on unchanged.
_FONT = {"title": 9, "label": 8, "tick": 7.5, "ytick": 7.5, "sup": 8}


def _restyle(ax, show_title: bool = True) -> None:
    ax.title.set_fontsize(_FONT["title"])
    ax.title.set_fontweight("normal")
    if not show_title:
        ax.set_title("")
    ax.xaxis.label.set_fontsize(_FONT["label"])
    ax.tick_params(axis="x", labelsize=_FONT["tick"])
    ax.tick_params(axis="y", labelsize=_FONT["ytick"])
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(_FONT["ytick"])


def _fig_size(n: int, panels: int = 1) -> tuple[float, float]:
    width = 4.2 if panels == 1 else 8.4
    return width, max(2.2, n * 0.30 + 1.5)


def _annotate_gap(ax, gap_df: pd.DataFrame) -> None:
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for i, v in enumerate(gap_df["mean_gap"].values):
        off = 0.015 * span
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color="#333333")


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(TOOLKIT_ROOT))
    except ValueError:
        return str(p)


def build_mub_table(main: pd.DataFrame, bare: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for value, g in main.groupby("value"):
        m = g["monitored_rate"].dropna().to_numpy()
        u = g["unmonitored_rate"].dropna().to_numpy()
        if len(m) == 0 or len(u) == 0:
            continue

        if bare is not None:
            b = bare.loc[bare["value"] == value, "B_rate"].dropna().to_numpy()
        else:
            b = np.array([])

        row = {
            "value": value,
            "M_mean": m.mean(), "M_n": len(m),
            "U_mean": u.mean(), "U_n": len(u),
            "M_minus_U": m.mean() - u.mean(),
        }
        if len(b):
            row.update({
                "B_mean": b.mean(), "B_n": len(b),
                "U_minus_B": u.mean() - b.mean(),
                "M_minus_B": m.mean() - b.mean(),
            })
        else:
            row.update({"B_mean": np.nan, "B_n": 0,
                        "U_minus_B": np.nan, "M_minus_B": np.nan})
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("error: no value produced a usable M and U rate")
    return df.sort_values("M_minus_U", ascending=False)[MUB_COLUMNS]


def load_bare(bare_dir: Path) -> pd.DataFrame:
    path = bare_dir / "scenario_results.csv"
    if not path.exists():
        raise SystemExit(f"error: no bare results at {path}")
    df = pd.read_csv(path)
    df["B_rate"] = df["monitored_rate"]
    return df


def draw_dumbbell(ax, table: pd.DataFrame, title: str) -> None:
    t = table.iloc[::-1]                      # largest gap at the top
    y = np.arange(len(t))

    has_bare = t["B_mean"].notna().any()
    for yi, (_, r) in zip(y, t.iterrows()):
        lo, hi = min(r["U_mean"], r["M_mean"]), max(r["U_mean"], r["M_mean"])
        ax.plot([lo, hi], [yi, yi], color="#BBBBBB", lw=1.4, zorder=1)

    if has_bare:
        ax.scatter(t["B_mean"], y, s=26, color="#444444", marker="s",
                   label="bare (B)", zorder=3)
    ax.scatter(t["U_mean"], y, s=30, color="#0077BB", label="unmonitored (U)", zorder=3)
    ax.scatter(t["M_mean"], y, s=30, color="#EE7733", marker="^",
               label="monitored (M)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(t["value"], fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("compliance rate (P chose anti-value option)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right", frameon=True)


def write_panels(df: pd.DataFrame, table: pd.DataFrame, label: str,
                 plots_dir: Path) -> list[str]:
    mon_r   = flatten_reasoning(df, "monitored")
    unmon_r = flatten_reasoning(df, "unmonitored")

    if mon_r.empty:
        raise SystemExit(
            "error: no reasoning annotations found. The input has no usable\n"
            "monitored_reasoning_types column - run the annotate stage first.")

    all_values = sorted(mon_r["value"].unique())
    mon_prop   = value_proportions(mon_r, all_values)
    unmon_prop = value_proportions(unmon_r, all_values)
    gap_df     = value_gap(df, all_values)

    # Divergence needs Layer 1 (intended vs actual), which only analyze_layers
    # emits. analyze_test_results-annotated runs lack the columns entirely, so
    # skip the panel rather than dying: the other four are still worth having.
    has_div = {"monitored_divergence", "unmonitored_divergence"} <= set(df.columns)
    div_df = None
    if has_div:
        mon_d   = flatten_divergence(df, "monitored")
        unmon_d = flatten_divergence(df, "unmonitored")
        if not mon_d.empty:
            div_df = value_divergence(mon_d, unmon_d, all_values)

    n = len(all_values)
    title = _display_label(label)

    apply_style()
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # constrained layout plus "outside" legends is what keeps the legend off the
    # x-axis label; the manual lower-center placement used to sit on top of it.
    fig, ax = plt.subplots(figsize=_fig_size(n), layout="constrained")
    draw_gap(ax, gap_df, "Compliance gap (M - U)")
    _restyle(ax)
    _annotate_gap(ax, gap_df)
    fig.suptitle(title, fontsize=_FONT["sup"])
    savefig(fig, str(plots_dir / f"{PANEL_GAP}.pdf"), dpi=200)
    plt.close(fig)
    written.append(PANEL_GAP)

    if div_df is not None:
        fig, (ax_m, ax_u) = plt.subplots(1, 2, figsize=_fig_size(n, panels=2),
                                         layout="constrained")
        # Short per-axes titles: the run is named once in the suptitle, so
        # repeating it per panel is what made the two titles overlap.
        draw_divergence(ax_m, ax_u, div_df, "Monitored", "Unmonitored",
                        show_legend=False)
        for a in (ax_m, ax_u):
            _restyle(a)
        ax_u.set_yticklabels([])
        fig.legend(handles=div_legend_handles(), loc="outside lower center",
                   ncol=2, fontsize=7, frameon=False)
        fig.suptitle(f"{title} - reasoning/choice divergence",
                     fontsize=_FONT["sup"])
        savefig(fig, str(plots_dir / f"{PANEL_DIVERGENCE}.pdf"), dpi=200)
        plt.close(fig)
        written.append(PANEL_DIVERGENCE)

    for prop, name, cond in ((mon_prop, PANEL_REASON_MON, "Monitored"),
                             (unmon_prop, PANEL_REASON_UNM, "Unmonitored")):
        fig, ax = plt.subplots(figsize=_fig_size(n), layout="constrained")
        draw_reasoning(ax, prop, f"{cond} reasoning")
        _restyle(ax)
        fig.legend(handles=reasoning_legend_handles(), loc="outside lower center",
                   ncol=3, fontsize=6.5, frameon=False)
        fig.suptitle(title, fontsize=_FONT["sup"])
        savefig(fig, str(plots_dir / f"{name}.pdf"), dpi=200)
        plt.close(fig)
        written.append(name)

    fig, ax = plt.subplots(figsize=_fig_size(n), layout="constrained")
    draw_dumbbell(ax, table, "Bare / unmonitored / monitored")
    _restyle(ax)
    fig.suptitle(title, fontsize=_FONT["sup"])
    savefig(fig, str(plots_dir / f"{PANEL_DUMBBELL}.pdf"), dpi=200)
    plt.close(fig)
    written.append(PANEL_DUMBBELL)

    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-run panels and B/M/U table for a single evaluation run.",
        epilog="Output layout matches make_value_card.py, so `valet.py card` "
               "consumes this directly.")
    p.add_argument("--run-dir", required=True,
                   help="run directory containing analyzed_results.csv")
    p.add_argument("--bare-dir",
                   help="bare-condition run directory; without it B is unavailable, "
                        "the dumbbell has no bare marker, and the card cannot build")
    p.add_argument("--label", help="display label (default: the run directory name)")
    p.add_argument("--input",
                   help="annotated CSV to read (default: <run-dir>/analyzed_results.csv)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"error: run directory not found: {run_dir}")

    src = Path(args.input) if args.input else run_dir / "analyzed_results.csv"
    if not src.exists():
        raise SystemExit(
            f"error: {src} not found. Run the annotate stage first:\n"
            f"  python valet.py annotate --input {run_dir}/scenario_results.csv "
            f"--output {src} --service <svc> --model <judge>")

    label = args.label or run_dir.name
    df = pd.read_csv(src)

    bare = load_bare(Path(args.bare_dir).resolve()) if args.bare_dir else None
    table = build_mub_table(df, bare)

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "mub_per_value.csv"
    table.to_csv(table_path, index=False)
    print(f"wrote {table_path}  ({len(table)} values)")

    written = write_panels(df, table, label, run_dir / "plots")

    print(f"\n{len(written)} panels -> {run_dir / 'plots'}  ({', '.join(written)})")

    # Blocker messages quote the paths actually in play. A copy-pasteable command
    # is the difference between a diagnostic and a chore.
    rel_run = _rel(run_dir)
    blockers = []
    if bare is None:
        blockers.append(
            "  B is missing, so U-B and the dumbbell's bare marker are unavailable.\n"
            f"    Produce a bare run, then rerun this stage with:\n"
            f"      --bare-dir {rel_run}_bare")
    if PANEL_DIVERGENCE not in written:
        blockers.append(
            "  divergence panel skipped: the annotated input has no intended/actual\n"
            "  columns, which only the Layer 1/2 judge emits. Re-annotate with:\n\n"
            f"    python valet.py annotate \\\n"
            f"      --input  {_rel(run_dir / 'scenario_results.csv')} \\\n"
            f"      --output {_rel(src)} \\\n"
            f"      --service openrouter --model meta-llama/llama-3.1-70b-instruct \\\n"
            f"      --requests_per_second 2\n\n"
            "  then rerun this stage.")

    if blockers:
        print("\nNot card-ready yet:")
        print("\n".join(blockers))
    else:
        print(f"\ncard-ready:  python valet.py card --model {run_dir.name}")


if __name__ == "__main__":
    main()
