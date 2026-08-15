#!/usr/bin/env python3
"""Toolkit entry point. No installation required."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# stage -> (module, one-line description). Order is pipeline order, which is also
# the order this file prints, so the listing doubles as the walkthrough.
STAGES: dict[str, tuple[str, str]] = {
    "run":       ("value_faking.run.run_test",
                  "monitored + unmonitored inference (M, U)"),
    "run-bare":  ("value_faking.run.run_bare",
                  "bare inference with --no_system_prompt (B), and the option-swap control"),
    # analyze_layers is the annotation the paper's figures are built on: Layer 1
    # emits intended/actual/divergence (the S5.2 validity check), Layer 2 the
    # six-category taxonomy. analyze_test_results predates it and has no
    # divergence columns, so it cannot feed the divergence panel or the card.
    "annotate":  ("value_faking.annotate.analyze_layers",
                  "judge pass: intended/actual, divergence, six-category reasoning"),
    "annotate-legacy": ("value_faking.annotate.analyze_test_results",
                  "older judge pass, no divergence; kept for reproducing old runs"),
    "card":      ("value_faking.report.make_value_card",
                  "build the per-model Value Card from a completed run"),
}

GROUPS: dict[str, dict[str, tuple[str, str]]] = {
    "build": {
        "topics": ("value_faking.build.topic_gen.run_all", "per-value topic generation"),
        "seeds":  ("value_faking.build.seed_scenario_gen.run_all", "seed dilemmas per topic"),
        "expand": ("value_faking.build.scenario_expansion.run_all", "expand and validate"),
    },
    "stats": {
        "bootstrap":     ("value_faking.stats.bootstrap_ci", "cluster bootstrap, B/M/U and differences"),
        "heterogeneity": ("value_faking.stats.heterogeneity", "variance decomposition of the gap matrix"),
        "sensitivity":   ("value_faking.stats.compute_sensitivity", "M-U / U-B decomposition"),
        "bare":          ("value_faking.stats.compute_bare_rate", "bare-condition compliance rates"),
        "aggregate":     ("value_faking.stats.compute_aggregate_stats", "per-model aggregate table"),
        "reasoning":     ("value_faking.stats.compute_reasoning_stats", "reasoning-category breakdown"),
        "crossmodel":    ("value_faking.stats.compute_cross_model", "cross-model comparison tables"),
        "mub":           ("value_faking.bf.mub_table", "per-value B/M/U table the Value Card reads"),
    },
    # Per-run: operates on one run directory, so it works on a single-value pilot.
    "figures": {
        "panels": ("value_faking.report.run_panels",
                   "gap, divergence, reasoning, dumbbell + B/M/U table for one run"),
    },
    # Cross-model: consumes all ten runs at once. These are the paper's figures;
    # none of them can run on a single evaluation.
    "crossmodel": {
        "pervalue":   ("value_faking.report.plot_per_value_all10",
                       "per-model per-value panels -> combined.png (x9 in paper)"),
        "heatmap":    ("value_faking.report.plot_all10_crossmodel",
                       "all10_heatmap, all10_spearman, all10_small_multiples"),
        "reasoning":  ("value_faking.report.plot_reasoning_all10", "all10_reasoning_bars"),
        "divergence": ("value_faking.report.plot_divergence_all10", "all10_divergence_bars"),
        "bf":         ("value_faking.bf.plot_bf_results",
                       "mub_dumbbell_<slug> (x11), bf3_forest, bf4_scatter"),
    },
    "ablate": {
        "prereg":      ("ablations.analysis.bf5_prereg_check", "arm parity checks, run before inference"),
        "consequence": ("ablations.run.run_consequence_ablation_SP_test", "F.6 consequence arms"),
        "tone":        ("ablations.run.run_tone_ablation_SP_test", "F.6 register arms"),
        "swap":        ("ablations.run.run_option_pro_value", "F.3 option-order control"),
        "recognition": ("ablations.run.run_binary_label", "F.4 scenario recognition check"),
        "likert":      ("ablations.run.run_likert", "F.4 value-endorsement probe"),
        "swapstats":   ("ablations.analysis.compute_swap_ablation", "F.3 REAL/POS_BIAS verdicts"),
    },
}


def usage() -> str:
    out = ["VALET — Value ALignment-faking Evaluation Tool", "",
           "usage: python valet.py <stage> [options]", "", "stages:"]
    for name, (_, desc) in STAGES.items():
        out.append(f"  {name:<22}{desc}")
    for group, subs in GROUPS.items():
        out.append(f"\n{group}:")
        for name, (_, desc) in subs.items():
            out.append(f"  {group} {name:<16}{desc}")
    out.append("\nEvery argument after the stage is passed through, so"
               "\n`python valet.py stats bootstrap --help` shows that stage's own options.")
    return "\n".join(out)


def dispatch(module: str, argv: list[str], display: str) -> None:
    sys.argv = [display, *argv]
    runpy.run_module(module, run_name="__main__")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return

    stage, rest = argv[0], argv[1:]

    if stage in STAGES:
        dispatch(STAGES[stage][0], rest, f"valet.py {stage}")
        return

    if stage in GROUPS:
        subs = GROUPS[stage]
        if not rest or rest[0] in ("-h", "--help"):
            print(f"usage: python valet.py {stage} <command> [options]\n")
            for name, (_, desc) in subs.items():
                print(f"  {name:<16}{desc}")
            return
        sub, rest = rest[0], rest[1:]
        if sub not in subs:
            sys.exit(f"error: unknown '{stage}' command '{sub}'. Known: {', '.join(subs)}")
        dispatch(subs[sub][0], rest, f"valet.py {stage} {sub}")
        return

    sys.exit(f"error: unknown stage '{stage}'. Run `python valet.py` for the list.")


if __name__ == "__main__":
    main()
