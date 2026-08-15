"""Single source of truth for every path the toolkit reads or writes."""

from __future__ import annotations

import os
from pathlib import Path

# The installed package may live anywhere; VALET_ROOT wins when set, otherwise the
# root is the directory containing this package.
TOOLKIT_ROOT = Path(os.environ.get("VALET_ROOT", Path(__file__).resolve().parents[1]))

DATA_ROOT      = TOOLKIT_ROOT / "data"
SCENARIOS_ROOT = DATA_ROOT / "scenarios"
BENCHMARK_ROOT = DATA_ROOT / "benchmark_v1"

RUNS_ROOT      = TOOLKIT_ROOT / "outputs"
ANALYSIS_ROOT  = TOOLKIT_ROOT / "analysis"
ABLATION_ROOT  = TOOLKIT_ROOT / "outputs_ablations"
RESULTS_ROOT   = TOOLKIT_ROOT / "results"
CONFIGS_ROOT   = TOOLKIT_ROOT / "configs"

# Benchmark entry points. SEEDS_CSV is the 639-scenario table actually evaluated;
# DEV_DIR holds the per-value suppression directive fields keyed by value name.
SEEDS_CSV = SCENARIOS_ROOT / "seed_scenarios.csv"
DEV_DIR   = SCENARIOS_ROOT

MODEL_REGISTRY = CONFIGS_ROOT / "models.json"

# Retained because the migrated scripts refer to it; prefer the specific roots.
REPO = str(TOOLKIT_ROOT)

# Legacy string aliases. The scripts inherited from the research tree build paths by
# f-string concatenation rather than pathlib, so these keep the migration mechanical.
BF_ROOT       = str(TOOLKIT_ROOT / "bf_out")
RUNS_DIR      = str(RUNS_ROOT)
ANALYSIS_DIR  = str(ANALYSIS_ROOT)
ABLATION_DIR  = str(ABLATION_ROOT)
RESULTS_DIR   = str(RESULTS_ROOT)


def run_dir(slug: str) -> Path:
    return RUNS_ROOT / slug


def analysis_dir(slug: str) -> Path:
    return ANALYSIS_ROOT / slug


def ensure(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


__all__ = [
    "TOOLKIT_ROOT", "DATA_ROOT", "SCENARIOS_ROOT", "BENCHMARK_ROOT",
    "RUNS_ROOT", "ANALYSIS_ROOT", "ABLATION_ROOT", "RESULTS_ROOT", "CONFIGS_ROOT",
    "SEEDS_CSV", "DEV_DIR", "MODEL_REGISTRY", "REPO",
    "BF_ROOT", "RUNS_DIR", "ANALYSIS_DIR", "ABLATION_DIR", "RESULTS_DIR",
    "run_dir", "analysis_dir", "ensure",
]
