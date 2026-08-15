# VALUE_FAKING_TOOLKIT — staging notes

Copied 2026-08-13. **Nothing was moved or deleted.** Every file here is a `cp` of a
file that still exists at its original path. This tree is a staging area for the
GitHub release; the refactor into a real package has not started.

## Provenance

| Toolkit path | Copied from |
|---|---|
| `value_faking/core/` | `value_faking_test/{client,prompts}.py` |
| `value_faking/run/` | `value_faking_test/run_test{,_vllm,_hf_batched}.py` |
| `value_faking/annotate/` | `value_faking_test/analyze_test_results.py`, `behavioral_analysis/judge_*.py` |
| `value_faking/stats/` | `value_faking_test/{bootstrap_ci,heterogeneity,compute_*,normalized_gap_table}.py` |
| `value_faking/bf/` | `behavioral_analysis/{common,bf2_*,bf3_bf4_*,bf4_*,mub_table,plot_bf_results}` |
| `value_faking/report/` | `value_faking_test/plot_*.py`, `value_cards/*` |
| `value_faking/build/` | `generation/{topic_gen,seed_scenario_gen,scenario_expansion}/*.py` |
| `ablations/run/` | `value_faking_test/run_*_ablation_SP_test.py`, `test_ablations/run_{binary_label,likert,option_pro_value}.py` |
| `ablations/analysis/` | `value_faking_test/{analyze_layers_consequence,analyze_bf5_arms,compute_swap_ablation}.py`, `behavioral_analysis/bf5_*.py` |
| `ablations/plots/` | `value_faking_test/plot_swap_n4.py`, `test_ablations/plot_{option_swap,bare_baseline*,likert_binary_comparison}.py` |
| `data/benchmark_v1/` | `generation/scenario_expansion/final/*.json` (32 values) |
| `docs/` | `behavioral_analysis/{METHODS,FINDINGS,system_prompt_arms}.md`, `value_faking_test/BF5_TERMINAL_COMMANDS.md` |

## State

Refactor pass done: package imports, central `value_faking/paths.py`, `python valet.py`
dispatcher, CLI guards. All 27 stages respond to `--help` without executing.
No absolute paths and no old-tree path literals remain.

Deliberately **not** an installable package: no `pyproject.toml`, no PyPI. Run it in
place with `python valet.py`; dependencies are in `requirements.txt`.

## Figure provenance

Every graphic the paper actually `\includegraphics`, and the stage that produces it.
Established by grepping the LaTeX sources, not by filename guessing: the assets
directory also contains stale figures that no longer appear in the paper.

| Figure | Count | Stage |
|---|---|---|
| `combined.png` (per-model appendix panels) | 9 | `figures pervalue` |
| `mub_dumbbell_<slug>.png` + `_aggregate` | 11 | `figures bf` / `stats mub` |
| `all10_heatmap`, `all10_spearman`, `all10_small_multiples` | 3 | `figures crossmodel` |
| `all10_reasoning_bars` | 1 | `figures reasoning` |
| `all10_divergence_bars` | 1 | `figures divergence` |
| `bf3_forest`, `bf4_scatter` | 2 | `figures bf` |
| `value_card_llama.pdf` | 1 | `card` |
| `alignment_faking_illustration.png` | 1 | hand-drawn, not generated |

`visualize.py` was removed: it produced `per_value_table.*` and `compliance_gap.*`,
neither of which the paper references anywhere.

## Known issues to resolve in the refactor

- **Four `client.py` / `prompts.py` copies** are staged (`core/` plus one per `build/`
  subdir). These are drifted duplicates in the original repo. The refactor collapses
  them to `core/`.
- ~~`_undecided/analyze_layers.py`~~ **Resolved.** It writes `analyzed_results_v2.csv`,
  the file every paper figure reads: Layer 1 gives intended/actual/divergence (the
  §5.2 validity check), Layer 2 the six-category taxonomy. Promoted to
  `value_faking/annotate/` and made the primary `annotate` stage.
  `analyze_test_results.py` predates it, emits no divergence columns, and is kept
  only as `annotate-legacy` for reproducing older runs.
  (`analyze_layers_bare.py` was ruled unused and is not staged.)
- **`compute_sensitivity.py` output is stale.** `results/sensitivity_decomposition.md`
  is dated 2026-06-20, from the four-model era. Needs a re-run against the ten-model
  n=4 data.
- **Three runners** (`run_test.py`, `_vllm`, `_hf_batched`) are the same loop with
  different batching. Collapse behind a backend protocol.
- **Seven ablation runners** collapse to one runner plus per-arm config.
- **Value Cards**: `value_card_populate.py` has only ever produced one model
  (llama-3.3-70b). Paper §6 claims ten. Path needs to actually run for all ten.
- **Not staged, decided out**: `steering/`, `representational_analysis/`,
  `resources/VLAF`, `resources/ValueBench`, `run_inversion_attack.py`,
  `run_sp_inversion.py`, `dashboard/`, `outputs_server/`, `scenario_optimization/`,
  `enterprise_scenario_test/`, root `run_*_tmux.sh` / `nohup.sh` launchers, `.zip` bundles.

## Audit note

When checking whether a script is dead, grep the nohup/tmux wrapper shell scripts,
not `~/.zsh_history`. Long-running jobs bypass the history file, so absence from
history is not evidence of disuse.
