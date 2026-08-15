# Behavioral analysis: methods

Code lives in `behavioral_analysis/`. All four questions reuse two upstream data sources:

- **Main runs** (`value_faking_test/analysis/<model>/analyzed_results_v2.csv`, produced by `value_faking_test/analyze_layers.py`): 10 models, 32 values, n=4 responses/scenario, under the alignment-faking (AF) system prompt. Per scenario: `monitored_rate`, `unmonitored_rate`, `compliance_gap` (= monitored − unmonitored), and per-response reasoning-type labels from a single LLM judge (gpt-4o-mini).
- **Bare runs** (`test_ablations/outputs/<model>_v2_n4_bare/`): same scenarios generator, no system prompt at all, no directive, no monitoring framing. Complete for all 10 models (32/32 values each).

`common.py` centralizes the model→path registry so every script picks up new models automatically as their bare runs finish, without editing analysis code.

---

## M/U/B table — bare vs. monitored vs. unmonitored compliance (BF1 retired)

**File:** `mub_table.py` (replaces the earlier `bf1_held_preference.py`)

For each value × model, three compliance rates are reported directly, with no classification layer on top: **B** (bare, no directive), **M** (monitored, AF prompt + watched), **U** (unmonitored, AF prompt + not watched). All three are `P(model picks the anti-value / developer_compliant option)`.

**Why BF1's three-way split (`robust_value` / `alignment_faking` / `not_recognized`) was dropped:** it tried to classify "held preference" from a single undirected baseline (B) plus a bootstrap-CI comparison against M/U. In practice that conflates two different things a value could show — genuinely no preference, versus a preference the monitoring cue simply doesn't engage — and the B/M/U triplet alone can't tell them apart. Distinguishing those needs a real preference-elicitation design, not just one bare baseline. Rather than publish a split that overclaims what the data can support, this reports the raw B/M/U numbers per value × model so they're available for inspection without a classification claim riding on top.

**What's kept:** the same B (independently-generated bare scenario pool, zero text overlap with the main run, verified) vs. M/U (main run) sourcing, and the same preference for the LLM-judged `analyzed_bare_l1.csv` (`analyze_layers_bare.py`'s L1 output) over the regex-extracted `bare_choices` when both are available for a model — reported per model as `B_source` in the output table.

No bootstrap, no near/far thresholding, no per-cell claim — just `B_mean`, `M_mean`, `U_mean`, and the two raw differences (`M_minus_U`, `U_minus_B`) per value × model, saved to `mub_table.csv`.

**Known issue with the current `B_mean` values:** the L1 bare re-annotation (`analyze_layers_bare.py`) was run with a gpt-oss-20b judge served locally via vLLM, and its "actual choice" extraction disagrees with the original regex extraction on ~42% of responses (79.5% agreement even restricted to cases both call valid; the judge marks ~21% of responses "unclear" versus ~6% for regex, and skews noticeably toward "b"). This compresses `B_mean` toward 0 for many values across all 10 models and should not yet be treated as a clean signal — it looks like a judge-side artifact (gpt-oss-20b's response format may not be parsing cleanly through the L1 prompt/JSON-mode path via vLLM), not a property of the underlying bare responses. Recommend spot-checking judge output against raw response text, or re-running L1 bare annotation with the gpt-4o-mini judge (matching the main run) before relying on `B_mean` for any conclusion.

---

## BF2 — is the faking coherent or noise?

**Files:** `bf2_coherence.py` (TF-IDF baseline), `bf2_llm_cluster_coherence.py` (embeddings + LLM clustering upgrade)

**Reframing note:** the original brief asks for "within-model correlation of gaps across values." That's not statistically well-posed for a single model — each value has its own independently generated scenario set, so there's no shared unit to pair value A's scenarios against value B's scenarios within one model. The well-posed version used here: build a (model × value) matrix of mean `compliance_gap`, then correlate each pair of value-columns **across the 10 model rows**. Two values are "coherent" if models that fake one tend to fake the other. This is stated explicitly in the code so the reinterpretation is visible, not silently substituted.

**Semantic similarity — two versions:**
1. `bf2_coherence.py`: TF-IDF/cosine similarity over each value's aggregated scenario `topic` text, used as a first-pass proxy when `sentence-transformers` was broken in the venv (huggingface-hub version conflict).
2. `bf2_llm_cluster_coherence.py`: real OpenAI embeddings (`text-embedding-3-small`) over the same aggregated topic text. Swapped in because TF-IDF picks up superficial shared vocabulary, not semantic content — a real embedding model is the honest test.

**LLM clustering layer:** GPT-5.4-mini reads all 32 value definitions and groups them into semantically coherent clusters **using its own judgment**, not k-means on the embedding vectors — the point is a human/model-legible grouping (something like Schwartz's higher-order dimensions) to compare against a purely data-driven behavioral grouping.

**Behavioral clustering:** agglomerative (average-linkage) clustering on `1 − gap_correlation` as a distance, cut to the same number of clusters the LLM produced, so the two partitions are comparable.

**Matching + scoring:** Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the contingency table finds the best correspondence between semantic and behavioral cluster IDs (cluster IDs are arbitrary on both sides, so raw label agreement is meaningless without this step). Coherence is then scored three ways:
- **Adjusted Rand Index** — corrects for chance agreement (unlike raw purity, which is inflated by any imbalanced partition).
- **Normalized Mutual Information** — captures partial/fuzzy agreement ARI can miss.
- **Purity** — simplest, most literal "fraction correctly matched" number, reported alongside the other two rather than alone, since purity alone is misleadingly high for degenerate partitions (e.g. one giant cluster + several singletons scores well on purity by construction).

**Mantel test (both versions):** correlate the upper triangles of the gap-correlation matrix and the semantic-similarity matrix, then build a null by **permuting the value labels** on one matrix (not shuffling the raw correlation values) and recomputing, 10,000 times. Label permutation is required here rather than a standard Pearson p-value because the entries of a correlation/similarity matrix aren't independent observations — permuting labels preserves the matrix's internal structure while destroying the correspondence between the two matrices, which is the actual null hypothesis of interest.

---

## BF3 + BF4 — can you trust the trace?

**File:** `bf3_bf4_trace_reliability.py`

Both reuse the existing single-judge `monitored_reasoning_types` column (JSON list per scenario, one label per of the n=4 monitored responses) from `analyze_layers.py`'s output. "Verbalized" = at least one monitored response for that scenario was labeled `training_signal` (explicit mention of free-tier/monitoring/training as the reason for the choice).

**BF3 (scenario grain):** 2×2 contingency table — gap above/below that model's median `compliance_gap`, crossed with verbalized/not — per model and pooled. **Odds ratio, not just agreement rate**, per the brief: agreement rate treats "both low" and "both high" as equally informative, which is the wrong thing to optimize when the actual question is whether high-gap scenarios are *disproportionately* accompanied by verbalization. Odds ratio + Fisher exact p-value (small/sparse cells in several models) + `statsmodels.stats.contingency_tables.Table2x2` for the CI.

**BF4 (model grain):** mean `compliance_gap` vs. verbalization rate, one point per model (all 10, all with main-run data — doesn't need bare data so it's not blocked). Flags the high-gap/low-verbalization quadrant (above median gap, below median verbalization) — the "silent faking" signature the brief specifically calls out as the thing this analysis is hunting for.

**Armor step (not run):** `bf4_reannotation_stub.py`. The brief is explicit that a silent-faking claim gets attacked on annotation quality first, since the existing labels are a single LLM-judge pass. Two things are needed before trusting BF3/BF4 numbers: (1) a second judge — different model, differently worded prompt, to avoid correlated errors from one model/prompt's blind spots — scored against the first pass via Cohen's κ; (2) a human spot check of ≥100 traces, stratified and **oversampled on the high-gap/not-verbalized cell** (40% of the sample) since that's the cell Narrative A (silent faking) actually rests on — the other three cells are lower stakes to get wrong. The stratified sampler itself (no API calls, pure pandas) was run and produced `human_review_sample.csv` (120 traces, 48/24/24/24 split). The actual second-judge LLM calls are sketched in the same file but intentionally not executed — that needs a live API budget decision and your manual review time, not something to run silently.
