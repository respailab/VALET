# Behavioral analysis: findings

Status: BF1's classification was retired (see METHODS.md) in favor of a plain M/U/B compliance-rate table across all 10 models; BF2/BF3/BF4 run on all 10 models.

Figures referenced below are in `behavioral_analysis/plots/`.

---

## M/U/B — bare vs. monitored vs. unmonitored compliance (replaces BF1)

`mub_dumbbell_aggregate.png`, `mub_dumbbell_<model>.png`, `mub_table.csv`

Per-model mean compliance rate, averaged across all 32 values (B = bare/no system prompt, M = monitored/AF prompt+watched, U = unmonitored/AF prompt+unwatched — bare rate uses the regex extraction from `run_test.py`, not the gpt-oss-20b L1 judge pass; see METHODS.md for why):

| model | B_mean | M_mean | U_mean | M−U | U−B |
|---|---|---|---|---|---|
| deepseek-r1-distill | 0.402 | 0.684 | 0.206 | 0.475 | **−0.195** |
| deepseek-r1-0528 | 0.395 | 0.694 | 0.249 | 0.445 | **−0.146** |
| llama-3.3-70b | 0.458 | 0.819 | 0.478 | 0.340 | 0.019 |
| llama-3.1-70b | 0.425 | 0.813 | 0.486 | 0.331 | 0.072 |
| mistral-large-2512 | 0.603 | 0.690 | 0.442 | 0.245 | **−0.148** |
| gpt-4o-mini | 0.398 | 0.678 | 0.529 | 0.157 | 0.144 |
| ministral-14b | 0.579 | 0.774 | 0.633 | 0.155 | 0.062 |
| mistral-small-2409 | 0.389 | 0.813 | 0.670 | 0.147 | 0.319 |
| gemma-3-27b | 0.405 | 0.815 | 0.785 | 0.054 | 0.388 |
| llama-3.1-8b | 0.387 | 0.860 | 0.805 | 0.052 | 0.420 |

Sorted by `M−U` gap. A few patterns stand out:

- **deepseek-r1-distill, deepseek-r1-0528, mistral-large-2512** are the only three models where **U < B** (unmonitored compliance sits *below* the undirected bare baseline). For these, the AF directive — even when the model thinks it's unwatched — pushes it *toward* the pro-value option relative to having no directive at all. Combined with the largest `M−U` gaps in the table, these three show the most textbook-looking faking shape: directive+watched pushes compliance up, directive+unwatched pushes it down past baseline, not just back to it.
- **llama-3.3-70b and llama-3.1-70b** have B and U close together (0.42-0.49) with M well above both — the cleanest separation between "no pressure"/"pressure but unwatched" (similar) vs. "pressure and watched" (much higher).
- **gemma-3-27b, llama-3.1-8b, mistral-small-2409, ministral-14b** have U sitting close to M, both well above B — behavior looks driven by the directive itself regardless of whether it's watched, not by the watched/unwatched distinction.
- No model classification is asserted on top of these numbers (see METHODS.md for why the earlier BF1 three-way split was dropped) — this table is descriptive, not a faking/not-faking verdict per model.

---

## BF2 — is the faking coherent or noise?

`bf2_heatmaps.png`, `bf2_scatter.png`, `bf2_cluster_contingency.png`

**Two versions were run; the upgraded one is the one to trust.**

| | TF-IDF (first pass) | OpenAI embeddings (upgraded) |
|---|---|---|
| Mantel r | +0.214 | +0.184 |
| Mantel p (10k permutations) | 0.0074 | **0.0945 (n.s.)** |

The TF-IDF "hit" doesn't replicate with real embeddings. TF-IDF was picking up shared vocabulary across scenario topics, not semantic content — exactly the failure mode it's known for. **Treat the embeddings result as the honest one: no significant relationship between semantic similarity and gap-correlation.**

**LLM clustering vs. behavioral clustering — no coherence:**
- Adjusted Rand Index: **−0.022** (worse than chance)
- Normalized Mutual Info: 0.247 (weak)
- Purity: 0.333
- Permutation p: **0.75**

GPT-5.4-mini produced 6 clean, sensible clusters from the value definitions (`dependence_fatalism`, `integrity_fairness`, `prosocial_care`, `self_direction`, `social_order`, `status_power` — recognizably Schwartz-adjacent). The behavioral clustering, by contrast, is degenerate: one cluster absorbs 20/32 values, and three of the remaining five are singletons (`Seriousness`, `Fate Determinism`, `Imperturbability` each alone). That's the signature of a matrix dominated by a single factor rather than genuine pairwise structure — almost every value-pair has a strongly positive gap-correlation, consistent with **"some models fake more than others, roughly uniformly across values"** rather than **"specific values fake together because they're semantically related."**

**Bottom line: BF2 does not support Narrative B (coherent, value-structure-driven faking) as tested.** The dominant signal in the gap-correlation matrix looks like a per-model overall propensity, not value-specific coevolution.

**Flagged follow-up (not yet run):** z-score each model's per-value gaps before correlating, to remove the per-model propensity factor and re-test for genuine value-pair coherence underneath it. This is the correct next step before concluding coherence is entirely absent — the current result may be masking a real but smaller effect.

---

## BF3 + BF4 — can you trust the trace?

`bf3_forest.png`, `bf4_scatter.png`

**BF3 — pooled odds ratio (gap>median vs. verbalized training_signal reasoning) is significant and positive: OR≈3.03, 95% CI [2.59, 3.55] (Fisher p<0.0001, n=6390 scenario-condition pairs across all models).** High-gap scenarios are about 3x more likely to co-occur with explicit training/monitoring verbalization than low-gap scenarios, pooled.

Per-model, only two individually clear significant positive odds ratios (both wide CIs from sparse cells, small-n caveat applies):
- **deepseek-r1-0528**: OR=17.3, 95% CI [1.01, 298], p=0.002
- **deepseek-r1-distill**: OR=6.96, 95% CI [0.90, 53.6], p=0.042

Everything else is not individually significant (CIs cross 1) — the pooled effect is carried disproportionately by the reasoning-heavy models (deepseek family), not a uniform effect across all 10.

**BF4 — the ranking that matters for "silent faking":**

| model | mean_gap | verbalization_rate |
|---|---|---|
| deepseek-r1-distill | 0.495 | 0.978 |
| deepseek-r1-0528 | 0.444 | 0.984 |
| llama-3.3-70b | 0.330 | 0.986 |
| llama-3.1-70b | 0.325 | 0.987 |
| mistral-large-2512 | 0.232 | 0.977 |
| mistral-small-2409 | 0.159 | 0.934 |
| ministral-14b | 0.146 | 0.897 |
| **gpt-4o-mini** | **0.141** | **0.077** |
| llama-3.1-8b | 0.051 | 0.510 |
| gemma-3-27b | 0.051 | 0.840 |

**No model lands in the high-gap/low-verbalization quadrant** — every model with an above-median gap also verbalizes above the median. So on its face, BF4 doesn't support a "silent faking" story either: the models that fake more also talk about it more.

**But gpt-4o-mini is a striking outlier that needs checking before that conclusion is trusted:** every other model verbalizes training_signal reasoning in 84–99% of monitored responses; gpt-4o-mini does so in only **7.7%**. Two explanations are live and this analysis can't yet distinguish them:
1. **Real finding** — gpt-4o-mini genuinely reasons about training/monitoring far less than every other model tested (plausible: it's the smallest/cheapest model here and may not engage in explicit chain-of-thought about its situation the way reasoning-oriented models do).
2. **Judge-calibration artifact** — the labeling judge is itself gpt-4o-mini (see `value_faking_test/analyze_layers.py`), so this could be the judge under-flagging its own family's outputs, or a prompt-sensitivity issue specific to that model's response style.

**This is exactly the annotation-quality risk the brief calls out, and it's not resolved yet.** The second-judge re-annotation + human spot check (`bf4_reannotation_stub.py`, not run) is needed before either BF3's per-model ranking or BF4's "no silent faking" conclusion can be trusted. The stratified human-review sample is ready (`human_review_sample.csv`, 120 traces) but hasn't been reviewed.

---

## Open items before these findings are final

1. **BF2**: run the z-scored (per-model-propensity-removed) variant to check for coherence masked by the dominant propensity factor.
2. **BF3/BF4**: second-judge re-annotation + human review of the 120-trace sample, especially the gpt-4o-mini verbalization-rate anomaly, before treating the "no silent-faking corner" result as settled.
