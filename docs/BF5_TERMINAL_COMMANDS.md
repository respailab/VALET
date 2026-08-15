# BF5 Part 1 (consequence ablation) — top-8-per-model, split across 5 terminals

Each block is meant to be pasted into its own terminal / tmux pane and run top-to-bottom.
`run_consequence_ablation_SP_test.py` runs all 4 arms (formal / sp1 / logged / contractor)
in one call; the analyze step below it uses **`analyze_layers_consequence.py`** (the
consequence-generalized judge — `consequence_awareness` label + `consequence_matches_prompt`
check) with `--arm` set per arm, so the judge is told the correct stated consequence for
each one. (Rule going forward: consequence ablation → `analyze_layers_consequence.py`;
tone ablation → plain `analyze_layers.py`, since tone doesn't change the consequence.)

**Judge model:** terminals 1-3 (HF/local generation) use a **local vllm-served
`openai/gpt-oss-20b`** as the judge instead of the OpenAI API — start the vllm server on
the same GPU(s) *after* the HF generation step finishes and frees the GPU, since both can't
hold VRAM at once on a single card. Terminals 4-5 (API generation) keep the default
`openai/gpt-4o-mini` judge.

**Before analyzing any HF terminal's results**, start the vllm server in that terminal:
```bash
vllm serve openai/gpt-oss-20b --port 8000 &
# wait until it logs "Application startup complete" before running the analyze commands
```
`client.py`'s vllm backend reads `VLLM_BASE_URL` (defaults to `http://localhost:8000/v1`) —
leave unset if using the default port. Kill the server (`kill %1`) before freeing the GPU
for the next model in a terminal that runs more than one (terminal 3).

---

## Terminal 1 — CUDA 1 — hf llama-3.1-8b

```bash
export CUDA_VISIBLE_DEVICES=1

# llama-3.1-8b — top8: ['Sympathy', 'Conformity', 'Cooperativeness', 'Resourcefulness', 'Order', 'Dominance', 'Change', 'Extravagance']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service hf --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --n 2 --tag bf5_part1_top8 \
  --value "Sympathy" "Conformity" "Cooperativeness" "Resourcefulness" "Order" "Dominance" "Change" "Extravagance"

# --- generation done, GPU free — start local judge ---
vllm serve openai/gpt-oss-20b --port 8000 &
# wait for "Application startup complete"

BASE=value_faking_test/outputs/hf__meta-llama_Meta-Llama-3.1-8B-Instruct
ANALYSIS=value_faking_test/analysis/hf__meta-llama_Meta-Llama-3.1-8B-Instruct

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service vllm --model openai/gpt-oss-20b

kill %1   # stop the vllm server, free GPU 1
```
---

## Terminal 2 — CUDA 2 — hf ministral-14b

⚠️ **Verify this HF repo id before running** — I could not confirm
`mistralai/Ministral-14B-Instruct-2512` actually exists on Hugging Face. Check
`huggingface.co/mistralai` and substitute the correct repo id if this one 404s.

```bash
export CUDA_VISIBLE_DEVICES=2

# ministral-14b — top8: ['Tolerance', 'Order', 'Depth', 'Complexity', 'Dependence', 'Change', 'Tradition', 'Social Confidence']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service hf --model "mistralai/Ministral-14B-Instruct-2512" \
  --n 2 --tag bf5_part1_top8 \
  --value "Tolerance" "Order" "Depth" "Complexity" "Dependence" "Change" "Tradition" "Social Confidence"

# --- generation done, GPU free — start local judge ---
vllm serve openai/gpt-oss-20b --port 8000 &
# wait for "Application startup complete"

BASE=value_faking_test/outputs/hf__mistralai_Ministral-14B-Instruct-2512
ANALYSIS=value_faking_test/analysis/hf__mistralai_Ministral-14B-Instruct-2512

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service vllm --model openai/gpt-oss-20b

kill %1   # stop the vllm server, free GPU 2
```
---

## Terminal 3 — CUDA 1,2 — hf mistral-small-22b, then hf gemma-3-27b

Two models sequentially on the same 2 GPUs — finish and analyze the first before starting the second.

```bash
export CUDA_VISIBLE_DEVICES=1,2

# ── model 1: mistral-small-2409 (22B) ──────────────────────────────────
# mistral-small-2409 — top8: ['Complexity', 'Order', 'Tolerance', 'Depth', 'Autonomy', 'Tradition', 'Sympathy', 'Compassion']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service hf --model "mistralai/Mistral-Small-Instruct-2409" \
  --n 2 --tag bf5_part1_top8 \
  --value "Complexity" "Order" "Tolerance" "Depth" "Autonomy" "Tradition" "Sympathy" "Compassion"

# --- generation done, GPU free — start local judge ---
vllm serve openai/gpt-oss-20b --port 8000 &
# wait for "Application startup complete"

BASE=value_faking_test/outputs/hf__mistralai_Mistral-Small-Instruct-2409
ANALYSIS=value_faking_test/analysis/hf__mistralai_Mistral-Small-Instruct-2409

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service vllm --model openai/gpt-oss-20b

kill %1   # stop the vllm server, free GPU 1,2 before the next model

# ── model 2: gemma-3-27b ────────────────────────────────────────────────
# gemma-3-27b — top8: ['Autonomy', 'Sympathy', 'Order', 'Orderliness', 'Complexity', 'Self-discipline', 'Social Confidence', 'Depth']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service hf --model "google/gemma-3-27b-it" \
  --n 2 --tag bf5_part1_top8 \
  --value "Autonomy" "Sympathy" "Order" "Orderliness" "Complexity" "Self-discipline" "Social Confidence" "Depth"

# --- generation done, GPU free — start local judge ---
vllm serve openai/gpt-oss-20b --port 8000 &
# wait for "Application startup complete"

BASE=value_faking_test/outputs/hf__google_gemma-3-27b-it
ANALYSIS=value_faking_test/analysis/hf__google_gemma-3-27b-it

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service vllm --model openai/gpt-oss-20b
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service vllm --model openai/gpt-oss-20b

kill %1
```
---

## Terminal 4 — openrouter llama-3.1-70b, then openrouter llama-3.3-70b

API-based — no GPU/CUDA needed. Judge stays the default `openai/gpt-4o-mini`.

```bash
# ── model 1: llama-3.1-70b ──────────────────────────────────────────────
# llama-3.1-70b — top8: ['Complexity', 'Depth', 'Dependence', 'Compassion', 'Order', 'Fairness', 'Change', 'Extravagance']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service openrouter --model "meta-llama/llama-3.1-70b-instruct" \
  --n 2 --tag bf5_part1_top8 \
  --value "Complexity" "Depth" "Dependence" "Compassion" "Order" "Fairness" "Change" "Extravagance"

BASE=value_faking_test/outputs/openrouter__meta-llama_llama-3.1-70b-instruct
ANALYSIS=value_faking_test/analysis/openrouter__meta-llama_llama-3.1-70b-instruct

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service openai --model gpt-4o-mini --requests_per_second 10

# ── model 2: llama-3.3-70b ──────────────────────────────────────────────
# llama-3.3-70b — top8: ['Complexity', 'Depth', 'Order', 'Tolerance', 'Compassion', 'Sympathy', 'Tradition', 'Friendliness']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service openrouter --model "meta-llama/llama-3.3-70b-instruct" \
  --n 2 --tag bf5_part1_top8 \
  --value "Complexity" "Depth" "Order" "Tolerance" "Compassion" "Sympathy" "Tradition" "Friendliness"

BASE=value_faking_test/outputs/openrouter__meta-llama_llama-3.3-70b-instruct
ANALYSIS=value_faking_test/analysis/openrouter__meta-llama_llama-3.3-70b-instruct

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service openai --model gpt-4o-mini --requests_per_second 10
```
---

## Terminal 5 — openrouter mistral-large, then openai gpt-4o-mini

```bash
# ── model 1: mistral-large-2512 ─────────────────────────────────────────
# mistral-large-2512 — top8: ['Sympathy', 'Order', 'Depth', 'Complexity', 'Tolerance', 'Compassion', 'Friendliness', 'Responsibility']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service openrouter --model "mistralai/mistral-large-2512" \
  --n 2 --tag bf5_part1_top8 \
  --value "Sympathy" "Order" "Depth" "Complexity" "Tolerance" "Compassion" "Friendliness" "Responsibility"

BASE=value_faking_test/outputs/openrouter__mistralai_mistral-large-2512
ANALYSIS=value_faking_test/analysis/openrouter__mistralai_mistral-large-2512

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service openai --model gpt-4o-mini --requests_per_second 10

# ── model 2: gpt-4o-mini ────────────────────────────────────────────────
# gpt-4o-mini — top8: ['Tolerance', 'Order', 'Complexity', 'Responsibility', 'Orderliness', 'Dependence', 'Sympathy', 'Depth']
python value_faking_test/run_consequence_ablation_SP_test.py \
  --service openai --model "gpt-4o-mini" \
  --n 2 --tag bf5_part1_top8 \
  --value "Tolerance" "Order" "Complexity" "Responsibility" "Orderliness" "Dependence" "Sympathy" "Depth"

BASE=value_faking_test/outputs/openai__gpt-4o-mini
ANALYSIS=value_faking_test/analysis/openai__gpt-4o-mini

python value_faking_test/analyze_layers_consequence.py --input "${BASE}__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__bf5_part1_top8/analyzed_results_v2.csv" --arm formal --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__sp1__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__sp1__bf5_part1_top8/analyzed_results_v2.csv" --arm sp1 --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__logged__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__logged__bf5_part1_top8/analyzed_results_v2.csv" --arm logged --service openai --model gpt-4o-mini --requests_per_second 10
python value_faking_test/analyze_layers_consequence.py --input "${BASE}__contractor__bf5_part1_top8/scenario_results.csv" --output "${ANALYSIS}__contractor__bf5_part1_top8/analyzed_results_v2.csv" --arm contractor --service openai --model gpt-4o-mini --requests_per_second 10
```
---

## Notes

- All commands assume the current working directory is the repo root (`conflict_research/`).
- Consequence ablation → `analyze_layers_consequence.py` with `--arm` set per arm (formal/sp1/logged/contractor),
  so the judge is told the correct stated consequence for that arm and can flag `consequence_matches_prompt`.
- Variable expansion uses `${BASE}__...` (braced) — `$BASE__...` without braces would be parsed
  as the single variable name `BASE__...` and expand to nothing. Always use `${BASE}` here.
- This only covers **BF5 Part 1 (consequence ablation)** — Part 2 (tone ablation) isn't included
  since it wasn't requested for this terminal split; if you add it, use the plain `analyze_layers.py`
  (no `--arm`), since tone ablation doesn't change the stated consequence.
- Set `OPENROUTER_API_KEY` / `OPENAI_API_KEY` (e.g. `source .env`) before terminals 4 and 5.
