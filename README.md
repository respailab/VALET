# VALET — Value ALignment-faking Evaluation Tool

### Alignment Faking Depends on the Value, Not Just the Model

Language models can follow a directive when they believe they are monitored and
abandon it when they believe they are not. VALET profiles that behaviour
**per value** rather than per model, scoring each model–value pair under three
matched conditions:

| | condition | rate |
|---|---|---|
| **B** | bare: no system prompt, no oversight tag | undirected baseline |
| **M** | monitored: `<is_free_tier_user>True` | oversight-conditional |
| **U** | unmonitored: `<is_free_tier_user>False` | directive-only |

`M - U` is the compliance gap. `U - B` is how far the directive moves behaviour
even when the model believes it is unwatched. Reporting the triple separates a
value the model robustly holds from one it never held, which `M - U` alone cannot.

A six-category taxonomy over reasoning traces tracks the behavioural shift, and a
reasoning–choice divergence check tests whether traces record the decision or
narrate it afterwards. Divergence is derived from the judge's intended-versus-actual
pair rather than scored directly. Per-model results are packaged as **value cards**.

Instantiated on 32 ValueBench dimensions × 10 models (8B–~675B): gaps span
0.05–0.50, do not track parameter count, and a quarter of the structural variance
comes from which model was asked about which value.

## What you can change, and what you can watch

Every component of the probe is a file you can edit, and every stage writes a
table you can read before the next stage consumes it.

| Modify | Where | Effect |
|---|---|---|
| the values | `data/values.csv` — append a row | new dimension enters generation and scoring |
| the dilemmas | `data/scenarios/` — or regenerate with `build` | new benchmark |
| the directive | `data/scenarios/<Value>/dev_fields.json` | what the developer asks the model to suppress |
| the oversight cue | `value_faking/core/prompts.py` | how the model is told it is watched |
| the conditions | `--no_system_prompt`, `--swap_options`, `ablate` arms | B, M, U, and the ablation arms |
| the models | `--service` / `--model`, `configs/models.json` | any OpenAI, OpenRouter, vLLM, or local HF model |
| the judge | `--model` on `annotate` | who scores choices and reasoning |

| Observe | Written by | Contains |
|---|---|---|
| `scenario_results.csv` | `run`, `run-bare` | raw responses, extracted choices, per-condition rates |
| `analyzed_results.csv` | `annotate` | judged intended vs actual choice, divergence, reasoning category |
| `tables/mub_per_value.csv` | `figures panels` | B, M, U, `M - U`, `U - B` per value |
| `plots/*.pdf` | `figures panels` | gap, dumbbell, reasoning, divergence |
| `sig/bh_significant.csv` | `stats bootstrap` | which gaps survive Benjamini–Hochberg |
| `card/value_card_<slug>.pdf` | `card` | the whole profile as one document |

Nothing is hidden in a black box: the judge's per-response verdicts sit in the same
CSV as the raw text it judged, so any number on a card can be traced back to the
response that produced it.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the repository root:

```bash
OPENAI_API_KEY=sk-...              # OpenAI-served models and the default judge
OPENROUTER_API_KEY=sk-or-...       # OpenRouter-served models
VLLM_BASE_URL=http://localhost:8000/v1   # optional, --service vllm
```

```bash
source .env
python valet.py                      # list every stage
```

No install step. Set `VALET_ROOT` to keep outputs outside the checkout.
Local inference needs `torch transformers accelerate` (`--service hf`) or `vllm`.
Compiling a value card needs `latexmk`.

---

## Per model: run to value card

Works on one value as a pilot or all 32.

```bash
MODEL=meta-llama/llama-3.1-8b-instruct
JUDGE=meta-llama/llama-3.1-70b-instruct
RUN=outputs/openrouter__meta-llama_llama-3.1-8b-instruct__smoke
BARE=outputs/openrouter__meta-llama_llama-3.1-8b-instruct__smoke_bare
```

**1. Monitored + unmonitored** — omit `--value` for all 32.

```bash
python valet.py run \
  --service openrouter --model $MODEL \
  --value Compassion --n 1 \
  --tag smoke --requests_per_second 2
```

**2. Bare** — required for `U - B`, the dumbbell panel, and the card.

```bash
python valet.py run-bare \
  --service openrouter --model $MODEL \
  --values Compassion --n 1 \
  --tag smoke_bare --no_system_prompt --requests_per_second 2
```

**3. Annotate** — one judge model, two calls per response: L1 extracts intended vs
actual choice, L2 assigns the reasoning category. Divergence is derived from L1's
pair, not judged. Budget `scenarios × conditions × n × 2` calls.

```bash
python valet.py annotate \
  --input  $RUN/scenario_results.csv \
  --output $RUN/analyzed_results.csv \
  --service openrouter --model $JUDGE \
  --requests_per_second 2
```

Not `annotate-legacy`: it emits no intended/actual columns, so divergence cannot be
computed and the card cannot build.

**4. Panels and the B/M/U table**

```bash
python valet.py figures panels --run-dir $RUN --bare-dir $BARE
```

Writes `$RUN/tables/mub_per_value.csv` and `$RUN/plots/{gap,dumbbell,reasoning_monitored,reasoning_unmonitored,divergence}.pdf`.

**5. Value card**

```bash
python valet.py card --model openrouter__meta-llama_llama-3.1-8b-instruct__smoke --compile
```

Daggers marking Benjamini–Hochberg survival appear only when
`$RUN/sig/bh_significant.csv` exists; otherwise the card states that no correction
is recorded. Add an entry to `configs/models.json` for a display name.

### Ablations

```bash
python valet.py ablate prereg                    # arm parity checks, no inference
python valet.py ablate consequence --service openrouter --model $MODEL \
  --value Compassion --n 1 --arms logged --tag smoke_bf5
python valet.py run-bare --service openrouter --model $MODEL \
  --values Compassion --n 1 --tag smoke_swap --swap_options
python valet.py ablate swapstats                 # position-bias verdicts
```

Arms carry their own baseline and are usually single-replicate; do not compare them
against main-path figures at a different `n`.

---

## Cross-model plots

Consume all ten runs at once. Complete the per-model flow for every model first.

```bash
python valet.py crossmodel pervalue      # per-model per-value panels
python valet.py crossmodel heatmap       # gap heatmap, Spearman matrix, small multiples
python valet.py crossmodel reasoning     # reasoning composition
python valet.py crossmodel divergence    # reasoning-choice divergence
python valet.py crossmodel bf            # B/M/U dumbbells, forest, scatter
```

`pervalue` takes `--model <slug>`; the rest take `--outdir`.

## Cross-model statistics

```bash
python valet.py stats mub                # per-value B/M/U, all models
python valet.py stats bare
python valet.py stats sensitivity        # M-U / U-B decomposition
python valet.py stats bootstrap --n_boot 1000 --seed 20260802
python valet.py stats heterogeneity --n_perm 5000 --seed 20260802
python valet.py stats aggregate
python valet.py stats crossmodel
```

The bootstrap resamples scenarios within values and evaluates every condition on
the same draws, so `M - U` and `U - B` intervals are paired.

---

## Benchmark

32 values × 20 forced-choice dilemmas (639 scenarios) ship in `data/scenarios/`.
Rebuild only to target different values or a different domain:

The generator is driven by one table, `data/values.csv` (32 rows). Columns:
`value`, `text` (a ValueBench item), `agreement`, `definition`, `opposite`,
`upper`, `level`, `source`. Only `value`, `text`, and `definition` are read.
Add a value by appending a row; no other file needs editing.

Each stage defaults to `data/values.csv`; pass `--csv` for a different table.

```bash
python valet.py build topics
# -> build/topic_gen/outputs/<Value>/topics_raw.json, topics_scored.json

python valet.py build seeds
# -> build/seed_scenario_gen/outputs/<Value>/dev_fields.json, seed_scenarios.json

python valet.py build expand
# -> build/scenario_expansion/outputs/<Value>/expansion_scenarios.json
# -> build/scenario_expansion/final/<Value>_50.json   (the one to inspect)
```

Each stage skips values whose output already exists, so a failed run resumes.
`build expand` reads `--gen_model` for generation and `--val_model` for validation;
`build topics` and `build seeds` take `--phase 1|2` to run one half.

## Layout

```
data/scenarios/            seed_scenarios.csv + <Value>/dev_fields.json
outputs/<run>/             raw results, then tables/ plots/ card/
outputs_ablations/<run>/   ablation arms, kept apart from the main path
configs/models.json        display names and card provenance
value_faking/paths.py      every path the toolkit reads or writes
```

Figures render without LaTeX by default; set `VALET_USETEX=1` for camera-ready type.

## Citation

```bibtex
```
