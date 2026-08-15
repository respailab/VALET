"""Shared model registry for the BF1-BF4 behavioral analyses."""
import os
import pandas as pd
from value_faking.paths import REPO
# name -> analyzed_results_v2.csv (main v2_n4, AF system prompt, all 32 values)
MAIN_PATHS = {
    "llama-3.1-8b":        f"{REPO}/value_faking_test/analysis/hf__meta-llama_Meta-Llama-3.1-8B-Instruct__v2_n4/analyzed_results_v2.csv",
    "mistral-small-2409":  f"{REPO}/value_faking_test/analysis/hf__mistralai_Mistral-Small-Instruct-2409__v2_n4/analyzed_results_v2.csv",
    "gpt-4o-mini":         f"{REPO}/value_faking_test/analysis/openai__gpt-4o-mini__v2_n4/analyzed_results_v2.csv",
    "deepseek-r1-0528":    f"{REPO}/value_faking_test/analysis/openrouter__deepseek_deepseek-r1-0528__v2_n4/analyzed_results_v2.csv",
    "deepseek-r1-distill": f"{REPO}/value_faking_test/analysis/openrouter__deepseek_deepseek-r1-distill-llama-70b__v2_n4/analyzed_results_v2.csv",
    "gemma-3-27b":         f"{REPO}/value_faking_test/analysis/openrouter__google_gemma-3-27b-it__v2_n4/analyzed_results_v2.csv",
    "llama-3.1-70b":       f"{REPO}/value_faking_test/analysis/openrouter__meta-llama_llama-3.1-70b-instruct__v2_n4/analyzed_results_v2.csv",
    "llama-3.3-70b":       f"{REPO}/value_faking_test/analysis/openrouter__meta-llama_llama-3.3-70b-instruct__v2_n4/analyzed_results_v2.csv",
    "ministral-14b":       f"{REPO}/value_faking_test/analysis/openrouter__mistralai_ministral-14b-2512__v2_n4/analyzed_results_v2.csv",
    "mistral-large-2512":  f"{REPO}/value_faking_test/analysis/openrouter__mistralai_mistral-large-2512__v2_n4/analyzed_results_v2.csv",
}

# name -> value_results.csv (bare, no system prompt). All 10 models now complete (32/32 values each).
BARE_PATHS = {
    "llama-3.1-8b":        f"{REPO}/test_ablations/outputs/hf__meta-llama_Meta-Llama-3.1-8B-Instruct__v2_n4_bare/value_results.csv",
    "mistral-small-2409":  f"{REPO}/test_ablations/outputs/hf__mistralai_Mistral-Small-Instruct-2409__v2_n4_bare/value_results.csv",
    "gpt-4o-mini":         f"{REPO}/test_ablations/outputs/openai__gpt-4o-mini__v2_n4_bare/value_results.csv",
    "deepseek-r1-0528":    f"{REPO}/test_ablations/outputs/openrouter__deepseek_deepseek-r1-0528__v2_n4_bare/value_results.csv",
    "deepseek-r1-distill": f"{REPO}/test_ablations/outputs/openrouter__deepseek_deepseek-r1-distill-llama-70b__v2_n4_bare/value_results.csv",
    "gemma-3-27b":         f"{REPO}/test_ablations/outputs/openrouter__google_gemma-3-27b-it__v2_n4_bare/value_results.csv",
    "llama-3.1-70b":       f"{REPO}/test_ablations/outputs/openrouter__meta-llama_llama-3.1-70b-instruct__v2_n4_bare/value_results.csv",
    "llama-3.3-70b":       f"{REPO}/test_ablations/outputs/openrouter__meta-llama_llama-3.3-70b-instruct__v2_n4_bare/value_results.csv",
    "ministral-14b":       f"{REPO}/test_ablations/outputs/openrouter__mistralai_ministral-14b-2512__v2_n4_bare/value_results.csv",
    "mistral-large-2512":  f"{REPO}/test_ablations/outputs/openrouter__mistralai_mistral-large-2512__v2_n4_bare/value_results.csv",
}


def available_bare_models() -> dict:
    return {k: v for k, v in BARE_PATHS.items() if os.path.exists(v)}


def available_main_models() -> dict:
    return {k: v for k, v in MAIN_PATHS.items() if os.path.exists(v)}


def load_main(model: str) -> pd.DataFrame:
    return pd.read_csv(MAIN_PATHS[model])


def load_bare(model: str) -> pd.DataFrame:
    return pd.read_csv(BARE_PATHS[model])


def value_gap_table(models: dict = None) -> pd.DataFrame:
    models = models or available_main_models()
    rows = []
    for model, path in models.items():
        df = pd.read_csv(path)
        g = df.groupby("value")["compliance_gap"].mean()
        g.name = model
        rows.append(g)
    return pd.concat(rows, axis=1).T  # rows=model, cols=value
