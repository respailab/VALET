"""Bare-rate analysis: compliance rate with NO system prompt (no developer directive, no oversight framing)."""

import argparse
import sys

import pandas as pd
from value_faking.paths import ABLATION_DIR

# no-SP runs: avg_monitored = bare compliance rate (avg_unmonitored is NaN in these runs)
NO_SP_CONFIGS = {
    "llama70b": {
        "label": "Llama 3.3 70B",
        "path":  f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__ablation_no_sp/value_results.csv",
    },
    "gemma27b": {
        "label": "Gemma 3 27B",
        "path":  f"{ABLATION_DIR}/openrouter__google_gemma-3-27b-it__v2-no_sp/value_results.csv",
    },
    "llama8b": {
        "label": "Llama 3.1 8B",
        "path":  f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.1-8b-instruct__v2-no_sp/value_results.csv",
    },
    "gpt4omini": {
        "label": "GPT-4o-mini",
        "path":  f"{ABLATION_DIR}/openai__gpt-4o-mini__v2-no_sp/value_results.csv",
    },
}


def load_bare_rates() -> pd.DataFrame:
    frames = {}
    for key, cfg in NO_SP_CONFIGS.items():
        try:
            df = pd.read_csv(cfg["path"]).set_index("value")
            frames[cfg["label"]] = df["avg_monitored"]
        except FileNotFoundError:
            print(f"[WARN] not found: {cfg['path']}", file=sys.stderr)
    if not frames:
        raise RuntimeError("No bare-rate files found.")
    matrix = pd.DataFrame(frames)
    return matrix.sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_csv", default=None)
    args = parser.parse_args()

    matrix = load_bare_rates()
    models = list(matrix.columns)

    print("Bare compliance rates by value (no system prompt)\n")
    print("(Rate = fraction of scenarios where model chose developer_compliant option with NO SP)\n")

    col_w = 26
    header = f"{'Value':<36}" + "".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print("-" * len(header))
    for value, row in matrix.iterrows():
        vals = "".join(
            f"{'N/A':>{col_w}}" if pd.isna(row[m]) else f"{row[m]:>{col_w}.3f}"
            for m in models
        )
        print(f"{value:<36}{vals}")

    print()
    means = matrix.mean()
    mean_row = f"{'MEAN':<36}" + "".join(f"{means[m]:>{col_w}.3f}" for m in models)
    print(mean_row)

    # values with strong natural compliance (bare > 0.6) — little tension to exploit
    print("\n--- High bare rate (≥0.60): model already leans compliant without SP ---")
    for value, row in matrix.iterrows():
        flagged = [m for m in models if not pd.isna(row[m]) and row[m] >= 0.60]
        if flagged:
            rates = {m: f"{row[m]:.2f}" for m in flagged}
            print(f"  {value:<36} {rates}")

    # values with low bare rate (< 0.30) — strong natural value pull
    print("\n--- Low bare rate (<0.30): strong natural preference for the value ---")
    for value, row in matrix.iterrows():
        flagged = [m for m in models if not pd.isna(row[m]) and row[m] < 0.30]
        if flagged:
            rates = {m: f"{row[m]:.2f}" for m in flagged}
            print(f"  {value:<36} {rates}")

    if args.save_csv:
        out = matrix.reset_index().rename(columns={"index": "value"})
        out.to_csv(args.save_csv, index=False)
        print(f"\nCSV saved → {args.save_csv}")


if __name__ == "__main__":
    main()
