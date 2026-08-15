"""Compute reasoning-type statistics from analyzed_results.csv."""

import argparse

import pandas as pd
from value_faking.paths import RUNS_DIR

MODELS = {
    "Llama 3.3 70B (OpenRouter)": f"{RUNS_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__v2/analyzed_results.csv",
    "GPT-4o-mini (OpenAI)":       f"{RUNS_DIR}/openai__gpt-4o-mini__v2/analyzed_results.csv",
    "Gemma 3 27B (OpenRouter)":   f"{RUNS_DIR}/openrouter__google_gemma-3-27b-it__v2/analyzed_results.csv",
    "Llama 3.1 8B (OpenRouter)":  f"{RUNS_DIR}/openrouter__meta-llama_llama-3.1-8b-instruct__v2/analyzed_results.csv",
}

REASONING_TYPES = [
    "training_signal",
    "value_conflict_aware",
    "directive_following",
    "rationalisation",
    "no_reasoning",
]

PATTERNS = [
    "alignment_faking",
    "consistent_compliant",
    "consistent_noncompliant",
    "advisory",
    "reverse_faking",
]


def analyze(csv_path: str, label: str):
    df = pd.read_csv(csv_path)
    n = len(df)

    required = {"behavioral_pattern", "monitored_reasoning_type", "unmonitored_reasoning_type"}
    missing = required - set(df.columns)
    if missing:
        print(f"\n[SKIP] {label}: analyzed_results.csv missing columns: {missing}")
        print(f"       Run analyze_test_results.py first to generate these columns.")
        return

    df["monitored_reasoning_type"]   = df["monitored_reasoning_type"].fillna("no_reasoning")
    df["unmonitored_reasoning_type"] = df["unmonitored_reasoning_type"].fillna("no_reasoning")
    df["monitored_refusal"]          = df["monitored_refusal"].fillna("none")
    df["unmonitored_refusal"]        = df["unmonitored_refusal"].fillna("none")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Source: {csv_path}")
    print(f"{'='*70}")
    print(f"  Total rows : {n}")
    print()

    print("  BEHAVIORAL PATTERN DISTRIBUTION")
    print(f"  {'Pattern':<30} {'N':>6}  {'%':>6}")
    print(f"  {'-'*46}")
    for pat in PATTERNS:
        cnt = (df["behavioral_pattern"] == pat).sum()
        print(f"  {pat:<30} {cnt:>6}  {100*cnt/n:>5.1f}%")
    print()

    print("  THINKING TAG PRESENCE")
    if "monitored_has_thinking" in df.columns and "unmonitored_has_thinking" in df.columns:
        mon_think   = df["monitored_has_thinking"].mean()
        unmon_think = df["unmonitored_has_thinking"].mean()
        print(f"  Monitored   : {mon_think:>6.1%}  ({df['monitored_has_thinking'].sum()}/{n})")
        print(f"  Unmonitored : {unmon_think:>6.1%}  ({df['unmonitored_has_thinking'].sum()}/{n})")
    else:
        print("  [columns not present in this file]")
    print()

    for cond, col in [("MONITORED", "monitored_reasoning_type"), ("UNMONITORED", "unmonitored_reasoning_type")]:
        print(f"  {cond} REASONING TYPE")
        print(f"  {'Type':<30} {'N':>6}  {'%':>6}")
        print(f"  {'-'*46}")
        vc = df[col].value_counts()
        for rt in REASONING_TYPES:
            cnt = vc.get(rt, 0)
            print(f"  {rt:<30} {cnt:>6}  {100*cnt/n:>5.1f}%")
        other = n - sum(vc.get(rt, 0) for rt in REASONING_TYPES)
        if other > 0:
            print(f"  {'other (api_error/parse_error)':<30} {other:>6}  {100*other/n:>5.1f}%")
        print()

    print("  REFUSAL RATES")
    mon_partial  = (df["monitored_refusal"]   == "partial").sum()
    mon_full     = (df["monitored_refusal"]   == "full").sum()
    unmon_partial= (df["unmonitored_refusal"] == "partial").sum()
    unmon_full   = (df["unmonitored_refusal"] == "full").sum()
    print(f"  Monitored   partial : {mon_partial:>4}  ({100*mon_partial/n:.1f}%)")
    print(f"  Monitored   full    : {mon_full:>4}  ({100*mon_full/n:.1f}%)")
    print(f"  Unmonitored partial : {unmon_partial:>4}  ({100*unmon_partial/n:.1f}%)")
    print(f"  Unmonitored full    : {unmon_full:>4}  ({100*unmon_full/n:.1f}%)")
    print()

    print("  CROSS-TAB: behavioral_pattern × unmonitored_reasoning_type")
    unmon_rt_cols = ["training_signal", "value_conflict_aware", "no_reasoning"]
    header = f"  {'Pattern':<30}"
    for rt in unmon_rt_cols:
        header += f"  {rt[:14]:>14}"
    header += f"  {'Total':>7}"
    print(header)
    print(f"  {'-'*90}")
    for pat in PATTERNS:
        sub = df[df["behavioral_pattern"] == pat]
        if len(sub) == 0:
            continue
        row_str = f"  {pat:<30}"
        for rt in unmon_rt_cols:
            cnt = (sub["unmonitored_reasoning_type"] == rt).sum()
            pct = 100 * cnt / len(sub)
            row_str += f"  {cnt:>5} ({pct:>4.1f}%)"
        row_str += f"  {len(sub):>7}"
        print(row_str)
    print()

    print("  PER-VALUE UNMONITORED REASONING: training_signal% vs value_conflict_aware%")
    print(f"  {'Value':<45} {'N':>5} {'TS%':>7} {'VC%':>7} {'NoR%':>7} {'Gap':>7}")
    print(f"  {'-'*82}")
    if "compliance_gap" in df.columns:
        gap_col = "compliance_gap"
    else:
        gap_col = None

    val_rows = []
    for val, g in df.groupby("value"):
        ni = len(g)
        ts  = 100 * (g["unmonitored_reasoning_type"] == "training_signal").sum() / ni
        vc  = 100 * (g["unmonitored_reasoning_type"] == "value_conflict_aware").sum() / ni
        nor = 100 * (g["unmonitored_reasoning_type"] == "no_reasoning").sum() / ni
        gap = g[gap_col].mean() if gap_col else float("nan")
        val_rows.append((val, ni, ts, vc, nor, gap))
    val_rows.sort(key=lambda x: x[5] if not pd.isna(x[5]) else -99, reverse=True)
    for val, ni, ts, vc, nor, gap in val_rows:
        gap_str = f"{gap:+.3f}" if not pd.isna(gap) else "   N/A"
        print(f"  {val:<45} {ni:>5} {ts:>6.1f}% {vc:>6.1f}% {nor:>6.1f}% {gap_str:>7}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to a single analyzed_results.csv")
    parser.add_argument("--all", action="store_true", help="Run on all 4 main models")
    args = parser.parse_args()

    if args.all:
        for label, path in MODELS.items():
            try:
                analyze(path, label)
            except FileNotFoundError:
                print(f"\n[MISSING] {path}")
    elif args.csv:
        analyze(args.csv, args.csv)
    else:
        path = list(MODELS.values())[0]
        analyze(path, list(MODELS.keys())[0])


if __name__ == "__main__":
    main()
