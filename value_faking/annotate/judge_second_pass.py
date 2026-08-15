"""Second-judge re-annotation of a per-value sample, for a small set of models."""
import argparse
import json
import os
import sys

import pandas as pd

from value_faking.bf.common import REPO, MAIN_PATHS, load_main
from value_faking.paths import REPO

sys.path.insert(0, f"{REPO}/value_faking_test")
from analyze_layers import run_analysis  # noqa: E402

# columns run_analysis() reads, plus the ones its _samples.csv writer copies through
INPUT_COLS = ["value", "topic", "scenario", "developer_compliant", "developer_noncompliant",
              "monitored_responses", "unmonitored_responses",
              "monitored_choices", "unmonitored_choices", "compliance_gap"]


def truncate_samples(df: pd.DataFrame, n_keep: int) -> pd.DataFrame:
    df = df.copy()
    for cond in ("monitored", "unmonitored"):
        for col in (f"{cond}_responses", f"{cond}_choices"):
            df[col] = df[col].apply(lambda cell: json.dumps(json.loads(cell)[:n_keep]))
    return df


def sample_per_value(model: str, per_value: int, seed: int) -> pd.DataFrame:
    full = load_main(model)
    missing = [c for c in INPUT_COLS if c not in full.columns]
    if missing:
        raise KeyError(f"{model}: analyzed_results_v2.csv missing {missing}")

    parts = []
    for value, g in full.groupby("value"):
        k = min(len(g), per_value)
        parts.append(g.sample(k, random_state=seed))
    # keep only input columns: the pass-1 annotation columns must not ride along,
    # run_analysis() would overwrite them and the pass-1 labels would be lost
    return pd.concat(parts, ignore_index=True)[INPUT_COLS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mistral-large-2512", "ministral-14b"],
                        help=f"model keys from common.MAIN_PATHS: {sorted(MAIN_PATHS)}")
    parser.add_argument("--per-value", type=int, default=10,
                        help="scenarios sampled per (model, value); 32 values per model")
    parser.add_argument("--samples-per-condition", type=int, default=1, choices=[1, 2, 3, 4],
                        help="how many of the 4 generations per condition to re-judge "
                             "(default 1: agreement is measured per response, so one "
                             "generation per condition is enough and costs 4x less)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-service", default="openrouter")
    parser.add_argument("--judge-model", required=True,
                        help="OpenRouter slug for the pass-2 judge. Must be a stronger model "
                             "than gpt-4o-mini and a different family; qwen-2.5-72b was tried "
                             "in judge_check_run.log and over-labels training_signal (74%%) "
                             "with a 0%% divergence rate — do not reuse it.")
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--outdir", default=f"{REPO}/behavioral_analysis/judge2")
    parser.add_argument("--subset-from", default=None, metavar="DIR",
                        help="reuse <DIR>/<model>_subset.csv instead of sampling afresh. Use when "
                             "judging the same rows with a second judge model, so the judges are "
                             "compared on identical inputs rather than on equal-seed redraws.")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the subset CSVs and print the call budget, make no API calls")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for model in args.models:
        if model not in MAIN_PATHS:
            print(f"SKIP unknown model '{model}'", file=sys.stderr)
            continue

        subset_csv = os.path.join(args.outdir, f"{model}_subset.csv")
        output_csv = os.path.join(args.outdir, f"{model}_judge2.csv")

        if os.path.exists(output_csv):
            # run_analysis() refuses to overwrite; say so here rather than after sampling
            print(f"SKIP {model}: {output_csv} already exists (delete to re-run)")
            continue

        if args.subset_from:
            src = os.path.join(args.subset_from, f"{model}_subset.csv")
            if not os.path.exists(src):
                print(f"SKIP {model}: no subset to reuse at {src}", file=sys.stderr)
                continue
            sample = pd.read_csv(src)
            n_kept = len(json.loads(sample.iloc[0]["monitored_responses"]))
            print(f"{model}: reusing {len(sample)} scenarios from {src} "
                  f"({n_kept} sample(s) per condition)")
        else:
            sample = sample_per_value(model, args.per_value, args.seed)
            sample = truncate_samples(sample, args.samples_per_condition)
            n_kept = args.samples_per_condition
        sample.to_csv(subset_csv, index=False)
        calls = len(sample) * 2 * n_kept * 2
        print(f"{model}: {len(sample)} scenarios across {sample['value'].nunique()} values "
              f"x {n_kept} sample(s) x 2 conditions x 2 layers "
              f"→ {subset_csv}  (~{calls:,} judge calls)")

        if args.dry_run:
            continue

        # run_analysis() prints per-row progress and checkpoints to
        # <output>_judge2_checkpoint.json every --checkpoint-every rows, so an
        # interrupted run resumes instead of re-paying for completed rows.
        run_analysis(
            input_csv=subset_csv,
            output_csv=output_csv,
            service=args.judge_service,
            model=args.judge_model,
            requests_per_second=args.requests_per_second,
            max_workers=args.max_workers,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"{model}: pass-2 annotations → {output_csv}")
        print(f"{model}: per-sample rows    → {output_csv.replace('.csv', '_samples.csv')}")

    print("\nNext: python behavioral_analysis/judge_agreement.py "
          f"--models {' '.join(args.models)} --pass2-dir {args.outdir}")


if __name__ == "__main__":
    main()
