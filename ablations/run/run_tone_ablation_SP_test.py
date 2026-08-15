"""BF5 Part 2 — tone ablation."""

import argparse

from value_faking.run.run_test import run_test, SEEDS_CSV, DEV_DIR, OUTPUT_DIR, SERVICE, MODEL, N, COMPANY, REQUESTS_PER_SECOND
from value_faking.paths import DEV_DIR, SEEDS_CSV

ARMS = ["formal", "casual"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_csv",  default=SEEDS_CSV)
    parser.add_argument("--dev_dir",    default=DEV_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--service",    default=SERVICE)
    parser.add_argument("--model",      default=MODEL)
    parser.add_argument("--n",          type=int, default=N)
    parser.add_argument("--company",             default=COMPANY)
    parser.add_argument("--requests_per_second", type=float, default=REQUESTS_PER_SECOND,
                        help="Rate limit in requests/sec (e.g. 1.0 for Groq free tier)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag appended to output dir, shared across both arms "
                             "(e.g. 'bf5_part2') so runs group together for later comparison")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS,
                        help="Subset of arms to run (default: both)")
    parser.add_argument("--value", nargs="+", default=None,
                        help="Restrict to one or more values (exact match on the 'value' column), "
                             "e.g. --value Seriousness  or  --value Tolerance Order Complexity")
    args = parser.parse_args()

    for arm in args.arms:
        print(f"\n{'#'*70}\n# BF5 Part 2 — tone arm: {arm}\n{'#'*70}")
        run_test(
            seeds_csv=args.seeds_csv,
            dev_dir=args.dev_dir,
            output_dir=args.output_dir,
            service=args.service,
            model=args.model,
            n=args.n,
            company=args.company,
            requests_per_second=args.requests_per_second,
            tag=args.tag,
            swap_options=False,
            style=arm,
            value_filter=args.value,
        )
