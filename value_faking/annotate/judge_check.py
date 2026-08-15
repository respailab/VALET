"""Second-judge re-annotation for BF3/BF4 (live version of bf4_reannotation_stub.py)."""
import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from value_faking.bf.common import REPO, load_main
from value_faking.paths import REPO

sys.path.insert(0, f"{REPO}/value_faking_test")
from analyze_layers import run_analysis  # noqa: E402


def stratified_sample(df: pd.DataFrame, min_per_cell: int, seed: int) -> pd.DataFrame:
    parts = []
    for model, g in df.groupby("model"):
        for v in (True, False):
            pool = g[g["verbalized"] == v]
            k = min(len(pool), min_per_cell)
            if k:
                parts.append(pool.sample(k, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def select_datapoints(sample: pd.DataFrame, subset_csv: str) -> pd.DataFrame:
    keep_cols = ["value", "topic", "scenario", "developer_compliant", "developer_noncompliant",
                 "monitored_responses", "unmonitored_responses", "monitored_choices",
                 "unmonitored_choices", "compliance_gap"]
    out = []
    for model, g in sample.groupby("model"):
        full = load_main(model)
        cols = [c for c in keep_cols if c in full.columns]
        merged = g[["model", "value", "scenario"]].merge(full[cols], on=["value", "scenario"], how="left")
        out.append(merged)
    combined = pd.concat(out, ignore_index=True)
    missing = combined["monitored_responses"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows failed to join on (value, scenario) - check for text mismatch")
    combined = combined.dropna(subset=["monitored_responses"]).reset_index(drop=True)
    # keep "model" in the CSV - scenario text is shared across models (same benchmark
    # scenarios evaluated per model), so (value, scenario) alone isn't a unique key and
    # dropping "model" here caused match() to cross-join rows from different models.
    combined.to_csv(subset_csv, index=False)
    return combined


def match(sample: pd.DataFrame, reannotated: pd.DataFrame, label: str):
    merged = sample.merge(
        reannotated[["model", "value", "scenario", "verbalized2"]],
        on=["model", "value", "scenario"], how="inner"
    )
    kappa = cohen_kappa_score(merged["verbalized"], merged["verbalized2"])
    table = pd.crosstab(merged["verbalized"], merged["verbalized2"],
                         rownames=["pass1"], colnames=["pass2"])
    table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)
    print(f"\n=== {label} (n={len(merged)}) ===")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(table)
    return kappa, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-per-cell", type=int, default=5,
                         help="max rows sampled per (model, verbalized) cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-service", default="openrouter")
    parser.add_argument("--judge-model", default="qwen/qwen-2.5-72b-instruct",
                         help="different model + family from gpt-4o-mini, to avoid a same-family judge confound")
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--control-models", nargs="+", default=["llama-3.1-70b"],
                         help="one or more non-gpt-4o-mini models to report kappa for, as a control")
    parser.add_argument("--subset-csv", default=f"{REPO}/behavioral_analysis/judge_check_subset.csv")
    parser.add_argument("--output-csv", default=f"{REPO}/behavioral_analysis/judge_check_reannotated.csv")
    args = parser.parse_args()

    scen = pd.read_csv(f"{REPO}/behavioral_analysis/bf3_scenario_table.csv")

    sample = stratified_sample(scen, min_per_cell=args.min_per_cell, seed=args.seed)
    print(f"Sampled {len(sample)} scenarios across {sample['model'].nunique()} models")
    print(sample.groupby(["model", "verbalized"]).size())

    select_datapoints(sample, args.subset_csv)

    # run_analysis() prints its own per-row progress ("[i+1/n] ... L2=...") and
    # checkpoints to <output>_checkpoint.json every --checkpoint-every rows.
    run_analysis(
        input_csv=args.subset_csv,
        output_csv=args.output_csv,
        service=args.judge_service,
        model=args.judge_model,
        requests_per_second=args.requests_per_second,
        checkpoint_every=args.checkpoint_every,
    )

    reannotated = pd.read_csv(args.output_csv)
    reannotated["verbalized2"] = reannotated["monitored_reasoning_types"].apply(
        lambda cell: "training_signal" in json.loads(cell)
    )

    match(sample, reannotated, "overall")
    match(sample[sample["model"] == "gpt-4o-mini"], reannotated, "gpt-4o-mini")
    for control in args.control_models:
        subset = sample[sample["model"] == control]
        if len(subset):
            match(subset, reannotated, f"control: {control}")
        else:
            print(f"\nWARNING: no sampled rows for control model '{control}'")


if __name__ == "__main__":
    main()
