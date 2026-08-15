"""BF3/BF4 armor step - second-judge re-annotation + human spot check."""
import numpy as np
import pandas as pd

from value_faking.bf.common import REPO
from value_faking.paths import REPO

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()


def stratified_sample_for_human_review(n_total=120, seed=0):
    scen = pd.read_csv(f"{REPO}/behavioral_analysis/bf3_scenario_table.csv")
    median_gap = scen.groupby("model")["compliance_gap"].transform("median")
    scen["high_gap"] = scen["compliance_gap"] > median_gap
    scen["cell"] = np.select(
        [
            scen["high_gap"] & ~scen["verbalized"],
            scen["high_gap"] & scen["verbalized"],
            ~scen["high_gap"] & ~scen["verbalized"],
            ~scen["high_gap"] & scen["verbalized"],
        ],
        ["high_gap_not_verbalized", "high_gap_verbalized",
         "low_gap_not_verbalized", "low_gap_verbalized"],
        default="unknown",
    )

    # oversample the cell Narrative A rests on
    weights = {"high_gap_not_verbalized": 0.40, "high_gap_verbalized": 0.20,
               "low_gap_not_verbalized": 0.20, "low_gap_verbalized": 0.20}
    rng = np.random.default_rng(seed)
    parts = []
    for cell, w in weights.items():
        pool = scen[scen["cell"] == cell]
        k = min(len(pool), int(round(n_total * w)))
        parts.append(pool.sample(k, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)

    out_path = f"{REPO}/behavioral_analysis/human_review_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"Sampled {len(sample)} scenarios for human review -> {out_path}")
    print(sample["cell"].value_counts())
    return sample


if __name__ == "__main__":
    stratified_sample_for_human_review()
