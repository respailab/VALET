"""BF3 + BF4 - Can you trust the trace?"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

from value_faking.bf.common import REPO, available_main_models, load_main
from value_faking.paths import REPO

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()


def scenario_table(model: str) -> pd.DataFrame:
    df = load_main(model)
    # monitored_reasoning_types is a JSON list (per-response); majority-vote column
    # doesn't exist in this schema (that's analyze_test_results.py's output) - here
    # (analyze_layers.py output) we derive "verbalized" as: any monitored response
    # for this scenario was classified training_signal.
    def any_training_signal(cell):
        try:
            types = json.loads(cell)
        except Exception:
            return False
        return "training_signal" in types

    df = df.copy()
    df["verbalized"] = df["monitored_reasoning_types"].apply(any_training_signal)
    df["model"] = model
    return df[["model", "value", "scenario", "compliance_gap", "verbalized"]]


def bf3_contingency(df: pd.DataFrame) -> dict:
    median_gap = df["compliance_gap"].median()
    high_gap = df["compliance_gap"] > median_gap

    table = pd.crosstab(high_gap, df["verbalized"])
    table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)

    odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())

    try:
        t2 = Table2x2(table.to_numpy())
        or_est = t2.oddsratio
        or_lo, or_hi = t2.oddsratio_confint()
    except Exception:
        or_est, or_lo, or_hi = odds_ratio, np.nan, np.nan

    return {
        "median_gap": median_gap,
        "table": table,
        "odds_ratio": or_est,
        "or_ci_lo": or_lo,
        "or_ci_hi": or_hi,
        "fisher_p": fisher_p,
        "n": len(df),
    }


def main():
    models = available_main_models()
    all_scen = pd.concat([scenario_table(m) for m in models], ignore_index=True)

    print("=" * 78)
    print("BF3 - per-model contingency: gap-above-median x verbalized (odds ratio)")
    print("=" * 78)
    bf3_rows = []
    for model, g in all_scen.groupby("model"):
        res = bf3_contingency(g)
        print(f"\n  {model}  (n={res['n']}, median_gap={res['median_gap']:.3f})")
        print(res["table"].rename(index={False: "gap<=median", True: "gap>median"},
                                   columns={False: "not_verbalized", True: "verbalized"}))
        print(f"    odds_ratio={res['odds_ratio']:.3f}  95% CI=[{res['or_ci_lo']:.3f}, {res['or_ci_hi']:.3f}]"
              f"  fisher_p={res['fisher_p']:.4f}")
        bf3_rows.append({"model": model, **{k: v for k, v in res.items() if k != "table"}})

    print("\n" + "=" * 78)
    print("BF3 - pooled contingency across all models")
    print("=" * 78)
    res = bf3_contingency(all_scen)
    print(res["table"].rename(index={False: "gap<=median", True: "gap>median"},
                               columns={False: "not_verbalized", True: "verbalized"}))
    print(f"  odds_ratio={res['odds_ratio']:.3f}  95% CI=[{res['or_ci_lo']:.3f}, {res['or_ci_hi']:.3f}]"
          f"  fisher_p={res['fisher_p']:.4f}")

    print("\n" + "=" * 78)
    print("BF4 - model grain: mean compliance_gap vs verbalization rate")
    print("=" * 78)
    bf4 = all_scen.groupby("model").agg(
        mean_gap=("compliance_gap", "mean"),
        verbalization_rate=("verbalized", "mean"),
        n_scenarios=("scenario", "count"),
    ).reset_index()
    bf4 = bf4.sort_values("mean_gap", ascending=False)
    print(bf4.to_string(index=False))

    # flag the high-gap / low-verbalization corner: gap in top half, verbalization in bottom half
    gap_med = bf4["mean_gap"].median()
    verb_med = bf4["verbalization_rate"].median()
    corner = bf4[(bf4["mean_gap"] > gap_med) & (bf4["verbalization_rate"] < verb_med)]
    print("\n  high-gap / low-verbalization corner (silent-faking candidates):")
    if len(corner):
        print(corner.to_string(index=False))
    else:
        print("  (none - every high-gap model also verbalizes above the median)")

    pd.DataFrame(bf3_rows).to_csv(f"{REPO}/behavioral_analysis/bf3_per_model.csv", index=False)
    bf4.to_csv(f"{REPO}/behavioral_analysis/bf4_model_grain.csv", index=False)
    all_scen.to_csv(f"{REPO}/behavioral_analysis/bf3_scenario_table.csv", index=False)
    print(f"\nSaved -> behavioral_analysis/bf3_per_model.csv, bf4_model_grain.csv, bf3_scenario_table.csv")


if __name__ == "__main__":
    main()
