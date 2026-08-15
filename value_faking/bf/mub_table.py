"""Plain M/U/B compliance-rate table, per value x model."""
import os
import pandas as pd

from value_faking.bf.common import REPO, BARE_PATHS, available_bare_models, available_main_models, load_main
from value_faking.paths import REPO

if __name__ == "__main__":
    import argparse as _argparse
    _argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0] if __doc__ else None,
        epilog="Takes no options yet; paths are resolved by value_faking.paths.",
    ).parse_args()


def load_bare_scenario_table(model: str) -> tuple[pd.DataFrame, str]:
    bare_dir = os.path.dirname(BARE_PATHS[model])
    scenario_csv = os.path.join(bare_dir, "scenario_results.csv")
    df = pd.read_csv(scenario_csv)
    df["B_rate"] = df["monitored_rate"]
    return df, "scenario_results (regex-extracted)"


def build_table() -> pd.DataFrame:
    bare_models = available_bare_models()
    if not bare_models:
        raise RuntimeError("No bare-condition result files found yet.")

    rows = []
    for model in bare_models:
        bare, source = load_bare_scenario_table(model)
        main = load_main(model)
        print(f"  {model:<22} B source: {source}")

        for value, g in main.groupby("value"):
            # dropna: a scenario's rate is NaN when all n=4 replicate responses for that
            # scenario+condition were unparseable choices - drop those scenarios rather
            # than letting a single bad one blank out the whole value's mean (np.mean
            # propagates NaN; pandas' default .mean() would have skipped it silently,
            # which is why this needs to be explicit).
            b = bare.loc[bare["value"] == value, "B_rate"].dropna().to_numpy()
            m = g["monitored_rate"].dropna().to_numpy()
            u = g["unmonitored_rate"].dropna().to_numpy()
            if len(b) == 0 or len(m) == 0 or len(u) == 0:
                continue
            rows.append({
                "model": model,
                "value": value,
                "B_mean": b.mean(), "B_n": len(b),
                "M_mean": m.mean(), "M_n": len(m),
                "U_mean": u.mean(), "U_n": len(u),
                "M_minus_U": m.mean() - u.mean(),
                "U_minus_B": u.mean() - b.mean(),
                "M_minus_B": m.mean() - b.mean(),
                "B_source": source,
            })
    return pd.DataFrame(rows)


def main():
    df = build_table()
    missing = sorted(set(available_main_models()) - set(df["model"].unique()))

    print(f"\nBuilt {len(df)} value x model rows across {df['model'].nunique()} models "
          f"x {df['value'].nunique()} values")
    if missing:
        print(f"  (missing bare data for: {', '.join(missing)})")

    print("\nPer-model mean B / M / U (averaged across all 32 values):")
    summary = df.groupby("model")[["B_mean", "M_mean", "U_mean", "M_minus_U", "U_minus_B"]].mean()
    summary = summary.sort_values("M_minus_U", ascending=False)
    print(summary.round(3).to_string())

    out_path = f"{REPO}/behavioral_analysis/mub_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full value x model table -> {out_path}")

    per_model_dir = f"{REPO}/behavioral_analysis/mub_per_model"
    os.makedirs(per_model_dir, exist_ok=True)
    model_order = summary.index.tolist()  # same M_minus_U-descending order as the aggregate table
    cols = ["value", "B_mean", "M_mean", "U_mean", "M_minus_U", "U_minus_B"]

    for model in model_order:
        g = df[df["model"] == model].sort_values("M_minus_U", ascending=False)
        print("\n" + "=" * 78)
        print(f"{model}  (32 values, sorted by M-U)")
        print("=" * 78)
        print(g[cols].round(3).to_string(index=False))

        g[cols].to_csv(f"{per_model_dir}/{model}.csv", index=False)

    print(f"\nSaved per-model per-value tables -> {per_model_dir}/<model>.csv")


if __name__ == "__main__":
    main()
