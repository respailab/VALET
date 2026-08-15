"""Sensitivity decomposition: decompose compliance into three components."""

import argparse
import sys

import pandas as pd
from value_faking.paths import ABLATION_DIR

MODELS = [
    {
        "label":   "Llama 3.3 70B",
        "normal":  f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__v2/value_results.csv",
        "no_sp":   f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__ablation_no_sp/value_results.csv",
    },
    {
        "label":   "Gemma 3 27B",
        "normal":  f"{ABLATION_DIR}/openrouter__google_gemma-3-27b-it__v2/value_results.csv",
        "no_sp":   f"{ABLATION_DIR}/openrouter__google_gemma-3-27b-it__v2-no_sp/value_results.csv",
    },
    {
        "label":   "Llama 3.1 8B",
        "normal":  f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.1-8b-instruct__v2/value_results.csv",
        "no_sp":   f"{ABLATION_DIR}/openrouter__meta-llama_llama-3.1-8b-instruct__v2-no_sp/value_results.csv",
    },
    {
        "label":   "GPT-4o-mini",
        "normal":  f"{ABLATION_DIR}/openai__gpt-4o-mini__v2/value_results.csv",
        "no_sp":   f"{ABLATION_DIR}/openai__gpt-4o-mini__v2-no_sp/value_results.csv",
    },
]


def load_model(cfg: dict) -> pd.DataFrame | None:
    try:
        norm = pd.read_csv(cfg["normal"]).set_index("value")
        nosp = pd.read_csv(cfg["no_sp"]).set_index("value")
    except FileNotFoundError as e:
        print(f"[WARN] {cfg['label']}: {e}", file=sys.stderr)
        return None

    shared = norm.index.intersection(nosp.index)
    df = pd.DataFrame(index=shared)
    df["B"] = nosp.loc[shared, "avg_monitored"]
    df["M"] = norm.loc[shared, "avg_monitored"]
    df["U"] = norm.loc[shared, "avg_unmonitored"]
    df["monitoring_sensitivity"] = df["M"] - df["U"]
    df["directive_sensitivity"]  = df["U"] - df["B"]
    df["total_sensitivity"]      = df["M"] - df["B"]
    df["model"] = cfg["label"]
    return df.dropna(subset=["B", "M", "U"])


def print_model_report(cfg: dict):
    df = load_model(cfg)
    if df is None:
        return None

    print(f"\n{'='*64}")
    print(f"  {cfg['label']}")
    print(f"{'='*64}")

    header = f"{'Value':<36} {'B':>6} {'M':>6} {'U':>6} {'M-U':>7} {'U-B':>7} {'M-B':>7}"
    print(header)
    print("-" * len(header))

    df_sorted = df.sort_values("monitoring_sensitivity", ascending=False)
    for v, row in df_sorted.iterrows():
        print(
            f"{v:<36} {row['B']:>6.3f} {row['M']:>6.3f} {row['U']:>6.3f}"
            f" {row['monitoring_sensitivity']:>+7.3f} {row['directive_sensitivity']:>+7.3f}"
            f" {row['total_sensitivity']:>+7.3f}"
        )

    means = df[["monitoring_sensitivity", "directive_sensitivity", "total_sensitivity"]].mean()
    print()
    print(f"  Mean M-U (monitoring sensitivity): {means['monitoring_sensitivity']:+.4f}")
    print(f"  Mean U-B (directive sensitivity):  {means['directive_sensitivity']:+.4f}")
    print(f"  Mean M-B (total sensitivity):      {means['total_sensitivity']:+.4f}")

    n_neg_mon = (df["monitoring_sensitivity"] < 0).sum()
    print(f"\n  Values with negative monitoring sensitivity (U > M): {n_neg_mon}/{len(df)}")
    if n_neg_mon > 0:
        neg = df[df["monitoring_sensitivity"] < 0].sort_values("monitoring_sensitivity")
        for v, row in neg.iterrows():
            print(f"    {v:<36} M-U={row['monitoring_sensitivity']:+.3f}")

    return df


def print_cross_model_summary(dfs: list[pd.DataFrame]):
    print(f"\n{'='*64}")
    print("  Cross-model sensitivity decomposition summary")
    print(f"{'='*64}")
    header = f"{'Model':<24} {'Mean M-U':>10} {'Mean U-B':>10} {'Mean M-B':>10} {'N(U-B<0)':>10}"
    print(header)
    print("-" * len(header))
    for df in dfs:
        if df is None:
            continue
        model = df["model"].iloc[0]
        m_u = df["monitoring_sensitivity"].mean()
        u_b = df["directive_sensitivity"].mean()
        m_b = df["total_sensitivity"].mean()
        n_neg = (df["directive_sensitivity"] < 0).sum()
        print(f"{model:<24} {m_u:>+10.4f} {u_b:>+10.4f} {m_b:>+10.4f} {n_neg:>10}")


def build_summary(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in dfs:
        if df is None:
            continue
        model = df["model"].iloc[0]
        rows.append({
            "model":     model,
            "mean_M_U":  round(df["monitoring_sensitivity"].mean(), 4),
            "mean_U_B":  round(df["directive_sensitivity"].mean(), 4),
            "mean_M_B":  round(df["total_sensitivity"].mean(), 4),
            "n_neg_U_B": int((df["directive_sensitivity"] < 0).sum()),  # U < B: SP suppresses below baseline
            "n_values":  len(df),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_csv",     default=None, help="Per-value CSV path")
    parser.add_argument("--save_summary", default=None, help="Cross-model summary CSV path")
    args = parser.parse_args()

    dfs = []
    for cfg in MODELS:
        df = print_model_report(cfg)
        dfs.append(df)

    print_cross_model_summary(dfs)

    if args.save_csv:
        valid = [df for df in dfs if df is not None]
        if valid:
            combined = pd.concat(valid, ignore_index=True)
            combined.to_csv(args.save_csv, index=False)
            print(f"\nCSV saved → {args.save_csv}")

    if args.save_summary:
        summary = build_summary(dfs)
        summary.to_csv(args.save_summary, index=False)
        print(f"Summary CSV saved → {args.save_summary}")


if __name__ == "__main__":
    main()
