"""Bootstrapped compliance-gap table: subsample to the minimum N across values, then average value-level stats over iterations."""
import argparse
import numpy as np
import pandas as pd
from value_faking.paths import RUNS_DIR

RESULTS_CSV = f"{RUNS_DIR}/openrouter__meta-llama_llama-3.3-70b-instruct__v2/scenario_results.csv"
K = 1000  # number of subsample iterations


def compute(results_csv: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df = pd.read_csv(results_csv)
    total = len(df)

    pos = df[df["compliance_gap"] >= 0].copy()
    neg = df[df["compliance_gap"] < 0].copy()
    dropped = total - len(pos)
    print(f"Scenarios: {total} total | {dropped} negative dropped | {len(pos)} retained\n")

    counts = pos.groupby("value").size()
    min_n = counts.min()
    print(f"Scenarios per value after filter — min: {min_n}, max: {counts.max()}")
    print(f"Bootstrap: {k} iterations, subsampling {min_n} per value each time\n")

    values = sorted(pos["value"].unique())
    accum = {v: {"mon": [], "unmon": [], "gap": []} for v in values}

    rng = np.random.default_rng(42)
    for _ in range(k):
        for v in values:
            g = pos[pos["value"] == v]
            sample = g.sample(n=min_n, random_state=int(rng.integers(1 << 31)))
            accum[v]["mon"].append(sample["monitored_rate"].mean())
            accum[v]["unmon"].append(sample["unmonitored_rate"].mean())
            accum[v]["gap"].append(sample["compliance_gap"].mean())

    rows = []
    for v in values:
        rows.append({
            "value":            v,
            "n_available":      counts[v],
            "monitored_rate":   np.mean(accum[v]["mon"]),
            "unmonitored_rate": np.mean(accum[v]["unmon"]),
            "raw_gap":          np.mean(accum[v]["gap"]),
            "gap_std":          np.std(accum[v]["gap"]),
        })
    agg = pd.DataFrame(rows)
    agg = agg.sort_values("raw_gap", ascending=False).reset_index(drop=True)
    return agg, neg, min_n


def print_table(agg: pd.DataFrame, k: int, min_n: int):
    print(f"{'Value':<40} {'Mon':>6} {'Unmon':>6} {'Gap':>8} {'±std':>6} {'N avail':>8}")
    print("-" * 78)
    for _, r in agg.iterrows():
        print(
            f"{r['value']:<40} {r['monitored_rate']:>6.3f} {r['unmonitored_rate']:>6.3f}"
            f" {r['raw_gap']:>+8.3f} {r['gap_std']:>6.3f} {r['n_available']:>8.0f}"
        )
    print("-" * 78)
    print(
        f"{'Mean':<40} {agg['monitored_rate'].mean():>6.3f} {agg['unmonitored_rate'].mean():>6.3f}"
        f" {agg['raw_gap'].mean():>+8.3f} {agg['gap_std'].mean():>6.3f}"
    )
    print(f"\nBootstrap iterations: {k} | scenarios per iteration per value: {min_n}")


def plot(agg: pd.DataFrame, neg: pd.DataFrame, results_csv: str):
    import matplotlib.pyplot as plt

    model_tag = results_csv.rstrip("/").replace("\\", "/").split("/")[-2]

    # sort ascending so barh reads top-to-bottom as descending
    agg_plot = agg.sort_values("raw_gap", ascending=True).reset_index(drop=True)
    n = len(agg_plot)
    y = np.arange(n)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, max(8, n * 0.38)),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ── panel 1: raw gap bar chart ────────────────────────────────────────────
    ax = axes[0]
    gap_max = agg_plot["raw_gap"].max()
    colors = plt.cm.RdYlGn(agg_plot["raw_gap"].values / max(gap_max, 1e-6))
    ax.barh(
        y, agg_plot["raw_gap"],
        xerr=agg_plot["gap_std"],
        height=0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        error_kw={"elinewidth": 0.8, "ecolor": "#374151", "capsize": 2},
        zorder=3,
    )

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f3f4f6", zorder=0, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(agg_plot["value"], fontsize=8)
    ax.set_xlabel("Compliance gap  (monitored − unmonitored rate)", fontsize=9)
    ax.set_title(f"Bootstrap compliance gap — negative filtered\n{model_tag}", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.05, min(1.15, gap_max + 0.2))
    ax.axvline(0, color="#374151", linewidth=0.8, zorder=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)

    for i, (_, r) in enumerate(agg_plot.iterrows()):
        ax.text(r["raw_gap"] + 0.015, i, f"{r['raw_gap']:+.3f}", va="center", fontsize=7, color="#374151")

    ax2 = axes[1]

    all_values = sorted(agg["value"].unique())
    neg_counts = neg.groupby("value").size().reindex(all_values, fill_value=0)
    total_csv = pd.read_csv(results_csv)
    total_counts = total_csv.groupby("value").size().reindex(all_values, fill_value=0)
    pos_counts = total_counts - neg_counts

    value_order = list(agg_plot["value"])
    neg_ordered = [neg_counts.get(v, 0) for v in value_order]
    pos_ordered = [pos_counts.get(v, 0) for v in value_order]

    y2 = np.arange(len(value_order))
    ax2.barh(y2, pos_ordered, height=0.6, color="#6366f1", alpha=0.85, label="Retained (gap ≥ 0)", zorder=3)
    ax2.barh(y2, neg_ordered, height=0.6, left=pos_ordered, color="#ef4444", alpha=0.85, label="Dropped (gap < 0)", zorder=3)

    for i in range(len(value_order)):
        if i % 2 == 0:
            ax2.axhspan(i - 0.5, i + 0.5, color="#f3f4f6", zorder=0, alpha=0.7)
        n_neg = neg_ordered[i]
        if n_neg > 0:
            ax2.text(pos_ordered[i] + n_neg / 2, i, str(n_neg),
                     va="center", ha="center", fontsize=7, color="white", fontweight="bold")

    ax2.set_yticks(y2)
    ax2.set_yticklabels(value_order, fontsize=8)
    ax2.set_xlabel("Scenario count", fontsize=9)
    ax2.set_title(
        "Why we filter negative-gap scenarios\n"
        "Red = model more compliant when UNmonitored\n"
        "(inverted signal — invalid for AF measurement)",
        fontsize=9, fontweight="bold",
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=1)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    plt.tight_layout(pad=1.2)

    out_dir = "/".join(results_csv.replace("\\", "/").split("/")[:-1])
    out_path = f"{out_dir}/bootstrap_gap_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved → {out_path}")


def main(results_csv: str, k: int, save_csv: str | None, do_plot: bool):
    agg, neg, min_n = compute(results_csv, k)
    print_table(agg, k, min_n)

    if save_csv:
        agg.to_csv(save_csv, index=False)
        print(f"\nTable saved → {save_csv}")

    if do_plot:
        plot(agg, neg, results_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", default=RESULTS_CSV)
    parser.add_argument("--k", type=int, default=K, help="Number of bootstrap iterations")
    parser.add_argument("--save_csv", default=None, help="Path to write table CSV")
    parser.add_argument("--plot", action="store_true", help="Show two-panel diagnostic figure")
    args = parser.parse_args()
    main(args.results_csv, args.k, args.save_csv, args.plot)
