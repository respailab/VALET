"""Heterogeneity tests on the compliance-gap matrix (analysis 7a-7b)."""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from value_faking.paths import REPO
BOOT_NPZ = os.path.join(REPO, "results", "bootstrap", "boot_cells.npz")
OUT_DIR = os.path.join(REPO, "results", "heterogeneity")

# models whose swap control classifies a majority of values REAL (\S F.3).
# The strongest form of the heterogeneity claim is restricted to these.
SWAP_REAL_MAJORITY = ["Llama 3.3 70B Instruct", "Mistral Small Instruct 2409"]
SWAP_TESTED = {
    "Llama 3.3 70B Instruct": 22, "Gemma 3 27B": 7,
    "Llama 3.1 8B Instruct": 1, "Mistral Small Instruct 2409": 19,
}


def decompose(G, noise_var):
    n_m, n_v = G.shape
    grand = G.mean()
    row = G.mean(axis=1)
    col = G.mean(axis=0)

    ss_model = n_v * ((row - grand) ** 2).sum()
    ss_value = n_m * ((col - grand) ** 2).sum()
    resid = G - row[:, None] - col[None, :] + grand
    ss_resid = (resid ** 2).sum()

    ms_model = ss_model / (n_m - 1)
    ms_value = ss_value / (n_v - 1)
    ms_resid = ss_resid / ((n_m - 1) * (n_v - 1))

    # expected mean squares for a random-effects layout
    var_model = max((ms_model - ms_resid) / n_v, 0.0)
    var_value = max((ms_value - ms_resid) / n_m, 0.0)
    var_inter = max(ms_resid - noise_var, 0.0)

    structural = var_model + var_value + var_inter
    return {
        "var_model": var_model,
        "var_value": var_value,
        "var_interaction": var_inter,
        "var_noise": noise_var,
        "ms_resid": ms_resid,
        "share_model": var_model / structural if structural else np.nan,
        "share_value": var_value / structural if structural else np.nan,
        "share_interaction": var_inter / structural if structural else np.nan,
        "noise_share_of_resid": noise_var / ms_resid if ms_resid else np.nan,
        "interaction_over_value": var_inter / var_value if var_value else np.inf,
    }


# ── 7b: permutation test on the Spearman matrix ───────────────────────────────

def mean_pairwise_rho(G, pairs):
    return float(np.mean([stats.spearmanr(G[i], G[j]).statistic for i, j in pairs]))


def permutation_test(G, pairs, n_perm, rng):
    observed = mean_pairwise_rho(G, pairs)
    n_v = G.shape[1]
    null = np.empty(n_perm)
    for b in range(n_perm):
        Gp = np.array([row[rng.permutation(n_v)] for row in G])
        null[b] = mean_pairwise_rho(Gp, pairs)
    p = (1.0 + np.sum(null >= observed)) / (1.0 + n_perm)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_lo": float(np.percentile(null, 2.5)),
        "null_hi": float(np.percentile(null, 97.5)),
        "p_one_sided": float(p),
        "n_pairs": len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    z = np.load(BOOT_NPZ, allow_pickle=True)
    models = [str(m) for m in z["models"]]
    values = [str(v) for v in z["values"]]
    boot_gap = z["boot_gap"]                      # (n_models, n_values, n_boot)
    G = z["point_M"] - z["point_U"]               # difference of cell means
    n_m, n_v, n_b = boot_gap.shape
    print(f"gap matrix: {n_m} models x {n_v} values, {n_b} bootstrap draws")

    # per-cell sampling variance, straight from the Hour-1 bootstrap
    cell_noise = np.nanvar(boot_gap, axis=2, ddof=1)
    noise_var = float(np.nanmean(cell_noise))
    print(f"mean per-cell bootstrap variance: {noise_var:.5f} "
          f"(sd {np.sqrt(noise_var):.3f})")

    point = decompose(G, noise_var)

    # CIs by re-running the decomposition on each resampled gap matrix. A
    # resampled matrix carries one extra unit of sampling noise relative to the
    # observed one, so the noise term doubles for those fits.
    boot_rows = []
    for b in range(n_b):
        Gb = boot_gap[:, :, b]
        if np.isnan(Gb).any():
            continue
        boot_rows.append(decompose(Gb, 2.0 * noise_var))
    boot_df = pd.DataFrame(boot_rows)

    rows = []
    for k in ["var_model", "var_value", "var_interaction",
              "share_model", "share_value", "share_interaction",
              "interaction_over_value"]:
        rows.append({
            "quantity": k, "estimate": point[k],
            "ci_lo": float(np.percentile(boot_df[k], 2.5)),
            "ci_hi": float(np.percentile(boot_df[k], 97.5)),
        })
    var_df = pd.DataFrame(rows)
    var_df.to_csv(os.path.join(args.outdir, "variance_decomposition.csv"), index=False)

    print("\n── 7a: variance decomposition of G[model, value] ──")
    print(f"  residual MS                {point['ms_resid']:.5f}")
    print(f"  of which measurement noise {point['var_noise']:.5f} "
          f"({100*point['noise_share_of_resid']:.1f}% of residual)")
    for k, lab in [("model", "model main effect"), ("value", "value main effect"),
                   ("interaction", "model x value interaction")]:
        r = var_df[var_df.quantity == f"share_{k}"].iloc[0]
        v = point[f"var_{k}"]
        print(f"  {lab:26s} var {v:.5f}   share {100*r.estimate:5.1f}%  "
              f"[{100*r.ci_lo:.1f}, {100*r.ci_hi:.1f}]")
    r = var_df[var_df.quantity == "interaction_over_value"].iloc[0]
    print(f"  interaction / value ratio  {r.estimate:.2f}  [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")

    idx = {m: i for i, m in enumerate(models)}
    gap_mean = G.mean(axis=1)
    high = [m for m in models if gap_mean[idx[m]] >= 0.20]
    weak = ["Llama 3.1 8B Instruct", "Gemma 3 27B"]
    others = [m for m in models if m not in weak]

    def allpairs(group):
        ii = [idx[m] for m in group]
        return [(a, b) for k, a in enumerate(ii) for b in ii[k + 1:]]

    groups = {
        "all 10 models": allpairs(models),
        "high-gap cluster (5)": allpairs(high),
        "Llama-3.1-8B + Gemma-3-27B pairings": (
            [(idx[w], idx[o]) for w in weak for o in others] + [(idx[weak[0]], idx[weak[1]])]
        ),
        "Llama-3.1-8B vs the other nine": [(idx[weak[0]], idx[m]) for m in models if m != weak[0]],
        "Gemma-3-27B vs the other nine": [(idx[weak[1]], idx[m]) for m in models if m != weak[1]],
        "Llama-3.1-8B vs Gemma-3-27B": [(idx[weak[0]], idx[weak[1]])],
    }
    print(f"\n  high-gap cluster: {', '.join(high)}")

    perm_rows = []
    for name, pairs in groups.items():
        res = permutation_test(G, pairs, args.n_perm, rng)
        res["group"] = name
        perm_rows.append(res)
    perm_df = pd.DataFrame(perm_rows)[
        ["group", "n_pairs", "observed", "null_mean", "null_lo", "null_hi", "p_one_sided"]]
    perm_df.to_csv(os.path.join(args.outdir, "spearman_permutation.csv"), index=False)

    print(f"\n── 7b: permutation test, {args.n_perm} iterations ──")
    print(f"  {'group':38s} {'pairs':>5s} {'obs rho':>8s} {'null 95%':>18s} {'p':>8s}")
    for _, r in perm_df.iterrows():
        print(f"  {r.group:38s} {r.n_pairs:5d} {r.observed:+8.3f} "
              f"  [{r.null_lo:+.3f}, {r.null_hi:+.3f}] {r.p_one_sided:8.4f}")

    write_report(args, point, var_df, perm_df, models, high, noise_var, cell_noise, values)


def write_report(args, point, var_df, perm_df, models, high, noise_var, cell_noise, values):
    g = lambda q: var_df[var_df.quantity == q].iloc[0]  # noqa: E731
    L = ["# Heterogeneity tests (7a-7b)\n"]
    L.append("Two-step substitute for a crossed random-effects logistic model: a variance "
             "decomposition of the gap matrix with the bootstrap noise term removed, and a "
             "permutation test on the cross-model rank agreement.\n")

    L.append("\n## 7a — Variance decomposition of G[model, value] = M - U\n")
    L.append("Two-way ANOVA without replication, 10 models x 32 values. The residual mean "
             "square carries model x value interaction plus per-cell measurement noise; the "
             "noise term comes from the cluster bootstrap and is subtracted.\n")
    L.append(f"Residual MS {point['ms_resid']:.5f}, of which measurement noise "
             f"{noise_var:.5f} ({100*point['noise_share_of_resid']:.1f}%). Per-cell bootstrap "
             f"sd ranges {np.sqrt(cell_noise).min():.3f} to {np.sqrt(cell_noise).max():.3f}.\n")
    L.append("| Component | Variance | Share of structural variance | 95% CI |")
    L.append("|---|---|---|---|")
    for k, lab in [("model", "Model main effect"), ("value", "Value main effect"),
                   ("interaction", "Model x value interaction")]:
        r = g(f"share_{k}")
        L.append(f"| {lab} | {point[f'var_{k}']:.5f} | {100*r.estimate:.1f}% | "
                 f"[{100*r.ci_lo:.1f}%, {100*r.ci_hi:.1f}%] |")
    L.append(f"| (measurement noise, excluded) | {noise_var:.5f} | n/a | n/a |")
    r = g("interaction_over_value")
    L.append(f"\nInteraction / value main effect ratio: {r.estimate:.2f} "
             f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}].\n")

    L.append("\n## 7b — Permutation test on the Spearman matrix (Figure 30)\n")
    L.append(f"Value labels shuffled independently within each model, {args.n_perm} iterations. "
             f"High-gap cluster: {', '.join(high)}.\n")
    L.append("| Group | Pairs | Observed mean rho | Null mean | Null 95% | p |")
    L.append("|---|---|---|---|---|---|")
    for _, r in perm_df.iterrows():
        L.append(f"| {r.group} | {r.n_pairs} | {r.observed:+.3f} | {r.null_mean:+.3f} | "
                 f"[{r.null_lo:+.3f}, {r.null_hi:+.3f}] | {r.p_one_sided:.4f} |")

    L.append("\n## Scope restriction from the swap control (\\S F.3)\n")
    L.append("The option-swap control classifies a majority of values REAL for only "
             f"{' and '.join(SWAP_REAL_MAJORITY)} "
             + "(" + ", ".join(f"{m}: {n}/32" for m, n in SWAP_TESTED.items()) + "). "
             "Llama-3.1-8B in particular has 1/32 REAL at n=4, so most of its measured gap is "
             "position bias rather than value-conditioned behavior. Its weak correlation with "
             "the rest of the roster is therefore consistent with two readings: a genuinely "
             "different value profile, or little real signal to correlate. The strongest form "
             "of the heterogeneity claim should be stated only for models where the swap "
             "control leaves a majority of values REAL.\n")

    path = os.path.join(args.outdir, "HETEROGENEITY.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nreport → {path}")


if __name__ == "__main__":
    main()
