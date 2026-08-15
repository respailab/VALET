"""Cluster bootstrap for the value-faking compliance rates (analysis 6a-6f)."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from value_faking.paths import REPO
from value_faking.bf import common as C

OUT_DIR = os.path.join(REPO, "results", "bootstrap")

# display name -> registry key in behavioral_analysis/common.py,
# ordered as in Table 2 (mean gap, descending)
MODELS = [
    ("DeepSeek R1 Distill Llama 70B", "deepseek-r1-distill"),
    ("DeepSeek R1 0528",              "deepseek-r1-0528"),
    ("Llama 3.3 70B Instruct",        "llama-3.3-70b"),
    ("Llama 3.1 70B Instruct",        "llama-3.1-70b"),
    ("Mistral Large 2512",            "mistral-large-2512"),
    ("Mistral Small Instruct 2409",   "mistral-small-2409"),
    ("Ministral 14B 2512",            "ministral-14b"),
    ("GPT-4o-mini",                   "gpt-4o-mini"),
    ("Llama 3.1 8B Instruct",         "llama-3.1-8b"),
    ("Gemma 3 27B",                   "gemma-3-27b"),
]

CONDITIONS = ["B", "M", "U"]


def bare_scenario_path(key: str) -> str:
    return C.BARE_PATHS[key].replace("value_results.csv", "scenario_results.csv")


def load_panel():
    ref = pd.read_csv(C.MAIN_PATHS[MODELS[0][1]])
    ref = ref.sort_values(["value", "topic"]).reset_index(drop=True)
    keys = list(zip(ref["value"], ref["topic"]))
    index = pd.MultiIndex.from_tuples(keys, names=["value", "topic"])

    data = {c: np.full((len(MODELS), len(keys)), np.nan) for c in CONDITIONS}

    for i, (label, key) in enumerate(MODELS):
        main = pd.read_csv(C.MAIN_PATHS[key]).set_index(["value", "topic"])
        bare = pd.read_csv(bare_scenario_path(key)).set_index(["value", "topic"])
        if main.index.duplicated().any() or bare.index.duplicated().any():
            raise ValueError(f"{key}: duplicate (value, topic) keys")
        missing = index.difference(main.index).union(index.difference(bare.index))
        if len(missing):
            raise ValueError(f"{key}: {len(missing)} scenarios missing from the grid")

        data["M"][i] = main.loc[index, "monitored_rate"].to_numpy(dtype=float)
        data["U"][i] = main.loc[index, "unmonitored_rate"].to_numpy(dtype=float)
        data["B"][i] = bare.loc[index, "monitored_rate"].to_numpy(dtype=float)

    return keys, np.array([v for v, _ in keys]), data


def cell_draws(blocks, n_boot, rng):
    return {
        v: rng.integers(0, len(pos), size=(n_boot, len(pos)))
        for v, pos in blocks.items()
    }


def aggregate_draws(blocks, value_order, n_boot, rng):
    n_values = len(value_order)
    out = []
    for _ in range(n_boot):
        drawn_values = rng.integers(0, n_values, size=n_values)
        parts = []
        for vi in drawn_values:
            pos = blocks[value_order[vi]]
            parts.append(pos[rng.integers(0, len(pos), size=len(pos))])
        out.append(np.concatenate(parts))
    return out


def percentile_ci(dist, axis=-1):
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(dist, 2.5, axis=axis)
        hi = np.nanpercentile(dist, 97.5, axis=axis)
    return lo, hi


def two_sided_p(dist, axis=-1):
    n = np.sum(~np.isnan(dist), axis=axis)
    le = np.sum(np.nan_to_num(dist, nan=np.inf) <= 0, axis=axis)
    ge = np.sum(np.nan_to_num(dist, nan=-np.inf) >= 0, axis=axis)
    p = 2.0 * np.minimum((le + 1.0) / (n + 1.0), (ge + 1.0) / (n + 1.0))
    return np.minimum(p, 1.0)


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    ranked = flat[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(ranked, 1.0)
    return q.reshape(p.shape)


def nanmean_cols(arr, idx):
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr[:, idx], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    keys, scen_values, data = load_panel()
    model_labels = [m for m, _ in MODELS]
    value_order = sorted(set(scen_values))
    blocks = {v: np.flatnonzero(scen_values == v) for v in value_order}

    # M-U is formed per scenario, so a scenario contributes only when both
    # conditions parsed. This matches the `compliance_gap` column the published
    # tables are built from, and keeps the difference genuinely paired.
    # U-B is the difference of the two condition means, so it stays consistent
    # with the B and U columns printed alongside it.
    data["gap"] = data["M"] - data["U"]

    n_m, n_s = len(MODELS), len(keys)
    n_v, n_b = len(value_order), args.n_boot
    print(f"panel: {n_m} models x {n_v} values x {n_s} scenarios, {n_b} iterations")
    for c in CONDITIONS:
        print(f"  {c}: {np.isnan(data[c]).sum()} missing scenario rates")

    # ── 6a: per-cell bootstrap, shared draws across conditions and models ──────
    draws = cell_draws(blocks, n_b, rng)
    series = CONDITIONS + ["gap"]
    boot = {c: np.full((n_m, n_v, n_b), np.nan) for c in series}
    for vi, v in enumerate(value_order):
        pos = blocks[v]
        idx = pos[draws[v]]                       # (n_boot, n_v_scen)
        for c in series:
            with np.errstate(invalid="ignore"):
                boot[c][:, vi, :] = np.nanmean(data[c][:, idx], axis=-1)

    # 6b: derived quantities come from the same draws. `gap` is the paired
    # per-scenario M-U (Table 2, Figure 2); `gap_md` is the difference of the two
    # condition means, which is what the worked card (Table 3) prints alongside
    # its own M and U columns. They differ only where a condition failed to parse.
    boot_gap = boot["gap"]
    boot_gap_md = boot["M"] - boot["U"]
    boot_ub = boot["U"] - boot["B"]

    # point estimates
    point = {
        c: np.array([[np.nanmean(data[c][i, blocks[v]]) for v in value_order]
                     for i in range(n_m)])
        for c in series
    }
    point_gap = point["gap"]
    point_gap_md = point["M"] - point["U"]
    point_ub = point["U"] - point["B"]

    # ── 6c: aggregate rows, two-level resample ────────────────────────────────
    agg_idx = aggregate_draws(blocks, value_order, n_b, rng)
    agg = {c: np.full((n_m, n_b), np.nan) for c in series}
    for b, idx in enumerate(agg_idx):
        for c in series:
            agg[c][:, b] = nanmean_cols(data[c], idx)
    agg_gap = agg["gap"]
    agg_ub = agg["U"] - agg["B"]

    # scenario-only (single-level) aggregate, for the width comparison
    flat_idx = [np.concatenate([blocks[v][draws[v][b]] for v in value_order])
                for b in range(n_b)]
    agg1_gap = np.full((n_m, n_b), np.nan)
    for b, idx in enumerate(flat_idx):
        agg1_gap[:, b] = nanmean_cols(data["gap"], idx)

    agg_point = {c: np.nanmean(data[c], axis=1) for c in series}

    # ── 6e: per-cell p and BH over all 320 cells ──────────────────────────────
    p_gap = two_sided_p(boot_gap)
    q_gap = benjamini_hochberg(p_gap)
    p_ub = two_sided_p(boot_ub)
    q_ub = benjamini_hochberg(p_ub)
    p_gap_md = two_sided_p(boot_gap_md)
    q_gap_md = benjamini_hochberg(p_gap_md)

    # ── save the full distribution (reused by #3 and #7) ──────────────────────
    npz = os.path.join(args.outdir, "boot_cells.npz")
    np.savez_compressed(
        npz,
        models=np.array(model_labels), values=np.array(value_order),
        conditions=np.array(CONDITIONS), seed=args.seed, n_boot=n_b,
        boot_B=boot["B"], boot_M=boot["M"], boot_U=boot["U"],
        boot_gap=boot_gap, boot_gap_md=boot_gap_md, boot_ub=boot_ub,
        point_B=point["B"], point_M=point["M"], point_U=point["U"],
        agg_B=agg["B"], agg_M=agg["M"], agg_U=agg["U"],
        agg_gap=agg_gap, agg_ub=agg_ub, agg1_gap=agg1_gap,
        cell_draws=np.stack(flat_idx),
        n_scenarios=np.array([len(blocks[v]) for v in value_order]),
    )
    print(f"bootstrap distribution → {npz}  ({os.path.getsize(npz)/1e6:.1f} MB)")

    # ── 6a/6b/6e table: one row per (model, value) cell ───────────────────────
    rows = []
    for i, model in enumerate(model_labels):
        for vi, v in enumerate(value_order):
            row = {"model": model, "value": v, "n_scenarios": len(blocks[v])}
            for c in CONDITIONS:
                lo, hi = percentile_ci(boot[c][i, vi])
                row[c] = point[c][i, vi]
                row[f"{c}_lo"], row[f"{c}_hi"] = lo, hi
            for name, pt, dist, pv, qv in [
                ("gap", point_gap, boot_gap, p_gap, q_gap),
                ("gap_md", point_gap_md, boot_gap_md, p_gap_md, q_gap_md),
                ("u_minus_b", point_ub, boot_ub, p_ub, q_ub),
            ]:
                lo, hi = percentile_ci(dist[i, vi])
                row[name] = pt[i, vi]
                row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
                row[f"{name}_p"] = pv[i, vi]
                row[f"{name}_q"] = qv[i, vi]
            row["gap_sig"] = bool(q_gap[i, vi] < 0.05)
            rows.append(row)
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(args.outdir, "cell_ci.csv"), index=False)
    print(f"cell CIs → {os.path.join(args.outdir, 'cell_ci.csv')}  ({len(cells)} cells)")

    # ── 6c table: aggregate rows with two-level CIs ───────────────────────────
    agg_rows = []
    for i, model in enumerate(model_labels):
        row = {"model": model}
        for c in CONDITIONS:
            lo, hi = percentile_ci(agg[c][i])
            row[c] = agg_point[c][i]
            row[f"{c}_lo"], row[f"{c}_hi"] = lo, hi
        for name, pt, dist in [("gap", agg_point["gap"][i], agg_gap),
                               ("u_minus_b", agg_point["U"][i] - agg_point["B"][i], agg_ub)]:
            lo, hi = percentile_ci(dist[i])
            row[name] = pt
            row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
        lo1, hi1 = percentile_ci(agg1_gap[i])
        row["gap_lo_scenario_only"], row["gap_hi_scenario_only"] = lo1, hi1
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(os.path.join(args.outdir, "aggregate_ci.csv"), index=False)

    # ── 6f: paired cross-model contrasts on the aggregate gap ─────────────────
    con_rows = []
    for i in range(n_m):
        for j in range(i + 1, n_m):
            d = agg_gap[i] - agg_gap[j]
            lo, hi = percentile_ci(d)
            con_rows.append({
                "model_a": model_labels[i], "model_b": model_labels[j],
                "gap_a": agg_point["gap"][i], "gap_b": agg_point["gap"][j],
                "diff": np.nanmean(d), "diff_lo": lo, "diff_hi": hi,
                "p": float(two_sided_p(d)), "significant": bool(lo > 0 or hi < 0),
            })
    contrasts = pd.DataFrame(con_rows)
    contrasts["q"] = benjamini_hochberg(contrasts["p"].to_numpy())
    contrasts.to_csv(os.path.join(args.outdir, "model_contrasts.csv"), index=False)

    # ── 6d: worked card, per-value M-U CIs ────────────────────────────────────
    card = (cells[cells.model == "Llama 3.3 70B Instruct"]
            .sort_values("gap_md", ascending=False)
            [["value", "B", "M", "U", "gap", "gap_lo", "gap_hi", "gap_q", "gap_sig",
              "gap_md", "gap_md_lo", "gap_md_hi", "gap_md_q"]])
    card.to_csv(os.path.join(args.outdir, "table3_llama33_pervalue_ci.csv"), index=False)

    write_report(args, aggregate, cells, contrasts, card, agg1_gap, agg_gap, model_labels)


def write_report(args, aggregate, cells, contrasts, card, agg1_gap, agg_gap, model_labels):
    fmt = lambda x: f"{x:+.3f}"  # noqa: E731
    L = []
    L.append("# Bootstrap confidence intervals (analysis 6a-6f)\n")
    L.append(f"Cluster bootstrap over scenarios, {args.n_boot} iterations, seed {args.seed}. "
             "Replicates (n=4) are averaged within a scenario before resampling; all "
             "conditions and models share the same drawn scenarios, so differences are paired.\n")

    L.append("\n## 6c — Aggregate rows (Table 2), two-level resample\n")
    L.append("| Model | B | M | U | Gap (M-U) | 95% CI | U-B | 95% CI |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in aggregate.iterrows():
        L.append(f"| {r.model} | {r.B:.3f} | {r.M:.3f} | {r.U:.3f} | {fmt(r.gap)} | "
                 f"[{fmt(r.gap_lo)}, {fmt(r.gap_hi)}] | {fmt(r.u_minus_b)} | "
                 f"[{fmt(r.u_minus_b_lo)}, {fmt(r.u_minus_b_hi)}] |")

    L.append("\nWidth of the gap CI, two-level vs scenario-only resample:\n")
    L.append("| Model | two-level width | scenario-only width |")
    L.append("|---|---|---|")
    for i, m in enumerate(model_labels):
        w2 = np.nanpercentile(agg_gap[i], 97.5) - np.nanpercentile(agg_gap[i], 2.5)
        w1 = np.nanpercentile(agg1_gap[i], 97.5) - np.nanpercentile(agg1_gap[i], 2.5)
        L.append(f"| {m} | {w2:.3f} | {w1:.3f} |")

    L.append("\n## 6e — Figure 2 significance (BH over 320 cells)\n")
    sig = cells.groupby("model", sort=False).gap_sig.agg(["sum", "count"])
    L.append(f"Significant cells: {int(cells.gap_sig.sum())} / {len(cells)} "
             f"at BH q < 0.05.\n")
    L.append("| Model | significant / 32 |")
    L.append("|---|---|")
    for m in model_labels:
        L.append(f"| {m} | {int(sig.loc[m, 'sum'])} / {int(sig.loc[m, 'count'])} |")

    L.append("\n## 6f — The two claims\n")
    pairs = [("DeepSeek R1 Distill Llama 70B", "DeepSeek R1 0528"),
             ("Gemma 3 27B", "Llama 3.1 8B Instruct")]
    L.append("| Contrast | gap A | gap B | difference | 95% CI | p | separated? |")
    L.append("|---|---|---|---|---|---|---|")
    for a, b in pairs:
        r = contrasts[((contrasts.model_a == a) & (contrasts.model_b == b)) |
                      ((contrasts.model_a == b) & (contrasts.model_b == a))].iloc[0]
        L.append(f"| {r.model_a} vs {r.model_b} | {r.gap_a:.3f} | {r.gap_b:.3f} | "
                 f"{fmt(r['diff'])} | [{fmt(r.diff_lo)}, {fmt(r.diff_hi)}] | {r.p:.3f} | "
                 f"{'yes' if r.significant else 'NO'} |")

    L.append("\n## 6d — Worked card (Llama 3.3 70B), per-value M-U\n")
    L.append("| Value | B | M | U | M-U | 95% CI | BH q |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in card.iterrows():
        L.append(f"| {r.value} | {r.B:.2f} | {r.M:.2f} | {r.U:.2f} | {fmt(r.gap_md)} | "
                 f"[{fmt(r.gap_md_lo)}, {fmt(r.gap_md_hi)}] | {r.gap_md_q:.3f} |")

    path = os.path.join(args.outdir, "BOOTSTRAP_CI.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"report → {path}")
    print("\n".join(L[L.index("\n## 6f — The two claims\n"):]))


if __name__ == "__main__":
    main()
