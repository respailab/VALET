"""Second-judge agreement, step 2 of 2 (run judge_second_pass.py first)."""
import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from value_faking.bf.common import REPO, MAIN_PATHS
from value_faking.paths import REPO

RTYPES = ["training_signal", "value_conflict_aware", "directive_following",
          "rationalisation", "no_reasoning", "parse_error"]

KEY = ["value", "scenario", "condition", "sample_idx"]


def pass1_path(model: str) -> str:
    return MAIN_PATHS[model].replace(".csv", "_samples.csv")


def pass2_path(model: str, pass2_dir: str) -> str:
    return os.path.join(pass2_dir, f"{model}_judge2_samples.csv")


def load_pair(model: str, pass2_dir: str) -> pd.DataFrame:
    p1_path = pass1_path(model)
    p2_path = pass2_path(model, pass2_dir)
    for p in (p1_path, p2_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    p1 = pd.read_csv(p1_path)
    p2 = pd.read_csv(p2_path)

    cols = ["intended", "actual", "divergence", "reasoning_type", "has_thinking"]
    p1 = p1[KEY + cols + ["behavioral_pattern", "compliance_gap"]]
    p2 = p2[KEY + cols]

    for df, tag in ((p1, "pass1"), (p2, "pass2")):
        dupes = df.duplicated(subset=KEY).sum()
        if dupes:
            print(f"WARNING: {model} {tag} has {dupes} duplicate keys — "
                  f"scenario text is not unique, agreement may be misaligned")

    # inner join: pass 2 is a per-value subset, so it selects its own rows out of pass 1
    merged = p1.merge(p2, on=KEY, how="inner", suffixes=("_1", "_2"))
    if len(merged) != len(p2):
        print(f"WARNING: {model} joined {len(merged)}/{len(p2)} pass-2 rows — "
              f"{len(p2) - len(merged)} pass-2 rows found no pass-1 match")

    merged["model"] = model
    # verbalized = the BF3/BF4 bit: did the judge see explicit training-signal reasoning
    for s in ("1", "2"):
        merged[f"verbalized_{s}"] = merged[f"reasoning_type_{s}"] == "training_signal"
        merged[f"divergence_{s}"] = merged[f"divergence_{s}"].astype(bool)
        merged[f"has_thinking_{s}"] = merged[f"has_thinking_{s}"].astype(bool)
    return merged


def kappa_block(df: pd.DataFrame, col: str, label: str, order=None, lines=None):
    a, b = df[f"{col}_1"], df[f"{col}_2"]
    # Cohen's kappa is 0.0 whenever either rater is constant, which reads as
    # "chance-level agreement" but actually means that rater never varied — a
    # different claim, and the one that matters when a judge collapses a label.
    # Report NaN and name the degenerate rater instead.
    degenerate = [tag for tag, s in (("pass1", a), ("pass2", b)) if s.nunique() < 2]
    if degenerate:
        k = float("nan")
        note = f" — {'/'.join(degenerate)} constant (={b.iloc[0] if 'pass2' in degenerate else a.iloc[0]}), kappa undefined"
    else:
        k = cohen_kappa_score(a, b)
        note = ""
    agree = (a == b).mean() * 100

    table = pd.crosstab(a, b, rownames=["pass1"], colnames=["pass2"])
    if order:
        present = [o for o in order if o in set(a) | set(b)]
        table = table.reindex(index=present, columns=present, fill_value=0)

    out = lines if lines is not None else []
    out.append(f"\n**{label}** — n={len(df)}, exact agreement {agree:.1f}%, "
               f"Cohen's kappa {k:.3f}{note}")
    out.append("```")
    out.append(table.to_string())
    out.append("```")
    return k, agree, out


def report_model(model: str, pass2_dir: str, lines: list):
    merged = load_pair(model, pass2_dir)
    n_full = len(pd.read_csv(pass1_path(model)))

    lines.append(f"\n---\n\n## {model}\n")
    lines.append(f"- pass 1 judge: gpt-4o-mini (openai) — `{pass1_path(model)}`")
    lines.append(f"- pass 2 judge: see judge_second_pass.py run — `{pass2_path(model, pass2_dir)}`")
    lines.append(f"- joined samples: {len(merged)} of {n_full} pass-1 samples "
                 f"({100*len(merged)/n_full:.0f}% coverage)")

    # sanity: has_thinking is regex-derived, not judged — must be identical if aligned
    think_match = (merged["has_thinking_1"] == merged["has_thinking_2"]).mean() * 100
    flag = "" if think_match == 100 else "  <-- JOIN MISALIGNED, results below are invalid"
    lines.append(f"- has_thinking match (regex, expect 100%): {think_match:.1f}%{flag}")

    for s in ("1", "2"):
        pe = (merged[f"reasoning_type_{s}"] == "parse_error").mean() * 100
        lines.append(f"- pass {s} L2 parse_error rate: {pe:.1f}%")
    lines.append(f"- divergence base rate: pass1 {merged['divergence_1'].mean()*100:.1f}% "
                 f"/ pass2 {merged['divergence_2'].mean()*100:.1f}%")
    lines.append(f"- training_signal base rate: pass1 {merged['verbalized_1'].mean()*100:.1f}% "
                 f"/ pass2 {merged['verbalized_2'].mean()*100:.1f}%")

    rows = []
    for scope, sub in [("overall", merged),
                       ("monitored", merged[merged.condition == "monitored"]),
                       ("unmonitored", merged[merged.condition == "unmonitored"])]:
        if not len(sub):
            continue
        lines.append(f"\n### {model} — {scope}")
        k_rt, a_rt, _ = kappa_block(sub, "reasoning_type", "reasoning_type (5-class)",
                                    order=RTYPES, lines=lines)
        k_vb, a_vb, _ = kappa_block(sub, "verbalized", "verbalized (training_signal vs rest)",
                                    order=[False, True], lines=lines)
        k_dv, a_dv, _ = kappa_block(sub, "divergence", "divergence (intended != actual)",
                                    order=[False, True], lines=lines)
        k_in, a_in, _ = kappa_block(sub, "intended", "L1 intended choice",
                                    order=["a", "b", "unclear", "parse_error"], lines=lines)
        k_ac, a_ac, _ = kappa_block(sub, "actual", "L1 actual choice",
                                    order=["a", "b", "unclear", "parse_error"], lines=lines)
        rows.append({"model": model, "scope": scope, "n": len(sub),
                     "kappa_reasoning_type": k_rt, "agree_reasoning_type": a_rt,
                     "kappa_verbalized": k_vb, "agree_verbalized": a_vb,
                     "kappa_divergence": k_dv, "agree_divergence": a_dv,
                     "kappa_intended": k_in, "agree_intended": a_in,
                     "kappa_actual": k_ac, "agree_actual": a_ac})

    # where pass 2 moved the label: which pass-1 classes get reassigned, and to what
    disagree = merged[merged["reasoning_type_1"] != merged["reasoning_type_2"]]
    if len(disagree):
        lines.append(f"\n### {model} — top reasoning_type disagreements")
        top = (disagree.groupby(["reasoning_type_1", "reasoning_type_2"])
               .size().sort_values(ascending=False).head(10))
        lines.append("```")
        lines.append(top.to_string())
        lines.append("```")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mistral-large-2512", "ministral-14b"],
                        help=f"model keys from common.MAIN_PATHS: {sorted(MAIN_PATHS)}")
    parser.add_argument("--pass2-dir", default=f"{REPO}/behavioral_analysis/judge2",
                        help="directory judge_second_pass.py wrote <model>_judge2_samples.csv into")
    parser.add_argument("--out", default=f"{REPO}/results/judge_agreement_pass2.md")
    parser.add_argument("--csv-out", default=f"{REPO}/results/judge_agreement_pass2.csv")
    args = parser.parse_args()

    lines = ["# Second-judge agreement — full re-judged runs",
             "",
             "Pass 1 = gpt-4o-mini (the judge behind every `analyzed_results_v2.csv`).",
             "Pass 2 = stronger OpenRouter judge, same L1/L2 prompts via `analyze_layers.run_analysis()`.",
             "Sample-level (scenario x condition x sample), not scenario-level."]

    summaries = []
    for model in args.models:
        if model not in MAIN_PATHS:
            print(f"SKIP unknown model '{model}'", file=sys.stderr)
            continue
        try:
            summaries.append(report_model(model, args.pass2_dir, lines))
        except FileNotFoundError as e:
            print(f"SKIP {model}: missing {e}", file=sys.stderr)

    if not summaries:
        sys.exit("no models processed")

    summary = pd.concat(summaries, ignore_index=True)
    lines.insert(5, "\n## Summary\n\n```\n" + summary.round(3).to_string(index=False) + "\n```")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    summary.to_csv(args.csv_out, index=False)
    print("\n".join(lines))
    print(f"\nwrote {args.out}\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
