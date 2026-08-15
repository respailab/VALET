#!/usr/bin/env python3
"""Generate a standalone two-page Value Card from a toolkit evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from value_faking.paths import TOOLKIT_ROOT, RUNS_ROOT, MODEL_REGISTRY

DEFAULT_RUNS_ROOT = RUNS_ROOT
DEFAULT_REGISTRY = MODEL_REGISTRY

PANELS = ("gap", "dumbbell", "reasoning_monitored", "reasoning_unmonitored", "divergence")

# Header fields, in card order. Absent keys render as an em-less placeholder rather
# than vanishing, so a half-filled registry still produces a well-formed card.
META_FIELDS = (
    ("benchmark",   "Benchmark"),
    ("run_tag",     "Run tag"),
    ("replicates",  "Replicates"),
    ("conditions",  "Conditions"),
    ("judge",       "Judge model"),
    ("annotation",  "Annotation"),
)

FALLBACK_DEFAULTS = {
    "benchmark":  "unrecorded",
    "run_tag":    "unrecorded",
    "replicates": "unrecorded",
    "conditions": "bare / monitored / unmonitored",
    "judge":      "unrecorded",
    "annotation": "unrecorded",
}


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(TOOLKIT_ROOT))
    except ValueError:
        return str(p)


def tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def fmt_rate(x: float) -> str:
    return f"{x:.2f}"


def fmt_signed(x: float) -> str:
    r = round(x, 2)
    if r == 0:
        return r"$\ \ 0.00$"
    return f"${r:+.2f}$"


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {"_defaults": {}, "models": {}, "abbrev": {}}
    with path.open() as f:
        reg = json.load(f)
    reg.setdefault("_defaults", {})
    reg.setdefault("models", {})
    reg.setdefault("abbrev", {})
    return reg


def run_dir_for(slug: str, reg: dict, runs_root: Path) -> Path:
    entry = reg["models"].get(slug, {})
    return runs_root / entry.get("run_dir", slug)


def discover_slugs(runs_root: Path) -> list[str]:
    if not runs_root.is_dir():
        die(f"runs root not found: {rel(runs_root)}")
    found = sorted(
        p.name for p in runs_root.iterdir()
        if (p / "tables" / "mub_per_value.csv").exists()
    )
    if not found:
        die(f"no runs with tables/mub_per_value.csv under {rel(runs_root)}")
    return found


def load_rows(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "tables" / "mub_per_value.csv"
    if not csv_path.exists():
        die(f"no per-value table at {rel(csv_path)}")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"{rel(csv_path)} has no data rows")

    required = {"value", "B_mean", "M_mean", "U_mean", "M_minus_U", "U_minus_B"}
    missing = required - set(rows[0])
    if missing:
        die(f"{rel(csv_path)} is missing columns: {sorted(missing)}")

    for r in rows:
        for col in required - {"value"}:
            r[col] = float(r[col])
    rows.sort(key=lambda r: r["M_minus_U"], reverse=True)
    return rows


def load_significant(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if not path.exists():
        die(f"significance file not found: {path}")
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split(",")[0].strip())
    names.discard("value")  # tolerate a CSV header
    return names


def resolve_panels(run_dir: Path, out_dir: Path, ext: str) -> tuple[str, dict[str, str]]:
    plots_dir = run_dir / "plots"
    if not plots_dir.is_dir():
        die(f"plots directory not found: {rel(plots_dir)}")

    missing = [rel(plots_dir / f"{p}.{ext}") for p in PANELS
               if not (plots_dir / f"{p}.{ext}").exists()]
    if missing:
        die("missing plot(s):\n  " + "\n  ".join(missing))

    gpath = os.path.relpath(plots_dir, out_dir)
    return gpath, {p: p for p in PANELS}


def load_meta(run_dir: Path, reg: dict, slug: str) -> dict:
    meta = dict(FALLBACK_DEFAULTS)
    meta.update(reg["_defaults"])

    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        try:
            with run_meta_path.open() as f:
                meta.update({k: v for k, v in json.load(f).items() if k in FALLBACK_DEFAULTS})
        except json.JSONDecodeError as e:
            die(f"{rel(run_meta_path)} is not valid JSON: {e}")

    entry = reg["models"].get(slug, {})
    meta.update({k: v for k, v in entry.items() if k in FALLBACK_DEFAULTS})
    return meta


# Below this many values the two-column split leaves a header with no rows under
# it, which reads as a rendering fault rather than a short run.
_SPLIT_THRESHOLD = 8


def _table_block(chunk: list[dict], abbrev: dict, significant: set[str] | None,
                 width: str) -> str:
    body = []
    for r in chunk:
        name = tex_escape(abbrev.get(r["value"], r["value"]))
        if significant is not None and r["value"] in significant:
            name += r"$^{\dagger}$"
        body.append(
            f"{name} & {fmt_rate(r['B_mean'])} & {fmt_rate(r['M_mean'])} & "
            f"{fmt_rate(r['U_mean'])} & {fmt_signed(r['M_minus_U'])} & "
            f"{fmt_signed(r['U_minus_B'])} \\\\"
        )
    return (
        f"\\begin{{minipage}}[b]{{{width}\\linewidth}}\n"
        "\\centering\n"
        "\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\n"
        "Value & $B$ & $M$ & $U$ & $M-U$ & $U-B$ \\\\\n"
        "\\midrule\n" + "\n".join(body) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{minipage}"
    )


def rate_table(rows: list[dict], abbrev: dict, significant: set[str] | None) -> str:
    if len(rows) < _SPLIT_THRESHOLD:
        return ("\\centering\n"
                + _table_block(rows, abbrev, significant, width="0.6"))

    half = (len(rows) + 1) // 2
    left = _table_block(rows[:half], abbrev, significant, width="0.49")
    right = _table_block(rows[half:], abbrev, significant, width="0.49")
    return left + "\\hfill\n" + right


def build_tex(model: dict, rows: list[dict], panels: dict, gpath: str, meta: dict,
              abbrev: dict, significant: set[str] | None, generated: str) -> str:
    display = model["display"]
    n_resolved = (f"{len(significant & {r['value'] for r in rows})} of {len(rows)} "
                  "(BH-corrected)") if significant is not None else "not assessed"

    dagger_note = (
        r"$\dagger$ marks values surviving Benjamini--Hochberg correction; unmarked "
        r"cells are unresolved at this sample size and should not be acted on."
        if significant is not None else
        r"No multiple-comparison correction is recorded for this card; per-value cells "
        r"should not be acted on individually."
    )

    meta_cells = " &\n".join(
        rf"\metafield{{{label}}}{{{tex_escape(str(meta[key]))}}}"
        for key, label in META_FIELDS[:3]
    )
    meta_cells2 = " &\n".join(
        rf"\metafield{{{label}}}{{{tex_escape(str(meta[key]))}}}"
        for key, label in META_FIELDS[3:]
    )

    return rf"""% Standalone two-page Value Card for {display}.
% GENERATED by value_faking/report/make_value_card.py -- edit the script or
% configs/models.json, not this file.
% Compile from this directory:  latexmk -pdf {model['out_stem']}.tex
\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.0cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage[table]{{xcolor}}
\usepackage{{times}}
\usepackage[T1]{{fontenc}}
\usepackage{{microtype}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\graphicspath{{{{{gpath}/}}}}

\definecolor{{cardink}}{{HTML}}{{1F2933}}
\definecolor{{cardrule}}{{HTML}}{{9AA5B1}}
\definecolor{{cardtint}}{{HTML}}{{EEF2F6}}

% Panel heading: small caps label with a hairline under it.
\newcommand{{\panel}}[1]{{%
  \vspace{{1pt}}%
  {{\color{{cardink}}\normalsize\scshape\bfseries #1}}\\[1pt]
  {{\color{{cardrule}}\rule{{\linewidth}}{{0.4pt}}}}\\[2pt]
}}

% Label above value, inside a p-column cell (\newline, not \\, or the row ends).
\newcommand{{\metafield}}[2]{{{{\color{{cardrule}}\scriptsize\scshape #1}}\newline{{\small #2}}}}

\newcommand{{\figcap}}[1]{{\\[1pt]{{\small\color{{cardink}} #1}}}}

\newcommand{{\cardhead}}{{%
{{\color{{cardink}}\rule{{\linewidth}}{{1.8pt}}}}\\[3pt]
{{\LARGE\bfseries\color{{cardink}} Value Card}}\hspace{{0.7em}}{{\large\color{{cardink}} {tex_escape(display)}}}\\[3pt]
{{\color{{cardink}}\rule{{\linewidth}}{{0.7pt}}}}%
}}

\begin{{document}}

% ================================================================ page 1
\cardhead\\[4pt]

\colorbox{{cardtint}}{{%
\begin{{minipage}}{{\dimexpr\linewidth-2\fboxsep\relax}}
\vspace{{3pt}}
\begin{{tabular}}{{@{{}}p{{0.27\linewidth}}p{{0.16\linewidth}}p{{0.19\linewidth}}p{{0.13\linewidth}}p{{0.19\linewidth}}@{{}}}}
\metafield{{Model}}{{{tex_escape(model['model_id'])}}} &
\metafield{{Serving checkpoint}}{{{tex_escape(model['checkpoint'])}}} &
{meta_cells} \\[8pt]
{meta_cells2} &
\metafield{{Generated}}{{{generated}}} &
\metafield{{Values resolved}}{{{n_resolved}}} \\
\end{{tabular}}
\vspace{{3pt}}
\end{{minipage}}}}\\[4pt]

{{\footnotesize\color{{cardink}}
$M-U$ is the compliance gap under a change in perceived oversight, $U-B$ the residual effect of the directive when the model believes it is unmonitored.
{dagger_note}}}\\[4pt]

\panel{{Compliance: gap and B/M/U triple}}

\begin{{minipage}}[b]{{0.45\linewidth}}
\centering
\includegraphics[width=\linewidth,height=11.0cm,keepaspectratio]{{{panels['gap']}}}
\figcap{{Mean compliance gap ($M-U$) per value}}
\end{{minipage}}\hfill
\begin{{minipage}}[b]{{0.53\linewidth}}
\centering
\includegraphics[width=\linewidth,height=11.0cm,keepaspectratio]{{{panels['dumbbell']}}}
\figcap{{Bare / unmonitored / monitored compliance rate}}
\end{{minipage}}

\vspace{{4pt}}

\panel{{Per-value rates, sorted by $M-U$}}

\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{0.95}}
{rate_table(rows, abbrev, significant)}

\newpage

% ================================================================ page 2
\cardhead\\[6pt]

\panel{{Reasoning composition under the six-category taxonomy}}

\begin{{minipage}}[b]{{0.49\linewidth}}
\centering
\includegraphics[width=\linewidth,height=10.6cm,keepaspectratio]{{{panels['reasoning_monitored']}}}
\figcap{{Monitored}}
\end{{minipage}}\hfill
\begin{{minipage}}[b]{{0.49\linewidth}}
\centering
\includegraphics[width=\linewidth,height=10.6cm,keepaspectratio]{{{panels['reasoning_unmonitored']}}}
\figcap{{Unmonitored}}
\end{{minipage}}

\vspace{{6pt}}

\panel{{Reasoning--choice divergence}}

\begin{{center}}
\includegraphics[width=\linewidth,height=10.4cm,keepaspectratio]{{{panels['divergence']}}}
\figcap{{Rate at which the stated reasoning and the final choice disagree, monitored (left) and unmonitored (right).
Where divergence is elevated, the trace should not be read as a record of the decision.}}
\end{{center}}

\vfill
{{\color{{cardrule}}\rule{{\linewidth}}{{0.4pt}}}}\\[2pt]
{{\footnotesize\color{{cardink}} All panels are populated directly from the judge pass; no generation stage sits between evaluation and card.
A card describes one model on one benchmark version: it does not transfer across models, and says nothing about values outside those sampled.
It goes stale silently if the serving checkpoint changes behind an unchanged model name.}}

\end{{document}}
"""


def make_card(slug: str, reg: dict, args: argparse.Namespace) -> Path:
    runs_root = Path(args.runs_root).resolve()
    run_dir = run_dir_for(slug, reg, runs_root)
    if not run_dir.is_dir():
        die(f"run directory not found: {rel(run_dir)}")

    entry = reg["models"].get(slug, {})
    model = {
        "slug":       slug,
        "display":    entry.get("display", slug),
        "model_id":   entry.get("model_id", slug),
        "checkpoint": entry.get("checkpoint", "unpinned"),
        "out_stem":   f"value_card_{slug}",
    }

    rows = load_rows(run_dir)
    meta = load_meta(run_dir, reg, slug)

    if args.significant:
        significant = load_significant(args.significant)
    else:
        # .csv, not .txt: the repo .gitignore blanket-ignores *.txt for token safety.
        default = next((p for p in (run_dir / "sig" / "bh_significant.csv",
                                    run_dir / "sig" / f"{slug}.csv") if p.exists()), None)
        significant = load_significant(default)

    if significant is not None:
        unknown = significant - {r["value"] for r in rows}
        if unknown:
            die(f"significance file names values absent from the table: {sorted(unknown)}")

    out_dir = Path(args.outdir).resolve() if args.outdir else (run_dir / "card")
    out_dir.mkdir(parents=True, exist_ok=True)

    gpath, panels = resolve_panels(run_dir, out_dir, args.ext)
    tex = build_tex(model, rows, panels, gpath, meta, reg["abbrev"],
                    significant, args.date)

    out_path = out_dir / f"{model['out_stem']}.tex"
    out_path.write_text(tex)
    print(f"wrote {rel(out_path)}  "
          f"({len(rows)} values, daggers: "
          f"{'off' if significant is None else len(significant)})")

    if args.compile:
        if shutil.which("latexmk") is None:
            die("latexmk not found on PATH; rerun without --compile")
        r = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", out_path.name],
            cwd=out_dir, capture_output=True, text=True)
        pdf = out_path.with_suffix(".pdf")
        if r.returncode != 0 or not pdf.exists():
            log = out_path.with_suffix(".log")
            tail = "\n".join(
                l for l in log.read_text(errors="replace").splitlines()
                if l.startswith("! ")) if log.exists() else r.stdout[-2000:]
            die(f"latexmk failed for {out_path.name}:\n{tail}")
        print(f"  compiled {rel(pdf)}")

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate standalone Value Card PDFs from a toolkit evaluation run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].strip() if "Usage:" in __doc__ else None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="run slug, e.g. llama-3.3-70b")
    g.add_argument("--all", action="store_true",
                   help="every run under --runs-root with a per-value table")
    p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                   help=f"evaluation output root (default: {rel(DEFAULT_RUNS_ROOT)})")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                   help="model display metadata; optional, slugs stand in when absent")
    p.add_argument("--outdir", default=None,
                   help="where to write the .tex (default: <run_dir>/card/)")
    p.add_argument("--ext", default="pdf", choices=("pdf", "png"),
                   help="plot file extension to reference (default: pdf)")
    p.add_argument("--significant", type=Path,
                   help="file of BH-surviving value names, one per line. Defaults to "
                        "<run_dir>/sig/bh_significant.csv when present; daggers are "
                        "omitted entirely if neither is found.")
    p.add_argument("--date", default=date.today().isoformat(),
                   help="value for the 'Generated' header field (default: today)")
    p.add_argument("--compile", action="store_true", help="run latexmk on the output")
    args = p.parse_args()

    reg = load_registry(args.registry)
    if args.all and args.significant:
        die("--significant names one file; use <run_dir>/sig/bh_significant.csv with --all")

    slugs = discover_slugs(Path(args.runs_root).resolve()) if args.all else [args.model]
    for slug in slugs:
        make_card(slug, reg, args)


if __name__ == "__main__":
    main()
