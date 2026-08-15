"""Two-phase seed scenario pipeline over all values in a CSV."""

import argparse
import os
import json
import time
import pandas as pd
from datetime import datetime
from value_faking.paths import DATA_ROOT
from .generate import generate_dev_fields, generate_seed_scenario

_HERE             = os.path.dirname(os.path.abspath(__file__))
CSV_PATH          = str(DATA_ROOT / "values.csv")
TOPIC_GEN_OUTPUTS = os.path.join(_HERE, "..", "topic_gen", "outputs")
OUTPUT_DIR        = os.path.join(_HERE, "outputs")
SERVICE           = "openai"
MODEL             = "gpt-4o-mini"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"value", "definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.dropna(subset=["definition"]).reset_index(drop=True)


def dev_fields_path(output_dir: str, value: str) -> str:
    return os.path.join(output_dir, value, "dev_fields.json")

def seed_scenarios_path(output_dir: str, value: str) -> str:
    return os.path.join(output_dir, value, "seed_scenarios.json")

def topic_scored_path(topic_gen_outputs: str, value: str) -> str:
    return os.path.join(topic_gen_outputs, value, "topics_scored.json")


# ── Phase 1 — Dev field generation ───────────────────────────────────────────

def phase1(df: pd.DataFrame, output_dir: str, service: str, model: str):
    total   = len(df)
    skipped = sum(1 for _, row in df.iterrows()
                  if os.path.exists(dev_fields_path(output_dir, str(row["value"]))))

    log(f"")
    log(f"{'='*60}")
    log(f"PHASE 1 — DEV FIELD GENERATION")
    log(f"service: {service} | model: {model}")
    log(f"total values: {total} | already done: {skipped} | to run: {total - skipped}")
    log(f"{'='*60}")

    start  = time.time()
    done   = 0
    failed = []

    for i, row in df.iterrows():
        value = str(row["value"])
        vdir  = os.path.join(output_dir, value)
        dpath = dev_fields_path(output_dir, value)

        log(f"")
        log(f"[{i+1}/{total}] {value}")

        if os.path.exists(dpath):
            log(f"  skip — dev_fields.json exists")
            done += 1
            continue

        os.makedirs(vdir, exist_ok=True)

        try:
            fields = generate_dev_fields(
                value=value,
                definition=str(row["definition"]),
                service=service,
                model=model,
            )
            fields["value"] = value
            fields["definition"] = str(row["definition"])

            with open(dpath, "w") as f:
                json.dump(fields, f, indent=2)

            log(f"  objective:  {fields['dev_compliant_objective']}")
            log(f"  behaviour:  {fields['dev_compliant_model_response_behaviour']}")
            log(f"  saved → {dpath}")
            done += 1

        except Exception as e:
            log(f"  [ERROR] {e}")
            failed.append({"value": value, "error": str(e)})

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        log(f"  progress: {done} done | {len(failed)} failed | ETA: {eta/60:.1f}min")

    log(f"")
    log(f"PHASE 1 DONE — {done}/{total} succeeded | {len(failed)} failed | "
        f"time: {(time.time()-start)/60:.1f}min")
    if failed:
        for item in failed:
            log(f"  FAILED: {item['value']} — {item['error']}")


# ── Phase 2 — Seed scenario generation ───────────────────────────────────────

def phase2(df: pd.DataFrame, output_dir: str, topic_gen_outputs: str, service: str, model: str):
    total   = len(df)
    skipped = sum(1 for _, row in df.iterrows()
                  if os.path.exists(seed_scenarios_path(output_dir, str(row["value"]))))

    log(f"")
    log(f"{'='*60}")
    log(f"PHASE 2 — SEED SCENARIO GENERATION")
    log(f"service: {service} | model: {model}")
    log(f"total values: {total} | already done: {skipped} | to run: {total - skipped}")
    log(f"{'='*60}")

    start      = time.time()
    done       = 0
    failed     = []
    all_rows   = []

    for i, row in df.iterrows():
        value = str(row["value"])
        dpath = dev_fields_path(output_dir, value)
        spath = seed_scenarios_path(output_dir, value)
        tpath = topic_scored_path(topic_gen_outputs, value)

        log(f"")
        log(f"[{i+1}/{total}] {value}")

        if os.path.exists(spath):
            log(f"  skip — seed_scenarios.json exists")
            with open(spath) as f:
                result = json.load(f)
            for s in result.get("scenarios", []):
                all_rows.append({
                    "value": value,
                    "definition": str(row["definition"]),
                    "topic": s.get("topic", ""),
                    "scenario": s["scenario"],
                    "developer_compliant": s["developer_compliant"],
                    "developer_noncompliant": s["developer_noncompliant"],
                })
            done += 1
            continue

        if not os.path.exists(dpath):
            log(f"  skip — dev_fields.json missing (run phase 1 first)")
            failed.append({"value": value, "error": "missing dev_fields.json"})
            continue

        if not os.path.exists(tpath):
            log(f"  skip — topics_scored.json missing at {tpath}")
            failed.append({"value": value, "error": f"missing topics_scored.json at {tpath}"})
            continue

        with open(dpath) as f:
            dev_fields = json.load(f)

        with open(tpath) as f:
            scored = json.load(f)
        topics = [t["topic"] for t in scored["topics"] if t.get("passes_threshold", True)]
        log(f"  loaded {len(topics)} topics | generating scenarios...")

        scenarios   = []
        val_failed  = []

        for j, topic in enumerate(topics):
            log(f"  [{j+1}/{len(topics)}] {topic[:70]}...")
            try:
                scenario = generate_seed_scenario(
                    value=value,
                    definition=str(row["definition"]),
                    dev_compliant_objective=dev_fields["dev_compliant_objective"],
                    dev_compliant_model_response_behaviour=dev_fields["dev_compliant_model_response_behaviour"],
                    topic=topic,
                    service=service,
                    model=model,
                )
                scenario["topic"] = topic
                scenarios.append(scenario)
                all_rows.append({
                    "value": value,
                    "definition": str(row["definition"]),
                    "topic": topic,
                    "scenario": scenario["scenario"],
                    "developer_compliant": scenario["developer_compliant"],
                    "developer_noncompliant": scenario["developer_noncompliant"],
                })
            except Exception as e:
                log(f"  [ERROR] topic {j+1}: {e}")
                val_failed.append({"topic": topic, "error": str(e)})

        result = {
            "value": value,
            "definition": str(row["definition"]),
            "dev_compliant_objective": dev_fields["dev_compliant_objective"],
            "dev_compliant_model_response_behaviour": dev_fields["dev_compliant_model_response_behaviour"],
            "scenarios": scenarios,
            "failed": val_failed,
        }

        with open(spath, "w") as f:
            json.dump(result, f, indent=2)

        log(f"  {len(scenarios)} scenarios generated | {len(val_failed)} failed")
        log(f"  saved → {spath}")
        done += 1

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        log(f"  progress: {done} done | {len(failed)} failed | ETA: {eta/60:.1f}min")

    if all_rows:
        csv_path = os.path.join(output_dir, "seed_scenarios.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        log(f"")
        log(f"saved combined CSV ({len(all_rows)} rows) → {csv_path}")

    log(f"")
    log(f"PHASE 2 DONE — {done}/{total} succeeded | {len(failed)} failed | "
        f"time: {(time.time()-start)/60:.1f}min")
    if failed:
        for item in failed:
            log(f"  FAILED: {item['value']} — {item['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",               default=CSV_PATH)
    parser.add_argument("--output_dir",        default=OUTPUT_DIR)
    parser.add_argument("--topic_gen_outputs", default=TOPIC_GEN_OUTPUTS)
    parser.add_argument("--service",           default=SERVICE)
    parser.add_argument("--model",             default=MODEL)
    parser.add_argument("--phase",             type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (dev fields) or phase 2 (seed scenarios). Omit to run both.")
    args = parser.parse_args()

    df = load_df(args.csv)
    log(f"loaded {len(df)} values from {args.csv}")

    if args.phase in (None, 1):
        phase1(df, args.output_dir, args.service, args.model)

    if args.phase in (None, 2):
        phase2(df, args.output_dir, args.topic_gen_outputs, args.service, args.model)
