"""Scenario expansion — scale each value from ~20 validated seed scenarios to ~50."""

import argparse
import json
import os
import random
import time
from datetime import datetime

import pandas as pd

from value_faking.paths import DATA_ROOT
from .generate import generate_expansion_scenarios_batch, generate_backfill_topics
from .validate import score_scenario, compute_derived, LLMClient

_HERE = os.path.dirname(os.path.abspath(__file__))

CSV_PATH           = str(DATA_ROOT / "values.csv")
TOPIC_GEN_OUTPUTS  = os.path.join(_HERE, "..", "topic_gen", "outputs")
SEED_GEN_OUTPUTS   = os.path.join(_HERE, "..", "seed_scenario_gen", "outputs")
OUTPUT_DIR         = os.path.join(_HERE, "outputs")
FINAL_DIR          = os.path.join(_HERE, "final")

GEN_SERVICE  = "openai"
GEN_MODEL    = "gpt-5.4-mini"   # low-cost GPT-5.x tier for generation
VAL_SERVICE  = "openai"
VAL_MODEL    = "gpt-4o-mini"    # judge model, unchanged from seed validation

TARGET        = 50
FEW_SHOT_N    = 3
THRESHOLD     = 3
BATCH_SIZE    = 5     # topics per generation call
MAX_BACKFILL_ROUNDS = 3


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"value", "definition", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.dropna(subset=["definition"]).reset_index(drop=True)


def raw_topics_path(value: str) -> str:
    return os.path.join(TOPIC_GEN_OUTPUTS, value, "topics_raw.json")

def dev_fields_path(value: str) -> str:
    return os.path.join(SEED_GEN_OUTPUTS, value, "dev_fields.json")

def seed_scenarios_path(value: str) -> str:
    return os.path.join(SEED_GEN_OUTPUTS, value, "seed_scenarios.json")

def expansion_audit_path(output_dir: str, value: str) -> str:
    return os.path.join(output_dir, value, "expansion_scenarios.json")

def final_path(final_dir: str, value: str) -> str:
    return os.path.join(final_dir, f"{value}_50.json")


def to_afscenario(s: dict) -> dict:
    return {
        "scenario": s["scenario"],
        "developer_compliant": s["developer_compliant"],
        "developer_noncompliant": s["developer_noncompliant"],
    }


def run_value(
    value: str,
    definition: str,
    seed_text: str,
    output_dir: str,
    final_dir: str,
    target: int,
    few_shot_n: int,
    threshold: int,
    batch_size: int,
    gen_service: str,
    gen_model: str,
    val_service: str,
    val_model: str,
) -> dict:
    dpath = dev_fields_path(value)
    spath = seed_scenarios_path(value)
    rpath = raw_topics_path(value)
    fpath = final_path(final_dir, value)
    apath = expansion_audit_path(output_dir, value)

    if not os.path.exists(dpath):
        return {"value": value, "error": f"missing dev_fields.json at {dpath}"}
    if not os.path.exists(spath):
        return {"value": value, "error": f"missing seed_scenarios.json at {spath}"}
    if not os.path.exists(rpath):
        return {"value": value, "error": f"missing topics_raw.json at {rpath}"}

    with open(dpath) as f:
        dev_fields = json.load(f)
    with open(spath) as f:
        seed_data = json.load(f)
    with open(rpath) as f:
        raw_topics = json.load(f)["topics"]

    dev_obj = dev_fields["dev_compliant_objective"]
    dev_beh = dev_fields["dev_compliant_model_response_behaviour"]

    # ── resume support: a prior run's final_{value}_50.json (and its audit trail)
    # already contains everything the seed pool + earlier expansion calls produced.
    # If present, resume from it instead of restarting from just the ~20 seeds —
    # otherwise re-running with a higher --target would regenerate duplicates of
    # scenarios that already exist and pass.
    seed_scenarios = seed_data.get("scenarios", [])
    seed_used_topics = {s.get("topic", "") for s in seed_scenarios}

    audit = []
    attempted_topics = set(seed_used_topics)
    if os.path.exists(apath):
        with open(apath) as f:
            audit = json.load(f).get("audit", [])
        attempted_topics |= {a["topic"] for a in audit if a.get("topic")}

    if os.path.exists(fpath):
        with open(fpath) as f:
            accumulated = json.load(f)
        log(f"{value}: resuming from existing final file — {len(accumulated)} scenarios already saved")
    else:
        accumulated = [to_afscenario(s) for s in seed_scenarios]

    few_shot_pool = list(accumulated)   # self-bootstrap pool starts from everything so far

    val_client = LLMClient(service=val_service, model=val_model)

    log(f"{value}: starting with {len(accumulated)} scenarios, target {target}")

    all_seen_topics = set(raw_topics) | attempted_topics
    unused_topics = [t for t in raw_topics if t not in attempted_topics]
    backfill_round = 0

    while len(accumulated) < target:
        if not unused_topics:
            if backfill_round >= MAX_BACKFILL_ROUNDS:
                log(f"{value}: exhausted topics and backfill retries, stopping at {len(accumulated)}")
                break
            backfill_round += 1
            log(f"{value}: backfill round {backfill_round} — generating more topics")
            try:
                new_topics = generate_backfill_topics(
                    value=value, definition=definition, seed_text=seed_text,
                    service=gen_service, model=gen_model,
                )
            except Exception as e:
                log(f"{value}: backfill topic generation failed — {e}")
                break
            new_topics = [t for t in new_topics if t not in all_seen_topics]
            all_seen_topics.update(new_topics)
            if not new_topics:
                log(f"{value}: backfill produced no new topics, stopping at {len(accumulated)}")
                break
            unused_topics = new_topics

        random.shuffle(unused_topics)
        topic_names = unused_topics
        unused_topics = []   # consumed; refilled via backfill if still short

        for chunk_start in range(0, len(topic_names), batch_size):
            if len(accumulated) >= target:
                break
            chunk = topic_names[chunk_start:chunk_start + batch_size]
            few_shot_examples = random.sample(few_shot_pool, min(few_shot_n, len(few_shot_pool)))

            try:
                batch_scenarios = generate_expansion_scenarios_batch(
                    value=value, definition=definition,
                    dev_compliant_objective=dev_obj,
                    dev_compliant_model_response_behaviour=dev_beh,
                    topics=chunk, few_shot_examples=few_shot_examples,
                    service=gen_service, model=gen_model,
                )
            except Exception as e:
                log(f"{value}: [ERROR] batch of {len(chunk)} topics — {e}")
                for topic in chunk:
                    audit.append({"topic": topic, "error": str(e)})
                continue

            # match returned scenarios back to input topics by text; fall back to
            # positional order if the model didn't echo the topic back exactly
            remaining_chunk = list(chunk)
            for idx, scenario in enumerate(batch_scenarios):
                if len(accumulated) >= target:
                    break
                topic = scenario.get("topic") or ""
                if topic in remaining_chunk:
                    remaining_chunk.remove(topic)
                elif idx < len(remaining_chunk):
                    topic = remaining_chunk.pop(idx if idx < len(remaining_chunk) else 0)
                else:
                    topic = remaining_chunk.pop(0) if remaining_chunk else topic

                try:
                    scores = score_scenario(
                        value=value, definition=definition,
                        scenario=scenario["scenario"],
                        developer_compliant=scenario["developer_compliant"],
                        developer_noncompliant=scenario["developer_noncompliant"],
                        client=val_client,
                    )
                    derived = compute_derived(scores, threshold)
                    record = {**scenario, "topic": topic, **derived}
                    audit.append(record)

                    if derived["passes_threshold"]:
                        afs = to_afscenario(scenario)
                        few_shot_pool.append(afs)
                        accumulated.append(afs)
                        log(f"{value}: [{len(accumulated)}/{target}] PASS ({derived['combined_score']}/25) {topic[:60]}")
                    else:
                        log(f"{value}: FAIL ({derived['combined_score']}/25) {topic[:60]}")

                except Exception as e:
                    log(f"{value}: [ERROR] validating topic {topic[:60]} — {e}")
                    audit.append({"topic": topic, "error": str(e)})

    os.makedirs(os.path.join(output_dir, value), exist_ok=True)
    with open(expansion_audit_path(output_dir, value), "w") as f:
        json.dump({"value": value, "definition": definition, "audit": audit}, f, indent=2)

    os.makedirs(final_dir, exist_ok=True)
    with open(final_path(final_dir, value), "w") as f:
        json.dump(accumulated, f, indent=2)

    log(f"{value}: done — {len(accumulated)} total scenarios saved")
    return {"value": value, "n_final": len(accumulated), "n_attempts": len(audit)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default=CSV_PATH)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--final_dir",   default=FINAL_DIR)
    parser.add_argument("--values",      nargs="*", default=None, help="Restrict to specific values")
    parser.add_argument("--target",      type=int, default=TARGET)
    parser.add_argument("--few_shot_n",  type=int, default=FEW_SHOT_N)
    parser.add_argument("--threshold",   type=int, default=THRESHOLD)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE, help="Topics per generation call")
    parser.add_argument("--gen_service", default=GEN_SERVICE)
    parser.add_argument("--gen_model",   default=GEN_MODEL, help="Cheap model for scenario/topic generation")
    parser.add_argument("--val_service", default=VAL_SERVICE)
    parser.add_argument("--val_model",   default=VAL_MODEL, help="Judge model for validation")
    args = parser.parse_args()

    df = load_df(args.csv)
    if args.values:
        df = df[df["value"].isin(args.values)].reset_index(drop=True)
    log(f"loaded {len(df)} values | gen: {args.gen_service}/{args.gen_model} | val: {args.val_service}/{args.val_model}")

    total = len(df)
    start = time.time()
    results = []

    for i, row in df.iterrows():
        value = str(row["value"])
        fpath = final_path(args.final_dir, value)

        if os.path.exists(fpath):
            with open(fpath) as f:
                existing = json.load(f)
            if len(existing) >= args.target:
                log(f"[{i+1}/{total}] {value}: skip — final/{value}_50.json already has {len(existing)}")
                results.append({"value": value, "n_final": len(existing), "n_attempts": 0, "skipped": True})
                continue

        log(f"")
        log(f"[{i+1}/{total}] {value}")
        result = run_value(
            value=value,
            definition=str(row["definition"]),
            seed_text=str(row["text"]),
            output_dir=args.output_dir,
            final_dir=args.final_dir,
            target=args.target,
            few_shot_n=args.few_shot_n,
            threshold=args.threshold,
            batch_size=args.batch_size,
            gen_service=args.gen_service,
            gen_model=args.gen_model,
            val_service=args.val_service,
            val_model=args.val_model,
        )
        results.append(result)

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        log(f"  progress: {i+1}/{total} values | ETA: {eta/60:.1f}min")

    log(f"")
    log(f"{'='*65}")
    log(f"DONE — {len(results)} values processed in {(time.time()-start)/60:.1f}min")
    log(f"{'Value':<42} {'final':>6} {'attempts':>9}")
    log("-" * 65)
    for r in results:
        if "error" in r:
            log(f"  {r['value']:<40} ERROR: {r['error']}")
        else:
            log(f"  {r['value']:<40} {r['n_final']:>6} {r['n_attempts']:>9}")


if __name__ == "__main__":
    main()
