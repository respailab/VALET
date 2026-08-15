"""End-to-end seed scenario generation pipeline for a single value."""

import json
import argparse
import os
import time
from datetime import datetime
from .generate import generate_dev_fields, generate_seed_scenario

MODEL   = "gpt-4o-mini"
SERVICE = "openai"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_pipeline(
    value: str,
    definition: str,
    topics_file: str,
    output_dir: str,
    service: str = SERVICE,
    model: str = MODEL,
) -> dict:
    pipeline_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    log(f"{'='*60}")
    log(f"PIPELINE START — value: '{value}'")
    log(f"output dir: {output_dir}")
    log(f"{'='*60}")

    with open(topics_file) as f:
        scored = json.load(f)
    topics = [t["topic"] for t in scored["topics"] if t.get("passes_threshold", True)]
    log(f"loaded {len(topics)} passing topics from {topics_file}")

    # ── Phase 1 — Dev fields ──────────────────────────────────────
    dev_fields_path = os.path.join(output_dir, "dev_fields.json")
    if os.path.exists(dev_fields_path):
        log(f"[phase 1] skip — dev_fields.json exists")
        with open(dev_fields_path) as f:
            dev_fields = json.load(f)
    else:
        log(f"[phase 1] generating dev fields...")
        start = time.time()
        dev_fields = generate_dev_fields(
            value=value,
            definition=definition,
            service=service,
            model=model,
        )
        dev_fields["value"] = value
        dev_fields["definition"] = definition
        with open(dev_fields_path, "w") as f:
            json.dump(dev_fields, f, indent=2)
        log(f"[phase 1] done in {time.time()-start:.1f}s — saved → {dev_fields_path}")
        log(f"  objective:   {dev_fields['dev_compliant_objective']}")
        log(f"  behaviour:   {dev_fields['dev_compliant_model_response_behaviour']}")

    # ── Phase 2 — Seed scenarios ──────────────────────────────────
    seed_scenarios_path = os.path.join(output_dir, "seed_scenarios.json")
    if os.path.exists(seed_scenarios_path):
        log(f"[phase 2] skip — seed_scenarios.json exists")
        with open(seed_scenarios_path) as f:
            result = json.load(f)
        return result

    log(f"[phase 2] generating {len(topics)} seed scenarios...")
    scenarios = []
    failed = []

    for i, topic in enumerate(topics):
        log(f"  [{i+1}/{len(topics)}] {topic[:80]}...")
        try:
            start = time.time()
            scenario = generate_seed_scenario(
                value=value,
                definition=definition,
                dev_compliant_objective=dev_fields["dev_compliant_objective"],
                dev_compliant_model_response_behaviour=dev_fields["dev_compliant_model_response_behaviour"],
                topic=topic,
                service=service,
                model=model,
            )
            scenario["topic"] = topic
            scenarios.append(scenario)
            log(f"  done in {time.time()-start:.1f}s")
        except Exception as e:
            log(f"  [ERROR] {e}")
            failed.append({"topic": topic, "error": str(e)})

    result = {
        "value": value,
        "definition": definition,
        "dev_compliant_objective": dev_fields["dev_compliant_objective"],
        "dev_compliant_model_response_behaviour": dev_fields["dev_compliant_model_response_behaviour"],
        "scenarios": scenarios,
        "failed": failed,
    }

    with open(seed_scenarios_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"")
    log(f"{'='*60}")
    log(f"PIPELINE DONE — {len(scenarios)} scenarios generated | {len(failed)} failed | "
        f"time: {(time.time()-pipeline_start):.1f}s")
    log(f"saved → {seed_scenarios_path}")
    log(f"{'='*60}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value",       required=True)
    parser.add_argument("--definition",  required=True)
    parser.add_argument("--topics_file", required=True,
                        help="Path to topics_scored.json from topic_gen outputs")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--service",     default=SERVICE)
    parser.add_argument("--model",       default=MODEL)
    args = parser.parse_args()

    run_pipeline(
        value=args.value,
        definition=args.definition,
        topics_file=args.topics_file,
        output_dir=args.output_dir,
        service=args.service,
        model=args.model,
    )
