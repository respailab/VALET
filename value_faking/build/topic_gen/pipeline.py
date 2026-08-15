"""End-to-end topic generation + validation pipeline for a single value."""

import json
import argparse
import os
import time
from datetime import datetime
from .generate import generate_topics
from .validate import validate_topics, passes_threshold

N_GENERATE   = 50
N_FINAL      = 20
MAX_RETRIES  = 3

MODEL_GEN    = "nvidia/nemotron-3-nano-30b-a3b:free"
MODEL_EVAL   = "openai/gpt-oss-20b:free"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_pipeline(
    value: str,
    definition: str,
    seed_text: str,
    output_dir: str,
    n_generate: int = N_GENERATE,
    n_final: int = N_FINAL,
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    pipeline_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    all_topics: list[str] = []
    top = []

    log(f"{'='*60}")
    log(f"PIPELINE START — value: '{value}'")
    log(f"target: {n_final} passing topics | max attempts: {max_retries}")
    log(f"output dir: {output_dir}")
    log(f"{'='*60}")

    for attempt in range(1, max_retries + 1):
        attempt_start = time.time()
        still_needed  = n_generate - len(all_topics)

        log(f"")
        log(f"--- ATTEMPT {attempt}/{max_retries} ---")
        log(f"topics already collected: {len(all_topics)} | requesting {still_needed} more")

        log(f"[gen] calling model ({MODEL_GEN}) for {still_needed} topics...")
        gen_start  = time.time()
        new_topics = generate_topics(
            value=value,
            definition=definition,
            seed_text=seed_text,
            n_topics=still_needed,
            model=MODEL_GEN,
        )
        log(f"[gen] received {len(new_topics)} topics in {time.time()-gen_start:.1f}s")

        before_dedup = len(all_topics)
        all_topics.extend(new_topics)
        all_topics = list(dict.fromkeys(all_topics))
        dupes_removed = before_dedup + len(new_topics) - len(all_topics)
        log(f"[gen] after dedup: {len(all_topics)} unique topics ({dupes_removed} duplicates removed)")

        log(f"[val] scoring {len(all_topics)} topics with model ({MODEL_EVAL})...")
        val_start = time.time()
        top = validate_topics(
            value=value,
            definition=definition,
            topics=all_topics,
            top_n=n_final,
            model=MODEL_EVAL,
        )
        log(f"[val] scoring complete in {time.time()-val_start:.1f}s")

        passing   = [t for t in top if t["passes_threshold"]]
        failing   = [t for t in top if not t["passes_threshold"]]
        avg_score = sum(t["combined_score"] for t in top) / len(top) if top else 0

        log(f"[val] top {n_final} topics — avg combined score: {avg_score:.1f}/15")
        log(f"[val] passing all thresholds: {len(passing)} | failing: {len(failing)}")

        if failing:
            weakest = {}
            for t in failing:
                w = t.get("weakest_criterion", "unknown")
                weakest[w] = weakest.get(w, 0) + 1
            log(f"[val] weakest criteria in failing topics: {weakest}")

        log(f"[attempt {attempt}] elapsed: {time.time()-attempt_start:.1f}s")

        if len(passing) >= n_final:
            log(f"")
            log(f"✓ threshold met ({len(passing)}/{n_final}) — stopping early")
            break

        if attempt < max_retries:
            shortfall = n_final - len(passing)
            log(f"✗ threshold not met — shortfall: {shortfall} topics")
            log(f"  re-generating {shortfall} more topics on next attempt...")
            n_generate = shortfall  # only ask for what's still needed
        else:
            log(f"✗ max retries reached — proceeding with {len(passing)}/{n_final} passing topics")

    log(f"")
    log(f"--- SAVING OUTPUTS ---")

    raw_path    = os.path.join(output_dir, "topics_raw.json")
    scored_path = os.path.join(output_dir, "topics_scored.json")

    with open(raw_path, "w") as f:
        json.dump({"value": value, "topics": all_topics}, f, indent=2)
    log(f"saved {len(all_topics)} raw topics → {raw_path}")

    with open(scored_path, "w") as f:
        json.dump({"value": value, "topics": top}, f, indent=2)
    log(f"saved {len(top)} scored topics → {scored_path}")

    total_time = time.time() - pipeline_start
    log(f"")
    log(f"{'='*60}")
    log(f"PIPELINE DONE — {len([t for t in top if t['passes_threshold']])} passing topics | total time: {total_time:.1f}s")
    log(f"{'='*60}")

    return top


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value",       required=True)
    parser.add_argument("--definition",  required=True)
    parser.add_argument("--seed_text",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_generate",  type=int, default=N_GENERATE)
    parser.add_argument("--n_final",     type=int, default=N_FINAL)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args()

    run_pipeline(
        value=args.value,
        definition=args.definition,
        seed_text=args.seed_text,
        output_dir=args.output_dir,
        n_generate=args.n_generate,
        n_final=args.n_final,
        max_retries=args.max_retries,
    )
