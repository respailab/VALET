"""Two-phase topic pipeline over all values in a CSV."""

import argparse
import os
import json
import time
import pandas as pd
from datetime import datetime
from value_faking.paths import DATA_ROOT
from .generate import generate_topics
from .validate import validate_topics

_HERE      = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = str(DATA_ROOT / "values.csv")
OUTPUT_DIR = os.path.join(_HERE, "outputs")
SERVICE_GEN  = "openai"
MODEL_GEN    = "gpt-4o-mini"
SERVICE_EVAL = "openai"
MODEL_EVAL   = "gpt-4o-mini"

# alternatives:
# SERVICE_GEN  = "groq";       MODEL_GEN  = "llama-3.3-70b-versatile"
# SERVICE_EVAL = "groq";       MODEL_EVAL = "llama-3.1-8b-instant"
# SERVICE_EVAL = "vllm";       MODEL_EVAL = "openai/gpt-oss-20b"      # set VLLM_BASE_URL
# SERVICE_GEN  = "openrouter"; MODEL_GEN  = "google/gemma-4-26b-a4b-it:free"
N_GENERATE  = 50
N_FINAL     = 20
MAX_RETRIES = 3


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"value", "text", "definition"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.dropna(subset=["definition", "text"]).reset_index(drop=True)


def raw_path(output_dir: str, value: str) -> str:
    return os.path.join(output_dir, value, "topics_raw.json")

def scored_path(output_dir: str, value: str) -> str:
    return os.path.join(output_dir, value, "topics_scored.json")


def phase1(df: pd.DataFrame, output_dir: str, n_generate: int, max_retries: int):
    total   = len(df)
    skipped = sum(1 for _, row in df.iterrows() if os.path.exists(raw_path(output_dir, str(row["value"]))))

    log(f"")
    log(f"{'='*60}")
    log(f"PHASE 1 — TOPIC GENERATION")
    log(f"service: {SERVICE_GEN} | model: {MODEL_GEN} | {n_generate} topics per value")
    log(f"total values: {total} | already done: {skipped} | to run: {total - skipped}")
    log(f"{'='*60}")

    start   = time.time()
    done    = 0
    failed  = []

    for i, row in df.iterrows():
        value     = str(row["value"])
        vdir      = os.path.join(output_dir, value)
        rpath     = raw_path(output_dir, value)

        log(f"")
        log(f"[{i+1}/{total}] {value}")

        if os.path.exists(rpath):
            log(f"  skip — topics_raw.json exists")
            done += 1
            continue

        os.makedirs(vdir, exist_ok=True)
        all_topics: list[str] = []

        for attempt in range(1, max_retries + 1):
            still_needed = n_generate - len(all_topics)
            log(f"  [attempt {attempt}] generating {still_needed} topics...")
            try:
                new_topics = generate_topics(
                    value=value,
                    definition=str(row["definition"]),
                    seed_text=str(row["text"]),
                    n_topics=still_needed,
                    service=SERVICE_GEN,
                    model=MODEL_GEN,
                )
                all_topics.extend(new_topics)
                all_topics = list(dict.fromkeys(all_topics))
                log(f"  received {len(new_topics)} | unique so far: {len(all_topics)}")

                if len(all_topics) >= n_generate:
                    log(f"  target reached — stopping generation")
                    break
            except Exception as e:
                log(f"  [ERROR] attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise

        with open(rpath, "w") as f:
            json.dump({"value": value, "topics": all_topics}, f, indent=2)
        log(f"  saved {len(all_topics)} topics → {rpath}")
        done += 1

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        log(f"  progress: {done} done | {len(failed)} failed | ETA: {eta/60:.1f}min")

    log(f"")
    log(f"PHASE 1 DONE — {done}/{total} succeeded | {len(failed)} failed | "
        f"time: {(time.time()-start)/60:.1f}min")
    if failed:
        for f in failed:
            log(f"  FAILED: {f['value']} — {f['error']}")


def phase2(df: pd.DataFrame, output_dir: str, n_final: int):
    total   = len(df)
    skipped = sum(1 for _, row in df.iterrows() if os.path.exists(scored_path(output_dir, str(row["value"]))))

    log(f"")
    log(f"{'='*60}")
    log(f"PHASE 2 — TOPIC VALIDATION")
    log(f"service: {SERVICE_EVAL} | model: {MODEL_EVAL} | keeping top {n_final} per value")
    log(f"total values: {total} | already done: {skipped} | to run: {total - skipped}")
    log(f"{'='*60}")

    start  = time.time()
    done   = 0
    failed = []

    for i, row in df.iterrows():
        value  = str(row["value"])
        rpath  = raw_path(output_dir, value)
        spath  = scored_path(output_dir, value)

        log(f"")
        log(f"[{i+1}/{total}] {value}")

        if os.path.exists(spath):
            log(f"  skip — topics_scored.json exists")
            done += 1
            continue

        if not os.path.exists(rpath):
            log(f"  skip — no topics_raw.json (run phase 1 first)")
            failed.append({"value": value, "error": "missing topics_raw.json"})
            continue

        with open(rpath) as f:
            data   = json.load(f)
        topics = data["topics"]
        log(f"  loaded {len(topics)} raw topics")

        try:
            top = validate_topics(
                value=value,
                definition=str(row["definition"]),
                topics=topics,
                top_n=n_final,
                service=SERVICE_EVAL,
                model=MODEL_EVAL,
            )

            passing = sum(1 for t in top if t["passes_threshold"])
            log(f"  {passing}/{n_final} topics pass all thresholds")

            with open(spath, "w") as f:
                json.dump({"value": value, "topics": top}, f, indent=2)
            log(f"  saved → {spath}")
            done += 1

        except Exception as e:
            log(f"  [ERROR] {e}")
            failed.append({"value": value, "error": str(e)})

        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        eta = avg * (total - i - 1)
        log(f"  progress: {done} done | {len(failed)} failed | ETA: {eta/60:.1f}min")

    log(f"")
    log(f"PHASE 2 DONE — {done}/{total} succeeded | {len(failed)} failed | "
        f"time: {(time.time()-start)/60:.1f}min")
    if failed:
        for f in failed:
            log(f"  FAILED: {f['value']} — {f['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default=CSV_PATH)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--phase",       type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (generation) or phase 2 (validation). Omit to run both.")
    parser.add_argument("--n_generate",  type=int, default=N_GENERATE)
    parser.add_argument("--n_final",     type=int, default=N_FINAL)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args()

    df = load_df(args.csv)
    log(f"loaded {len(df)} values from {args.csv}")

    if args.phase in (None, 1):
        phase1(df, args.output_dir, args.n_generate, args.max_retries)

    if args.phase in (None, 2):
        phase2(df, args.output_dir, args.n_final)
