"""Topic validation for a list of generated topics."""

import json
import argparse
from .client import LLMClient
from .prompts import TOPIC_VALIDATION_SYSTEM, TOPIC_VALIDATION_USER

SERVICE     = "openai"
MODEL       = "gpt-4o-mini"
TEMPERATURE = 0.0

THRESHOLD_ISOLATION      = 3
THRESHOLD_DIRECTIONALITY = 3
THRESHOLD_OPPOSABILITY   = 3
THRESHOLD_CORRECTNESS    = 3
THRESHOLD_PLAUSIBILITY   = 3

AXES = ["value_isolation", "value_directionality", "opposability", "correctness", "plausibility"]


def score_topic(
    value: str,
    definition: str,
    topic: str,
    service: str = SERVICE,
    model: str   = MODEL,
) -> dict:
    client = LLMClient(service=service, model=model)

    raw = client.chat(
        messages=[
            {"role": "system", "content": TOPIC_VALIDATION_SYSTEM},
            {
                "role": "user",
                "content": TOPIC_VALIDATION_USER.format(
                    value=value,
                    definition=definition,
                    topic=topic,
                ),
            },
        ],
        temperature=TEMPERATURE,
        json_mode=True,
    )
    return json.loads(raw)


def passes_threshold(
    scores: dict,
    threshold_isolation: int      = THRESHOLD_ISOLATION,
    threshold_directionality: int = THRESHOLD_DIRECTIONALITY,
    threshold_opposability: int   = THRESHOLD_OPPOSABILITY,
    threshold_correctness: int    = THRESHOLD_CORRECTNESS,
    threshold_plausibility: int   = THRESHOLD_PLAUSIBILITY,
) -> bool:
    return (
        scores.get("value_isolation", 0)          >= threshold_isolation
        and scores.get("value_directionality", 0) >= threshold_directionality
        and scores.get("opposability", 0)         >= threshold_opposability
        and scores.get("correctness", 0)          >= threshold_correctness
        and scores.get("plausibility", 0)         >= threshold_plausibility
    )


def validate_topics(
    value: str,
    definition: str,
    topics: list[str],
    top_n: int   = 20,
    service: str = SERVICE,
    model: str   = MODEL,
    threshold_isolation: int      = THRESHOLD_ISOLATION,
    threshold_directionality: int = THRESHOLD_DIRECTIONALITY,
    threshold_opposability: int   = THRESHOLD_OPPOSABILITY,
    threshold_correctness: int    = THRESHOLD_CORRECTNESS,
    threshold_plausibility: int   = THRESHOLD_PLAUSIBILITY,
) -> list[dict]:
    scored = []

    for topic in topics:
        try:
            scores = score_topic(value, definition, topic, service, model)
            scores["topic"] = topic
            scores["combined_score"] = sum(scores.get(ax, 0) for ax in AXES)
            scores["passes_threshold"] = passes_threshold(
                scores,
                threshold_isolation,
                threshold_directionality,
                threshold_opposability,
                threshold_correctness,
                threshold_plausibility,
            )
            scored.append(scores)
        except Exception as e:
            print(f"  [warn] failed to score topic: {topic!r} — {e}")

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top = scored[:top_n]

    n_passing = sum(1 for t in top if t["passes_threshold"])
    print(f"Scored {len(scored)} topics | top {top_n} selected | {n_passing} pass all thresholds")

    if top:
        avg = sum(t["combined_score"] for t in top) / len(top)
        print(f"Avg combined score (top {top_n}): {avg:.1f}/25")

        failing = [t for t in top if not t["passes_threshold"]]
        if failing:
            weakest = {}
            for t in failing:
                w = t.get("weakest_criterion", "unknown")
                weakest[w] = weakest.get(w, 0) + 1
            print(f"Weakest criteria in failing topics: {weakest}")

    return top


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",                    required=True)
    parser.add_argument("--definition",               required=True)
    parser.add_argument("--output",                   required=True)
    parser.add_argument("--top_n",                    type=int, default=20)
    parser.add_argument("--service",                  default=SERVICE)
    parser.add_argument("--model",                    default=MODEL)
    parser.add_argument("--threshold_isolation",      type=int, default=THRESHOLD_ISOLATION)
    parser.add_argument("--threshold_directionality", type=int, default=THRESHOLD_DIRECTIONALITY)
    parser.add_argument("--threshold_opposability",   type=int, default=THRESHOLD_OPPOSABILITY)
    parser.add_argument("--threshold_correctness",    type=int, default=THRESHOLD_CORRECTNESS)
    parser.add_argument("--threshold_plausibility",   type=int, default=THRESHOLD_PLAUSIBILITY)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    top_topics = validate_topics(
        value=data["value"],
        definition=args.definition,
        topics=data["topics"],
        top_n=args.top_n,
        service=args.service,
        model=args.model,
        threshold_isolation=args.threshold_isolation,
        threshold_directionality=args.threshold_directionality,
        threshold_opposability=args.threshold_opposability,
        threshold_correctness=args.threshold_correctness,
        threshold_plausibility=args.threshold_plausibility,
    )

    with open(args.output, "w") as f:
        json.dump({"value": data["value"], "topics": top_topics}, f, indent=2)

    print(f"Saved {len(top_topics)} scored topics → {args.output}")
