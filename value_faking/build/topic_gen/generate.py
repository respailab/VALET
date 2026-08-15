"""Topic generation for a single value dimension."""

import json
import argparse
from .client import LLMClient
from .prompts import TOPIC_GENERATION_SYSTEM, TOPIC_GENERATION_USER

SERVICE     = "openai"
MODEL       = "gpt-4o-mini"
N_TOPICS    = 50
TEMPERATURE = 0.9


def generate_topics(
    value: str,
    definition: str,
    seed_text: str,
    n_topics: int   = N_TOPICS,
    service: str    = SERVICE,
    model: str      = MODEL,
) -> list[str]:
    client = LLMClient(service=service, model=model)

    raw = client.chat(
        messages=[
            {
                "role": "system",
                "content": TOPIC_GENERATION_SYSTEM.format(n_topics=n_topics),
            },
            {
                "role": "user",
                "content": TOPIC_GENERATION_USER.format(
                    value=value,
                    definition=definition,
                    seed_text=seed_text,
                    n_topics=n_topics,
                ),
            },
        ],
        temperature=TEMPERATURE,
        json_mode=True,
    )

    parsed = json.loads(raw)

    # handle both {"topics": [...]} and plain [...] responses
    if isinstance(parsed, list):
        return parsed
    for key in parsed:
        if isinstance(parsed[key], list):
            return parsed[key]
    raise ValueError(f"Unexpected response format: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value",      required=True)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--seed_text",  required=True)
    parser.add_argument("--n_topics",   type=int, default=N_TOPICS)
    parser.add_argument("--service",    default=SERVICE)
    parser.add_argument("--model",      default=MODEL)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    topics = generate_topics(
        value=args.value,
        definition=args.definition,
        seed_text=args.seed_text,
        n_topics=args.n_topics,
        service=args.service,
        model=args.model,
    )

    with open(args.output, "w") as f:
        json.dump({"value": args.value, "topics": topics}, f, indent=2)

    print(f"Generated {len(topics)} topics → {args.output}")
