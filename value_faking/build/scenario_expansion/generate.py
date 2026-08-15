"""Generation function for expansion scenarios (self-bootstrapped few-shot)."""

import json
from datetime import datetime
from .client import LLMClient
from .prompts import (
    build_expansion_scenario_messages,
    build_expansion_scenario_batch_messages,
    TOPIC_GENERATION_SYSTEM, TOPIC_GENERATION_USER,
)

SERVICE     = "openai"
MODEL       = "gpt-5.4-mini"   # low-cost GPT-5.x tier for generation
TEMPERATURE = 0.9
N_BACKFILL_TOPICS = 20
BATCH_SIZE  = 5


def _log_call(tag: str, model: str, detail: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    suffix = f" — {detail}" if detail else ""
    print(f"[{ts}] [call] {tag} ({model}){suffix}", flush=True)


def generate_expansion_scenario(
    value: str,
    definition: str,
    dev_compliant_objective: str,
    dev_compliant_model_response_behaviour: str,
    topic: str,
    few_shot_examples: list[dict],
    service: str = SERVICE,
    model: str = MODEL,
) -> dict:
    client = LLMClient(service=service, model=model)
    messages = build_expansion_scenario_messages(
        value=value,
        definition=definition,
        dev_compliant_objective=dev_compliant_objective,
        dev_compliant_model_response_behaviour=dev_compliant_model_response_behaviour,
        topic=topic,
        few_shot_examples=few_shot_examples,
    )
    _log_call("generate_scenario", model, topic[:50])
    raw = client.chat(messages=messages, temperature=TEMPERATURE)

    # strip markdown fences if present
    content = raw.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    parsed = json.loads(content)

    # handle both list and bare object responses
    if isinstance(parsed, list):
        parsed = parsed[0]

    required = {"scenario", "developer_compliant", "developer_noncompliant"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Expansion scenario response missing keys: {missing}. Got: {raw}")
    return {k: parsed[k] for k in required}


def generate_expansion_scenarios_batch(
    value: str,
    definition: str,
    dev_compliant_objective: str,
    dev_compliant_model_response_behaviour: str,
    topics: list[str],
    few_shot_examples: list[dict],
    service: str = SERVICE,
    model: str = MODEL,
) -> list[dict]:
    client = LLMClient(service=service, model=model)
    messages = build_expansion_scenario_batch_messages(
        value=value,
        definition=definition,
        dev_compliant_objective=dev_compliant_objective,
        dev_compliant_model_response_behaviour=dev_compliant_model_response_behaviour,
        topics=topics,
        few_shot_examples=few_shot_examples,
    )
    _log_call("generate_scenarios_batch", model, f"{len(topics)} topics")
    raw = client.chat(messages=messages, temperature=TEMPERATURE)

    content = raw.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        for key in parsed:
            if isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array of scenarios, got: {raw}")

    required = {"scenario", "developer_compliant", "developer_noncompliant"}
    results = []
    for item in parsed:
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Batch scenario item missing keys: {missing}. Got: {item}")
        results.append({
            "scenario": item["scenario"],
            "developer_compliant": item["developer_compliant"],
            "developer_noncompliant": item["developer_noncompliant"],
            "topic": item.get("topic", ""),
        })
    return results


def generate_backfill_topics(
    value: str,
    definition: str,
    seed_text: str,
    n_topics: int = N_BACKFILL_TOPICS,
    service: str = SERVICE,
    model: str = MODEL,
) -> list[str]:
    client = LLMClient(service=service, model=model)
    _log_call("generate_backfill_topics", model, f"{n_topics} topics")
    raw = client.chat(
        messages=[
            {"role": "system", "content": TOPIC_GENERATION_SYSTEM.format(n_topics=n_topics)},
            {"role": "user", "content": TOPIC_GENERATION_USER.format(
                value=value, definition=definition, seed_text=seed_text, n_topics=n_topics,
            )},
        ],
        temperature=TEMPERATURE,
        json_mode=True,
    )
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    for key in parsed:
        if isinstance(parsed[key], list):
            return parsed[key]
    raise ValueError(f"Unexpected backfill topic response format: {raw}")
