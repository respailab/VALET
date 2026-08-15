"""Generation functions for dev fields and seed scenarios."""

import json
from .client import LLMClient
from .prompts import DEV_FIELD_SYSTEM, DEV_FIELD_USER, build_seed_scenario_messages

SERVICE     = "openai"
MODEL       = "gpt-4o-mini"
TEMPERATURE = 0.9


def generate_dev_fields(
    value: str,
    definition: str,
    service: str = SERVICE,
    model: str = MODEL,
) -> dict:
    client = LLMClient(service=service, model=model)
    raw = client.chat(
        messages=[
            {"role": "system", "content": DEV_FIELD_SYSTEM},
            {"role": "user",   "content": DEV_FIELD_USER.format(value=value, definition=definition)},
        ],
        temperature=0.0,
        json_mode=True,
    )
    parsed = json.loads(raw)
    required = {"dev_compliant_objective", "dev_compliant_model_response_behaviour"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Dev field response missing keys: {missing}. Got: {raw}")
    return {k: parsed[k] for k in required}


def generate_seed_scenario(
    value: str,
    definition: str,
    dev_compliant_objective: str,
    dev_compliant_model_response_behaviour: str,
    topic: str,
    service: str = SERVICE,
    model: str = MODEL,
) -> dict:
    client = LLMClient(service=service, model=model)
    messages = build_seed_scenario_messages(
        value=value,
        definition=definition,
        dev_compliant_objective=dev_compliant_objective,
        dev_compliant_model_response_behaviour=dev_compliant_model_response_behaviour,
        topic=topic,
    )
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
        raise ValueError(f"Seed scenario response missing keys: {missing}. Got: {raw}")
    return {k: parsed[k] for k in required}
