"""Unified LLM client supporting openai, openrouter, groq, and vllm backends. Loads only the env variable required for the chosen service."""

import os
import openai


SERVICES = {
    "openai": {
        "base_url": None,                              # default OpenAI endpoint
        "env_var":  "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var":  "OPENROUTER_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_var":  "GROQ_API_KEY",
    },
    "vllm": {
        "base_url": None,                              # read from VLLM_BASE_URL env var
        "env_var":  None,                              # local — no key required by default
    },
}


class LLMClient:
    def __init__(self, service: str, model: str):
        if service not in SERVICES:
            raise ValueError(f"Unknown service '{service}'. Choose from: {list(SERVICES)}")

        self.service = service
        self.model   = model
        cfg          = SERVICES[service]

        # resolve api key
        env_var = cfg["env_var"]
        if env_var:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise EnvironmentError(
                    f"Service '{service}' requires {env_var} to be set.\n"
                    f"Run: export {env_var}=your_key"
                )
        else:
            api_key = os.environ.get("VLLM_API_KEY", "EMPTY")   # vllm accepts any non-empty key

        # resolve base url
        if service == "vllm":
            base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        else:
            base_url = cfg["base_url"]

        # extra headers for openrouter
        extra_headers = {}
        if service == "openrouter":
            extra_headers = {
                "HTTP-Referer":      os.environ.get("OPENROUTER_SITE_URL", ""),
                "X-OpenRouter-Title": os.environ.get("OPENROUTER_SITE_NAME", "conflict_research"),
            }

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        self._client = openai.OpenAI(**kwargs)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        kwargs = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
