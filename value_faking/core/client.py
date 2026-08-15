"""Unified LLM client supporting openai, openrouter, groq, vllm (HTTP server), and hf (HuggingFace transformers) backends."""

import os
import openai
import time

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
    "clod": {
        "base_url": "https://api.clod.io/v1",
        "env_var":  "CLOD_API_KEY",
    },
    "vllm": {
        "base_url": None,                              # read from VLLM_BASE_URL env var
        "env_var":  None,                              # local — no key required by default
    },
    "hf": {
        "base_url": None,
        "env_var":  None,
    },
}


class LLMClient:
    def __init__(
        self,
        service: str,
        model: str,
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ):
        if service not in SERVICES:
            raise ValueError(f"Unknown service '{service}'. Choose from: {list(SERVICES)}")

        self.service   = service
        self.model     = model
        self._hf_model = None

        if service == "hf":
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            try:
                from transformers import AutoModelForVision2Seq
            except ImportError:
                AutoModelForVision2Seq = AutoModelForCausalLM
            print(f"[hf] Loading model: {model}")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            _cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
            _vlm_types = {"mistral3", "pixtral", "llava", "idefics", "blip", "git", "paligemma"}
            _model_cls = AutoModelForVision2Seq if getattr(_cfg, "model_type", "") in _vlm_types else AutoModelForCausalLM
            # Gemma models overflow fp16's dynamic range and emit NaN/inf logits;
            # bfloat16 is what Google trained/recommends them in.
            _model_type = getattr(_cfg, "model_type", "")
            _dtype = torch.bfloat16 if "gemma" in _model_type else torch.float16
            self._hf_model = _model_cls.from_pretrained(
                model, device_map="auto", dtype=_dtype, trust_remote_code=True,
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self._hf_tokenizer.padding_side = "left"
            self._hf_model.eval()
            self._hf_batch_size     = batch_size
            self._hf_max_new_tokens = max_new_tokens
            return

        cfg = SERVICES[service]

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
                "HTTP-Referer":       os.environ.get("OPENROUTER_SITE_URL", ""),
                "X-OpenRouter-Title": os.environ.get("OPENROUTER_SITE_NAME", "conflict_research"),
            }

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        self._client = openai.OpenAI(**kwargs)

    def _hf_messages_to_prompt(self, messages: list[dict]) -> str:
        if self._hf_tokenizer.chat_template:
            return self._hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(m["content"] for m in messages)

    def _hf_generate_batch(self, prompts: list[str], temperature: float) -> list[str]:
        import torch
        inputs = self._hf_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self._hf_model.device)
        do_sample = temperature > 0.0
        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=self._hf_max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self._hf_tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return [
            self._hf_tokenizer.decode(out[input_len:], skip_special_tokens=True)
            for out in outputs
        ]

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        if self.service == "hf":
            prompt = self._hf_messages_to_prompt(messages)
            return self._hf_generate_batch([prompt], temperature)[0]

        kwargs = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Retry transient server-side errors (e.g. vLLM's gpt-oss harmony-format
        # parser occasionally 500s under sustained load — intermittent, not tied
        # to specific content, so a retry usually succeeds).
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except (openai.APIStatusError, openai.APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[client] {type(e).__name__} on attempt {attempt + 1}/{max_retries}, retrying in {wait}s: {e}")
                time.sleep(wait)

        if not response.choices:
            raise RuntimeError(f"Empty choices from {self.model} — model may not support json_mode or was rate-limited. Response: {response}")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"None content from {self.model} — finish_reason: {response.choices[0].finish_reason}")
        return content

    def _safe_chat(self, messages: list[dict], temperature: float, json_mode: bool) -> str:
        try:
            return self.chat(messages, temperature=temperature, json_mode=json_mode)
        except Exception as e:
            print(f"[client] giving up on one item after retries — marking parse_error: {e}")
            return '{"parse_error": true}'

    def batch(
        self,
        messages_list: list[list[dict]],
        temperature: float = 0.0,
        json_mode: bool = False,
        max_workers: int = 8,
        requests_per_second: float = None,
    ) -> list[str]:
        if self.service == "hf":
            from tqdm import tqdm
            prompts = [self._hf_messages_to_prompt(m) for m in messages_list]
            results = []
            n_batches = (len(prompts) + self._hf_batch_size - 1) // self._hf_batch_size
            for i in tqdm(range(0, len(prompts), self._hf_batch_size),
                          desc="HF batches", unit="batch", total=n_batches):
                results.extend(self._hf_generate_batch(prompts[i:i + self._hf_batch_size], temperature))
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        if requests_per_second is not None:
            # sequential with fixed interval — simplest correct throttle
            results = []
            interval = 1.0 / requests_per_second
            for messages in messages_list:
                t0 = time.time()
                results.append(self._safe_chat(messages, temperature=temperature, json_mode=json_mode))
                elapsed = time.time() - t0
                remaining = interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            return results

        def _call(args):
            idx, messages = args
            return idx, self._safe_chat(messages, temperature=temperature, json_mode=json_mode)

        results = [None] * len(messages_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call, (i, m)): i for i, m in enumerate(messages_list)}
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response
        return results
