"""Provider-agnostic LLM client.

Groq, Ollama, OpenAI and Azure OpenAI all speak the OpenAI Chat Completions
API, so one client covers them; switching provider is a config change.
This matters because hosted models get retired (Groq shut down its Llama 3.x
models for free accounts in August 2026) and clients have different
constraints (e.g. "data must not leave our network" -> local Ollama).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


class LLMClient:
    def __init__(self, provider: str, base_url: str, model: str, api_key_env: str = "",
                 temperature: float = 0.1, max_tokens: int = 1024, extra_body: dict | None = None):
        api_key = os.getenv(api_key_env) if api_key_env else "not-needed"  # Ollama needs no key
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set. Add it to your .env file.")
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body = extra_body or {}
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_config(cls, llm_cfg: dict) -> "LLMClient":
        name = llm_cfg["provider"]
        p = llm_cfg["providers"][name]
        return cls(provider=name, base_url=p["base_url"], model=p["model"],
                   api_key_env=p.get("api_key_env", ""),
                   temperature=llm_cfg.get("temperature", 0.1),
                   max_tokens=llm_cfg.get("max_tokens", 1024),
                   extra_body=p.get("extra_body"))

    def complete(self, messages: list[dict]) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature,
            max_tokens=self.max_tokens, extra_body=self.extra_body or None,
        )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=(resp.choices[0].message.content or "").strip(),
            model=self.model,
            latency_s=round(time.perf_counter() - t0, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
