"""Provider-agnostic async LLM client (OpenAI-compatible endpoints).

Endpoints come from configs/models.yaml, never hardcoded. Every request and
raw completion is appended to a JSONL log for debugging and ablation analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str | None = None  # env var holding the key; local servers need none

    @property
    def api_key(self) -> str:
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"provider '{self.name}' needs API key in env var {self.api_key_env}"
                )
            return key
        return "not-needed"


@dataclass(frozen=True)
class TierConfig:
    name: str  # e.g. "fast" | "strong"
    provider: str  # ProviderConfig.name
    model: str
    weight: float = 1.0  # ensemble sampling weight
    temperature: float = 0.8
    max_tokens: int = 4096


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    tier: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLMClient:
    def __init__(self, provider: ProviderConfig, log_path: Path | None = None) -> None:
        self.provider = provider
        self._client = openai.AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key)
        self._log_path = log_path
        self._log_lock = asyncio.Lock()

    async def probe(self) -> list[str]:
        """List served model ids; used at startup to fail fast and clearly."""
        try:
            models = await self._client.models.list()
        except openai.OpenAIError as exc:
            raise RuntimeError(
                f"provider '{self.provider.name}' unreachable at {self.provider.base_url}: {exc}"
            ) from exc
        return [m.id for m in models.data]

    async def complete(
        self,
        prompt: str,
        *,
        tier: TierConfig,
        system: str | None = None,
        purpose: str = "evolve",
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        started = time.monotonic()
        response = await self._client.chat.completions.create(
            model=tier.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=tier.temperature,
            max_tokens=tier.max_tokens,
        )
        seconds = time.monotonic() - started
        text = response.choices[0].message.content or ""
        usage = response.usage
        completion = Completion(
            text=text,
            model=tier.model,
            tier=tier.name,
            provider=self.provider.name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            seconds=seconds,
        )
        await self._log(prompt, system, completion, purpose)
        return completion

    async def _log(
        self, prompt: str, system: str | None, completion: Completion, purpose: str
    ) -> None:
        if self._log_path is None:
            return
        record = {
            "ts": time.time(),
            "purpose": purpose,
            "provider": completion.provider,
            "model": completion.model,
            "tier": completion.tier,
            "system": system,
            "prompt": prompt,
            "completion": completion.text,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "seconds": completion.seconds,
        }
        async with self._log_lock:
            with self._log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")


@dataclass
class ModelRegistry:
    """Parsed configs/models.yaml: named providers + named tiers."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    tiers: dict[str, TierConfig] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> ModelRegistry:
        providers = {
            name: ProviderConfig(
                name=name,
                base_url=str(p["base_url"]).rstrip("/"),
                api_key_env=p.get("api_key_env"),
            )
            for name, p in cfg.get("providers", {}).items()
        }
        tiers = {}
        for name, t in cfg.get("tiers", {}).items():
            if t["provider"] not in providers:
                raise ValueError(f"tier '{name}' references unknown provider '{t['provider']}'")
            tiers[name] = TierConfig(
                name=name,
                provider=t["provider"],
                model=str(t["model"]),
                weight=float(t.get("weight", 1.0)),
                temperature=float(t.get("temperature", 0.8)),
                max_tokens=int(t.get("max_tokens", 4096)),
            )
        return cls(providers=providers, tiers=tiers)
