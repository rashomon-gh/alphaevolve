"""Fast/strong model ensemble (paper §2.3): weight-sampled tier per request.

Retries happen only on transport errors (connection, timeout, rate limit),
with jittered backoff. Malformed content is never retried here — that is a
measurable signal handled upstream (CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import random

import openai

from alphaevolve.generation.llm import Completion, LLMClient, TierConfig

_TRANSPORT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class Ensemble:
    def __init__(
        self,
        tiers: list[TierConfig],
        clients: dict[str, LLMClient],  # provider name -> client
        *,
        rng: random.Random | None = None,
        max_retries: int = 3,
        backoff_s: float = 1.0,
    ) -> None:
        if not tiers:
            raise ValueError("ensemble needs at least one tier")
        for tier in tiers:
            if tier.provider not in clients:
                raise ValueError(f"no client for provider '{tier.provider}'")
        self.tiers = tiers
        self.clients = clients
        self.rng = rng or random.Random()
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def pick_tier(self) -> TierConfig:
        return self.rng.choices(self.tiers, weights=[t.weight for t in self.tiers], k=1)[0]

    async def generate(
        self, prompt: str, *, system: str | None = None, purpose: str = "evolve"
    ) -> Completion:
        tier = self.pick_tier()
        client = self.clients[tier.provider]
        attempt = 0
        while True:
            try:
                return await client.complete(prompt, tier=tier, system=system, purpose=purpose)
            except _TRANSPORT_ERRORS:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.backoff_s * (2 ** (attempt - 1)) * (0.5 + self.rng.random())
                await asyncio.sleep(delay)
