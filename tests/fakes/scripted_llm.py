"""Deterministic fake LLM so the entire loop runs offline (PLAN.md Phase 0).

Satisfies the controller's Generator protocol. Completions are served from a
script, in order; when the script is exhausted it either repeats the last
entry or raises, depending on `repeat_last`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from alphaevolve.generation.llm import Completion


@dataclass
class ScriptedLLM:
    script: list[str]
    repeat_last: bool = True
    delay_s: float = 0.0  # simulate latency to exercise pipeline concurrency
    calls: list[tuple[str, str]] = field(default_factory=list)  # (purpose, prompt)
    _cursor: int = 0

    async def generate(
        self, prompt: str, *, system: str | None = None, purpose: str = "evolve"
    ) -> Completion:
        self.calls.append((purpose, prompt))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self._cursor >= len(self.script):
            if not (self.repeat_last and self.script):
                raise RuntimeError("ScriptedLLM: script exhausted")
            text = self.script[-1]
        else:
            text = self.script[self._cursor]
            self._cursor += 1
        return Completion(
            text=text,
            model="scripted",
            tier="fake",
            provider="fake",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            seconds=self.delay_s,
        )
