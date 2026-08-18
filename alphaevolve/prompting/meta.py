"""Meta-prompt evolution (paper §2.2): a second, small evolutionary database
of instruction snippets. Snippets are sampled into evolution prompts; credit
is the mean score improvement of the children they helped produce; the LLM
itself occasionally proposes new snippets.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from alphaevolve.database.programs import new_id

PROPOSE_PROMPT = """You are improving the prompts of an evolutionary coding agent.
Below are instruction snippets currently in use, with their measured credit
(mean score improvement of programs generated under them):

{snippets}

Propose ONE new, different instruction snippet (a single imperative sentence,
no numbering, no quotes) that could push the code evolution toward better
scores. Respond with only the snippet text."""


@dataclass
class Snippet:
    id: str
    text: str
    uses: int = 0
    total_improvement: float = 0.0

    @property
    def credit(self) -> float:
        return self.total_improvement / self.uses if self.uses else 0.0


@dataclass
class MetaPromptDB:
    snippets: dict[str, Snippet] = field(default_factory=dict)

    @classmethod
    def from_seed_texts(cls, texts: list[str]) -> MetaPromptDB:
        db = cls()
        for text in texts:
            db.add(text)
        return db

    def add(self, text: str) -> Snippet:
        snippet = Snippet(id=new_id(), text=text.strip())
        self.snippets[snippet.id] = snippet
        return snippet

    def sample(self, k: int, rng: random.Random) -> list[Snippet]:
        """Credit-weighted sampling with a floor so unproven snippets still
        get explored."""
        pool = list(self.snippets.values())
        if not pool or k <= 0:
            return []
        chosen: list[Snippet] = []
        for _ in range(min(k, len(pool))):
            weights = [1.0 + max(0.0, s.credit) * 10.0 for s in pool]
            pick = rng.choices(pool, weights=weights, k=1)[0]
            chosen.append(pick)
            pool.remove(pick)
        return chosen

    def credit_children(self, snippet_ids: list[str], improvement: float) -> None:
        for sid in snippet_ids:
            if (snippet := self.snippets.get(sid)) is not None:
                snippet.uses += 1
                snippet.total_improvement += improvement

    def propose_prompt(self) -> str:
        listing = (
            "\n".join(
                f"- (credit {s.credit:+.4f}, uses {s.uses}) {s.text}"
                for s in self.snippets.values()
            )
            or "- (none yet)"
        )
        return PROPOSE_PROMPT.format(snippets=listing)

    # -- persistence in the run directory -----------------------------------

    def save(self, path: Path) -> None:
        payload = [
            {"id": s.id, "text": s.text, "uses": s.uses, "total_improvement": s.total_improvement}
            for s in self.snippets.values()
        ]
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> MetaPromptDB:
        db = cls()
        for item in json.loads(path.read_text()):
            db.snippets[item["id"]] = Snippet(**item)
        return db
