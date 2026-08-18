"""Optional LLM-graded auxiliary scores (paper §2.4). Off by default.

Auxiliary only (CLAUDE.md invariant 1): keys are namespaced ``llm_*`` and can
never be the sole basis of a program's fitness.
"""

from __future__ import annotations

from typing import Protocol

GRADE_PROMPT = """You are grading a Python program on the criterion: {criterion}.
Respond with only a number between 0.0 (worst) and 1.0 (best).

Program:
```python
{code}
```
"""


class _Completer(Protocol):
    async def complete(self, prompt: str) -> str: ...


async def grade(code: str, criteria: list[str], llm: _Completer) -> dict[str, float]:
    """Grade each criterion in [0, 1]; unparseable responses are skipped
    (missing key = no signal, never a fabricated score)."""
    scores: dict[str, float] = {}
    for criterion in criteria:
        response = await llm.complete(GRADE_PROMPT.format(criterion=criterion, code=code))
        try:
            value = float(response.strip().split()[0])
        except (ValueError, IndexError):
            continue
        key = "llm_" + criterion.lower().replace(" ", "_")
        scores[key] = max(0.0, min(1.0, value))
    return scores
