"""SEARCH/REPLACE diff parsing and application (paper §2.3).

Semantics (CLAUDE.md invariant 3):
- a SEARCH text must match the current program exactly once;
- the match must lie entirely inside one evolve block;
- blocks apply sequentially, each against the result of the previous one.

Malformed output is never retried or repaired here — it raises, and the
caller records it as a failed candidate (it is a measurable signal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from alphaevolve.task.markers import EVOLVE, parse_program

SEARCH_START = "<<<<<<< SEARCH"
DIVIDER = "======="
REPLACE_END = ">>>>>>> REPLACE"


class DiffError(ValueError):
    """reason is a stable, countable failure code."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class DiffBlock:
    search: str
    replace: str


_BLOCK_RE = re.compile(
    rf"^[ \t]*{re.escape(SEARCH_START)}[ \t]*\n(.*?)^[ \t]*{re.escape(DIVIDER)}[ \t]*\n"
    rf"(.*?)^[ \t]*{re.escape(REPLACE_END)}[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def parse_diff(text: str) -> list[DiffBlock]:
    """Extract ordered SEARCH/REPLACE blocks from LLM output. Prose around
    blocks is ignored; zero well-formed blocks is a parse failure."""
    blocks = [DiffBlock(search=m.group(1), replace=m.group(2)) for m in _BLOCK_RE.finditer(text)]
    if not blocks:
        raise DiffError("parse", "no well-formed SEARCH/REPLACE block found")
    return blocks


def _find_unique(code: str, search: str) -> int:
    first = code.find(search)
    if first == -1:
        raise DiffError("not_found", search[:120])
    if code.find(search, first + 1) != -1:
        raise DiffError("ambiguous", search[:120])
    return first


def apply_diff(code: str, blocks: list[DiffBlock]) -> str:
    """Apply blocks sequentially; evolve regions are recomputed after each
    application because offsets shift."""
    from alphaevolve.task.markers import evolve_regions

    for block in blocks:
        if block.search == "" and code != "":
            raise DiffError("empty_search")
        start = _find_unique(code, block.search)
        end = start + len(block.search)
        if not any(rs <= start and end <= re_ for rs, re_ in evolve_regions(code)):
            raise DiffError("outside_evolve_block", block.search[:120])
        code = code[:start] + block.replace + code[end:]
    return code


def apply_full_rewrite(code: str, new_block: str) -> str:
    """Full-rewrite mode (paper §2.3): replace the content of the program's
    single evolve block with ``new_block``. The skeleton never changes."""
    segments = parse_program(code)
    evolve_indices = [i for i, seg in enumerate(segments) if seg.kind == EVOLVE]
    if len(evolve_indices) != 1:
        raise DiffError(
            "full_rewrite_ambiguous",
            f"full-rewrite mode requires exactly 1 evolve block, found {len(evolve_indices)}",
        )
    if not new_block.endswith("\n"):
        new_block += "\n"
    out: list[str] = []
    for i, seg in enumerate(segments):
        out.append(new_block if i == evolve_indices[0] else seg.text)
    return "".join(out)


def extract_code_block(text: str) -> str:
    """Pull the payload out of the first fenced code block in a full-rewrite
    completion; if there is no fence, the whole completion is the payload."""
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text
