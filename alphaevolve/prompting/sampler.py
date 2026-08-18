"""Evolution-prompt assembly (paper §2.2, Fig. 3b).

Shape: system instructions → explicit task context → prior programs with
scores → current program with scores + eval output → output-format rules →
task instruction (with stochastic variants and meta-prompt snippets).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from alphaevolve.database.programs import Program, new_id
from alphaevolve.prompting.context import TaskContext
from alphaevolve.prompting.meta import MetaPromptDB
from alphaevolve.prompting.stochastic import StochasticSlot, sample_slots

SYSTEM_INSTRUCTIONS = (
    "You are an expert software engineer inside an evolutionary code-optimization "
    "loop. You will see prior attempts and their measured scores (higher is "
    "better). Propose a change to the current program that increases its scores. "
    "Only code between '# EVOLVE-BLOCK-START' and '# EVOLVE-BLOCK-END' markers "
    "may be modified; everything else is immutable infrastructure."
)

DIFF_RULES = """Propose your change as one or more SEARCH/REPLACE blocks, exactly in this format:

<<<<<<< SEARCH
(verbatim excerpt of the current program, inside an evolve block)
=======
(replacement text)
>>>>>>> REPLACE

Rules: the SEARCH text must appear exactly once in the current program and must
lie entirely inside an evolve block. Keep changes focused; several small blocks
are better than one huge one."""

FULL_REWRITE_RULES = """Output the complete new content of the evolve block (the code between the
markers, not the markers themselves) in a single ```python fenced code block.
Everything outside the evolve block is immutable and will be kept as is."""


@dataclass(frozen=True)
class PromptBundle:
    id: str
    text: str
    meta: dict[str, object] = field(default_factory=dict)


def _render_program(title: str, program: Program) -> str:
    scores = ", ".join(f"{k}={v:.6g}" for k, v in sorted(program.scores.items())) or "unscored"
    return f"### {title} (scores: {scores})\n```python\n{program.code}\n```"


def build_prompt(
    parent: Program,
    inspirations: list[Program],
    *,
    context: TaskContext,
    full_rewrite: bool = False,
    slots: list[StochasticSlot] | None = None,
    meta_db: MetaPromptDB | None = None,
    num_meta_snippets: int = 2,
    rng: random.Random | None = None,
    include_context: bool = True,
) -> PromptBundle:
    rng = rng or random.Random()
    sections: list[str] = [SYSTEM_INSTRUCTIONS]

    rendered_context = context.render()
    if include_context and rendered_context:
        sections.append("## Problem context\n" + rendered_context)

    if inspirations:
        rendered = "\n\n".join(
            _render_program(f"Prior program {i + 1}", p) for i, p in enumerate(inspirations)
        )
        sections.append("## Prior programs\n" + rendered)

    current = _render_program("Current program", parent)
    eval_output = (parent.artifacts.get("stdout") or "").strip()
    if eval_output:
        current += f"\n\nEvaluation output of the current program:\n```\n{eval_output}\n```"
    sections.append("## Current program\n" + current)

    sections.append("## Output format\n" + (FULL_REWRITE_RULES if full_rewrite else DIFF_RULES))

    slot_values = sample_slots(slots or [], rng)
    meta_snippets = meta_db.sample(num_meta_snippets, rng) if meta_db is not None else []
    instruction_lines = [
        "## Task",
        "Improve the current program so its scores increase.",
        *slot_values.values(),
        *(s.text for s in meta_snippets),
    ]
    sections.append("\n".join(instruction_lines))

    return PromptBundle(
        id=new_id(),
        text="\n\n".join(sections),
        meta={
            "parent_id": parent.id,
            "inspiration_ids": [p.id for p in inspirations],
            "slot_values": slot_values,
            "meta_snippet_ids": [s.id for s in meta_snippets],
            "full_rewrite": full_rewrite,
            "include_context": include_context,
        },
    )
