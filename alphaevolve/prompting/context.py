"""Explicit task context for prompts (paper §2.2): problem statement,
equations, literature snippets — anything the task author attaches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskContext:
    problem_statement: str = ""
    extra_snippets: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict, base_dir: Path) -> TaskContext:
        statement = str(cfg.get("problem_statement", ""))
        if "problem_statement_file" in cfg:
            statement = (base_dir / cfg["problem_statement_file"]).read_text()
        snippets = [str(s) for s in cfg.get("extra_snippets", [])]
        for rel in cfg.get("extra_files", []):
            snippets.append((base_dir / rel).read_text())
        return cls(problem_statement=statement, extra_snippets=snippets)

    def render(self) -> str:
        parts = [self.problem_statement.strip()] if self.problem_statement.strip() else []
        parts += [s.strip() for s in self.extra_snippets if s.strip()]
        return "\n\n".join(parts)
