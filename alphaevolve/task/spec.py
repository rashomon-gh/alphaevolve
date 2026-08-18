"""Task specification (paper §2.1): initial program, evaluator, cascade config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alphaevolve.task.markers import parse_program


@dataclass(frozen=True)
class CascadeStage:
    """One evaluation-cascade stage (paper §2.4)."""

    seeds: int = 1
    timeout_s: float = 60.0
    promote_if: dict[str, float] = field(default_factory=dict)  # score_key -> min value


@dataclass(frozen=True)
class FeatureSpec:
    """MAP-Elites feature descriptor: bins over code length or a score key."""

    name: str  # "code_length" or "score:<key>"
    bins: list[float]  # ascending bin edges


@dataclass(frozen=True)
class TaskSpec:
    name: str
    initial_program: str  # full source with EVOLVE-BLOCK markers
    evaluator_path: Path  # file exposing evaluate(program_path, seed, stage) -> dict[str, float]
    cascade: list[CascadeStage]
    full_rewrite: bool = False  # paper §2.3: whole-evolve-block output for short programs
    context: str = ""  # explicit problem context for prompts (paper §2.2)
    features: list[FeatureSpec] = field(default_factory=list)
    cpu_time_s: float | None = None
    memory_mb: int | None = None

    def __post_init__(self) -> None:
        parse_program(self.initial_program)  # validate markers early

    @classmethod
    def from_config(cls, cfg: dict, base_dir: Path) -> TaskSpec:
        """Build a TaskSpec from the ``task:`` section of a run config."""
        program_path = base_dir / cfg["initial_program"]
        evaluator_path = base_dir / cfg["evaluator"]
        context = cfg.get("context", "")
        if "context_file" in cfg:
            context = (base_dir / cfg["context_file"]).read_text()
        cascade = [
            CascadeStage(
                seeds=int(s.get("seeds", 1)),
                timeout_s=float(s.get("timeout_s", 60.0)),
                promote_if={k: float(v) for k, v in s.get("promote_if", {}).items()},
            )
            for s in cfg.get("cascade", [{}])
        ]
        features = [
            FeatureSpec(name=f["name"], bins=[float(b) for b in f["bins"]])
            for f in cfg.get("features", [])
        ]
        return cls(
            name=cfg["name"],
            initial_program=program_path.read_text(),
            evaluator_path=evaluator_path,
            cascade=cascade,
            full_rewrite=bool(cfg.get("full_rewrite", False)),
            context=context,
            features=features,
            cpu_time_s=cfg.get("cpu_time_s"),
            memory_mb=cfg.get("memory_mb"),
        )
