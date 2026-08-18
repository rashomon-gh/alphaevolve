"""Evaluation cascade (paper §2.4): staged budgets with promotion predicates.

Stage 0 is always a cheap smoke run that filters faulty programs. A candidate
proceeds to stage k+1 only if stage k succeeded AND met its promote_if
thresholds. Scores of the last completed stage are the program's scores.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from alphaevolve.evaluation.executor import evaluate_stage
from alphaevolve.task.spec import TaskSpec


@dataclass(frozen=True)
class EvalResult:
    scores: dict[str, float]  # {} iff failed
    stage_reached: int  # highest stage index that completed successfully; -1 if none
    failure_reason: str | None
    seconds: float
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure_reason is None


def promoted(scores: dict[str, float], thresholds: dict[str, float]) -> bool:
    """All listed keys must be present and >= their threshold."""
    return all(key in scores and scores[key] >= min_value for key, min_value in thresholds.items())


async def run_cascade(spec: TaskSpec, code: str, *, base_seed: int = 0) -> EvalResult:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="ae_candidate_", delete=False
    ) as tmp:
        tmp.write(code)
        program_path = Path(tmp.name)
    try:
        total_seconds = 0.0
        scores: dict[str, float] = {}
        artifacts: dict[str, str] = {}
        stage_reached = -1
        for index, stage in enumerate(spec.cascade):
            stage_scores, failure, stage_artifacts, seconds = await evaluate_stage(
                spec, program_path, index, stage, base_seed=base_seed
            )
            total_seconds += seconds
            if stage_scores is None:
                return EvalResult(
                    scores={},
                    stage_reached=stage_reached,
                    failure_reason=f"stage {index}: {failure}",
                    seconds=total_seconds,
                    artifacts=stage_artifacts,
                )
            scores, artifacts, stage_reached = stage_scores, stage_artifacts, index
            if not promoted(stage_scores, stage.promote_if):
                break  # evaluated but not promising enough for costlier stages
        return EvalResult(
            scores=scores,
            stage_reached=stage_reached,
            failure_reason=None,
            seconds=total_seconds,
            artifacts=artifacts,
        )
    finally:
        program_path.unlink(missing_ok=True)
