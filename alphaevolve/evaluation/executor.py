"""Async evaluation over parallel seeds (paper §2.4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from alphaevolve.task.sandbox import SandboxResult, run_evaluation
from alphaevolve.task.spec import CascadeStage, TaskSpec


def aggregate(per_seed: list[dict[str, float]], how: str = "mean") -> dict[str, float]:
    """Aggregate per-seed score dicts key-wise. Keys are the union; a seed
    missing a key simply doesn't contribute to that key."""
    keys = {k for scores in per_seed for k in scores}
    out: dict[str, float] = {}
    for key in sorted(keys):
        values = [s[key] for s in per_seed if key in s]
        if how == "mean":
            out[key] = sum(values) / len(values)
        elif how == "max":
            out[key] = max(values)
        elif how == "min":
            out[key] = min(values)
        else:
            raise ValueError(f"unknown aggregation: {how}")
    return out


async def evaluate_stage(
    spec: TaskSpec,
    program_path: Path,
    stage_index: int,
    stage: CascadeStage,
    *,
    base_seed: int = 0,
    aggregation: str = "mean",
) -> tuple[dict[str, float] | None, str | None, dict[str, str], float]:
    """Evaluate one cascade stage over `stage.seeds` parallel seeds.

    Returns (scores, failure_reason, artifacts, seconds). Any failing seed
    fails the stage — a program that crashes on some seeds is faulty.
    """
    results: list[SandboxResult] = await asyncio.gather(
        *(
            run_evaluation(
                spec.evaluator_path,
                program_path,
                seed=base_seed + i,
                stage=stage_index,
                timeout_s=stage.timeout_s,
                cpu_time_s=spec.cpu_time_s,
                memory_mb=spec.memory_mb,
            )
            for i in range(stage.seeds)
        )
    )
    seconds = max(r.seconds for r in results)
    artifacts = dict(results[0].artifacts)  # first seed's output is the prompt artifact
    for i, res in enumerate(results):
        if res.scores is None:
            return None, f"seed {base_seed + i}: {res.failure_reason}", artifacts, seconds
    return (
        aggregate([r.scores for r in results if r.scores is not None], aggregation),
        None,
        artifacts,
        seconds,
    )
