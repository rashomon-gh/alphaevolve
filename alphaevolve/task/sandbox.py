"""Sandboxed evaluation (paper §2.1/§2.4): subprocess + rlimits + wall clock.

Ground truth is execution (CLAUDE.md invariant 1): scores only ever come from
running the task's evaluate() here. Failures are returned as data, never
masked with heuristic scores (invariant 4).

The evaluator runs in its own session (process group) so a wall-clock timeout
kills any grandchildren it spawned, not just the direct child. Known accepted
gap (PLAN.md risks: "subprocess + rlimits first"): network access is NOT
blocked; move to containers if evolved code starts doing surprising things.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import resource
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_ARTIFACT_LIMIT = 8_000  # chars of stdout/stderr kept per stream


@dataclass(frozen=True)
class SandboxResult:
    scores: dict[str, float] | None  # None iff the evaluation failed
    failure_reason: str | None
    seconds: float
    artifacts: dict[str, str] = field(default_factory=dict)


def _make_preexec(cpu_time_s: float | None, memory_mb: int | None):
    def preexec() -> None:
        if cpu_time_s is not None:
            limit = int(cpu_time_s) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
        if memory_mb is not None:
            try:  # RLIMIT_AS is unreliable on macOS; best effort by design
                limit = memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass

    return preexec


async def run_evaluation(
    evaluator_path: Path,
    program_path: Path,
    *,
    seed: int,
    stage: int,
    timeout_s: float,
    cpu_time_s: float | None = None,
    memory_mb: int | None = None,
    task_params: dict[str, object] | None = None,
    state_dir: Path | None = None,
) -> SandboxResult:
    """Run evaluate(program_path, seed, stage) in a fresh subprocess.

    ``task_params`` reaches the evaluator as JSON in $AE_TASK_PARAMS (task
    sizes, budgets). ``state_dir`` is exported as $AE_STATE_DIR — a run-scoped
    directory where evaluators may persist state between generations (the
    paper's §3.2 iterative-refinement chain, e.g. best-construction-so-far).
    Evaluators run concurrently, so state writers must use atomic replace.
    """
    started = time.monotonic()
    env = dict(os.environ)
    if task_params:
        env["AE_TASK_PARAMS"] = json.dumps(task_params)
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        env["AE_STATE_DIR"] = str(state_dir)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "alphaevolve.task._runner",
            str(evaluator_path),
            str(program_path),
            str(seed),
            str(stage),
            str(out_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_make_preexec(cpu_time_s, memory_mb),
            start_new_session=True,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            # Kill the whole process group so grandchildren die too.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.kill()
            await proc.wait()
            return SandboxResult(
                scores=None,
                failure_reason=f"timeout after {timeout_s}s",
                seconds=time.monotonic() - started,
            )
        artifacts = {
            "stdout": stdout_b.decode(errors="replace")[-_ARTIFACT_LIMIT:],
            "stderr": stderr_b.decode(errors="replace")[-_ARTIFACT_LIMIT:],
        }
        elapsed = time.monotonic() - started
        if proc.returncode != 0:
            tail = artifacts["stderr"].strip().splitlines()[-3:]
            return SandboxResult(
                scores=None,
                failure_reason=f"crash (exit {proc.returncode}): {' | '.join(tail)}"[:500],
                seconds=elapsed,
                artifacts=artifacts,
            )
        try:
            scores = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return SandboxResult(
                scores=None,
                failure_reason=f"invalid_scores: {exc}",
                seconds=elapsed,
                artifacts=artifacts,
            )
        return SandboxResult(
            scores={str(k): float(v) for k, v in scores.items()},
            failure_reason=None,
            seconds=elapsed,
            artifacts=artifacts,
        )
    finally:
        out_path.unlink(missing_ok=True)
