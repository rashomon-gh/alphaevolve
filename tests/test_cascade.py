from pathlib import Path

from alphaevolve.evaluation.cascade import promoted, run_cascade
from alphaevolve.evaluation.executor import aggregate
from alphaevolve.task.spec import CascadeStage, TaskSpec

# Evaluator that scores from the program's own value of X and records which
# stages actually ran (so tests can prove stage 1 was skipped).
TRACKING_EVALUATOR = """
import json, pathlib

def evaluate(program_path, seed, stage):
    marker_dir = pathlib.Path(__file__).parent / "stages_run"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / f"stage{stage}_seed{seed}").touch()
    namespace = {}
    exec(pathlib.Path(program_path).read_text(), namespace)
    return {"quality": float(namespace["X"]) + seed * 0.1, "stage": float(stage)}
"""

PROGRAM_TEMPLATE = "# EVOLVE-BLOCK-START\nX = {value}\n# EVOLVE-BLOCK-END\n"


def make_spec(tmp_path: Path, cascade: list[CascadeStage]) -> TaskSpec:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(TRACKING_EVALUATOR)
    return TaskSpec(
        name="toy",
        initial_program=PROGRAM_TEMPLATE.format(value=0.0),
        evaluator_path=evaluator,
        cascade=cascade,
    )


def stages_run(tmp_path: Path) -> set[str]:
    marker_dir = tmp_path / "stages_run"
    return {p.name for p in marker_dir.iterdir()} if marker_dir.exists() else set()


def test_promoted_predicate():
    assert promoted({"a": 0.6}, {"a": 0.5})
    assert not promoted({"a": 0.4}, {"a": 0.5})
    assert not promoted({}, {"a": 0.5})  # missing key never promotes
    assert promoted({"a": 0.1}, {})  # no thresholds -> always promote


async def test_bad_candidate_pruned_at_stage0(tmp_path):
    spec = make_spec(
        tmp_path,
        [
            CascadeStage(seeds=1, timeout_s=20, promote_if={"quality": 0.5}),
            CascadeStage(seeds=2, timeout_s=20),
        ],
    )
    result = await run_cascade(spec, PROGRAM_TEMPLATE.format(value=0.1))
    assert result.ok
    assert result.stage_reached == 0
    assert result.scores["quality"] == 0.1
    assert stages_run(tmp_path) == {"stage0_seed0"}  # stage 1 never ran


async def test_good_candidate_promoted_through_cascade(tmp_path):
    spec = make_spec(
        tmp_path,
        [
            CascadeStage(seeds=1, timeout_s=20, promote_if={"quality": 0.5}),
            CascadeStage(seeds=2, timeout_s=20),
        ],
    )
    result = await run_cascade(spec, PROGRAM_TEMPLATE.format(value=0.9))
    assert result.ok
    assert result.stage_reached == 1
    # stage 1, seeds 0 and 1: mean of (0.9, 1.0)
    assert abs(result.scores["quality"] - 0.95) < 1e-9
    assert result.scores["stage"] == 1.0
    assert stages_run(tmp_path) == {"stage0_seed0", "stage1_seed0", "stage1_seed1"}


async def test_crash_records_failure_and_stage(tmp_path):
    spec = make_spec(tmp_path, [CascadeStage(seeds=1, timeout_s=20)])
    result = await run_cascade(
        spec, "# EVOLVE-BLOCK-START\nraise ValueError('bad')\n# EVOLVE-BLOCK-END\n"
    )
    assert not result.ok
    assert result.stage_reached == -1
    assert result.scores == {}
    assert result.failure_reason is not None and "stage 0" in result.failure_reason


def test_aggregate_modes():
    per_seed = [{"a": 1.0, "b": 0.0}, {"a": 3.0}]
    assert aggregate(per_seed, "mean") == {"a": 2.0, "b": 0.0}
    assert aggregate(per_seed, "max") == {"a": 3.0, "b": 0.0}
    assert aggregate(per_seed, "min") == {"a": 1.0, "b": 0.0}
