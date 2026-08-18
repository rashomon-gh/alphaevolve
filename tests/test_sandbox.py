from pathlib import Path

from alphaevolve.task.sandbox import run_evaluation

GOOD_EVALUATOR = """
def evaluate(program_path, seed, stage):
    print("evaluating", program_path)
    return {"score": 1.0 + seed, "stage": float(stage)}
"""

CRASHING_EVALUATOR = """
def evaluate(program_path, seed, stage):
    raise RuntimeError("deliberate crash")
"""

HANGING_EVALUATOR = """
def evaluate(program_path, seed, stage):
    while True:
        pass
"""

BAD_TYPE_EVALUATOR = """
def evaluate(program_path, seed, stage):
    return [1.0, 2.0]
"""


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


async def test_good_evaluation(tmp_path):
    evaluator = write(tmp_path, "evaluator.py", GOOD_EVALUATOR)
    program = write(tmp_path, "program.py", "x = 1\n")
    result = await run_evaluation(evaluator, program, seed=3, stage=1, timeout_s=20)
    assert result.failure_reason is None
    assert result.scores == {"score": 4.0, "stage": 1.0}
    assert "evaluating" in result.artifacts["stdout"]


async def test_crash_is_failure_with_reason(tmp_path):
    evaluator = write(tmp_path, "evaluator.py", CRASHING_EVALUATOR)
    program = write(tmp_path, "program.py", "x = 1\n")
    result = await run_evaluation(evaluator, program, seed=0, stage=0, timeout_s=20)
    assert result.scores is None
    assert result.failure_reason is not None and "crash" in result.failure_reason
    assert "deliberate crash" in result.failure_reason


async def test_infinite_loop_killed(tmp_path):
    evaluator = write(tmp_path, "evaluator.py", HANGING_EVALUATOR)
    program = write(tmp_path, "program.py", "x = 1\n")
    result = await run_evaluation(evaluator, program, seed=0, stage=0, timeout_s=2)
    assert result.scores is None
    assert result.failure_reason is not None and "timeout" in result.failure_reason
    assert result.seconds < 15


ENV_ECHO_EVALUATOR = """
import json, os, pathlib

def evaluate(program_path, seed, stage):
    params = json.loads(os.environ.get("AE_TASK_PARAMS", "{}"))
    state_dir = os.environ.get("AE_STATE_DIR")
    marker = pathlib.Path(state_dir) / "touched" if state_dir else None
    if marker is not None:
        marker.write_text("yes")
    return {"param_value": float(params.get("value", -1.0)), "has_state": float(marker is not None)}
"""


async def test_task_params_and_state_dir_reach_subprocess(tmp_path):
    evaluator = write(tmp_path, "evaluator.py", ENV_ECHO_EVALUATOR)
    program = write(tmp_path, "program.py", "x = 1\n")
    state_dir = tmp_path / "state"
    result = await run_evaluation(
        evaluator,
        program,
        seed=0,
        stage=0,
        timeout_s=20,
        task_params={"value": 42.0},
        state_dir=state_dir,
    )
    assert result.failure_reason is None
    assert result.scores == {"param_value": 42.0, "has_state": 1.0}
    assert (state_dir / "touched").read_text() == "yes"


async def test_non_dict_scores_rejected(tmp_path):
    evaluator = write(tmp_path, "evaluator.py", BAD_TYPE_EVALUATOR)
    program = write(tmp_path, "program.py", "x = 1\n")
    result = await run_evaluation(evaluator, program, seed=0, stage=0, timeout_s=20)
    assert result.scores is None
    assert result.failure_reason is not None
