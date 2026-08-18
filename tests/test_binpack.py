import importlib.util
from pathlib import Path

TASK_DIR = Path(__file__).parent.parent / "alphaevolve" / "tasks" / "binpack"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_initial_program_deterministic():
    evaluator = load_module(TASK_DIR / "evaluator.py")
    program = str(TASK_DIR / "initial_program.py")
    a = evaluator.evaluate(program, seed=1, stage=0)
    b = evaluator.evaluate(program, seed=1, stage=0)
    assert a == b
    assert 0.0 < a["utilization"] <= 1.0
    assert a["worst_trace_utilization"] <= a["utilization"]


def test_stage1_uses_held_out_traces():
    evaluator = load_module(TASK_DIR / "evaluator.py")
    program = str(TASK_DIR / "initial_program.py")
    smoke = evaluator.evaluate(program, seed=1, stage=0)
    full = evaluator.evaluate(program, seed=1, stage=1)
    assert smoke != full  # different trace stream and sizes


def test_best_fit_beats_first_fit():
    evaluator = load_module(TASK_DIR / "evaluator.py")
    first_fit = lambda item, remaining: 0.0  # noqa: E731 — mirrors the initial program
    total_ff, total_bf = 0.0, 0.0
    import random

    for trace_idx in range(20):
        items = evaluator._make_trace(random.Random(trace_idx), 150)
        total_ff += evaluator.simulate(first_fit, items)
        total_bf += evaluator.simulate(evaluator.best_fit_priority, items)
    # The reference best-fit baseline must clearly beat naive first fit,
    # leaving headroom for evolution to discover something better.
    assert total_bf > total_ff
