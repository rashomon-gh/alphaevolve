"""Tests for the kissing and packing construction tasks (Phase 7)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

TASKS = Path(__file__).parent.parent / "alphaevolve" / "tasks"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_" + path.parent.name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kissing = load_module(TASKS / "kissing" / "evaluator.py")
packing = load_module(TASKS / "packing" / "evaluator.py")


def icosahedron() -> np.ndarray:
    """The 12 icosahedron vertices: the classic d=3 kissing configuration."""
    phi = (1 + math.sqrt(5)) / 2
    points = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            points += [(0, a, b), (a, b, 0), (b, 0, a)]
    arr = np.array(points)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# -- kissing ----------------------------------------------------------------


def test_icosahedron_is_valid_kissing_12():
    assert kissing.validate(icosahedron(), dim=3) == 12


def test_duplicate_vector_rejected():
    points = np.vstack([icosahedron(), icosahedron()[:1]])
    with pytest.raises(ValueError, match="overlap"):
        kissing.validate(points, dim=3)


def test_non_unit_vector_rejected():
    points = icosahedron()
    points[0] *= 1.5
    with pytest.raises(ValueError, match="unit"):
        kissing.validate(points, dim=3)


def test_empty_configuration_is_zero():
    assert kissing.validate(np.zeros((0, 3)), dim=3) == 0


def test_kissing_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AE_STATE_DIR", str(tmp_path))
    kissing.save_best(3, icosahedron())
    loaded = kissing.load_best(3)
    assert loaded.shape == (12, 3)
    assert kissing.validate(loaded, dim=3) == 12


def test_kissing_evaluate_runs_initial_program(tmp_path, monkeypatch):
    monkeypatch.setenv("AE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AE_TASK_PARAMS", json.dumps({"dim": 3, "stage_budgets_s": [0.3]}))
    program = TASKS / "kissing" / "initial_program.py"
    scores = kissing.evaluate(str(program), seed=1, stage=0)
    assert scores["kissing_number"] >= 3  # greedy random finds a few instantly
    assert (tmp_path / "kissing_best_d3.json").exists()  # state persisted
    # A second generation starts from the persisted best, never regresses.
    scores2 = kissing.evaluate(str(program), seed=2, stage=0)
    assert scores2["kissing_number"] >= scores["kissing_number"]


# -- packing ----------------------------------------------------------------


def test_single_centered_circle_valid():
    assert packing.validate(np.array([[0.5, 0.5, 0.5]]), 1) == pytest.approx(0.5)


def test_overlapping_circles_rejected():
    circles = np.array([[0.3, 0.5, 0.25], [0.6, 0.5, 0.25]])  # dist 0.3 < 0.5
    with pytest.raises(ValueError, match="overlap"):
        packing.validate(circles, 2)


def test_circle_outside_square_rejected():
    with pytest.raises(ValueError, match="square"):
        packing.validate(np.array([[0.1, 0.5, 0.2]]), 1)


def test_wrong_count_rejected():
    with pytest.raises(ValueError, match="shape"):
        packing.validate(np.array([[0.5, 0.5, 0.1]]), 2)


def test_packing_evaluate_runs_initial_program(tmp_path, monkeypatch):
    monkeypatch.setenv("AE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AE_TASK_PARAMS", json.dumps({"n_circles": 4, "stage_budgets_s": [0.3]}))
    program = TASKS / "packing" / "initial_program.py"
    scores = packing.evaluate(str(program), seed=1, stage=0)
    # 4 circles on a 2x2 grid grow to r≈0.25 each: sum near 1.0.
    assert scores["sum_radii"] > 0.8
    assert (tmp_path / "packing_best_n4.json").exists()
    scores2 = packing.evaluate(str(program), seed=2, stage=0)
    assert scores2["sum_radii"] >= scores["sum_radii"] - 1e-9
