"""Evaluator for kissing-number lower bounds (paper §3.2).

A valid kissing configuration in dimension d is a set of unit vectors with
all pairwise dot products ≤ 1/2 (centers of unit spheres touching the central
unit sphere, angular separation ≥ 60°). The evolved program receives the best
known configuration (persisted in $AE_STATE_DIR between generations) plus a
time budget and must return a configuration; we validate it exactly and score
its size. An invalid construction is an evaluation failure — never silently
repaired (CLAUDE.md: failures are data).
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np

_DOT_TOL = 1e-9
_NORM_TOL = 1e-9


def validate(points: np.ndarray, dim: int) -> int:
    """Return the configuration size; raise ValueError if invalid."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 0
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"expected shape (k, {dim}), got {points.shape}")
    norms = np.linalg.norm(points, axis=1)
    if not np.all(np.abs(norms - 1.0) <= _NORM_TOL):
        raise ValueError("all vectors must be unit length")
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    worst = float(dots.max(initial=0.0))
    if worst > 0.5 + _DOT_TOL:
        raise ValueError(f"pairwise dot {worst:.6f} exceeds 1/2: spheres overlap")
    return points.shape[0]


def _state_path(dim: int) -> Path | None:
    state_dir = os.environ.get("AE_STATE_DIR")
    return Path(state_dir) / f"kissing_best_d{dim}.json" if state_dir else None


def load_best(dim: int) -> np.ndarray:
    path = _state_path(dim)
    if path is None or not path.exists():
        return np.zeros((0, dim))
    return np.array(json.loads(path.read_text()), dtype=float).reshape(-1, dim)


def save_best(dim: int, points: np.ndarray) -> None:
    path = _state_path(dim)
    if path is None:
        return
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:  # concurrent evaluators: atomic replace, last-writer-wins
        json.dump(np.asarray(points).tolist(), tmp)
    os.replace(tmp.name, path)


def evaluate(program_path: str, seed: int, stage: int) -> dict[str, float]:
    params = json.loads(os.environ.get("AE_TASK_PARAMS", "{}"))
    dim = int(params.get("dim", 3))
    budgets = params.get("stage_budgets_s", [2.0, 20.0, 100.0])
    budget_s = float(budgets[min(stage, len(budgets) - 1)])

    spec = importlib.util.spec_from_file_location("ae_kissing_program", program_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    best = load_best(dim)
    rng = np.random.default_rng(seed)
    result = np.asarray(module.improve(best.copy(), dim, rng, budget_s), dtype=float)
    count = validate(result, dim)
    if count > best.shape[0]:
        save_best(dim, result)
    return {"kissing_number": float(count)}
