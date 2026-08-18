"""Evaluator for circle packing in the unit square (paper App. B.12).

A valid packing is exactly n_circles circles (x, y, r) with r ≥ 0, each disk
inside [0,1]², and no pairwise overlap. Score = sum of radii (maximize). The
best packing persists in $AE_STATE_DIR between generations. Invalid packings
are evaluation failures, never repaired here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

_TOL = 1e-9


def validate(circles: np.ndarray, n_circles: int) -> float:
    """Return the sum of radii; raise ValueError if the packing is invalid."""
    circles = np.asarray(circles, dtype=float)
    if circles.shape != (n_circles, 3):
        raise ValueError(f"expected shape ({n_circles}, 3), got {circles.shape}")
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -_TOL):
        raise ValueError("negative radius")
    if np.any((x - r < -_TOL) | (x + r > 1 + _TOL) | (y - r < -_TOL) | (y + r > 1 + _TOL)):
        raise ValueError("circle leaves the unit square")
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = math.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - _TOL:
                raise ValueError(f"circles {i} and {j} overlap")
    return float(r.sum())


def _state_path(n_circles: int) -> Path | None:
    state_dir = os.environ.get("AE_STATE_DIR")
    return Path(state_dir) / f"packing_best_n{n_circles}.json" if state_dir else None


def load_best(n_circles: int) -> np.ndarray:
    path = _state_path(n_circles)
    if path is None or not path.exists():
        return np.zeros((0, 3))
    return np.array(json.loads(path.read_text()), dtype=float).reshape(-1, 3)


def save_best(n_circles: int, circles: np.ndarray) -> None:
    path = _state_path(n_circles)
    if path is None:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as tmp:
        json.dump(np.asarray(circles).tolist(), tmp)
    os.replace(tmp.name, path)


def evaluate(program_path: str, seed: int, stage: int) -> dict[str, float]:
    params = json.loads(os.environ.get("AE_TASK_PARAMS", "{}"))
    n_circles = int(params.get("n_circles", 26))
    budgets = params.get("stage_budgets_s", [2.0, 20.0, 100.0])
    budget_s = float(budgets[min(stage, len(budgets) - 1)])

    spec = importlib.util.spec_from_file_location("ae_packing_program", program_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    best = load_best(n_circles)
    best_sum = validate(best, n_circles) if best.shape[0] == n_circles else 0.0
    rng = np.random.default_rng(seed)
    result = np.asarray(module.improve(best.copy(), n_circles, rng, budget_s), dtype=float)
    total = validate(result, n_circles)
    if total > best_sum:
        save_best(n_circles, result)
    return {"sum_radii": total}
