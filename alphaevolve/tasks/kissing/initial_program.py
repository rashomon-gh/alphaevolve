"""Search heuristic for kissing configurations (paper §3.2).

The evaluator hands this program the best known construction so far (a list
of unit vectors in dimension `dim`, pairwise dot products ≤ 1/2) and a time
budget in seconds. It must return an equal-or-better valid construction —
the returned points are validated and counted by the evaluator, and the best
construction persists between generations, so later heuristics refine what
earlier ones found (the paper's iterative-refinement chain).

Only the code between the EVOLVE markers may change.
"""

import time

import numpy as np


# EVOLVE-BLOCK-START
def improve(points: np.ndarray, dim: int, rng: np.random.Generator, budget_s: float) -> np.ndarray:
    """Deliberately simple initial heuristic: greedy random augmentation.
    Keep the incoming configuration and try random unit vectors, adding any
    that keeps all pairwise dot products ≤ 1/2."""
    kept = [p / np.linalg.norm(p) for p in np.asarray(points).reshape(-1, dim)]
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        candidate = rng.standard_normal(dim)
        candidate /= np.linalg.norm(candidate)
        if all(float(np.dot(candidate, q)) <= 0.5 - 1e-9 for q in kept):
            kept.append(candidate)
    return np.array(kept) if kept else np.zeros((0, dim))


# EVOLVE-BLOCK-END
