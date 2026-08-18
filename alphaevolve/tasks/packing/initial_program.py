"""Search heuristic for circle packing in the unit square (paper App. B.12).

The evaluator hands this program the best known packing of exactly
`n_circles` circles (rows of (x, y, r)) and a time budget; it must return a
valid packing whose sum of radii is at least as large. The best packing
persists between generations, so heuristics refine earlier results.

Only the code between the EVOLVE markers may change.
"""

import math
import time

import numpy as np


# EVOLVE-BLOCK-START
def improve(
    circles: np.ndarray, n_circles: int, rng: np.random.Generator, budget_s: float
) -> np.ndarray:
    """Deliberately simple initial heuristic: if no packing exists yet, lay
    the circles on a square grid; then grow every radius to the largest value
    its neighbors and the walls allow, in random order."""
    deadline = time.monotonic() + budget_s
    circles = np.asarray(circles, dtype=float).reshape(-1, 3)
    if circles.shape[0] != n_circles:
        side = math.ceil(math.sqrt(n_circles))
        pitch = 1.0 / side
        centers = [((i % side + 0.5) * pitch, (i // side + 0.5) * pitch) for i in range(n_circles)]
        circles = np.array([[x, y, pitch / 2 * 0.95] for x, y in centers])

    while time.monotonic() < deadline:
        for i in rng.permutation(n_circles):
            x, y, _ = circles[i]
            limit = min(x, y, 1.0 - x, 1.0 - y)
            for j in range(n_circles):
                if j == int(i):
                    continue
                dist = math.hypot(x - circles[j, 0], y - circles[j, 1])
                limit = min(limit, dist - circles[j, 2])
            circles[i, 2] = max(limit - 1e-12, 0.0)
    return circles


# EVOLVE-BLOCK-END
