"""Priority heuristic for online 2-resource (CPU, mem) vector bin packing.

The simulator presents items one at a time. For each item it calls
priority(item, remaining) once per open bin that can still fit the item, and
places the item into the feasible bin with the highest score (first bin wins
ties). If no open bin fits, a new bin is opened. Bins have capacity (1.0, 1.0).

Only the code between the EVOLVE markers may change; this docstring and the
function signature are the immutable contract with the evaluator.
"""


# EVOLVE-BLOCK-START
def priority(item: tuple[float, float], remaining: tuple[float, float]) -> float:
    """Score placing `item` (cpu, mem demand) into a bin with `remaining`
    (cpu, mem) capacity. Higher is better. Naive initial heuristic:
    every feasible bin is equally fine (i.e. first fit)."""
    return 0.0


# EVOLVE-BLOCK-END
