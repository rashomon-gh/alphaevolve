"""Evaluator for the online 2-resource bin-packing task (open analog of
paper §3.3.1, the Borg scheduling heuristic).

evaluate(program_path, seed, stage) loads the evolved program, runs the
trace-driven simulator, and returns maximized scores:
  - utilization: mean over traces of (total demand) / (2 * bins opened)
  - worst_trace_utilization: min over traces (robustness signal)

Stage 0 is a cheap smoke run; stage >= 1 uses more, longer, held-out traces
(disjoint seed stream). Deterministic given (seed, stage).
"""

from __future__ import annotations

import importlib.util
import random

CAPACITY = 1.0
_HELD_OUT_OFFSET = 1_000_003  # stage >= 1 draws from a disjoint seed stream


def _load_priority(program_path: str):
    spec = importlib.util.spec_from_file_location("ae_binpack_program", program_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.priority


def _make_trace(rng: random.Random, n_items: int) -> list[tuple[float, float]]:
    """Mixed workload: cpu-heavy, mem-heavy, and small balanced items."""
    items: list[tuple[float, float]] = []
    for _ in range(n_items):
        kind = rng.random()
        if kind < 0.3:  # cpu-heavy
            items.append((rng.uniform(0.3, 0.7), rng.uniform(0.05, 0.2)))
        elif kind < 0.6:  # mem-heavy
            items.append((rng.uniform(0.05, 0.2), rng.uniform(0.3, 0.7)))
        else:  # balanced small
            items.append((rng.uniform(0.05, 0.35), rng.uniform(0.05, 0.35)))
    return items


def simulate(priority, items: list[tuple[float, float]]) -> float:
    """Run one online packing episode; return utilization in (0, 1]."""
    bins: list[list[float]] = []  # remaining (cpu, mem) per open bin
    for cpu, mem in items:
        best_idx = -1
        best_score = float("-inf")
        for idx, (rc, rm) in enumerate(bins):
            if rc >= cpu and rm >= mem:
                score = float(priority((cpu, mem), (rc, rm)))
                if score > best_score:  # strict: first feasible bin wins ties
                    best_score = score
                    best_idx = idx
        if best_idx < 0:
            bins.append([CAPACITY - cpu, CAPACITY - mem])
        else:
            bins[best_idx][0] -= cpu
            bins[best_idx][1] -= mem
    total_demand = sum(c + m for c, m in items)
    return total_demand / (2.0 * CAPACITY * len(bins))


def evaluate(program_path: str, seed: int, stage: int) -> dict[str, float]:
    priority = _load_priority(program_path)
    if stage == 0:
        n_traces, n_items, offset = 2, 40, 0
    else:
        n_traces, n_items, offset = 8, 150, _HELD_OUT_OFFSET
    utilizations = []
    for trace_idx in range(n_traces):
        rng = random.Random(offset + seed * 10_007 + trace_idx)
        utilizations.append(simulate(priority, _make_trace(rng, n_items)))
    return {
        "utilization": sum(utilizations) / len(utilizations),
        "worst_trace_utilization": min(utilizations),
    }


# Reference baseline (not used by evaluate; kept for tests and as a target the
# evolution should rediscover or beat): best-fit on the summed remaining
# capacity, i.e. prefer the tightest bin that still fits. The paper's
# published `alpha_evolve_score` heuristic (Fig. 6) is a refinement of this
# idea; transcribe it here verbatim when validating Phase 8 reports.
def best_fit_priority(item: tuple[float, float], remaining: tuple[float, float]) -> float:
    return -((remaining[0] - item[0]) + (remaining[1] - item[1]))
