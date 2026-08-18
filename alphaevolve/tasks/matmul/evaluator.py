"""Evaluator for matmul tensor decomposition (paper §3.1).

Runs the evolved optimizer from several random seeds against the ⟨m,n,p⟩
matmul tensor, rounds factor entries to the nearest (half-)integer, verifies
the tensor equation EXACTLY in integer arithmetic, and reports:
  - neg_best_rank: -(lowest rank with an exact decomposition); the paper's
    primary signal, negated so higher is better
  - fraction_seeds: fraction of seeds that achieved that best rank
  - neg_final_loss: -(best unrounded residual seen); a dense auxiliary
    signal so evolution gets gradient before exactness is reached

Task parameters come from $AE_TASK_PARAMS (set from the run config):
  m, n, p        target tensor sizes
  start_rank     first rank attempted (descend from here)
  min_rank       stop descending below this
  stage_budgets  [[seeds, iters], ...] indexed by cascade stage
Stage 0 always evaluates the tiny ⟨2,2,2⟩ tensor as a sanity filter for
broken optimizers (paper §2.4), regardless of the target size.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np


def matmul_tensor(m: int, n: int, p: int) -> np.ndarray:
    """T[i*n+j, j*p+k, k*m+i] = 1: the bilinear map of (m×n)·(n×p) matmul."""
    tensor = np.zeros((m * n, n * p, p * m))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                tensor[i * n + j, j * p + k, k * m + i] = 1.0
    return tensor


def round_half_integer(x: np.ndarray) -> np.ndarray:
    """Round entries (real or complex) to the nearest half-integer."""
    if np.iscomplexobj(x):
        return round_half_integer(np.real(x)) + 1j * round_half_integer(np.imag(x))
    return np.round(2.0 * x) / 2.0


def verify_exact(tensor: np.ndarray, factors) -> bool:
    """Exact check: scale rounded half-integer factors by 2 so entries are
    integers, then require Σ_r U2⊗V2⊗W2 == 8·T exactly (paper §3.1)."""
    U, V, W = (round_half_integer(np.asarray(f)) for f in factors)
    a, b, c = tensor.shape
    if U.shape[0] != a or V.shape[0] != b or W.shape[0] != c:
        return False
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
    if np.iscomplexobj(U) or np.iscomplexobj(V) or np.iscomplexobj(W):
        u2 = (2 * U).astype(np.complex128)
        v2 = (2 * V).astype(np.complex128)
        w2 = (2 * W).astype(np.complex128)
        recon = np.einsum("ar,br,cr->abc", u2, v2, w2)
        return bool(np.all(recon == 8 * tensor.astype(np.complex128)))
    u2 = np.rint(2 * U).astype(np.int64)
    v2 = np.rint(2 * V).astype(np.int64)
    w2 = np.rint(2 * W).astype(np.int64)
    recon = np.einsum("ar,br,cr->abc", u2, v2, w2)
    return bool(np.all(recon == 8 * np.rint(tensor).astype(np.int64)))


def _load_optimize(program_path: str):
    spec = importlib.util.spec_from_file_location("ae_matmul_program", program_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.optimize


def _residual(tensor: np.ndarray, factors) -> float:
    U, V, W = (np.asarray(f) for f in factors)
    recon = np.einsum("ar,br,cr->abc", U, V, W)
    value = float(np.linalg.norm(recon - tensor))
    return value if math.isfinite(value) else float("inf")


def evaluate(program_path: str, seed: int, stage: int) -> dict[str, float]:
    params = json.loads(os.environ.get("AE_TASK_PARAMS", "{}"))
    budgets = params.get("stage_budgets", [[1, 150], [2, 800], [4, 3000]])
    seeds_per_eval, iters = budgets[min(stage, len(budgets) - 1)]

    if stage == 0:  # sanity filter on the tiny tensor, whatever the target
        m = n = p = 2
        start_rank, min_rank = 8, 7
    else:
        m, n, p = int(params.get("m", 2)), int(params.get("n", 2)), int(params.get("p", 2))
        start_rank = int(params.get("start_rank", m * n * p))
        min_rank = int(params.get("min_rank", 1))

    tensor = matmul_tensor(m, n, p)
    optimize = _load_optimize(program_path)

    best_ranks: list[int] = []
    best_loss = float("inf")
    for s in range(int(seeds_per_eval)):
        rng = np.random.default_rng(seed * 100_003 + s)
        achieved = start_rank + 1  # sentinel: nothing exact found
        rank = start_rank
        while rank >= min_rank:
            factors = optimize(tensor, rank, rng, int(iters))
            best_loss = min(best_loss, _residual(tensor, factors))
            if verify_exact(tensor, factors):
                achieved = rank
                rank -= 1  # push lower with the remaining budget
            else:
                break
        best_ranks.append(achieved)

    overall_best = min(best_ranks)
    fraction = sum(r == overall_best for r in best_ranks) / len(best_ranks)
    return {
        "neg_best_rank": -float(overall_best),
        "fraction_seeds": fraction if overall_best <= start_rank else 0.0,
        "neg_final_loss": -best_loss,
    }
