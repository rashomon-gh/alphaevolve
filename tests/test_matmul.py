import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TASK_DIR = Path(__file__).parent.parent / "alphaevolve" / "tasks" / "matmul"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_mod", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_module(TASK_DIR / "evaluator.py")


def strassen_factors():
    """Strassen's rank-7 decomposition of the 2x2 matmul tensor, in the
    evaluator's index convention (a=i*n+j, b=j*p+k, c=k*m+i)."""
    U = np.array(
        [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ],
        dtype=float,
    )
    V = np.array(
        [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    W = np.array(
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    return U, V, W


def test_matmul_tensor_shape_and_count():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    assert tensor.shape == (4, 4, 4)
    assert tensor.sum() == 8  # m*n*p ones
    tensor434 = evaluator.matmul_tensor(4, 3, 4)
    assert tensor434.shape == (12, 12, 16)
    assert tensor434.sum() == 48


def test_strassen_rank7_verifies_exactly():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    assert evaluator.verify_exact(tensor, strassen_factors())


def test_truncated_rank6_rejected():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    U, V, W = strassen_factors()
    assert not evaluator.verify_exact(tensor, (U[:, :6], V[:, :6], W[:, :6]))


def test_perturbed_factors_rejected():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    U, V, W = strassen_factors()
    U = U.copy()
    U[0, 0] += 1.0
    assert not evaluator.verify_exact(tensor, (U, V, W))


def test_near_half_integer_factors_round_and_verify():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    U, V, W = strassen_factors()
    noisy = (U + 0.03, V - 0.02, W + 0.01)  # rounding must recover exactness
    assert evaluator.verify_exact(tensor, noisy)


def test_complex_factors_supported():
    tensor = evaluator.matmul_tensor(2, 2, 2)
    U, V, W = (f.astype(complex) for f in strassen_factors())
    assert evaluator.verify_exact(tensor, (U, V, W))
    # i * conj(i) trick: multiplying one column of U by i and of V by -i
    # keeps the product exact (Gaussian half-integers are allowed).
    U[:, 0] *= 1j
    V[:, 0] *= -1j
    assert evaluator.verify_exact(tensor, (U, V, W))


def test_round_half_integer():
    x = np.array([0.24, 0.26, -0.74, 1.1])
    assert np.allclose(evaluator.round_half_integer(x), [0.0, 0.5, -0.5, 1.0])


PERFECT_PROGRAM = """
import numpy as np
from tests.test_matmul import strassen_factors

def optimize(tensor, rank, rng, iters):
    if rank == 7 and tensor.shape == (4, 4, 4):
        return strassen_factors()
    a, b, c = tensor.shape
    return (np.zeros((a, rank)), np.zeros((b, rank)), np.zeros((c, rank)))
"""


def test_evaluate_reports_best_rank_and_fraction(tmp_path, monkeypatch):
    program = tmp_path / "program.py"
    program.write_text(PERFECT_PROGRAM)
    monkeypatch.setenv(
        "AE_TASK_PARAMS",
        json.dumps(
            {
                "m": 2,
                "n": 2,
                "p": 2,
                "start_rank": 7,
                "min_rank": 6,
                "stage_budgets": [[1, 5], [3, 5]],
            }
        ),
    )
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent))
    scores = evaluator.evaluate(str(program), seed=0, stage=1)
    assert scores["neg_best_rank"] == -7.0  # found 7 exactly, 6 unreachable
    assert scores["fraction_seeds"] == 1.0


def test_initial_program_optimizer_reduces_loss():
    program = load_module(TASK_DIR / "initial_program.py")
    tensor = evaluator.matmul_tensor(2, 2, 2)
    rng = np.random.default_rng(0)
    factors = program.optimize(tensor, rank=8, rng=rng, iters=300)
    loss = np.linalg.norm(np.einsum("ar,br,cr->abc", *factors) - tensor)
    initial_norm = np.linalg.norm(tensor)
    assert loss < initial_norm * 0.5  # Adam made real progress from random init


def test_initial_program_deterministic_given_seed():
    program = load_module(TASK_DIR / "initial_program.py")
    tensor = evaluator.matmul_tensor(2, 2, 2)
    f1 = program.optimize(tensor, 8, np.random.default_rng(7), iters=50)
    f2 = program.optimize(tensor, 8, np.random.default_rng(7), iters=50)
    for a, b in zip(f1, f2, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("m,n,p", [(2, 2, 2), (3, 3, 3)])
def test_trivial_decomposition_verifies(m, n, p):
    # The rank-mnp "schoolbook" decomposition: one rank-1 term per (i,j,k).
    tensor = evaluator.matmul_tensor(m, n, p)
    cols = []
    for i in range(m):
        for j in range(n):
            for k in range(p):
                u = np.zeros(m * n)
                v = np.zeros(n * p)
                w = np.zeros(p * m)
                u[i * n + j] = 1
                v[j * p + k] = 1
                w[k * m + i] = 1
                cols.append((u, v, w))
    U = np.stack([c[0] for c in cols], axis=1)
    V = np.stack([c[1] for c in cols], axis=1)
    W = np.stack([c[2] for c in cols], axis=1)
    assert evaluator.verify_exact(tensor, (U, V, W))
