"""Optimizer program for matrix-multiplication tensor decomposition (§3.1).

The evaluator builds the ⟨m,n,p⟩ matmul tensor T of shape (m·n, n·p, p·m) and
calls optimize(tensor, rank, rng, iters), which must return three real or
complex NumPy factor matrices U (m·n × rank), V (n·p × rank), W (p·m × rank)
approximating T ≈ Σ_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]. The evaluator then rounds the
entries to the nearest half-integer and checks the tensor equation exactly.

Only the code between the EVOLVE markers may change; the function signature
is the immutable contract with the evaluator.
"""

import numpy as np


# EVOLVE-BLOCK-START
def optimize(
    tensor: np.ndarray, rank: int, rng: np.random.Generator, iters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deliberately simple initial optimizer (as in the paper): random init +
    squared reconstruction loss + full-batch Adam."""
    a, b, c = tensor.shape
    U = rng.standard_normal((a, rank)) * 0.3
    V = rng.standard_normal((b, rank)) * 0.3
    W = rng.standard_normal((c, rank)) * 0.3

    lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8
    m_state = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    v_state = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]

    for t in range(1, iters + 1):
        residual = np.einsum("ar,br,cr->abc", U, V, W) - tensor
        grads = [
            2.0 * np.einsum("abc,br,cr->ar", residual, V, W),
            2.0 * np.einsum("abc,ar,cr->br", residual, U, W),
            2.0 * np.einsum("abc,ar,br->cr", residual, U, V),
        ]
        params = [U, V, W]
        for i in range(3):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grads[i]
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * grads[i] ** 2
            m_hat = m_state[i] / (1 - beta1**t)
            v_hat = v_state[i] / (1 - beta2**t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        U, V, W = params

    return U, V, W


# EVOLVE-BLOCK-END
