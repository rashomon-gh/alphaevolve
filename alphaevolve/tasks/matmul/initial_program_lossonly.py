"""Restricted variant of the matmul optimizer for the §4 'no full-file
evolution' ablation: only the loss-and-gradient function may evolve; the
Adam loop is frozen skeleton.
"""

import numpy as np


# EVOLVE-BLOCK-START
def loss_and_grads(
    tensor: np.ndarray, U: np.ndarray, V: np.ndarray, W: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Squared reconstruction loss and its gradients w.r.t. the factors."""
    residual = np.einsum("ar,br,cr->abc", U, V, W) - tensor
    loss = float(np.sum(residual**2))
    gU = 2.0 * np.einsum("abc,br,cr->ar", residual, V, W)
    gV = 2.0 * np.einsum("abc,ar,cr->br", residual, U, W)
    gW = 2.0 * np.einsum("abc,ar,br->cr", residual, U, V)
    return loss, gU, gV, gW


# EVOLVE-BLOCK-END


def optimize(
    tensor: np.ndarray, rank: int, rng: np.random.Generator, iters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = tensor.shape
    U = rng.standard_normal((a, rank)) * 0.3
    V = rng.standard_normal((b, rank)) * 0.3
    W = rng.standard_normal((c, rank)) * 0.3

    lr, beta1, beta2, eps = 0.05, 0.9, 0.999, 1e-8
    m_state = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]
    v_state = [np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)]

    for t in range(1, iters + 1):
        _, gU, gV, gW = loss_and_grads(tensor, U, V, W)
        grads = [gU, gV, gW]
        params = [U, V, W]
        for i in range(3):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grads[i]
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * grads[i] ** 2
            m_hat = m_state[i] / (1 - beta1**t)
            v_hat = v_state[i] / (1 - beta2**t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        U, V, W = params

    return U, V, W
