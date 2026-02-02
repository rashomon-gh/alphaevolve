"""
Example tasks for AlphaEvolve.

This module provides example optimization tasks of varying difficulty
to demonstrate AlphaEvolve's capabilities.
"""

import numpy as np
from typing import Tuple

from alphaevolve.search import NumericalEvaluator


def logistic_function_evolve_block_task() -> Tuple[NumericalEvaluator, str]:
    """
    Create a logistic function fitting task using EVOLVE-BLOCK markers.

    The task is to find a function that approximates:
    y = L / (1 + exp(-k*(x - x0))) + noise

    This is a sigmoid (S-curve) with:
    - L = 100 (maximum value)
    - k = 0.5 (steepness)
    - x0 = 5 (midpoint)

    The initial guess is a linear function, which will be a poor fit.
    This requires discovering the non-linear sigmoidal relationship.
    """
    np.random.seed(42)

    # Generate synthetic data: logistic function + noise
    L = 100.0  # maximum value
    k = 0.5  # steepness
    x0 = 5.0  # midpoint

    X = np.linspace(0, 10, 30)
    y = L / (1 + np.exp(-k * (X - x0))) + np.random.normal(0, 3, size=X.shape)

    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean(
            (np.array(preds) - np.array(targets)) ** 2
        ),  # type: ignore
    )

    # Initial code with EVOLVE-BLOCK markers
    initial_code = """
# EVOLVE-BLOCK: optimize this function to better fit the target data
def solve(x):
    # Initial guess: simple sigmoid approximation
    # This is a crude approximation that will need significant improvement
    return 50 * x / (x + 5)
# END-EVOLVE-BLOCK
"""

    return evaluator, initial_code


def composite_function_no_block_task() -> Tuple[NumericalEvaluator, str]:
    """
    Create a composite function fitting task WITHOUT EVOLVE-BLOCK markers.

    The task is to find a function that approximates:
    y = x^2 * sin(x) + 2*cos(x/2) + noise

    This is a complex composite function combining:
    - Quadratic growth (x^2)
    - Oscillatory behavior (sin(x), cos(x/2))
    - Multiple frequencies

    The initial guess is a simple polynomial, which will be a poor fit.
    This requires discovering both the polynomial and trigonometric components.
    """
    np.random.seed(123)

    # Generate synthetic data: composite function + noise
    X = np.linspace(0, 8, 25)
    y = X**2 * np.sin(X) + 2 * np.cos(X / 2) + np.random.normal(0, 2, size=X.shape)

    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean(
            (np.array(preds) - np.array(targets)) ** 2
        ),  # type: ignore
    )

    # Initial code WITHOUT evolve block markers - entire function will be rewritten
    initial_code = """
def solve(x):
    # Initial guess: simple quadratic polynomial
    # This will be a very poor fit for the complex target function
    return x * x
"""

    return evaluator, initial_code


def damped_sine_wave_task() -> Tuple[NumericalEvaluator, str]:
    """
    Create a damped sine wave fitting task using EVOLVE-BLOCK markers.

    The task is to find a function that approximates:
    y = A * exp(-bx) * sin(cx + d) + noise

    This is a damped oscillatory function with:
    - A = 10 (amplitude)
    - b = 0.3 (damping coefficient)
    - c = 3 (angular frequency)
    - d = 0 (phase shift)

    The initial guess is a simple sine wave without damping.
    This requires discovering both the exponential decay and correct frequency.
    """
    np.random.seed(456)

    # Generate synthetic data: damped sine wave + noise
    A = 10.0
    b = 0.3
    c = 3.0
    d = 0.0

    X = np.linspace(0, 10, 35)
    y = A * np.exp(-b * X) * np.sin(c * X + d) + np.random.normal(0, 0.5, size=X.shape)

    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean(
            (np.array(preds) - np.array(targets)) ** 2
        ),  # type: ignore
    )

    # Initial code with EVOLVE-BLOCK markers
    initial_code = """
# EVOLVE-BLOCK: optimize this function to better fit the target data
def solve(x):
    # Initial guess: simple undamped sine wave
    # Missing the exponential decay component
    return 5 * np.sin(2 * x)
# END-EVOLVE-BLOCK
"""

    return evaluator, initial_code


def piecewise_function_task() -> Tuple[NumericalEvaluator, str]:
    """
    Create a piecewise function fitting task WITHOUT EVOLVE-BLOCK markers.

    The task is to find a function that approximates:
    y = {
        x^2 for x < 3
        9 + 2*(x-3) for 3 <= x < 6
        15 + 3*(x-6) for x >= 6
    } + noise

    This is a piecewise linear/quadratic function with multiple regimes.
    The initial guess is a simple linear function across the entire domain.
    This requires discovering both the piecewise nature and the correct functions
    in each region.
    """
    np.random.seed(789)

    # Generate synthetic data: piecewise function + noise
    X = np.linspace(0, 10, 40)

    y = np.zeros_like(X)
    for i, x in enumerate(X):
        if x < 3:
            y[i] = x**2
        elif x < 6:
            y[i] = 9 + 2 * (x - 3)
        else:
            y[i] = 15 + 3 * (x - 6)

    y += np.random.normal(0, 1, size=X.shape)

    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean(
            (np.array(preds) - np.array(targets)) ** 2
        ),  # type: ignore
    )

    # Initial code WITHOUT evolve block markers
    initial_code = """
def solve(x):
    # Initial guess: simple linear function
    # Fails to capture the piecewise nature and non-linear regions
    return 2.5 * x
"""

    return evaluator, initial_code


__all__ = [
    "logistic_function_evolve_block_task",
    "composite_function_no_block_task",
    "damped_sine_wave_task",
    "piecewise_function_task",
]
