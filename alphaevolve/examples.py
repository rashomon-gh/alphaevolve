"""
Example tasks for AlphaEvolve.

This module provides example optimization tasks of varying difficulty
to demonstrate AlphaEvolve's capabilities.
"""

import numpy as np
from typing import Tuple
from sympy import symbols, sin, cos, exp, log, sqrt, pi, E

from alphaevolve.search import (
    NumericalEvaluator,
    SymbolicEvaluator,
    SymbolicRegressionEvaluator,
)


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
        optimization_strategy="minimize",
    )

    # Initial code with EVOLVE-BLOCK markers
    initial_code = """
# EVOLVE-BLOCK: rewrite this function to correctly fit the target data
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
    y = X**2 * np.sin(X) + 2 * np.cos(X / 2)

    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean(
            (np.array(preds) - np.array(targets)) ** 2
        ),  # type: ignore
        optimization_strategy="minimize",
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
        optimization_strategy="minimize",
    )

    # Initial code with EVOLVE-BLOCK markers
    initial_code = """
# EVOLVE-BLOCK: rewrite this function to correctly fit the target data
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
        optimization_strategy="minimize",
    )

    # Initial code WITHOUT evolve block markers
    initial_code = """
def solve(x):
    # Initial guess: simple linear function
    # Fails to capture the piecewise nature and non-linear regions
    return 2.5 * x
"""

    return evaluator, initial_code


def symbolic_simplification_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a symbolic simplification task.

    The task is to find an expression equivalent to the target:
    (x + 1)^2 = x^2 + 2x + 1

    The initial guess is the expanded form, which is correct but not simplified.
    """
    x = symbols("x")
    target = (x + 1) ** 2

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x},
        complexity_weight=0.05,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols, expand

def solve(x):
    # Initial: expanded form - correct but verbose
    return x**2 + 2*x + 1
"""

    return evaluator, initial_code


def symbolic_trig_identity_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a trigonometric identity discovery task.

    The task is to discover the identity:
    sin(x)^2 + cos(x)^2 = 1

    The initial guess is an incorrect expression.
    """
    x = symbols("x")
    target = 1

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x},
        complexity_weight=0.1,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols, sin, cos

def solve(x):
    # Initial: incorrect guess
    return sin(x) + cos(x)
"""

    return evaluator, initial_code


def symbolic_derivative_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a symbolic derivative discovery task.

    The task is to find the derivative of x^3 * sin(x):
    d/dx(x^3 * sin(x)) = 3*x^2*sin(x) + x^3*cos(x)

    The initial guess is a simple approximation.
    """
    x = symbols("x")
    target = 3 * x**2 * sin(x) + x**3 * cos(x)

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x},
        complexity_weight=0.05,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols, sin, cos

def solve(x):
    # Initial: simple approximation
    return 3 * x**2 * sin(x)
"""

    return evaluator, initial_code


def symbolic_integral_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a symbolic integral discovery task.

    The task is to find the integral of 2*x*sin(x) + x^2*cos(x):
    integral = x^2 * sin(x)

    The initial guess is incorrect.
    """
    x = symbols("x")
    target = x**2 * sin(x)

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x},
        complexity_weight=0.05,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols, sin, cos

def solve(x):
    # Initial: incorrect guess
    return x**2 * cos(x)
"""

    return evaluator, initial_code


def symbolic_regression_quadratic_task() -> Tuple[SymbolicRegressionEvaluator, str]:
    """
    Create a symbolic regression task to discover y = x^2 + 2*x + 1.

    Given data points, discover the underlying formula.
    """
    x = symbols("x")

    data_points = [
        (0, 1),
        (1, 4),
        (2, 9),
        (3, 16),
        (4, 25),
        (5, 36),
        (-1, 0),
        (-2, 1),
        (-3, 4),
    ]

    evaluator = SymbolicRegressionEvaluator(
        data_points=data_points,
        symbols_dict={"x": x},
        error_metric="mse",
        parsimony_pressure=0.01,
        max_complexity=20,
    )

    initial_code = """
from sympy import symbols

def solve(x):
    # Initial: linear approximation
    return 5 * x + 1
"""

    return evaluator, initial_code


def symbolic_regression_trig_task() -> Tuple[SymbolicRegressionEvaluator, str]:
    """
    Create a symbolic regression task to discover y = 2*sin(x) + 1.

    Given noisy data points, discover the trigonometric formula.
    """
    x = symbols("x")

    np.random.seed(42)
    data_points = []
    for xi in np.linspace(-np.pi, np.pi, 20):
        yi = 2 * np.sin(xi) + 1 + np.random.normal(0, 0.1)
        data_points.append((float(xi), float(yi)))

    evaluator = SymbolicRegressionEvaluator(
        data_points=data_points,
        symbols_dict={"x": x},
        error_metric="mse",
        parsimony_pressure=0.02,
        max_complexity=15,
    )

    initial_code = """
from sympy import symbols, sin

def solve(x):
    # Initial: simple sine approximation
    return sin(x)
"""

    return evaluator, initial_code


def symbolic_expression_rewrite_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a task to rewrite expressions into equivalent forms.

    Target: Rewrite sin(2*x) as 2*sin(x)*cos(x)
    """
    x = symbols("x")
    target = 2 * sin(x) * cos(x)

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x},
        complexity_weight=0.1,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols, sin

def solve(x):
    # Initial: original form
    return sin(2*x)
"""

    return evaluator, initial_code


def symbolic_multi_variable_task() -> Tuple[SymbolicEvaluator, str]:
    """
    Create a multi-variable symbolic task.

    Target: x^2 + y^2 + 2*x*y = (x + y)^2
    """
    x, y = symbols("x y")
    target = (x + y) ** 2

    evaluator = SymbolicEvaluator(
        target_expression=target,
        symbols_dict={"x": x, "y": y},
        complexity_weight=0.05,
        equivalence_bonus=100.0,
    )

    initial_code = """
from sympy import symbols

def solve(x, y):
    # Initial: expanded form
    return x**2 + y**2 + 2*x*y
"""

    return evaluator, initial_code


__all__ = [
    "logistic_function_evolve_block_task",
    "composite_function_no_block_task",
    "damped_sine_wave_task",
    "piecewise_function_task",
    "symbolic_simplification_task",
    "symbolic_trig_identity_task",
    "symbolic_derivative_task",
    "symbolic_integral_task",
    "symbolic_regression_quadratic_task",
    "symbolic_regression_trig_task",
    "symbolic_expression_rewrite_task",
    "symbolic_multi_variable_task",
]
