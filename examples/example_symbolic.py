"""
Symbolic regression task for AlphaEvolve.

Discover the formula y = x^2 + 2*x + 1 from data points.
"""

import numpy as np
from sympy import symbols, sin, cos, exp, sqrt, simplify


def load_data():
    """Load training data points."""
    x_sym = symbols("x")
    target_expr = x_sym**2 + 2 * x_sym + 1

    data_points = []
    for xi in np.linspace(-5, 5, 20):
        yi = float(target_expr.subs(x_sym, xi))
        data_points.append((float(xi), yi))

    return data_points, target_expr


# EVOLVE-BLOCK-START
def solve(x):
    """
    Discover the symbolic expression that fits the data.
    This function will be evolved by AlphaEvolve.

    The goal is to find an expression in terms of x that
    correctly models the relationship in the training data.
    """
    return x * 2 + 1


# EVOLVE-BLOCK-END


def evaluate():
    """
    Evaluate the current symbolic solution.
    Returns a dictionary of scalar metrics (higher is better).
    """
    data_points, target_expr = load_data()
    x_sym = symbols("x")

    total_error = 0.0
    for xi, yi in data_points:
        try:
            pred = float(solve(x_sym).subs(x_sym, xi))
            total_error += (pred - yi) ** 2
        except Exception:
            total_error += 1e6

    mse = total_error / len(data_points)
    accuracy = 1.0 / (1.0 + mse)

    try:
        result_expr = solve(x_sym)
        simplified_diff = simplify(result_expr - target_expr)
        is_exact = simplified_diff == 0
    except Exception:
        is_exact = False

    return {
        "accuracy": float(accuracy),
        "negative_mse": float(-mse),
        "exact_match": float(is_exact) * 100,
    }


if __name__ == "__main__":
    metrics = evaluate()
    print(f"Evaluation metrics: {metrics}")
    print(f"Current expression: {solve(symbols('x'))}")
