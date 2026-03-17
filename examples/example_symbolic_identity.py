"""
Symbolic expression equivalence task for AlphaEvolve.

Find an expression equivalent to sin(x)^2 + cos(x)^2.
The target is 1 (Pythagorean trigonometric identity).
"""

from sympy import symbols, sin, cos, simplify, expand, trigsimp


def get_target():
    """Return the target expression."""
    return 1


# EVOLVE-BLOCK-START
def solve(x):
    """
    Find an expression that equals 1 using trigonometric identities.
    This function will be evolved by AlphaEvolve.

    The goal is to discover that sin(x)^2 + cos(x)^2 = 1.
    """
    return sin(x) ** 2 + cos(x) ** 2


# EVOLVE-BLOCK-END


def evaluate():
    """
    Evaluate the current symbolic solution.
    Returns a dictionary of scalar metrics (higher is better).
    """
    x = symbols("x")
    target = get_target()

    try:
        result_expr = solve(x)

        diff = simplify(result_expr - target)
        is_exact = diff == 0

        if not is_exact:
            diff = trigsimp(result_expr - target)
            is_exact = diff == 0

        if is_exact:
            from sympy import count_ops

            complexity = count_ops(result_expr)
            complexity_score = 1.0 / (1.0 + complexity)
            fitness = 100.0 + complexity_score
        else:
            try:
                from sympy import Float
                import numpy as np

                errors = []
                for val in np.linspace(-2 * 3.14159, 2 * 3.14159, 20):
                    try:
                        pred = complex(result_expr.subs(x, Float(val)))
                        targ = complex(target.subs(x, Float(val)))
                        errors.append(abs(pred.real - targ.real))
                    except Exception:
                        errors.append(1e6)

                avg_error = sum(errors) / len(errors)
                fitness = 1.0 / (1.0 + avg_error)
            except Exception:
                fitness = 0.0

    except Exception:
        fitness = -1000.0
        is_exact = False

    return {
        "fitness": float(fitness),
        "exact_match": float(is_exact) * 100,
    }


if __name__ == "__main__":
    metrics = evaluate()
    print(f"Evaluation metrics: {metrics}")
    x = symbols("x")
    print(f"Current expression: {solve(x)}")
    print(f"Simplified: {simplify(solve(x))}")
