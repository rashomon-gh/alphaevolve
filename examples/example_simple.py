"""
My custom optimization task.
"""

import numpy as np


# Static helper functions (not evolved)
def load_data():
    """Load training data."""
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    return X, y


# EVOLVE-BLOCK-START
def solve(x):
    """
    Transform input x to produce the correct output.
    This function will be evolved by AlphaEvolve.
    """
    # Initial implementation - needs improvement
    return x * 2


# EVOLVE-BLOCK-END


def evaluate():
    """
    Evaluate the current solution.
    Returns a dictionary of scalar metrics (higher is better).
    """
    X, y = load_data()
    predictions = solve(X)

    # Calculate metrics
    mse = np.mean((predictions - y) ** 2)
    accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like score

    return {
        "accuracy": float(accuracy),
        "negative_mse": float(-mse),
    }


if __name__ == "__main__":
    metrics = evaluate()
    print(f"Evaluation metrics: {metrics}")
