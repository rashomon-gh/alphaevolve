"""
Example task for AlphaEvolve with EVOLVE-BLOCK markers.

This file demonstrates how to structure a task for AlphaEvolve.
The code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END will be evolved.
"""
import numpy as np


# Static helper functions (not evolved)
def load_data():
    """Load training data."""
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    # y = x^2 + noise
    y = X**2 + np.random.normal(0, 3, size=X.shape)
    return X, y


def normalize(x):
    """Normalize input data."""
    return (x - x.mean()) / x.std()


# EVOLVE-BLOCK-START
def solve(x):
    """
    Transform input x to produce the correct output.
    This function will be evolved by AlphaEvolve.
    
    The target pattern is: y = x^2 (approximately)
    """
    # Initial implementation - needs improvement
    return x * 5
# EVOLVE-BLOCK-END


def evaluate():
    """
    Evaluate the current solution.
    Returns a dictionary of scalar metrics (higher is better).
    """
    X, y = load_data()
    X_norm = normalize(X)
    
    predictions = solve(X_norm)
    
    # Calculate metrics
    mse = np.mean((predictions - y) ** 2)
    accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like score
    
    return {
        "accuracy": float(accuracy),
        "negative_mse": float(-mse),
    }


if __name__ == "__main__":
    # Test the initial solution
    metrics = evaluate()
    print(f"Initial evaluation metrics: {metrics}")
