from dataclasses import dataclass


@dataclass
class Program:
    """
    Represents a candidate solution (an 'Individual' in evolutionary terms).
    AlphaEvolve stores these in a Program Database.
    """

    code: str
    # initial score to inf
    # requires cuda (since torch inf isn't defined for cpus)
    fitness: float = -float("inf")

    def __repr__(self):
        return f"Program(fitness={self.fitness:.4f})"


class Evaluator:
    """
    The automated evaluator that assigns a scalar score to code.
    In this demo, we want the agent to discover the function: f(x) = x^2 + 2x + 1
    """

    # TODO: make this extensible, instead of being hardcoded
    def __init__(self):
        # Ground truth data (x, y) pairs
        self.test_inputs = [-5, -2, 0, 2, 5, 10]
        self.test_targets = [x**2 + 2 * x + 1 for x in self.test_inputs]

    def evaluate(self, code_str: str) -> float:
        """
        Executes the code securely (mocked here with exec) and calculates error.
        Higher fitness is better (fitness = -error).
        """
        # Define a local scope to run the generated code
        local_scope = {}

        try:
            # TODO: find an alternative to exec, should be fine for
            # offline runs though!
            exec(code_str, {}, local_scope)

            # We expect the LLM to define a function named 'solve'
            if "solve" not in local_scope:
                return -float("inf")

            candidate_func = local_scope["solve"]

            # Calculate Mean Squared Error
            total_error = 0
            for x, target in zip(self.test_inputs, self.test_targets):
                prediction = candidate_func(x)
                if not isinstance(prediction, (int, float)):
                    return -float("inf")
                total_error += (prediction - target) ** 2

            # Return negative error (maximization problem)
            return -total_error

        except Exception:
            # Code that crashes gets the lowest fitness
            return -float("inf")
