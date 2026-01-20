from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union


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


class Evaluator(ABC):
    """
    Abstract base class for evaluators that assign a scalar score to code.
    
    Subclasses must implement the `evaluate` method to provide custom
    evaluation logic.
    """

    @abstractmethod
    def evaluate(self, code_str: str) -> float:
        """
        Evaluates the code and returns a fitness score.
        
        Args:
            code_str: The code string to evaluate.
            
        Returns:
            A fitness score (higher is better).
        """
        pass


class NumericalEvaluator(Evaluator):
    """
    An evaluator that tests numerical functions against ground truth data.
    
    Users can provide custom test inputs, targets, and an optional custom
    evaluation function. By default, it calculates Mean Squared Error (MSE)
    between predictions and targets.
    
    Example usage:
    https://gist.github.com/rashomon-gh/c951d651985a573c49fe6384aa82a948
    """

    def __init__(
        self,
        test_inputs: List[Union[int, float]],
        test_targets: List[Union[int, float]],
        evaluation_func: Optional[Callable[[str], List[Union[int, float]]]] = None,
        error_metric: Optional[Callable[[List[float], List[float]], float]] = None,
    ):
        """
        Initialize the NumericalEvaluator.
        
        Args:
            test_inputs: List of input values to test the candidate function with.
            test_targets: List of expected output values corresponding to test_inputs.
            evaluation_func: Optional custom function that takes a code string and
                returns a list of predictions. If None, expects the code to define
                a function named 'solve' that takes a single argument.
            error_metric: Optional custom function that takes predictions and targets
                and returns an error value. If None, uses Mean Squared Error (MSE).
        """
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.evaluation_func = evaluation_func
        self.error_metric = error_metric

    def evaluate(self, code_str: str) -> float:
        """
        Evaluates the code by executing it and comparing predictions against targets.
        
        Args:
            code_str: The code string to evaluate.
            
        Returns:
            A fitness score (higher is better, negative error), or -inf if evaluation fails.
        """
        try:
            # Get predictions using either custom evaluation function or default
            if self.evaluation_func is not None:
                predictions = self.evaluation_func(code_str)
            else:
                predictions = self._get_default_predictions(code_str)

            # Calculate error using either custom error metric or default MSE
            if self.error_metric is not None:
                error = self.error_metric(predictions, self.test_targets)
            else:
                error = self._calculate_mse(predictions, self.test_targets)

            # Return negative error (maximization problem)
            return -error

        except Exception:
            # Code that crashes gets the lowest fitness
            return -float("inf")

    def _get_default_predictions(self, code_str: str) -> List[float]:
        """
        Gets predictions using the default evaluation logic.
        
        Expects the code to define a function named 'solve' that takes a single
        argument and returns a numerical value.
        
        Args:
            code_str: The code string to execute.
            
        Returns:
            List of predictions for each test input.
        """
        local_scope = {}
        exec(code_str, {}, local_scope)

        if "solve" not in local_scope:
            raise ValueError("Code must define a 'solve' function")

        candidate_func = local_scope["solve"]
        predictions = []

        for x in self.test_inputs:
            prediction = candidate_func(x)
            if not isinstance(prediction, (int, float)):
                raise ValueError(f"Prediction must be a number, got {type(prediction)}")
            predictions.append(float(prediction))

        return predictions

    def _calculate_mse(
        self, predictions: List[float], targets: List[float]
    ) -> float:
        """
        Calculates Mean Squared Error between predictions and targets.
        
        Args:
            predictions: List of predicted values.
            targets: List of target values.
            
        Returns:
            The Mean Squared Error.
        """
        if len(predictions) != len(targets):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of targets ({len(targets)})"
            )

        total_error = 0.0
        for pred, target in zip(predictions, targets):
            total_error += (pred - target) ** 2

        return total_error / len(predictions)
