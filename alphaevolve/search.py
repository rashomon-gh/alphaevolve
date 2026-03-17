from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Literal, Dict, Any
import numpy as np
from sympy import (
    Expr,
    Symbol,
    symbols,
    simplify,
    expand,
    sympify,
    count_ops,
    nsimplify,
    Rational,
    Float,
    Integer,
    srepr,
)


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
        optimization_strategy: Literal["maximize", "minimize"] = "maximize",
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
            optimization_strategy: Whether to 'maximize' (higher is better) or 'minimize'
                (lower error is better) the metric. Use 'minimize' for error metrics like MSE.
        """
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.evaluation_func = evaluation_func
        self.error_metric = error_metric
        self.optimization_strategy = optimization_strategy

    def evaluate(self, code_str: str) -> float:
        """
        Evaluates the code by executing it and comparing predictions against targets.

        Args:
            code_str: The code string to evaluate.

        Returns:
            A fitness score (higher is better), or -inf if evaluation fails.
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

            # Convert error to fitness based on optimization strategy
            if self.optimization_strategy == "maximize":
                # For metrics where higher is better (e.g., accuracy)
                fitness = error
            else:  # minimize
                # For error metrics where lower is better (e.g., MSE)
                # Use negative log transformation to ensure fitness increases as error decreases
                # Adding a small epsilon to avoid log(0)
                epsilon = 1e-10
                fitness = -np.log(error + epsilon)

            return fitness

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

    def _calculate_mse(self, predictions: List[float], targets: List[float]) -> float:
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


class SymbolicEvaluator(Evaluator):
    """
    An evaluator for symbolic mathematical problems using SymPy.

    This evaluator tests if evolved code produces symbolic expressions
    that are equivalent to or approximate target symbolic expressions.

    Supports:
    - Exact symbolic equivalence checking
    - Structural similarity scoring
    - Expression complexity penalties
    - Multi-expression targets

    Example usage:
        x = symbols('x')
        evaluator = SymbolicEvaluator(
            target_expression=x**2 + 2*x + 1,
            symbols_dict={'x': x},
            complexity_weight=0.1
        )
    """

    def __init__(
        self,
        target_expression: Union[Expr, List[Expr]],
        symbols_dict: Dict[str, Symbol],
        evaluation_func: Optional[Callable[[str], Union[Expr, List[Expr]]]] = None,
        complexity_weight: float = 0.1,
        equivalence_bonus: float = 100.0,
        similarity_weight: float = 1.0,
        optimization_strategy: Literal["maximize", "minimize"] = "maximize",
        simplify_timeout: int = 5,
    ):
        """
        Initialize the SymbolicEvaluator.

        Args:
            target_expression: Target sympy expression(s) to match.
            symbols_dict: Dictionary mapping symbol names to sympy Symbol objects.
            evaluation_func: Optional custom function that takes code string and
                returns a sympy expression. If None, expects code to define
                a 'solve' function that returns a symbolic expression.
            complexity_weight: Weight for complexity penalty in fitness calculation.
                Higher values penalize complex expressions more.
            equivalence_bonus: Bonus fitness when expression is exactly equivalent.
            similarity_weight: Weight for structural similarity scoring.
            optimization_strategy: Whether to 'maximize' or 'minimize' fitness.
            simplify_timeout: Timeout in seconds for simplification operations.
        """
        self.target_expression = (
            target_expression
            if isinstance(target_expression, list)
            else [target_expression]
        )
        self.symbols_dict = symbols_dict
        self.evaluation_func = evaluation_func
        self.complexity_weight = complexity_weight
        self.equivalence_bonus = equivalence_bonus
        self.similarity_weight = similarity_weight
        self.optimization_strategy = optimization_strategy
        self.simplify_timeout = simplify_timeout

        self._target_complexity = sum(
            count_ops(expr) for expr in self.target_expression
        )

    def evaluate(self, code_str: str) -> float:
        """
        Evaluates code by comparing its symbolic output to the target expression.

        Args:
            code_str: The code string to evaluate.

        Returns:
            A fitness score (higher is better), or -inf if evaluation fails.
        """
        try:
            if self.evaluation_func is not None:
                candidate_expressions = self.evaluation_func(code_str)
            else:
                candidate_expressions = self._get_default_expressions(code_str)

            if not isinstance(candidate_expressions, list):
                candidate_expressions = [candidate_expressions]

            if len(candidate_expressions) != len(self.target_expression):
                return -float("inf")

            total_fitness = 0.0

            for candidate, target in zip(candidate_expressions, self.target_expression):
                fitness = self._evaluate_single(candidate, target)
                total_fitness += fitness

            return total_fitness / len(self.target_expression)

        except Exception:
            return -float("inf")

    def _get_default_expressions(self, code_str: str) -> List[Expr]:
        """
        Gets expressions using default evaluation logic.

        Expects code to define a 'solve' function that returns
        a sympy expression when called with symbolic arguments.

        Args:
            code_str: The code string to execute.

        Returns:
            List of sympy expressions.
        """
        local_scope = {"symbols": symbols, **self.symbols_dict}
        for key in [
            "Expr",
            "Symbol",
            "simplify",
            "expand",
            "sympify",
            "Rational",
            "Float",
            "Integer",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pi",
            "E",
            "I",
            "oo",
        ]:
            try:
                import sympy

                local_scope[key] = getattr(sympy, key)
            except (ImportError, AttributeError):
                pass

        exec(code_str, local_scope, local_scope)

        if "solve" not in local_scope:
            raise ValueError("Code must define a 'solve' function")

        solve_func = local_scope["solve"]

        symbol_args = [
            self.symbols_dict[name] for name in sorted(self.symbols_dict.keys())
        ]

        result = solve_func(*symbol_args)

        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

    def _evaluate_single(self, candidate: Expr, target: Expr) -> float:
        """
        Evaluate a single candidate expression against a target.

        Args:
            candidate: The candidate sympy expression.
            target: The target sympy expression.

        Returns:
            Fitness score for this expression pair.
        """
        if not isinstance(candidate, Expr):
            try:
                candidate = sympify(candidate)
            except Exception:
                return -float("inf")

        try:
            diff = simplify(candidate - target)
            if diff == 0:
                complexity_score = self._complexity_score(candidate)
                return self.equivalence_bonus + complexity_score
        except Exception:
            pass

        try:
            if candidate.equals(target):
                complexity_score = self._complexity_score(candidate)
                return self.equivalence_bonus + complexity_score
        except Exception:
            pass

        similarity = self._structural_similarity(candidate, target)
        complexity_penalty = self._complexity_penalty(candidate)

        fitness = (similarity * self.similarity_weight) - (
            complexity_penalty * self.complexity_weight
        )

        return fitness

    def _complexity_score(self, expr: Expr) -> float:
        """
        Calculate a positive score based on expression simplicity.

        Simpler expressions get higher scores.

        Args:
            expr: The sympy expression.

        Returns:
            Complexity score (higher is simpler).
        """
        try:
            ops = count_ops(expr)
            return 1.0 / (1.0 + ops)
        except Exception:
            return 0.0

    def _complexity_penalty(self, expr: Expr) -> float:
        """
        Calculate penalty based on expression complexity.

        More complex expressions get higher penalties.

        Args:
            expr: The sympy expression.

        Returns:
            Complexity penalty.
        """
        try:
            candidate_ops = count_ops(expr)
            target_ops = self._target_complexity

            if candidate_ops <= target_ops:
                return 0.0
            return float(candidate_ops - target_ops)
        except Exception:
            return 100.0

    def _structural_similarity(self, candidate: Expr, target: Expr) -> float:
        """
        Calculate structural similarity between expressions.

        Uses multiple heuristics:
        1. Tree structure comparison
        2. Symbol overlap
        3. Operation type overlap
        4. Numerical evaluation comparison

        Args:
            candidate: Candidate expression.
            target: Target expression.

        Returns:
            Similarity score between 0 and 1.
        """
        scores = []

        candidate_symbols = set(str(s) for s in candidate.free_symbols)
        target_symbols = set(str(s) for s in target.free_symbols)

        if target_symbols:
            symbol_overlap = len(candidate_symbols & target_symbols) / len(
                target_symbols
            )
        else:
            symbol_overlap = 1.0 if not candidate_symbols else 0.0
        scores.append(symbol_overlap)

        candidate_ops = self._get_operation_types(candidate)
        target_ops = self._get_operation_types(target)

        if target_ops:
            ops_overlap = len(candidate_ops & target_ops) / len(target_ops)
        else:
            ops_overlap = 1.0 if not candidate_ops else 0.0
        scores.append(ops_overlap)

        numerical_sim = self._numerical_similarity(candidate, target)
        scores.append(numerical_sim)

        tree_sim = self._tree_similarity(candidate, target)
        scores.append(tree_sim)

        return sum(scores) / len(scores)

    def _get_operation_types(self, expr: Expr) -> set:
        """
        Extract the types of operations used in an expression.

        Args:
            expr: The sympy expression.

        Returns:
            Set of operation type names.
        """
        ops = set()

        def traverse(e):
            if hasattr(e, "func"):
                ops.add(e.func.__name__)
            for arg in e.args:
                traverse(arg)

        try:
            traverse(expr)
        except Exception:
            pass

        return ops

    def _numerical_similarity(self, candidate: Expr, target: Expr) -> float:
        """
        Compare expressions by numerical evaluation at random points.

        Args:
            candidate: Candidate expression.
            target: Target expression.

        Returns:
            Similarity score based on numerical agreement.
        """
        try:
            symbol_list = list(self.symbols_dict.values())
            if not symbol_list:
                return 1.0

            test_values = []
            for _ in range(10):
                point = {}
                for sym in symbol_list:
                    point[sym] = Float(np.random.uniform(-5, 5))
                test_values.append(point)

            errors = []
            for point in test_values:
                try:
                    candidate_val = complex(candidate.subs(point))
                    target_val = complex(target.subs(point))

                    if abs(target_val) > 1e-10:
                        rel_error = abs(candidate_val - target_val) / abs(target_val)
                    else:
                        rel_error = abs(candidate_val - target_val)

                    errors.append(rel_error)
                except Exception:
                    errors.append(float("inf"))

            if not errors:
                return 0.0

            avg_error = np.mean(errors)
            return float(np.exp(-avg_error))

        except Exception:
            return 0.0

    def _tree_similarity(self, candidate: Expr, target: Expr) -> float:
        """
        Calculate similarity based on expression tree structure.

        Uses srepr to get string representations and compares them.

        Args:
            candidate: Candidate expression.
            target: Target expression.

        Returns:
            Tree structure similarity score.
        """
        try:
            candidate_repr = srepr(candidate)
            target_repr = srepr(target)

            if candidate_repr == target_repr:
                return 1.0

            candidate_tokens = set(candidate_repr.split("("))
            target_tokens = set(target_repr.split("("))

            if not target_tokens:
                return 0.0

            overlap = len(candidate_tokens & target_tokens)
            return overlap / len(target_tokens)

        except Exception:
            return 0.0


class SymbolicRegressionEvaluator(Evaluator):
    """
    An evaluator for symbolic regression problems.

    Unlike SymbolicEvaluator which checks symbolic equivalence,
    this evaluator finds expressions that fit numerical data points
    while preferring simpler expressions.

    This is useful for:
    - Discovering mathematical formulas from data
    - Symbolic function approximation
    - Equation discovery

    Example usage:
        evaluator = SymbolicRegressionEvaluator(
            data_points=[(1, 1), (2, 4), (3, 9), (4, 16)],
            symbols_dict={'x': symbols('x')},
            parsimony_pressure=0.01
        )
    """

    def __init__(
        self,
        data_points: List[tuple],
        symbols_dict: Dict[str, Symbol],
        evaluation_func: Optional[Callable[[str], Expr]] = None,
        error_metric: str = "mse",
        parsimony_pressure: float = 0.01,
        max_complexity: int = 50,
        optimization_strategy: Literal["maximize", "minimize"] = "maximize",
    ):
        """
        Initialize the SymbolicRegressionEvaluator.

        Args:
            data_points: List of (input_values, output_value) tuples.
                For single variable: [(x1, y1), (x2, y2), ...]
                For multiple variables: [((x1, z1), y1), ...]
            symbols_dict: Dictionary mapping symbol names to sympy Symbols.
            evaluation_func: Optional custom evaluation function.
            error_metric: Error metric to use ('mse', 'mae', 'rmse').
            parsimony_pressure: Penalty per operation to favor simpler expressions.
            max_complexity: Maximum allowed operations in expression.
            optimization_strategy: 'maximize' or 'minimize'.
        """
        self.data_points = data_points
        self.symbols_dict = symbols_dict
        self.evaluation_func = evaluation_func
        self.error_metric = error_metric
        self.parsimony_pressure = parsimony_pressure
        self.max_complexity = max_complexity
        self.optimization_strategy = optimization_strategy

        self._symbol_list = list(symbols_dict.values())

    def evaluate(self, code_str: str) -> float:
        """
        Evaluate code by testing its expression against data points.

        Args:
            code_str: The code string to evaluate.

        Returns:
            Fitness score (higher is better).
        """
        try:
            if self.evaluation_func is not None:
                expr = self.evaluation_func(code_str)
            else:
                expr = self._get_expression(code_str)

            if not isinstance(expr, Expr):
                expr = sympify(expr)

            complexity = count_ops(expr)
            if complexity > self.max_complexity:
                return -float("inf")

            error = self._calculate_error(expr)

            epsilon = 1e-10
            fitness = -np.log(error + epsilon) - (complexity * self.parsimony_pressure)

            return fitness

        except Exception:
            return -float("inf")

    def _get_expression(self, code_str: str) -> Expr:
        """
        Execute code and extract the symbolic expression.

        Args:
            code_str: Code to execute.

        Returns:
            Sympy expression from the 'solve' function.
        """
        local_scope = {"symbols": symbols, **self.symbols_dict}
        for key in [
            "Expr",
            "Symbol",
            "simplify",
            "expand",
            "sympify",
            "Rational",
            "Float",
            "Integer",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pi",
            "E",
            "I",
            "oo",
        ]:
            try:
                import sympy

                local_scope[key] = getattr(sympy, key)
            except (ImportError, AttributeError):
                pass

        exec(code_str, local_scope, local_scope)

        if "solve" not in local_scope:
            raise ValueError("Code must define a 'solve' function")

        solve_func = local_scope["solve"]
        return solve_func(*self._symbol_list)

    def _calculate_error(self, expr: Expr) -> float:
        """
        Calculate error between expression predictions and data points.

        Args:
            expr: The sympy expression to evaluate.

        Returns:
            Error value based on selected metric.
        """
        errors = []

        for inputs, target in self.data_points:
            if isinstance(inputs, (int, float)):
                inputs = (inputs,)

            if len(inputs) != len(self._symbol_list):
                raise ValueError(
                    f"Input dimension mismatch: expected {len(self._symbol_list)}, "
                    f"got {len(inputs)}"
                )

            subs_dict = dict(zip(self._symbol_list, inputs))

            try:
                prediction = float(expr.subs(subs_dict))
                errors.append((prediction - target) ** 2)
            except Exception:
                errors.append(float("inf"))

        if self.error_metric == "mse":
            return np.mean(errors)
        elif self.error_metric == "rmse":
            return np.sqrt(np.mean(errors))
        elif self.error_metric == "mae":
            return np.mean([np.sqrt(e) for e in errors])
        else:
            return np.mean(errors)
