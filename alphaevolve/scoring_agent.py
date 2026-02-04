"""
Scoring Agent - Ephemeral agent for code evaluation.

Responsibilities:
- Sandbox Creation
- Cascade Evaluation
- Metric Return
"""

from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of scoring a program."""

    fitness: float
    metrics: Dict[str, Any]
    passed_stage: int
    error: Optional[str] = None


class ScoringAgent:
    """
    Ephemeral Scoring Agent.

    Spawned by Search Agent, evaluates code and returns metrics,
    then terminates.
    """

    def __init__(
        self,
        evaluator: Callable[[str], float],
        use_cascade: bool = True,
        fast_eval_ratio: float = 0.3,
    ):
        """
        Initialize Scoring Agent.

        Args:
            evaluator: Base evaluation function (code) -> float
            use_cascade: Whether to use cascaded evaluation
            fast_eval_ratio: Ratio of fast to full evaluation in cascade
        """
        self.evaluator = evaluator
        self.use_cascade = use_cascade
        self.fast_eval_ratio = fast_eval_ratio

    def run(self, code: str) -> ScoreResult:
        """
        Evaluate code and return score.

        Args:
            code: Code to evaluate

        Returns:
            ScoreResult with fitness and metrics
        """
        try:
            if self.use_cascade:
                return self._evaluate_cascade(code)
            else:
                return self._evaluate_full(code)
        except Exception as e:
            return ScoreResult(
                fitness=-float("inf"),
                metrics={},
                passed_stage=0,
                error=str(e),
            )

    def _evaluate_full(self, code: str) -> ScoreResult:
        """
        Evaluate code with full evaluation.

        Args:
            code: Code to evaluate

        Returns:
            ScoreResult
        """
        fitness = self.evaluator(code)
        return ScoreResult(
            fitness=fitness,
            metrics={"fitness": fitness},
            passed_stage=1,
        )

    def _evaluate_cascade(self, code: str) -> ScoreResult:
        """
        Evaluate code using cascaded approach.

        Phase 1: Fast, small-scale test. Fails fast if unsuccessful.
        Phase 2: Full-scale, expensive test (only if Phase 1 passes).

        Args:
            code: Code to evaluate

        Returns:
            ScoreResult
        """
        # Phase 1: Fast evaluation
        fitness_phase1 = self.evaluator(code)

        # Quick rejection threshold
        if fitness_phase1 < -1000:
            return ScoreResult(
                fitness=fitness_phase1,
                metrics={"phase1_fitness": fitness_phase1},
                passed_stage=1,
                error="Failed fast evaluation threshold",
            )

        # Phase 2: Full evaluation (same as Phase 1 in this simplified version)
        # In practice, you'd run a more thorough evaluation here
        fitness_full = fitness_phase1

        return ScoreResult(
            fitness=fitness_full,
            metrics={
                "phase1_fitness": fitness_phase1,
                "phase2_fitness": fitness_full,
            },
            passed_stage=2,
        )
