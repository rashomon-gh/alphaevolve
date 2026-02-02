"""
Evaluation Engine module for AlphaEvolve.

Implements cascaded evaluation and parallel execution for efficiency.
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from alphaevolve.program_database import Program


@dataclass
class EvaluationResult:
    """
    Result of evaluating a program.
    
    Attributes:
        program: The evaluated program
        fitness: Overall fitness score
        metrics: Dictionary of individual metrics
        passed_stage: Which evaluation stages passed
        error: Error message if evaluation failed
    """
    program: Program
    fitness: float
    metrics: Dict[str, float]
    passed_stage: int
    error: Optional[str] = None
    
    def __repr__(self):
        return f"EvaluationResult(fitness={self.fitness:.4f}, stage={self.passed_stage})"


class CascadedEvaluator:
    """
    Implements cascaded (multi-stage) evaluation for efficiency.
    
    Strategy:
    - Stage 1: Fast, small-scale tests. Quick filter for obviously bad code.
    - Stage 2: Medium-scale tests. More thorough evaluation.
    - Stage 3: Full-scale, expensive tests. Only for promising candidates.
    """
    
    def __init__(
        self,
        stage1_func: Callable[[str], Tuple[bool, float]],
        stage2_func: Optional[Callable[[str], Tuple[bool, float]]] = None,
        stage3_func: Optional[Callable[[str], Tuple[bool, float]]] = None,
        stage2_threshold: float = 0.0,
        stage3_threshold: float = 0.5,
    ):
        """
        Initialize the cascaded evaluator.
        
        Args:
            stage1_func: Fast evaluation function (code) -> (passed, score)
            stage2_func: Medium evaluation function (code) -> (passed, score)
            stage3_func: Full evaluation function (code) -> (passed, score)
            stage2_threshold: Minimum score to proceed to stage 2
            stage3_threshold: Minimum score to proceed to stage 3
        """
        self.stage1_func = stage1_func
        self.stage2_func = stage2_func
        self.stage3_func = stage3_func
        self.stage2_threshold = stage2_threshold
        self.stage3_threshold = stage3_threshold
    
    def evaluate(self, code: str) -> EvaluationResult:
        """
        Evaluate code using cascaded approach.
        
        Args:
            code: Code to evaluate
            
        Returns:
            EvaluationResult with fitness and metrics
        """
        try:
            # Stage 1: Fast test
            passed1, score1 = self.stage1_func(code)
            if not passed1:
                return EvaluationResult(
                    program=Program(code=code),
                    fitness=-float("inf"),
                    metrics={},
                    passed_stage=0,
                    error="Failed stage 1 (fast test)",
                )
            
            # Stage 2: Medium test
            if self.stage2_func is not None:
                passed2, score2 = self.stage2_func(code)
                if not passed2 or score2 < self.stage2_threshold:
                    return EvaluationResult(
                        program=Program(code=code),
                        fitness=score1,
                        metrics={"stage1_score": score1, "stage2_score": score2},
                        passed_stage=1,
                    )
            
            # Stage 3: Full test
            if self.stage3_func is not None:
                passed3, score3 = self.stage3_func(code)
                if not passed3 or score3 < self.stage3_threshold:
                    return EvaluationResult(
                        program=Program(code=code),
                        fitness=score2,
                        metrics={
                            "stage1_score": score1,
                            "stage2_score": score2,
                            "stage3_score": score3,
                        },
                        passed_stage=2,
                    )
                
                return EvaluationResult(
                    program=Program(code=code),
                    fitness=score3,
                    metrics={
                        "stage1_score": score1,
                        "stage2_score": score2,
                        "stage3_score": score3,
                    },
                    passed_stage=3,
                )
            
            # If only stage 1 and 2
            return EvaluationResult(
                program=Program(code=code),
                fitness=score2,
                metrics={"stage1_score": score1, "stage2_score": score2},
                passed_stage=2,
            )
            
        except Exception as e:
            return EvaluationResult(
                program=Program(code=code),
                fitness=-float("inf"),
                metrics={},
                passed_stage=0,
                error=str(e),
            )


class ParallelEvaluator:
    """
    Evaluates multiple programs in parallel for throughput.
    """
    
    def __init__(
        self,
        evaluate_func: Callable[[str], EvaluationResult],
        max_workers: int = 4,
    ):
        """
        Initialize the parallel evaluator.
        
        Args:
            evaluate_func: Function to evaluate a single program
            max_workers: Maximum number of parallel workers
        """
        self.evaluate_func = evaluate_func
        self.max_workers = max_workers
    
    def evaluate_batch(self, codes: List[str]) -> List[EvaluationResult]:
        """
        Evaluate multiple codes in parallel.
        
        Args:
            codes: List of code strings to evaluate
            
        Returns:
            List of EvaluationResults in the same order as input
        """
        results = [None] * len(codes)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_index = {
                executor.submit(self.evaluate_func, code): idx
                for idx, code in enumerate(codes)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = EvaluationResult(
                        program=Program(code=codes[idx]),
                        fitness=-float("inf"),
                        metrics={},
                        passed_stage=0,
                        error=str(e),
                    )
        
        return results
    
    def evaluate_programs(self, programs: List[Program]) -> List[EvaluationResult]:
        """
        Evaluate multiple programs in parallel.
        
        Args:
            programs: List of Program objects to evaluate
            
        Returns:
            List of EvaluationResults
        """
        codes = [p.code for p in programs]
        results = self.evaluate_batch(codes)
        
        # Update programs with fitness
        for program, result in zip(programs, results):
            program.fitness = result.fitness
            program.metadata.update(result.metrics)
        
        return results


class MultiObjectiveEvaluator:
    """
    Evaluates programs on multiple objectives.
    
    Supports:
    - Pareto front selection
    - Weighted scalarization
    - Constraint handling
    """
    
    def __init__(
        self,
        evaluate_func: Callable[[str], Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Initialize the multi-objective evaluator.
        
        Args:
            evaluate_func: Function that returns dictionary of metrics
            weights: Optional weights for scalarization
            constraints: Optional constraints (metric_name: (min, max))
        """
        self.evaluate_func = evaluate_func
        self.weights = weights or {}
        self.constraints = constraints or {}
    
    def evaluate(self, code: str) -> EvaluationResult:
        """
        Evaluate code with multiple objectives.
        
        Args:
            code: Code to evaluate
            
        Returns:
            EvaluationResult with scalarized fitness
        """
        try:
            # Get all metrics
            metrics = self.evaluate_func(code)
            
            # Check constraints
            for metric_name, (min_val, max_val) in self.constraints.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if value < min_val or value > max_val:
                        return EvaluationResult(
                            program=Program(code=code),
                            fitness=-float("inf"),
                            metrics=metrics,
                            passed_stage=0,
                            error=f"Constraint violation: {metric_name}={value}",
                        )
            
            # Scalarize if weights provided
            if self.weights:
                fitness = 0.0
                for metric_name, weight in self.weights.items():
                    if metric_name in metrics:
                        fitness += weight * metrics[metric_name]
                
                # Normalize by total weight
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    fitness /= total_weight
            else:
                # Default: use first metric or average
                if len(metrics) == 1:
                    fitness = list(metrics.values())[0]
                else:
                    fitness = np.mean(list(metrics.values()))
            
            return EvaluationResult(
                program=Program(code=code),
                fitness=fitness,
                metrics=metrics,
                passed_stage=1,
            )
            
        except Exception as e:
            return EvaluationResult(
                program=Program(code=code),
                fitness=-float("inf"),
                metrics={},
                passed_stage=0,
                error=str(e),
            )
    
    def find_pareto_front(self, programs: List[Program]) -> List[Program]:
        """
        Find the Pareto front among programs.
        
        Args:
            programs: List of programs to analyze
            
        Returns:
            List of non-dominated programs
        """
        if not programs:
            return []
        
        # Extract metrics
        metrics_list = [p.metadata for p in programs]
        
        # Find non-dominated points
        pareto_front = []
        for i, program in enumerate(programs):
            metrics_i = metrics_list[i]
            is_dominated = False
            
            for j, other_program in enumerate(programs):
                if i == j:
                    continue
                
                metrics_j = metrics_list[j]
                
                # Check if j dominates i
                dominates = True
                for metric_name in metrics_i:
                    if metric_name in metrics_j:
                        if metrics_j[metric_name] < metrics_i[metric_name]:
                            dominates = False
                            break
                
                if dominates:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(program)
        
        return pareto_front


class EvaluationEngine:
    """
    Main evaluation engine that combines cascaded, parallel, and multi-objective evaluation.
    """
    
    def __init__(
        self,
        base_evaluator: Callable[[str], float],
        use_cascaded: bool = False,
        use_parallel: bool = False,
        use_multi_objective: bool = False,
        max_workers: int = 4,
        fast_eval_ratio: float = 0.3,
    ):
        """
        Initialize the evaluation engine.
        
        Args:
            base_evaluator: Base evaluation function (code) -> float
            use_cascaded: Whether to use cascaded evaluation
            use_parallel: Whether to use parallel evaluation
            use_multi_objective: Whether to use multi-objective evaluation
            max_workers: Maximum parallel workers
            fast_eval_ratio: Ratio of fast to full evaluation in cascaded mode
        """
        self.base_evaluator = base_evaluator
        self.use_cascaded = use_cascaded
        self.use_parallel = use_parallel
        self.use_multi_objective = use_multi_objective
        self.max_workers = max_workers
        self.fast_eval_ratio = fast_eval_ratio
        self.use_multi_objective = use_multi_objective
        self.max_workers = max_workers
        self.fast_eval_ratio = fast_eval_ratio
        
        # Setup evaluator
        if use_cascaded:
            # Setup cascaded evaluator (simplified version)
            self.cascaded_evaluator = CascadedEvaluator(
                stage1_func=lambda code: (True, base_evaluator(code)),
            )
        elif use_multi_objective:
            # Setup multi-objective evaluator
            self.multi_objective_evaluator = MultiObjectiveEvaluator(
                evaluate_func=lambda code: {"fitness": base_evaluator(code)},
            )
        
        if use_parallel:
            self.parallel_evaluator = ParallelEvaluator(
                evaluate_func=self._evaluate_single,
                max_workers=max_workers,
            )
    
    def _evaluate_single(self, code: str) -> EvaluationResult:
        """Evaluate a single code string."""
        if self.use_cascaded:
            return self.cascaded_evaluator.evaluate(code)
        elif self.use_multi_objective:
            return self.multi_objective_evaluator.evaluate(code)
        else:
            try:
                fitness = self.base_evaluator(code)
                return EvaluationResult(
                    program=Program(code=code),
                    fitness=fitness,
                    metrics={"fitness": fitness},
                    passed_stage=1,
                )
            except Exception as e:
                return EvaluationResult(
                    program=Program(code=code),
                    fitness=-float("inf"),
                    metrics={},
                    passed_stage=0,
                    error=str(e),
                )
    
    def evaluate(self, code: str) -> float:
        """
        Evaluate code and return fitness.
        
        Args:
            code: Code to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        result = self._evaluate_single(code)
        return result.fitness
    
    def evaluate_batch(self, codes: List[str]) -> List[float]:
        """
        Evaluate multiple codes.
        
        Args:
            codes: List of codes to evaluate
            
        Returns:
            List of fitness scores
        """
        if self.use_parallel:
            results = self.parallel_evaluator.evaluate_batch(codes)
            return [r.fitness for r in results]
        else:
            return [self.evaluate(code) for code in codes]
    
    def evaluate_programs(self, programs: List[Program]) -> List[Program]:
        """
        Evaluate multiple programs and update their fitness.
        
        Args:
            programs: List of programs to evaluate
            
        Returns:
            List of evaluated programs
        """
        codes = [p.code for p in programs]
        fitnesses = self.evaluate_batch(codes)
        
        for program, fitness in zip(programs, fitnesses):
            program.fitness = fitness
        
        return programs
