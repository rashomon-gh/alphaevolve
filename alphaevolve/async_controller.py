"""
Asynchronous Distributed Controller for AlphaEvolve.

Implements an asyncio-based pipeline optimized for throughput with:
- Non-blocking concurrent LLM samplers
- Parallel evaluation worker pool
- Evaluation cascade for resource efficiency
- Queue-based communication between components
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from alphaevolve.config import SearchConfig
from alphaevolve.program_database import ProgramDatabase, Program
from alphaevolve.prompt_sampler import PromptSampler
from alphaevolve.llm_ensemble import LLMEnsemble


class ControllerState(Enum):
    """States of the async controller."""

    IDLE = "idle"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    STOPPED = "stopped"


@dataclass
class GenerationRequest:
    """Request for LLM to generate a new program."""

    parent: Program
    inspirations: List[Program]
    generation_idx: int
    request_id: int


@dataclass
class EvaluationRequest:
    """Request for evaluation of a program."""

    code: str
    generation_idx: int
    request_id: int


@dataclass
class ProgramCandidate:
    """A generated program awaiting evaluation."""

    code: str
    generation_idx: int
    parent_fitness: float
    generation_time: float


@dataclass
class AsyncGenerationResult:
    """Result from LLM generation."""

    code: str
    generation_idx: int
    request_id: int
    latency: float
    parent: Program


@dataclass
class AsyncEvaluationResult:
    """Result from async evaluation."""

    code: str
    fitness: float
    metrics: Dict[str, float]
    passed_stage: int
    error: Optional[str]
    generation_idx: int
    request_id: int
    evaluation_time: float


class AsyncLLMSampler:
    """
    Asynchronous LLM sampler that continuously generates candidate programs.

    Runs as a coroutine that polls a generation request queue and
    produces programs for evaluation without blocking the controller.
    """

    def __init__(
        self,
        llm_ensemble: LLMEnsemble,
        prompt_sampler: PromptSampler,
        task_description: str = "",
        use_diff_format: bool = False,
    ):
        """
        Initialize the async LLM sampler.

        Args:
            llm_ensemble: LLM ensemble for generation
            prompt_sampler: Prompt sampler for constructing prompts
            task_description: Description of the task
            use_diff_format: Whether to use diff format for mutations
        """
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.task_description = task_description
        self.use_diff_format = use_diff_format
        self.request_counter = 0
        self.total_generated = 0
        self.total_generation_time = 0.0

    async def generate_program(
        self,
        parent: Program,
        inspirations: List[Program],
        generation_idx: int,
        generations_without_improvement: int = 0,
    ) -> AsyncGenerationResult:
        """
        Generate a single program asynchronously.

        Args:
            parent: Parent program to mutate
            inspirations: Context programs for prompt
            generation_idx: Current generation index
            generations_without_improvement: Count of stagnant generations

        Returns:
            AsyncGenerationResult with generated code
        """
        request_id = self.request_counter
        self.request_counter += 1

        start_time = time.time()

        try:
            # Construct prompt
            if self.use_diff_format:
                prompt = self.prompt_sampler.construct_diff_prompt(
                    current_program=parent,
                    prior_programs=inspirations,
                    task_description=self.task_description,
                )
            else:
                # Get evaluation feedback for parent
                evaluation_feedback = parent.metadata if parent.metadata else None

                prompt = self.prompt_sampler.construct_prompt(
                    current_program=parent,
                    prior_programs=inspirations,
                    task_description=self.task_description,
                    evaluation_feedback=evaluation_feedback,
                )

            # Generate mutation (LLM calls can be blocking, so we run in executor)
            loop = asyncio.get_event_loop()

            if self.use_diff_format:
                # Run diff-based generation in executor
                new_code = await loop.run_in_executor(
                    None,
                    lambda: self.llm_ensemble.mutate_with_diff(
                        original_code=parent.code,
                        prompt=prompt,
                        generation=generation_idx,
                        num_generations_without_improvement=generations_without_improvement,
                    ),
                )
            else:
                # Run standard generation in executor
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm_ensemble.mutate(
                        prompt=prompt,
                        generation=generation_idx,
                        num_generations_without_improvement=generations_without_improvement,
                    ),
                )

                # Extract code from response
                new_code = self.llm_ensemble.extract_code(response)

            generation_time = time.time() - start_time
            self.total_generated += 1
            self.total_generation_time += generation_time

            return AsyncGenerationResult(
                code=new_code,
                generation_idx=generation_idx,
                request_id=request_id,
                latency=generation_time,
                parent=parent,
            )

        except Exception as e:
            logger.error(f"Error generating program: {e}")
            # Return a failed result
            return AsyncGenerationResult(
                code="",  # Empty code indicates failure
                generation_idx=generation_idx,
                request_id=request_id,
                latency=time.time() - start_time,
                parent=parent,
            )


class AsyncEvaluatorWorker:
    """
    Asynchronous evaluation worker that processes programs from a queue.

    Implements evaluation cascade:
    - Stage 1: Fast, small-scale test
    - Stage 2: Medium-scale test (if Stage 1 passes)
    - Stage 3: Full evaluation (if Stage 2 passes)
    """

    def __init__(
        self,
        worker_id: int,
        base_evaluator: Callable[[str], float],
        use_cascaded: bool = True,
        fast_eval_ratio: float = 0.3,
    ):
        """
        Initialize an async evaluation worker.

        Args:
            worker_id: Unique identifier for this worker
            base_evaluator: Base evaluation function
            use_cascaded: Whether to use cascaded evaluation
            fast_eval_ratio: Ratio of data for fast evaluation
        """
        self.worker_id = worker_id
        self.base_evaluator = base_evaluator
        self.use_cascaded = use_cascaded
        self.fast_eval_ratio = fast_eval_ratio
        self.total_evaluated = 0
        self.total_evaluation_time = 0.0

    async def evaluate(
        self,
        code: str,
        generation_idx: int,
        request_id: int,
    ) -> AsyncEvaluationResult:
        """
        Evaluate a program asynchronously.

        Args:
            code: Code to evaluate
            generation_idx: Generation index
            request_id: Unique request identifier

        Returns:
            AsyncEvaluationResult with fitness and metrics
        """
        start_time = time.time()

        try:
            if self.use_cascaded:
                # Run cascaded evaluation
                result = await self._evaluate_cascaded(code)
            else:
                # Run full evaluation directly
                loop = asyncio.get_event_loop()
                fitness = await loop.run_in_executor(None, self.base_evaluator, code)

                result = {
                    "fitness": fitness,
                    "metrics": {"fitness": fitness},
                    "passed_stage": 1,
                    "error": None,
                }

            evaluation_time = time.time() - start_time
            self.total_evaluated += 1
            self.total_evaluation_time += evaluation_time

            return AsyncEvaluationResult(
                code=code,
                fitness=result["fitness"],
                metrics=result["metrics"],
                passed_stage=result["passed_stage"],
                error=result["error"],
                generation_idx=generation_idx,
                request_id=request_id,
                evaluation_time=evaluation_time,
            )

        except Exception as e:
            logger.error(f"Worker {self.worker_id} evaluation error: {e}")
            return AsyncEvaluationResult(
                code=code,
                fitness=-float("inf"),
                metrics={},
                passed_stage=0,
                error=str(e),
                generation_idx=generation_idx,
                request_id=request_id,
                evaluation_time=time.time() - start_time,
            )

    async def _evaluate_cascaded(self, code: str) -> Dict[str, Any]:
        """
        Perform cascaded evaluation.

        Args:
            code: Code to evaluate

        Returns:
            Dictionary with evaluation results
        """
        loop = asyncio.get_event_loop()

        # Stage 1: Fast evaluation (subset of data)
        # This is a simplified version - in practice, you'd modify the evaluator
        # to support partial evaluation
        try:
            fitness_stage1 = await loop.run_in_executor(None, self.base_evaluator, code)

            # Simple cascade: if Stage 1 is too low, reject early
            if fitness_stage1 < -1000:  # Threshold for quick rejection
                return {
                    "fitness": fitness_stage1,
                    "metrics": {"stage1_fitness": fitness_stage1},
                    "passed_stage": 1,
                    "error": "Failed fast evaluation threshold",
                }

            # Stage 2: Full evaluation (same as Stage 1 in this simplified version)
            # In practice, you'd run a more thorough evaluation here
            fitness_full = fitness_stage1

            return {
                "fitness": fitness_full,
                "metrics": {
                    "stage1_fitness": fitness_stage1,
                    "stage2_fitness": fitness_full,
                },
                "passed_stage": 2,
                "error": None,
            }

        except Exception as e:
            return {
                "fitness": -float("inf"),
                "metrics": {},
                "passed_stage": 0,
                "error": str(e),
            }


class AsyncController:
    """
    Asynchronous distributed controller for AlphaEvolve.

    Implements a non-blocking pipeline with:
    - Concurrent LLM samplers
    - Parallel evaluation worker pool
    - Queue-based communication
    - Throughput optimization
    """

    def __init__(
        self,
        config: SearchConfig,
        database: ProgramDatabase,
        llm_ensemble: LLMEnsemble,
        prompt_sampler: PromptSampler,
        evaluator: Callable[[str], float],
        task_description: str = "",
    ):
        """
        Initialize the async controller.

        Args:
            config: Search configuration
            database: Program database
            llm_ensemble: LLM ensemble for generation
            prompt_sampler: Prompt sampler
            evaluator: Base evaluation function
            task_description: Task description
        """
        self.config = config
        self.database = database
        self.task_description = task_description

        # State tracking
        self.state = ControllerState.IDLE
        self.current_generation = 0
        self.best_fitness = -float("inf")
        self.generations_without_improvement = 0

        # Initialize LLM sampler
        self.llm_sampler = AsyncLLMSampler(
            llm_ensemble=llm_ensemble,
            prompt_sampler=prompt_sampler,
            task_description=task_description,
            use_diff_format=config.use_diff_format,
        )

        # Initialize evaluation worker pool
        self.eval_workers = [
            AsyncEvaluatorWorker(
                worker_id=i,
                base_evaluator=evaluator,
                use_cascaded=config.use_cascaded_evaluation,
                fast_eval_ratio=config.fast_eval_ratio,
            )
            for i in range(config.max_workers)
        ]

        # Queues for async communication
        self.generation_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.evaluation_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.result_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []

        # Statistics
        self.stats = {
            "total_generated": 0,
            "total_evaluated": 0,
            "total_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_evaluation_time": 0.0,
            "throughput": 0.0,
        }

    async def start_workers(self):
        """Start the background worker tasks."""
        # Start evaluation workers
        for worker in self.eval_workers:
            task = asyncio.create_task(self._evaluation_worker_loop(worker))
            self.worker_tasks.append(task)

        logger.info(f"Started {len(self.eval_workers)} evaluation workers")

    async def stop_workers(self):
        """Stop all background worker tasks."""
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to cancel
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()

        logger.info("Stopped all workers")

    async def _evaluation_worker_loop(self, worker: AsyncEvaluatorWorker):
        """
        Main loop for an evaluation worker.

        Continuously pulls from the evaluation queue and processes programs.
        """
        while True:
            try:
                # Get work from queue (non-blocking with timeout)
                try:
                    request = await asyncio.wait_for(
                        self.evaluation_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Evaluate the program
                result = await worker.evaluate(
                    code=request.code,
                    generation_idx=request.generation_idx,
                    request_id=request.request_id,
                )

                # Put result in result queue
                await self.result_queue.put(result)

                # Mark task as done
                self.evaluation_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker.worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker.worker_id} error: {e}")

    async def generate_and_evaluate_batch(
        self,
        parent: Program,
        inspirations: List[Program],
        generation_idx: int,
        batch_size: int,
    ) -> List[Program]:
        """
        Generate and evaluate a batch of programs asynchronously.

        Args:
            parent: Parent program to mutate
            inspirations: Context programs
            generation_idx: Current generation index
            batch_size: Number of programs to generate

        Returns:
            List of evaluated programs
        """
        self.state = ControllerState.GENERATING

        # Create generation tasks
        generation_tasks = []
        for _ in range(batch_size):
            task = asyncio.create_task(
                self.llm_sampler.generate_program(
                    parent=parent,
                    inspirations=inspirations,
                    generation_idx=generation_idx,
                    generations_without_improvement=self.generations_without_improvement,
                )
            )
            generation_tasks.append(task)

        # Wait for all generations to complete
        logger.info(f"Generating {batch_size} programs...")
        generation_results = await asyncio.gather(*generation_tasks)

        # Filter successful generations and queue for evaluation
        successful_generations = [r for r in generation_results if r.code]
        logger.info(
            f"Successfully generated {len(successful_generations)}/{batch_size} programs"
        )

        self.state = ControllerState.EVALUATING

        # Queue successful programs for evaluation
        for i, gen_result in enumerate(successful_generations):
            eval_request = EvaluationRequest(
                code=gen_result.code,
                generation_idx=generation_idx,
                request_id=gen_result.request_id,
            )
            await self.evaluation_queue.put(eval_request)

        # Wait for all evaluations to complete
        logger.info(f"Evaluating {len(successful_generations)} programs...")
        evaluation_results = []
        for _ in range(len(successful_generations)):
            result = await self.result_queue.get()
            evaluation_results.append(result)
            self.result_queue.task_done()

        logger.info(f"Completed evaluation of {len(evaluation_results)} programs")

        # Convert to Program objects
        programs = []
        for result in evaluation_results:
            program = Program(
                code=result.code,
                fitness=result.fitness,
                metadata=result.metrics,
                generation=result.generation_idx,
            )
            if result.error:
                program.metadata["error"] = result.error
            program.metadata["passed_stage"] = result.passed_stage
            program.metadata["evaluation_time"] = result.evaluation_time
            programs.append(program)

        return programs

    async def run_generation_async(
        self,
        generation_idx: int,
    ) -> bool:
        """
        Run a single generation asynchronously.

        Args:
            generation_idx: Generation index

        Returns:
            True to continue, False to stop
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Generation {generation_idx}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # Advance generation in database
        self.database.advance_generation()
        self.current_generation = generation_idx

        # Select parents
        num_parents = min(self.config.num_parent_context, len(self.database.population))
        parents = self.database.select_parents(num_parents)

        logger.info(f"Selected {len(parents)} parents for mutation")
        for i, parent in enumerate(parents):
            logger.info(
                f"  Parent {i + 1}: fitness={parent.fitness:.4f}, gen={parent.generation}"
            )

        # Sample from archive for context
        context_programs = self.database.sample_for_context(
            self.config.num_context_programs
        )

        # Use the best parent for mutation
        parent = parents[0]

        # Generate and evaluate batch asynchronously
        new_programs = await self.generate_and_evaluate_batch(
            parent=parent,
            inspirations=context_programs,
            generation_idx=generation_idx,
            batch_size=self.config.population_size,
        )

        # Add new programs to database
        for program in new_programs:
            self.database.add_program(program)

        # Prune population
        self.database.prune_population()

        # Get statistics
        stats = self.database.get_population_stats()
        current_best = self.database.get_best_program()

        generation_time = time.time() - start_time

        logger.info(f"\nGeneration {generation_idx} Statistics:")
        logger.info(f"  Population size: {stats['population_size']}")
        logger.info(f"  Best fitness: {stats['best_fitness']:.4f}")
        logger.info(f"  Mean fitness: {stats['mean_fitness']:.4f}")
        logger.info(f"  Std fitness: {stats['std_fitness']:.4f}")
        logger.info(f"  Generation time: {generation_time:.2f}s")

        # Update statistics
        self.stats["total_generated"] += len(new_programs)
        self.stats["total_evaluated"] += len(new_programs)
        self.stats["total_time"] += generation_time
        if self.stats["total_evaluated"] > 0:
            self.stats["avg_generation_time"] = (
                self.llm_sampler.total_generation_time / self.stats["total_generated"]
            )
            self.stats["avg_evaluation_time"] = (
                sum(w.total_evaluation_time for w in self.eval_workers)
                / self.stats["total_evaluated"]
            )
            self.stats["throughput"] = (
                self.stats["total_evaluated"] / self.stats["total_time"]
            )

        # Check for improvement
        if current_best and current_best.fitness > self.best_fitness:
            improvement = current_best.fitness - self.best_fitness
            self.best_fitness = current_best.fitness
            self.generations_without_improvement = 0
            logger.info(f"  New best! Improvement: +{improvement:.4f}")
        else:
            self.generations_without_improvement += 1
            logger.info(
                f"  No improvement for {self.generations_without_improvement} generation(s)"
            )

        # Check early stopping
        if self.generations_without_improvement >= self.config.early_stopping_threshold:
            logger.info(
                f"\nEarly stopping: No improvement for {self.generations_without_improvement} generations"
            )
            return False

        return True

    async def run_async(
        self,
        num_generations: int,
    ) -> Dict[str, Any]:
        """
        Run the entire evolutionary search asynchronously.

        Args:
            num_generations: Number of generations to run

        Returns:
            Dictionary with final statistics
        """
        logger.info("Starting async evolutionary search...")
        logger.info(f"Configuration: async=True, workers={self.config.max_workers}")

        # Start workers
        await self.start_workers()

        try:
            # Run generations
            for gen in range(1, num_generations + 1):
                should_continue = await self.run_generation_async(gen)
                if not should_continue:
                    break

        finally:
            # Stop workers
            await self.stop_workers()

        logger.info("\n" + "=" * 70)
        logger.info("ASYNC EVOLUTIONARY SEARCH COMPLETE")
        logger.info("=" * 70)

        return self.stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return self.stats.copy()

    def get_best_program(self) -> Optional[Program]:
        """Get the best program found."""
        return self.database.get_best_program()
