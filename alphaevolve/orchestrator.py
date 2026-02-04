"""
Orchestrator - Persistent master agent for AlphaEvolve.

Responsibilities:
- Budget monitoring
- Spawn Search Agents in parallel
- Coordinate evolutionary search
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Any
from alphaevolve.database import Database, SelectionStrategy
from alphaevolve.llm_client import LLMClient, LLMConfig
from alphaevolve.mutation_agent import MutationAgent
from alphaevolve.scoring_agent import ScoringAgent
from alphaevolve.search_agent import SearchAgent


class Orchestrator:
    """
    Master Orchestrator - Persistent agent.

    Monitors budget and spawns Search Agents in parallel.
    Does NOT execute code or query LLMs itself.
    """

    def __init__(
        self,
        config: LLMConfig,
        database: Database,
        evaluator: Callable[[str], float],
        task_description: str = "",
        parallel_slots: int = 50,
        use_cascade: bool = True,
    ):
        """
        Initialize Orchestrator.

        Args:
            config: LLM configuration
            database: Program database (shared state)
            evaluator: Base evaluation function
            task_description: Description of the optimization task
            parallel_slots: Maximum parallel Search Agents
            use_cascade: Whether to use cascaded evaluation
        """
        self.config = config
        self.database = database
        self.evaluator = evaluator
        self.task_description = task_description
        self.parallel_slots = parallel_slots
        self.use_cascade = use_cascade

        # Initialize LLM client (shared by all mutation agents)
        self.llm_client = LLMClient(config)

        # State tracking
        self.generation = 0
        self.best_fitness = -float("inf")
        self.generations_without_improvement = 0

    def run(
        self,
        num_generations: int,
        population_size: int = 5,
        early_stopping_threshold: int = 5,
    ) -> Dict[str, Any]:
        """
        Run evolutionary search.

        Args:
            num_generations: Number of generations to run
            population_size: Number of offspring per generation
            early_stopping_threshold: Stop if no improvement for this many generations

        Returns:
            Dictionary with search statistics
        """
        stats = {
            "total_generated": 0,
            "total_evaluated": 0,
            "generations_run": 0,
            "best_fitness": self.best_fitness,
        }

        print(f"\n{'=' * 60}")
        print(f"Starting evolutionary search for {num_generations} generations")
        print(f"Population size: {population_size}")
        print(f"Parallel slots: {self.parallel_slots}")
        print(f"{'=' * 60}\n")

        for gen in range(1, num_generations + 1):
            print(f"\n{'=' * 60}")
            print(f"Generation {gen}")
            print(f"{'=' * 60}")

            # Advance generation in database
            self.database.advance_generation()

            # Run Search Agents in parallel
            program_ids = self._run_parallel_search(population_size)

            # Update statistics
            stats["total_generated"] += len(program_ids)
            stats["total_evaluated"] += len(program_ids)
            stats["generations_run"] += 1

            # Prune population
            self.database.prune()

            # Get statistics
            db_stats = self.database.get_stats()
            current_best = self.database.get_best()

            print(f"\nGeneration {gen} Statistics:")
            print(f"  Population size: {db_stats['population_size']}")
            print(f"  Best fitness: {db_stats['best_fitness']:.4f}")
            print(f"  Mean fitness: {db_stats['mean_fitness']:.4f}")
            print(f"  Std fitness: {db_stats['std_fitness']:.4f}")

            # Check for improvement
            if current_best and current_best.fitness > self.best_fitness:
                improvement = current_best.fitness - self.best_fitness
                self.best_fitness = current_best.fitness
                self.generations_without_improvement = 0
                stats["best_fitness"] = self.best_fitness
                print(f"  New best! Improvement: +{improvement:.4f}")
            else:
                self.generations_without_improvement += 1
                print(
                    f"  No improvement for {self.generations_without_improvement} generation(s)"
                )

            # Check early stopping
            if self.generations_without_improvement >= early_stopping_threshold:
                print(
                    f"\nEarly stopping: No improvement for {self.generations_without_improvement} generations"
                )
                break

        print(f"\n{'=' * 60}")
        print("EVOLUTIONARY SEARCH COMPLETE")
        print(f"{'=' * 60}")

        return stats

    def _run_parallel_search(self, num_offspring: int) -> list:
        """
        Run Search Agents in parallel.

        Args:
            num_offspring: Total number of offspring to generate

        Returns:
            List of program IDs added to database
        """
        # Calculate how many Search Agents to run
        num_agents = min(num_offspring, self.parallel_slots)

        # Divide offspring among agents
        offspring_per_agent = [num_offspring // num_agents] * num_agents
        for i in range(num_offspring % num_agents):
            offspring_per_agent[i] += 1

        # Create agents
        agents = []
        for i in range(num_agents):
            mutation_agent = MutationAgent(self.llm_client, use_diff=self.config.use_diff)
            scoring_agent = ScoringAgent(
                self.evaluator,
                use_cascade=self.use_cascade,
            )
            search_agent = SearchAgent(
                database=self.database,
                mutation_agent=mutation_agent,
                scoring_agent=scoring_agent,
                task_description=self.task_description,
                use_diff=self.config.use_diff,
            )
            agents.append((search_agent, offspring_per_agent[i]))

        # Run agents in parallel
        program_ids = []
        with ThreadPoolExecutor(max_workers=self.parallel_slots) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(agent.run, num_offspring): agent
                for agent, num_offspring in agents
            }

            # Collect results as they complete
            for future in as_completed(future_to_agent):
                try:
                    ids = future.result()
                    program_ids.extend(ids)
                except Exception as e:
                    print(f"Search Agent failed: {e}")

        return program_ids

    def get_best_program(self):
        """Get the best program found."""
        return self.database.get_best()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.database.get_stats()
