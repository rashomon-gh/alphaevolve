"""
Program Database - Shared State for AlphaEvolve.

Simple storage for programs, scores, and lineage accessible by all agents.
"""

import random
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class SelectionStrategy(Enum):
    """Selection strategies for parent selection."""

    ELITISM = "elitism"
    TOURNAMENT = "tournament"
    MAP_ELITES = "map_elites"
    ISLAND_MODEL = "island"


@dataclass
class Program:
    """Represents a candidate solution."""

    code: str
    fitness: float = -float("inf")
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[int] = None  # Lineage tracking
    id: int = 0  # Unique identifier for lineage tracking

    def __repr__(self):
        return f"Program(fitness={self.fitness:.4f}, gen={self.generation})"


class Database:
    """
    Program Database - shared state accessible by all agents.

    Stores programs, scores, and lineage information.
    """

    def __init__(
        self,
        population_size: int = 100,
        selection_strategy: SelectionStrategy = SelectionStrategy.MAP_ELITES,
        diversity_weight: float = 0.3,
        archive_size: int = 1000,
        num_islands: int = 3,
    ):
        """
        Initialize the database.

        Args:
            population_size: Maximum population size
            selection_strategy: Strategy for selecting parents
            diversity_weight: Weight for diversity in MAP-Elites (0-1)
            archive_size: Size of archive for resurfacing old solutions
            num_islands: Number of islands for island model
        """
        self.population_size = population_size
        self.selection_strategy = selection_strategy
        self.diversity_weight = diversity_weight
        self.archive_size = archive_size
        self.num_islands = num_islands

        self.population: List[Program] = []
        self.archive: List[Program] = []
        self.best_fitness = -float("inf")
        self.best_program: Optional[Program] = None
        self.generation = 0
        self._next_id = 0

    def add_program(
        self,
        code: str,
        fitness: float,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[int] = None,
    ) -> int:
        """
        Add a program to the database.

        Args:
            code: Program code
            fitness: Fitness score
            metadata: Optional metadata dictionary
            parent_id: ID of parent program (for lineage tracking)

        Returns:
            The ID of the added program
        """
        program = Program(
            code=code,
            fitness=fitness,
            metadata=metadata or {},
            generation=self.generation,
            parent_id=parent_id,
        )
        program.id = self._next_id
        self._next_id += 1

        self.population.append(program)

        # Update best
        if program.fitness > self.best_fitness:
            self.best_fitness = program.fitness
            self.best_program = program

        # Add to archive
        self.archive.append(program)

        # Prune archive if too large
        if len(self.archive) > self.archive_size:
            self.archive.sort(key=lambda p: p.fitness, reverse=True)
            self.archive = self.archive[: self.archive_size]

        return program.id

    def seed(self, code: str, fitness: float = -float("inf")) -> int:
        """
        Seed the database with initial code.

        Args:
            code: Initial code
            fitness: Fitness of initial code

        Returns:
            The ID of the seeded program
        """
        # Seed at generation 0
        old_gen = self.generation
        self.generation = 0
        program_id = self.add_program(code, fitness, parent_id=None)
        self.generation = old_gen
        return program_id

    def select_parent(self) -> Program:
        """
        Select a parent program using the configured strategy.

        Returns:
            Selected parent program
        """
        if len(self.population) == 0:
            raise ValueError("Population is empty")

        if self.selection_strategy == SelectionStrategy.ELITISM:
            return self._select_elitism()
        elif self.selection_strategy == SelectionStrategy.TOURNAMENT:
            return self._select_tournament()
        elif self.selection_strategy == SelectionStrategy.MAP_ELITES:
            return self._select_map_elites()
        elif self.selection_strategy == SelectionStrategy.ISLAND_MODEL:
            return self._select_island_model()
        else:
            return self._select_elitism()

    def _select_elitism(self) -> Program:
        """Select top performing program."""
        return max(self.population, key=lambda p: p.fitness)

    def _select_tournament(self) -> Program:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(
            self.population, min(tournament_size, len(self.population))
        )
        return max(tournament, key=lambda p: p.fitness)

    def _select_map_elites(self) -> Program:
        """
        Select parent using MAP-Elites inspired quality-diversity selection.
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)

        # 70% chance to select from top performers (exploitation)
        # 30% chance to select from diverse performers (exploration)
        if random.random() < (1.0 - self.diversity_weight):
            return sorted_pop[0]
        else:
            # Select from lower fitness programs
            if len(sorted_pop) > 1:
                return random.choice(sorted_pop[1:])
            return sorted_pop[0]

    def _select_island_model(self) -> Program:
        """Select parent using island model."""
        # Divide population into islands based on generation
        generations = sorted(set(p.generation for p in self.population))

        if len(generations) < self.num_islands:
            island_gens = generations
        else:
            island_size = len(generations) // self.num_islands
            island_gens = []
            for i in range(self.num_islands):
                start = i * island_size
                end = (
                    start + island_size
                    if i < self.num_islands - 1
                    else len(generations)
                )
                mid = (start + end - 1) // 2
                island_gens.append(generations[mid])

        # Select random island
        island_gen = random.choice(island_gens)
        island = [p for p in self.population if p.generation == island_gen]

        if island:
            return max(island, key=lambda p: p.fitness)
        else:
            # Fallback to best overall
            if self.best_program is None:
                raise ValueError("No program available for selection")
            return self.best_program

    def sample_context(self, num_samples: int) -> List[Program]:
        """
        Sample programs from archive for context.

        Args:
            num_samples: Number of programs to sample

        Returns:
            List of sampled programs
        """
        if len(self.archive) == 0:
            return []

        # Prefer recent and high-fitness programs
        weights = []
        for program in self.archive:
            fitness_weight = (
                program.fitness - min(p.fitness for p in self.archive)
            ) / (
                max(p.fitness for p in self.archive)
                - min(p.fitness for p in self.archive)
                + 1e-6
            )
            recency_weight = program.generation / (self.generation + 1)
            weights.append(fitness_weight * 0.7 + recency_weight * 0.3)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.archive)] * len(self.archive)

        # Sample based on weights
        sampled_indices = np.random.choice(
            len(self.archive),
            size=min(num_samples, len(self.archive)),
            replace=False,
            p=weights,
        )

        return [self.archive[i] for i in sampled_indices]

    def prune(self) -> None:
        """Prune population to maintain size limit."""
        if len(self.population) <= self.population_size:
            return

        if self.selection_strategy == SelectionStrategy.MAP_ELITES:
            groups = {}
            for program in self.population:
                length_bucket = len(program.code) // 50 * 50
                if length_bucket not in groups:
                    groups[length_bucket] = []
                groups[length_bucket].append(program)

            selected = []
            for length_bucket in sorted(groups.keys()):
                bucket = sorted(
                    groups[length_bucket], key=lambda p: p.fitness, reverse=True
                )
                selected.append(bucket[0])
                if len(selected) >= self.population_size:
                    break

            if len(selected) < self.population_size:
                all_sorted = sorted(
                    self.population, key=lambda p: p.fitness, reverse=True
                )
                added_ids = {p.id for p in selected}
                for prog in all_sorted:
                    if prog.id not in added_ids:
                        selected.append(prog)
                        if len(selected) >= self.population_size:
                            break

            self.population = selected[: self.population_size]
        else:
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            self.population = self.population[: self.population_size]

    def advance_generation(self) -> None:
        """Advance to the next generation."""
        self.generation += 1

    def get_best(self) -> Optional[Program]:
        """Get the best program."""
        return self.best_program

    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        if len(self.population) == 0:
            return {}

        fitnesses = [p.fitness for p in self.population]

        return {
            "population_size": len(self.population),
            "archive_size": len(self.archive),
            "best_fitness": self.best_fitness,
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "current_generation": self.generation,
        }

    def reset(self) -> None:
        """Reset the database."""
        self.population = []
        self.archive = []
        self.best_fitness = -float("inf")
        self.best_program = None
        self.generation = 0
        self._next_id = 0
