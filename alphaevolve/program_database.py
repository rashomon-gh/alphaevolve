"""
Program Database module for AlphaEvolve.

Implements MAP-Elites inspired selection algorithm to balance
exploitation (selecting best programs) and exploration (maintaining diversity).
"""

import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SelectionStrategy(Enum):
    """Selection strategies for parent selection."""

    ELITISM = "elitism"  # Select top performers
    TOURNAMENT = "tournament"  # Tournament selection
    MAP_ELITES = "map_elites"  # Quality-diversity based selection
    ISLAND_MODEL = "island"  # Island model selection


@dataclass
class Program:
    """
    Represents a candidate solution.

    Attributes:
        code: The program code
        fitness: Fitness score (higher is better)
        metadata: Additional metadata (e.g., behavioral descriptors)
        generation: Generation when this program was created
    """

    code: str
    fitness: float = -float("inf")
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation: int = 0

    def __repr__(self):
        return f"Program(fitness={self.fitness:.4f}, gen={self.generation})"


class ProgramDatabase:
    """
    Manages the population of programs with MAP-Elites inspired selection.

    Features:
    - Balance exploitation (best performers) and exploration (diversity)
    - Support multiple selection strategies
    - Resurface previous solutions for context
    - Track program metadata for diversity analysis
    """

    def __init__(
        self,
        population_size: int = 100,
        selection_strategy: SelectionStrategy = SelectionStrategy.MAP_ELITES,
        tournament_size: int = 3,
        diversity_weight: float = 0.3,
        archive_size: int = 1000,
        num_islands: int = 3,
    ):
        """
        Initialize the program database.

        Args:
            population_size: Maximum population size
            selection_strategy: Strategy for selecting parents
            tournament_size: Size of tournament for tournament selection
            diversity_weight: Weight for diversity in selection (0-1)
            archive_size: Size of archive for resurfacing old solutions
            num_islands: Number of islands for island model selection (default: 3)
        """
        self.population_size = population_size
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.diversity_weight = diversity_weight
        self.archive_size = archive_size
        self.num_islands = num_islands

        self.population: List[Program] = []
        self.archive: List[Program] = []  # Archive of all evaluated programs
        self.best_fitness = -float("inf")
        self.best_program: Optional[Program] = None
        self.generation = 0

    def add_program(self, program: Program) -> None:
        """
        Add a program to the database.

        Args:
            program: The program to add
        """
        program.generation = self.generation
        self.population.append(program)

        # Update best
        if program.fitness > self.best_fitness:
            self.best_fitness = program.fitness
            self.best_program = program

        # Add to archive
        self.archive.append(program)

        # Prune archive if too large
        if len(self.archive) > self.archive_size:
            # Keep the best programs from the archive
            self.archive.sort(key=lambda p: p.fitness, reverse=True)
            self.archive = self.archive[: self.archive_size]

    def seed_population(
        self, initial_code: str, fitness: float = -float("inf")
    ) -> None:
        """
        Initialize the database with a user-provided starting point.

        Args:
            initial_code: Initial code to seed with
            fitness: Fitness of the initial code
        """
        program = Program(code=initial_code, fitness=fitness, generation=0)
        self.add_program(program)

    def select_parents(self, num_parents: int) -> List[Program]:
        """
        Select parent programs for mutation based on the configured strategy.

        Args:
            num_parents: Number of parents to select

        Returns:
            List of selected parent programs
        """
        if len(self.population) == 0:
            raise ValueError("Population is empty")

        if self.selection_strategy == SelectionStrategy.ELITISM:
            return self._select_elitism(num_parents)
        elif self.selection_strategy == SelectionStrategy.TOURNAMENT:
            return self._select_tournament(num_parents)
        elif self.selection_strategy == SelectionStrategy.MAP_ELITES:
            return self._select_map_elites(num_parents)
        elif self.selection_strategy == SelectionStrategy.ISLAND_MODEL:
            return self._select_island_model(num_parents)
        else:
            return self._select_elitism(num_parents)

    def _select_elitism(self, num_parents: int) -> List[Program]:
        """Select top performing programs (pure exploitation)."""
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
        return sorted_pop[:num_parents]

    def _select_tournament(self, num_parents: int) -> List[Program]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(
                self.population, min(self.tournament_size, len(self.population))
            )
            winner = max(tournament, key=lambda p: p.fitness)
            parents.append(winner)
        return parents

    def _select_map_elites(self, num_parents: int) -> List[Program]:
        """
        Select parents using MAP-Elites inspired quality-diversity selection.

        Combines fitness and diversity:
        - Top performers (exploitation)
        - Diverse performers (exploration)
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)

        # Top performers for exploitation
        num_exploit = int(num_parents * (1.0 - self.diversity_weight))
        exploit_parents = sorted_pop[:num_exploit]

        # Diverse performers for exploration
        num_explore = num_parents - num_exploit
        if num_explore > 0:
            # Calculate diversity based on code length or other simple metrics
            remaining = sorted_pop[num_exploit:]
            if len(remaining) >= num_explore:
                # Sample diverse programs
                explore_parents = random.sample(remaining, num_explore)
            else:
                explore_parents = remaining
        else:
            explore_parents = []

        return exploit_parents + explore_parents

    def _select_island_model(self, num_parents: int) -> List[Program]:
        """
        Select parents using island model (divide population into islands).

        Each island maintains its own elite, and we rotate between islands.

        The island model divides the population into subgroups (islands) based
        on generation. Each island evolves somewhat independently, and we
        select parents by rotating through islands to promote diversity.
        """
        # Divide population into islands based on generation
        generations = sorted(set(p.generation for p in self.population))

        # If we have fewer distinct generations than islands, merge some
        if len(generations) < self.num_islands:
            # Use each generation as its own island
            island_gens = generations
        else:
            # Distribute generations across num_islands islands
            # Group consecutive generations into islands
            island_size = len(generations) // self.num_islands
            island_gens = []
            for i in range(self.num_islands):
                start = i * island_size
                end = (
                    start + island_size
                    if i < self.num_islands - 1
                    else len(generations)
                )
                # Use the middle generation of each group as representative
                mid = (start + end - 1) // 2
                island_gens.append(generations[mid])

        parents = []
        current_island_idx = 0

        for i in range(num_parents):
            # Select an island (rotate through islands)
            island_gen = island_gens[current_island_idx]
            current_island_idx = (current_island_idx + 1) % len(island_gens)

            # Get programs from this island's generation range
            island = [p for p in self.population if p.generation == island_gen]

            if island:
                # Select best program from this island
                parent = max(island, key=lambda p: p.fitness)
                parents.append(parent)
            else:
                # Fallback: select best from nearest generation
                nearest_gen = min(generations, key=lambda g: abs(g - island_gen))
                island = [p for p in self.population if p.generation == nearest_gen]
                if island:
                    parent = max(island, key=lambda p: p.fitness)
                    parents.append(parent)
                else:
                    # Ultimate fallback to best overall
                    parents.append(self.best_program)

        return parents

    def sample_for_context(self, num_samples: int) -> List[Program]:
        """
        Sample programs from archive to include in prompts for context.

        This allows resurfacing of previous ideas.

        Args:
            num_samples: Number of programs to sample

        Returns:
            List of sampled programs
        """
        if len(self.archive) == 0:
            return []

        # Prefer recent and high-fitness programs
        # Weight by fitness and recency
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

    def prune_population(self) -> None:
        """
        Prune population to maintain size limit while preserving diversity.
        """
        if len(self.population) <= self.population_size:
            return

        if self.selection_strategy == SelectionStrategy.MAP_ELITES:
            # MAP-Elites: keep best diverse programs
            self._prune_map_elites()
        else:
            # Simple: keep best programs
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            self.population = self.population[: self.population_size]

    def _prune_map_elites(self) -> None:
        """
        Prune population while maintaining diversity.

        Uses a simple heuristic: group by code length and keep best from each group.
        """
        # Group by code length as a simple diversity metric
        groups = {}
        for program in self.population:
            length_bucket = len(program.code) // 50 * 50  # Bucket by 50 chars
            if length_bucket not in groups:
                groups[length_bucket] = []
            groups[length_bucket].append(program)

        # Select best from each group until we reach population size
        selected = []
        for length_bucket in sorted(groups.keys()):
            bucket = sorted(
                groups[length_bucket], key=lambda p: p.fitness, reverse=True
            )
            selected.append(bucket[0])

            if len(selected) >= self.population_size:
                break

        self.population = selected[: self.population_size]

    def get_best_program(self) -> Optional[Program]:
        """Get the best program in the population."""
        return self.best_program

    def get_population_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current population.

        Returns:
            Dictionary with population statistics
        """
        if len(self.population) == 0:
            return {}

        fitnesses = [p.fitness for p in self.population]

        return {
            "population_size": len(self.population),
            "archive_size": len(self.archive),
            "best_fitness": self.best_fitness,
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "current_generation": self.generation,
        }

    def advance_generation(self) -> None:
        """Advance to the next generation."""
        self.generation += 1

    def reset(self) -> None:
        """Reset the database."""
        self.population = []
        self.archive = []
        self.best_fitness = -float("inf")
        self.best_program = None
        self.generation = 0
