"""
AlphaEvolve: LLM-Guided Evolutionary Coding Agent

A framework for evolutionary program synthesis using LLMs.
"""

# New agentic architecture modules
from alphaevolve.database import Database, Program, SelectionStrategy
from alphaevolve.config import Config
from alphaevolve.llm_client import LLMClient, LLMConfig, DiffParser
from alphaevolve.mutation_agent import MutationAgent
from alphaevolve.scoring_agent import ScoringAgent, ScoreResult
from alphaevolve.search_agent import SearchAgent
from alphaevolve.orchestrator import Orchestrator

# Keep existing utility modules
from alphaevolve.task_loader import TaskLoader, TaskSpecification
from alphaevolve.search import NumericalEvaluator
from alphaevolve.program_validator import (
    ProgramValidator,
    validate_program,
    validate_program_file,
)
from alphaevolve.utils import write_solution_to_file
from alphaevolve.secrets import values

__version__ = "0.3.0"

__all__ = [
    # New agentic modules
    "Database",
    "Program",
    "SelectionStrategy",
    "Config",
    "LLMClient",
    "LLMConfig",
    "DiffParser",
    "MutationAgent",
    "ScoringAgent",
    "ScoreResult",
    "SearchAgent",
    "Orchestrator",
    # Utility modules (kept)
    "TaskLoader",
    "TaskSpecification",
    "NumericalEvaluator",
    "ProgramValidator",
    "validate_program",
    "validate_program_file",
    "write_solution_to_file",
    "values",
]
