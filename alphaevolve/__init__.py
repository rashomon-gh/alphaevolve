"""
AlphaEvolve: LLM-Guided Evolutionary Coding Agent

A framework for evolutionary program synthesis using LLMs.
"""

from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig
from alphaevolve.program_database import ProgramDatabase, Program, SelectionStrategy
from alphaevolve.prompt_sampler import PromptSampler, PromptStyle
from alphaevolve.llm_ensemble import LLMEnsemble, ModelConfig, ModelTier
from alphaevolve.evaluation_engine import (
    EvaluationEngine,
    CascadedEvaluator,
    EvaluationResult,
)
from alphaevolve.task_loader import TaskLoader, TaskSpecification
from alphaevolve.search import NumericalEvaluator
from alphaevolve.program_validator import (
    ProgramValidator,
    validate_program,
    validate_program_file,
)

__version__ = "0.2.0"

__all__ = [
    "AlphaEvolveAgent",
    "SearchConfig",
    "ProgramDatabase",
    "Program",
    "SelectionStrategy",
    "PromptSampler",
    "PromptStyle",
    "LLMEnsemble",
    "ModelConfig",
    "ModelTier",
    "EvaluationEngine",
    "CascadedEvaluator",
    "EvaluationResult",
    "TaskLoader",
    "TaskSpecification",
    "NumericalEvaluator",
    "ProgramValidator",
    "validate_program",
    "validate_program_file",
]
