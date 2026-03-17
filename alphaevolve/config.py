"""
Configuration for AlphaEvolve Agentic Architecture.

Simplified configuration for the new agentic system.
"""

from dataclasses import dataclass
from typing import Optional
from alphaevolve.database import SelectionStrategy
from alphaevolve.llm_client import BackendType


@dataclass
class Config:
    """
    Configuration for AlphaEvolve.

    Simplified configuration for the agentic architecture.
    """

    # LLM settings
    model_id: str
    backend: BackendType = BackendType.HUGGINGFACE
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_diff: bool = False
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    # Search settings
    population_size: int = 5
    num_generations: int = 50
    parallel_slots: int = 50  # Max parallel Search Agents
    early_stopping_threshold: int = 5

    # Database settings
    selection_strategy: SelectionStrategy = SelectionStrategy.MAP_ELITES
    diversity_weight: float = 0.3
    archive_size: int = 1000
    num_islands: int = 3

    # Evaluation settings
    use_cascade: bool = True
    fast_eval_ratio: float = 0.3

    # Task settings
    task_file: Optional[str] = None
    use_evolve_blocks: bool = False
    task_description: str = ""
