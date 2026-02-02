from dataclasses import dataclass
from typing import Optional
from alphaevolve.program_database import SelectionStrategy
from alphaevolve.prompt_sampler import PromptStyle


@dataclass
class SearchConfig:
    # model to be loaded from huggingface
    model_id: str
    # number of candidates
    population_size: int
    # generations
    num_generations: int
    # how many best programs to add to the LLM context for generation
    # in paper terms, inspiration to the LLM
    num_parent_context: int
    # early stopping: stop if fitness doesn't improve after this many generations
    early_stopping_threshold: int = 5

    # Program Database settings
    selection_strategy: SelectionStrategy = SelectionStrategy.MAP_ELITES
    diversity_weight: float = 0.3
    archive_size: int = 1000
    num_islands: int = 3  # Number of islands for island model selection

    # Prompt Sampler settings
    prompt_style: PromptStyle = PromptStyle.STANDARD
    use_dynamic_formatting: bool = True
    num_context_programs: int = 3
    include_evaluation_feedback: bool = True

    # LLM Ensemble settings
    use_ensemble: bool = False
    strong_model_id: Optional[str] = None
    use_diff_format: bool = False

    # Evaluation Engine settings
    use_cascaded_evaluation: bool = False
    fast_eval_ratio: float = 0.3
    use_parallel_evaluation: bool = False
    max_workers: int = 4

    # Task settings
    task_file: Optional[str] = None
    use_evolve_blocks: bool = False
