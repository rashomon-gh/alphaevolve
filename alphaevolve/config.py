from dataclasses import dataclass


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
