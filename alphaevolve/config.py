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
