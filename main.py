import argparse
from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaEvolve: Evolutionary search for program synthesis using LLMs"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2b-it",
        help="HuggingFace model ID to use (default: google/gemma-2b-it)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Number of candidate programs in the population (default: 5)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=50,
        help="Number of generations to run the evolutionary search (default: 50)",
    )
    parser.add_argument(
        "--num-parent-context",
        type=int,
        default=2,
        help="Number of best programs to include in LLM context for generation (default: 2)",
    )

    args = parser.parse_args()

    search_config = SearchConfig(
        model_id=args.model_id,
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_parent_context=args.num_parent_context,
    )

    initial_heuristic = """
        def solve(x):
            # Initial guess: linear relationship
            return x * 2
    """

    agent = AlphaEvolveAgent(search_config)
    agent.seed_population(initial_heuristic)

    for gen in range(1, search_config.num_generations + 1):
        agent.step(gen)

    # TODO: export to runnable python file
    # TODO: also ask LLM to add dependencies. solution alone isn't enough
    print("\n=== Final Discovered Solution ===")
    print(agent.population[0].code)
