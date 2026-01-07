from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig
from alphaevolve.cli import create_cli_args

if __name__ == "__main__":
    args = create_cli_args()

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
