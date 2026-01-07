from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig

if __name__ == "__main__":
    # TODO: cmd args
    search_config = SearchConfig(
        model_id="google/gemma-2b-it",
        population_size=5,
        num_generations=50,
        num_parent_context=2,
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
