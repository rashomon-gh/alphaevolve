from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig
from alphaevolve.cli import create_cli_args
from alphaevolve.utils import write_solution_to_file

if __name__ == "__main__":
    args = create_cli_args()

    search_config = SearchConfig(
        model_id=args.model_id,
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_parent_context=args.num_parent_context,
        early_stopping_threshold=args.early_stopping_threshold,
    )

    initial_heuristic = """
        def solve(x):
            # Initial guess: linear relationship
            return x * 2
    """

    agent = AlphaEvolveAgent(search_config)
    agent.seed_population(initial_heuristic)

    for gen in range(1, search_config.num_generations + 1):
        should_continue = agent.step(gen)
        if not should_continue:
            break

    print("\n=== Final Discovered Solution ===")
    print(agent.population[0].code)

    # Export final solution to a runnable Python file
    output_file = f"solution_gen_{search_config.num_generations}.py"
    try:
        write_solution_to_file(agent.population[0].code, output_file)
        print(f"\nSolution exported to: {output_file}")
    except IOError as e:
        print(f"\nWarning: Failed to export solution to file: {e}")
