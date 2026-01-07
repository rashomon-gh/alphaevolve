import argparse


def create_cli_args():
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
    parser.add_argument(
        "--early-stopping-threshold",
        type=int,
        default=5,
        help="Stop if fitness doesn't improve after this many generations (default: 5)",
    )

    args = parser.parse_args()

    return args
