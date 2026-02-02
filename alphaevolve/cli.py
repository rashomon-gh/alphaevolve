import argparse


def create_cli_args():
    parser = argparse.ArgumentParser(
        description="AlphaEvolve: Evolutionary search for program synthesis using LLMs"
    )

    # Basic arguments
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

    # Program Database arguments
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["elitism", "tournament", "map_elites", "island"],
        default="map_elites",
        help="Selection strategy for parent programs (default: map_elites)",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Weight for diversity in selection (0-1, default: 0.3)",
    )

    # Prompt Sampler arguments
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["standard", "concise", "verbose", "analytical", "creative"],
        default="standard",
        help="Prompt style for LLM (default: standard)",
    )
    parser.add_argument(
        "--no-dynamic-formatting",
        action="store_true",
        help="Disable dynamic prompt formatting",
    )

    # LLM Ensemble arguments
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use LLM ensemble with fast and strong models",
    )
    parser.add_argument(
        "--strong-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID for strong model in ensemble",
    )
    parser.add_argument(
        "--use-diff-format",
        action="store_true",
        help="Use Search/Replace diff format for mutations",
    )

    # Evaluation Engine arguments
    parser.add_argument(
        "--use-cascaded-evaluation",
        action="store_true",
        help="Use cascaded (multi-stage) evaluation",
    )
    parser.add_argument(
        "--use-parallel-evaluation",
        action="store_true",
        help="Use parallel evaluation",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )

    # Task arguments
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Path to task file with EVOLVE-BLOCK markers",
    )
    parser.add_argument(
        "--use-evolve-blocks",
        action="store_true",
        help="Enable EVOLVE-BLOCK marker parsing",
    )

    args = parser.parse_args()

    return args
