import argparse

from alphaevolve.llm_client import BackendType


def create_cli_args():
    parser = argparse.ArgumentParser(
        description="AlphaEvolve: Evolutionary search for program synthesis using LLMs (Agentic Mode)"
    )

    # Backend arguments
    parser.add_argument(
        "--backend",
        type=str,
        choices=["huggingface", "openai"],
        default="huggingface",
        help="LLM backend to use: huggingface (local) or openai (Ollama/VLLM/OpenAI) (default: huggingface)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI-compatible API (optional for local servers like Ollama)",
    )

    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2b-it",
        help="Model ID to use (HuggingFace model ID or OpenAI model name) (default: google/gemma-2b-it)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        help="Number of candidate programs in population (default: 5)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=50,
        help="Number of generations to run evolutionary search (default: 50)",
    )
    parser.add_argument(
        "--parallel-slots",
        type=int,
        default=50,
        help="Maximum number of parallel Search Agents (default: 50)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=int,
        default=5,
        help="Stop if fitness doesn't improve after this many generations (default: 5)",
    )

    # Database arguments
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
    parser.add_argument(
        "--archive-size",
        type=int,
        default=1000,
        help="Size of archive for resurfacing old solutions (default: 1000)",
    )
    parser.add_argument(
        "--num-islands",
        type=int,
        default=3,
        help="Number of islands for island model (default: 3)",
    )

    # LLM arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="LLM nucleus sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--use-diff-format",
        action="store_true",
        help="Use Search/Replace diff format for mutations",
    )

    # Evaluation arguments
    parser.add_argument(
        "--use-cascaded-evaluation",
        action="store_true",
        help="Use cascaded (multi-stage) evaluation",
    )
    parser.add_argument(
        "--fast-eval-ratio",
        type=float,
        default=0.3,
        help="Ratio of fast to full evaluation in cascade (default: 0.3)",
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

    args.backend = BackendType(args.backend)

    return args
