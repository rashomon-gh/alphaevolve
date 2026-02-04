"""
AlphaEvolve: LLM-Guided Evolutionary Coding Agent

This script demonstrates AlphaEvolve capabilities with the new agentic architecture.
"""

from loguru import logger

from alphaevolve.config import Config
from alphaevolve.database import Database, SelectionStrategy
from alphaevolve.llm_client import LLMConfig
from alphaevolve.orchestrator import Orchestrator
from alphaevolve.task_loader import TaskLoader
from alphaevolve.utils import write_solution_to_file
from alphaevolve import examples
from alphaevolve.cli import create_cli_args


def main():
    """Main entry point for AlphaEvolve."""
    args = create_cli_args()

    # Map string arguments to enums
    selection_strategy = SelectionStrategy(args.selection_strategy)

    # Create configuration
    config = Config(
        model_id=args.model_id,
        max_tokens=args.max_tokens if hasattr(args, "max_tokens") else 512,
        temperature=args.temperature if hasattr(args, "temperature") else 0.7,
        top_p=args.top_p if hasattr(args, "top_p") else 0.9,
        use_diff=args.use_diff_format,
        population_size=args.population_size,
        num_generations=args.num_generations,
        parallel_slots=args.parallel_slots if hasattr(args, "parallel_slots") else 50,
        early_stopping_threshold=args.early_stopping_threshold,
        selection_strategy=selection_strategy,
        diversity_weight=args.diversity_weight,
        archive_size=args.archive_size if hasattr(args, "archive_size") else 1000,
        num_islands=args.num_islands if hasattr(args, "num_islands") else 3,
        use_cascade=args.use_cascaded_evaluation,
        fast_eval_ratio=args.fast_eval_ratio if hasattr(args, "fast_eval_ratio") else 0.3,
        task_file=args.task_file,
        use_evolve_blocks=args.use_evolve_blocks,
        task_description="",
    )

    # Print configuration
    logger.info("=" * 70)
    logger.info("AlphaEvolve: LLM-Guided Evolutionary Coding Agent (Agentic Mode)")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Model ID: {config.model_id}")
    logger.info(f"  Population size: {config.population_size}")
    logger.info(f"  Generations: {config.num_generations}")
    logger.info(f"  Parallel slots: {config.parallel_slots}")
    logger.info(f"  Selection strategy: {config.selection_strategy.value}")
    logger.info(f"  Use diff format: {config.use_diff}")
    logger.info(f"  Use cascaded evaluation: {config.use_cascade}")
    logger.info("=" * 70)

    # Determine task and log it
    task_spec = None
    if config.use_evolve_blocks and config.task_file:
        task_loader = TaskLoader(config.task_file)
        task_spec = task_loader.parse()
        if task_spec.evaluate_function:
            task_name = f"User-provided task: {config.task_file}"
            logger.info(f"  Task file: {config.task_file}")
            logger.info(f"  Task: {task_name}")
        else:
            task_name = "Logistic function approximation (with EVOLVE-BLOCK markers)"
            logger.info(f"  Task file: {config.task_file}")
            logger.info(f"  Task: {task_name}")
    else:
        task_name = "Composite function approximation (without EVOLVE-BLOCK markers)"
        logger.info(f"  Task: {task_name}")

    if config.use_evolve_blocks:
        logger.info(f"  Using EVOLVE-BLOCK markers: {config.use_evolve_blocks}")
    logger.info("=" * 70)

    # Set evaluator and task
    if task_spec and task_spec.evaluate_function:
        # Use user-provided evaluate function
        evaluator = task_spec.evaluate_function
        initial_code = (
            task_spec.evolve_blocks[0]
            if task_spec.evolve_blocks
            else task_spec.original_code
        )
        config.task_description = "Rewrite the code within EVOLVE-BLOCK markers to correctly fit the target data."
    elif config.use_evolve_blocks and config.task_file:
        # Fallback to default evaluator with evolve block example
        evaluator_obj, initial_code = examples.logistic_function_evolve_block_task()
        evaluator = evaluator_obj.evaluate
    else:
        # Use default example task without evolve blocks
        evaluator_obj, initial_code = examples.composite_function_no_block_task()
        evaluator = evaluator_obj.evaluate

    # Initialize database
    database = Database(
        population_size=config.population_size,
        selection_strategy=config.selection_strategy,
        diversity_weight=config.diversity_weight,
        archive_size=config.archive_size,
        num_islands=config.num_islands,
    )

    # Seed population
    logger.info("Seeding initial population...")
    initial_fitness = evaluator(initial_code)
    database.seed(initial_code, initial_fitness)
    logger.info(f"Seeded with fitness: {initial_fitness:.4f}")

    # Create LLM config
    llm_config = LLMConfig(
        model_id=config.model_id,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        use_diff=config.use_diff,
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(
        config=llm_config,
        database=database,
        evaluator=evaluator,
        task_description=config.task_description,
        parallel_slots=config.parallel_slots,
        use_cascade=config.use_cascade,
    )

    # Run evolutionary search
    stats = orchestrator.run(
        num_generations=config.num_generations,
        population_size=config.population_size,
        early_stopping_threshold=config.early_stopping_threshold,
    )

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("EVOLUTIONARY SEARCH COMPLETE")
    logger.info("=" * 70)

    best_program = orchestrator.get_best_program()
    if best_program:
        logger.info(f"Best fitness achieved: {best_program.fitness:.4f}")
        logger.info(f"Found at generation: {best_program.generation}")

        logger.info("\n" + "-" * 70)
        logger.info("Best Solution:")
        logger.info("-" * 70)
        logger.info(best_program.code)
        logger.info("-" * 70)

        # Export solution
        output_file = f"solution_gen_{best_program.generation}.py"
        try:
            write_solution_to_file(best_program.code, output_file)
            logger.success(f"Solution exported to: {output_file}")
        except IOError as e:
            logger.error(f"Failed to export solution: {e}")

        # Print statistics
        logger.info("\nSearch Statistics:")
        logger.info(f"  Total generated: {stats['total_generated']}")
        logger.info(f"  Total evaluated: {stats['total_evaluated']}")
        logger.info(f"  Generations run: {stats['generations_run']}")
        logger.info(f"  Best fitness: {stats['best_fitness']:.4f}")

        # Print population statistics
        db_stats = orchestrator.get_database_stats()
        logger.info("\nFinal Population Statistics:")
        logger.info(f"  Population size: {db_stats['population_size']}")
        logger.info(f"  Archive size: {db_stats['archive_size']}")
        logger.info(f"  Mean fitness: {db_stats['mean_fitness']:.4f}")
        logger.info(f"  Std fitness: {db_stats['std_fitness']:.4f}")
    else:
        logger.warning("No valid solution found!")


if __name__ == "__main__":
    main()
