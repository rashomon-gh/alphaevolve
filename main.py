"""
AlphaEvolve: LLM-Guided Evolutionary Coding Agent

This script demonstrates AlphaEvolve capabilities with an example optimization task.
"""

import asyncio
from loguru import logger

from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig
from alphaevolve.cli import create_cli_args
from alphaevolve.task_loader import TaskLoader
from alphaevolve.utils import write_solution_to_file
from alphaevolve import examples


async def run_async_main(args, search_config):
    """
    Run AlphaEvolve with async controller.

    Args:
        args: Command-line arguments
        search_config: Search configuration
    """
    from loguru import logger

    logger.info("=" * 70)
    logger.info("AlphaEvolve: LLM-Guided Evolutionary Coding Agent (Async Mode)")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Model ID: {search_config.model_id}")
    logger.info(f"  Population size: {search_config.population_size}")
    logger.info(f"  Generations: {search_config.num_generations}")
    logger.info(f"  Selection strategy: {search_config.selection_strategy.value}")
    logger.info(f"  Prompt style: {search_config.prompt_style.value}")
    logger.info(f"  Use ensemble: {search_config.use_ensemble}")
    logger.info(f"  Use diff format: {search_config.use_diff_format}")
    logger.info(f"  Use cascaded evaluation: {search_config.use_cascaded_evaluation}")
    logger.info(f"  Max workers: {search_config.max_workers}")
    logger.info("  ASYNC MODE: ENABLED")
    if search_config.use_evolve_blocks and search_config.task_file:
        logger.info(f"  Task file: {search_config.task_file}")
        logger.info(f"  Using EVOLVE-BLOCK markers: {search_config.use_evolve_blocks}")
    logger.info("=" * 70)

    # Initialize agent
    agent = AlphaEvolveAgent(search_config)

    # Set evaluator and task
    if search_config.use_evolve_blocks and search_config.task_file:
        # Use task file with EVOLVE-BLOCK markers
        task_loader = TaskLoader(search_config.task_file)
        task_spec = task_loader.parse()

        if task_spec.evaluate_function:
            # Use user-provided evaluate function
            agent.set_evaluator(task_spec.evaluate_function)
            initial_code = (
                task_spec.evolve_blocks[0]
                if task_spec.evolve_blocks
                else task_spec.original_code
            )
            task_description = "Optimize the code within EVOLVE-BLOCK markers."
        else:
            # Fallback to default evaluator with evolve block example
            evaluator, initial_code = examples.logistic_function_evolve_block_task()
            agent.set_evaluator(evaluator)
            task_description = ""
    else:
        # Use default example task without evolve blocks
        evaluator, initial_code = examples.composite_function_no_block_task()
        agent.set_evaluator(evaluator)
        task_description = ""

    # Seed population
    logger.info("Seeding initial population...")
    agent.seed_population(initial_code)

    # Initialize async controller
    logger.info("Initializing async controller...")
    agent.initialize_async_controller(
        evaluator=agent.evaluator,
        task_description=task_description,
    )

    # Run async evolutionary search
    logger.info(
        f"Starting async evolutionary search for {search_config.num_generations} generations..."
    )
    logger.info("-" * 70)

    stats = await agent.run_async_search(search_config.num_generations)

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("EVOLUTIONARY SEARCH COMPLETE")
    logger.info("=" * 70)

    best_program = agent.get_best_program()
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
        logger.info(f"  Total time: {stats['total_time']:.2f}s")
        logger.info(f"  Avg generation time: {stats['avg_generation_time']:.2f}s")
        logger.info(f"  Avg evaluation time: {stats['avg_evaluation_time']:.2f}s")
        logger.info(f"  Throughput: {stats['throughput']:.2f} programs/s")

        # Print population statistics
        pop_stats = agent.get_population_stats()
        logger.info("\nFinal Population Statistics:")
        logger.info(f"  Population size: {pop_stats['population_size']}")
        logger.info(f"  Archive size: {pop_stats['archive_size']}")
        logger.info(f"  Mean fitness: {pop_stats['mean_fitness']:.4f}")
        logger.info(f"  Std fitness: {pop_stats['std_fitness']:.4f}")
    else:
        logger.warning("No valid solution found!")

    return stats


def main():
    """Main entry point for AlphaEvolve."""
    args = create_cli_args()

    # Map string arguments to enums
    from alphaevolve.program_database import SelectionStrategy
    from alphaevolve.prompt_sampler import PromptStyle

    # Create configuration
    search_config = SearchConfig(
        model_id=args.model_id,
        population_size=args.population_size,
        num_generations=args.num_generations,
        num_parent_context=args.num_parent_context,
        early_stopping_threshold=args.early_stopping_threshold,
        selection_strategy=SelectionStrategy(args.selection_strategy),
        diversity_weight=args.diversity_weight,
        prompt_style=PromptStyle(args.prompt_style),
        use_dynamic_formatting=not args.no_dynamic_formatting,
        use_ensemble=args.use_ensemble,
        strong_model_id=args.strong_model_id,
        use_diff_format=args.use_diff_format,
        use_cascaded_evaluation=args.use_cascaded_evaluation,
        use_parallel_evaluation=args.use_parallel_evaluation,
        max_workers=args.max_workers,
        task_file=args.task_file,
        use_evolve_blocks=args.use_evolve_blocks,
        use_async=not args.use_sync,  # Async is default, use_sync overrides to False
        async_queue_size=args.async_queue_size
        if hasattr(args, "async_queue_size")
        else 100,
    )

    # Choose execution mode (async is default)
    if not args.use_sync:
        # Run async mode (default)
        asyncio.run(run_async_main(args, search_config))
    else:
        # Run sync mode (explicitly requested)
        run_sync_main(args, search_config)


def run_sync_main(args, search_config):
    """
    Run AlphaEvolve in synchronous mode (original implementation).

    Args:
        args: Command-line arguments
        search_config: Search configuration
    """

    # Print configuration
    logger.info("=" * 70)
    logger.info("AlphaEvolve: LLM-Guided Evolutionary Coding Agent")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Model ID: {search_config.model_id}")
    logger.info(f"  Population size: {search_config.population_size}")
    logger.info(f"  Generations: {search_config.num_generations}")
    logger.info(f"  Selection strategy: {search_config.selection_strategy.value}")
    logger.info(f"  Prompt style: {search_config.prompt_style.value}")
    logger.info(f"  Use ensemble: {search_config.use_ensemble}")
    logger.info(f"  Use diff format: {search_config.use_diff_format}")
    logger.info(f"  Use cascaded evaluation: {search_config.use_cascaded_evaluation}")
    logger.info(f"  Use parallel evaluation: {search_config.use_parallel_evaluation}")
    if search_config.use_evolve_blocks and search_config.task_file:
        logger.info(f"  Task file: {search_config.task_file}")
        logger.info(f"  Using EVOLVE-BLOCK markers: {search_config.use_evolve_blocks}")
    logger.info("=" * 70)

    # Initialize agent
    agent = AlphaEvolveAgent(search_config)

    # Set evaluator
    if search_config.use_evolve_blocks and search_config.task_file:
        # Use task file with EVOLVE-BLOCK markers
        task_loader = TaskLoader(search_config.task_file)
        task_spec = task_loader.parse()

        if task_spec.evaluate_function:
            # Use user-provided evaluate function
            agent.set_evaluator(task_spec.evaluate_function)
            initial_code = (
                task_spec.evolve_blocks[0]
                if task_spec.evolve_blocks
                else task_spec.original_code
            )
        else:
            # Fallback to default evaluator with evolve block example
            evaluator, initial_code = examples.logistic_function_evolve_block_task()
            agent.set_evaluator(evaluator)
    else:
        # Use default example task without evolve blocks
        evaluator, initial_code = examples.composite_function_no_block_task()
        agent.set_evaluator(evaluator)

    # Seed population
    logger.info("Seeding initial population...")
    agent.seed_population(initial_code)

    # Run evolutionary search
    logger.info(
        f"Starting evolutionary search for {search_config.num_generations} generations..."
    )
    logger.info("-" * 70)

    for gen in range(1, search_config.num_generations + 1):
        should_continue = agent.step(gen)
        if not should_continue:
            break

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("EVOLUTIONARY SEARCH COMPLETE")
    logger.info("=" * 70)

    best_program = agent.get_best_program()
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

        # Print population statistics
        stats = agent.get_population_stats()
        logger.info("\nFinal Population Statistics:")
        logger.info(f"  Population size: {stats['population_size']}")
        logger.info(f"  Archive size: {stats['archive_size']}")
        logger.info(f"  Mean fitness: {stats['mean_fitness']:.4f}")
        logger.info(f"  Std fitness: {stats['std_fitness']:.4f}")
    else:
        logger.warning("No valid solution found!")


if __name__ == "__main__":
    main()
