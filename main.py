from loguru import logger

from alphaevolve.config import Config
from alphaevolve.database import Database, SelectionStrategy
from alphaevolve.llm_client import LLMConfig
from alphaevolve.orchestrator import Orchestrator
from alphaevolve.task_loader import TaskLoader
from alphaevolve.utils import write_solution_to_file
from alphaevolve import examples
from alphaevolve.cli import create_cli_args


def _extract_sample_data(task_file: str) -> dict:
    """
    Extract sample data from a task file by executing load_data().

    Args:
        task_file: Path to the task file

    Returns:
        Dictionary with 'inputs' and 'outputs' keys
    """
    try:
        from pathlib import Path

        code = Path(task_file).read_text()
        namespace = {}
        exec(code, namespace, namespace)

        if "load_data" not in namespace:
            return {}

        X, y = namespace["load_data"]()
        return {
            "inputs": X.tolist() if hasattr(X, "tolist") else list(X),
            "outputs": y.tolist() if hasattr(y, "tolist") else list(y),
        }

    except Exception as e:
        logger.warning(f"Failed to extract sample data: {e}")
        return {}


def main():
    """Main entry point for prog_search."""
    args = create_cli_args()

    # Map string arguments to enums
    selection_strategy = SelectionStrategy(args.selection_strategy)

    # Create configuration
    config = Config(
        model_id=args.model_id,
        backend=args.backend,
        max_tokens=args.max_tokens if hasattr(args, "max_tokens") else 512,
        temperature=args.temperature if hasattr(args, "temperature") else 0.7,
        top_p=args.top_p if hasattr(args, "top_p") else 0.9,
        use_diff=args.use_diff_format,
        base_url=args.base_url if hasattr(args, "base_url") else None,
        api_key=args.api_key if hasattr(args, "api_key") else None,
        population_size=args.population_size,
        num_generations=args.num_generations,
        parallel_slots=args.parallel_slots if hasattr(args, "parallel_slots") else 50,
        early_stopping_threshold=args.early_stopping_threshold,
        selection_strategy=selection_strategy,
        diversity_weight=args.diversity_weight,
        archive_size=args.archive_size if hasattr(args, "archive_size") else 1000,
        num_islands=args.num_islands if hasattr(args, "num_islands") else 3,
        use_cascade=args.use_cascaded_evaluation,
        fast_eval_ratio=args.fast_eval_ratio
        if hasattr(args, "fast_eval_ratio")
        else 0.3,
        task_file=args.task_file,
        use_evolve_blocks=args.use_evolve_blocks,
        task_description="",
    )

    # Print configuration
    logger.info("=" * 70)
    logger.info("prog_search: LLM-Guided Evolutionary Coding Agent (Agentic Mode)")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  Backend: {config.backend.value}")
    logger.info(f"  Model ID: {config.model_id}")
    if config.backend.value == "openai" and config.base_url:
        logger.info(f"  Base URL: {config.base_url}")
    logger.info(f"  Population size: {config.population_size}")
    logger.info(f"  Generations: {config.num_generations}")
    logger.info(f"  Parallel slots: {config.parallel_slots}")
    logger.info(f"  Selection strategy: {config.selection_strategy.value}")
    logger.info(f"  Use diff format: {config.use_diff}")
    logger.info(f"  Use cascaded evaluation: {config.use_cascade}")
    logger.info("=" * 70)

    # Determine task and log it
    task_spec = None
    task_loader = None
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
        # The evaluate function from task files takes no arguments - it expects
        # the code to be loaded in its namespace. We need to wrap it to:
        # 1. Take the evolved code block as input
        # 2. Reconstruct the full code by replacing EVOLVE-BLOCK
        # 3. Execute the code
        # 4. Call the evaluate function
        task_loader_obj = task_loader

        def make_evaluator_wrapper(loader):
            def wrapper(code: str) -> float:
                # Reconstruct full code with evolved block
                full_code = loader.reconstruct_code([code])
                # Execute in a namespace - use same dict for globals and locals
                # so functions can find other functions defined in the same scope
                namespace = {}
                exec(full_code, namespace, namespace)
                # Call the evaluate function from the namespace
                evaluate_func = namespace.get("evaluate")
                if evaluate_func is None or not callable(evaluate_func):
                    raise ValueError("evaluate function not found in code")
                result = evaluate_func()
                # Handle dict return values - extract first numeric value
                if isinstance(result, dict):
                    for value in result.values():
                        if isinstance(value, (int, float)):
                            return float(value)
                    return 0.0
                return float(result) if isinstance(result, (int, float)) else 0.0

            return wrapper

        evaluator = make_evaluator_wrapper(task_loader_obj)
        initial_code = (
            task_spec.evolve_blocks[0]
            if task_spec.evolve_blocks
            else task_spec.original_code
        )
        config.task_description = "Rewrite the code within EVOLVE-BLOCK markers to correctly fit the target data."
    elif config.use_evolve_blocks and config.task_file:
        # Fallback to default evaluator with evolve block example
        evaluator_obj, initial_code = examples.logistic_function_evolve_block_task()
        # The evaluator from examples returns (evaluator_obj, initial_code)
        # We need to use the evaluator_obj.evaluate method
        evaluator = evaluator_obj.evaluate
    else:
        # Use default example task without evolve blocks
        evaluator_obj, initial_code = examples.composite_function_no_block_task()
        # The evaluator from examples returns (evaluator_obj, initial_code)
        # We need to use the evaluator_obj.evaluate method
        evaluator = evaluator_obj.evaluate

    # Initialize database
    database = Database(
        population_size=config.population_size,
        selection_strategy=config.selection_strategy,
        diversity_weight=config.diversity_weight,
        archive_size=config.archive_size,
        num_islands=config.num_islands,
    )

    # Extract skeleton code and sample data for prompts
    skeleton_code = ""
    sample_data = {}

    if config.use_evolve_blocks and config.task_file and task_spec:
        # Use original_code (includes EVOLVE-BLOCK markers) for prediction reconstruction
        # Use skeleton_code for showing helper functions in prompts
        # We pass both to the SearchAgent
        skeleton_code = task_spec.original_code
        # Extract sample data by running load_data() from the task file
        sample_data = _extract_sample_data(config.task_file)

    # Seed population
    logger.info("Seeding initial population...")
    initial_fitness = evaluator(initial_code)
    database.seed(initial_code, initial_fitness)
    logger.info(f"Seeded with fitness: {initial_fitness:.4f}")

    # Create LLM config
    llm_config = LLMConfig(
        model_id=config.model_id,
        backend=config.backend,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        use_diff=config.use_diff,
        base_url=config.base_url,
        api_key=config.api_key,
    )

    # Initialize orchestrator
    orchestrator = Orchestrator(
        config=llm_config,
        database=database,
        evaluator=evaluator,
        task_description=config.task_description,
        parallel_slots=config.parallel_slots,
        use_cascade=config.use_cascade,
        skeleton_code=skeleton_code,
        sample_data=sample_data,
    )

    # Run search
    stats = orchestrator.run(
        num_generations=config.num_generations,
        population_size=config.population_size,
        early_stopping_threshold=config.early_stopping_threshold,
    )

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("Program Search Completed.")
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
