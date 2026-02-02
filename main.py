"""
AlphaEvolve: LLM-Guided Evolutionary Coding Agent

This script demonstrates AlphaEvolve capabilities with an example optimization task.
"""
from alphaevolve.agent import AlphaEvolveAgent
from alphaevolve.config import SearchConfig
from alphaevolve.cli import create_cli_args
from alphaevolve.task_loader import TaskLoader
from alphaevolve.search import NumericalEvaluator
from alphaevolve.utils import write_solution_to_file


def create_example_task():
    """
    Create an example optimization task for AlphaEvolve.
    
    The task is to find a function that transforms input x to produce
    the correct output y = x^2 (with some noise added).
    """
    import numpy as np
    
    # Generate synthetic data: y = x^2 + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 20)
    y = X**2 + np.random.normal(0, 2, size=X.shape)
    
    # Create evaluator
    evaluator = NumericalEvaluator(
        test_inputs=list(X),
        test_targets=list(y),
        error_metric=lambda preds, targets: np.mean((np.array(preds) - np.array(targets))**2)
    )
    
    # Initial heuristic
    initial_code = """
def solve(x):
    # Initial guess: linear relationship
    return x * 5
"""
    
    return evaluator, initial_code


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
    )
    
    # Print configuration
    print("="*70)
    print("AlphaEvolve: LLM-Guided Evolutionary Coding Agent")
    print("="*70)
    print(f"Configuration:")
    print(f"  Model ID: {search_config.model_id}")
    print(f"  Population size: {search_config.population_size}")
    print(f"  Generations: {search_config.num_generations}")
    print(f"  Selection strategy: {search_config.selection_strategy.value}")
    print(f"  Prompt style: {search_config.prompt_style.value}")
    print(f"  Use ensemble: {search_config.use_ensemble}")
    print(f"  Use diff format: {search_config.use_diff_format}")
    print(f"  Use cascaded evaluation: {search_config.use_cascaded_evaluation}")
    print(f"  Use parallel evaluation: {search_config.use_parallel_evaluation}")
    if search_config.use_evolve_blocks and search_config.task_file:
        print(f"  Task file: {search_config.task_file}")
        print(f"  Using EVOLVE-BLOCK markers: True")
    print("="*70)
    
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
            initial_code = task_spec.evolve_blocks[0] if task_spec.evolve_blocks else task_spec.original_code
        else:
            # Fallback to default evaluator
            evaluator, initial_code = create_example_task()
            agent.set_evaluator(evaluator)
    else:
        # Use default example task
        evaluator, initial_code = create_example_task()
        agent.set_evaluator(evaluator)
    
    # Seed population
    print("\nSeeding initial population...")
    agent.seed_population(initial_code)
    
    # Run evolutionary search
    print(f"\nStarting evolutionary search for {search_config.num_generations} generations...")
    print("-"*70)
    
    for gen in range(1, search_config.num_generations + 1):
        should_continue = agent.step(gen)
        if not should_continue:
            break
    
    # Print final results
    print("\n" + "="*70)
    print("EVOLUTIONARY SEARCH COMPLETE")
    print("="*70)
    
    best_program = agent.get_best_program()
    if best_program:
        print(f"\nBest fitness achieved: {best_program.fitness:.4f}")
        print(f"Found at generation: {best_program.generation}")
        
        print("\n" + "-"*70)
        print("Best Solution:")
        print("-"*70)
        print(best_program.code)
        print("-"*70)
        
        # Export solution
        output_file = f"solution_gen_{best_program.generation}.py"
        try:
            write_solution_to_file(best_program.code, output_file)
            print(f"\n✓ Solution exported to: {output_file}")
        except IOError as e:
            print(f"\n✗ Failed to export solution: {e}")
        
        # Print population statistics
        stats = agent.get_population_stats()
        print(f"\nFinal Population Statistics:")
        print(f"  Population size: {stats['population_size']}")
        print(f"  Archive size: {stats['archive_size']}")
        print(f"  Mean fitness: {stats['mean_fitness']:.4f}")
        print(f"  Std fitness: {stats['std_fitness']:.4f}")
    else:
        print("\nNo valid solution found!")


if __name__ == "__main__":
    main()
