"""
Test script to verify AlphaEvolve imports and basic functionality.
"""

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from alphaevolve import AlphaEvolveAgent
        print("✓ AlphaEvolveAgent imported")
    except Exception as e:
        print(f"✗ Failed to import AlphaEvolveAgent: {e}")
        return False
    
    try:
        from alphaevolve import SearchConfig
        print("✓ SearchConfig imported")
    except Exception as e:
        print(f"✗ Failed to import SearchConfig: {e}")
        return False
    
    try:
        from alphaevolve import ProgramDatabase, Program, SelectionStrategy
        print("✓ ProgramDatabase, Program, SelectionStrategy imported")
    except Exception as e:
        print(f"✗ Failed to import ProgramDatabase: {e}")
        return False
    
    try:
        from alphaevolve import PromptSampler, PromptStyle
        print("✓ PromptSampler, PromptStyle imported")
    except Exception as e:
        print(f"✗ Failed to import PromptSampler: {e}")
        return False
    
    try:
        from alphaevolve import LLMEnsemble, ModelConfig, ModelTier
        print("✓ LLMEnsemble, ModelConfig, ModelTier imported")
    except Exception as e:
        print(f"✗ Failed to import LLMEnsemble: {e}")
        return False
    
    try:
        from alphaevolve import EvaluationEngine, CascadedEvaluator
        print("✓ EvaluationEngine, CascadedEvaluator imported")
    except Exception as e:
        print(f"✗ Failed to import EvaluationEngine: {e}")
        return False
    
    try:
        from alphaevolve import TaskLoader, TaskSpecification
        print("✓ TaskLoader, TaskSpecification imported")
    except Exception as e:
        print(f"✗ Failed to import TaskLoader: {e}")
        return False
    
    try:
        from alphaevolve import NumericalEvaluator
        print("✓ NumericalEvaluator imported")
    except Exception as e:
        print(f"✗ Failed to import NumericalEvaluator: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_config():
    """Test SearchConfig creation."""
    print("\nTesting SearchConfig...")
    
    try:
        from alphaevolve import SearchConfig, SelectionStrategy, PromptStyle
        
        config = SearchConfig(
            model_id="google/gemma-2b-it",
            population_size=10,
            num_generations=50,
            num_parent_context=3,
            selection_strategy=SelectionStrategy.MAP_ELITES,
            prompt_style=PromptStyle.STANDARD,
        )
        
        print(f"✓ SearchConfig created")
        print(f"  Model ID: {config.model_id}")
        print(f"  Population size: {config.population_size}")
        print(f"  Selection strategy: {config.selection_strategy}")
        return True
    except Exception as e:
        print(f"✗ Failed to create SearchConfig: {e}")
        return False


def test_program_database():
    """Test ProgramDatabase basic operations."""
    print("\nTesting ProgramDatabase...")
    
    try:
        from alphaevolve import ProgramDatabase, Program, SelectionStrategy
        
        db = ProgramDatabase(
            population_size=10,
            selection_strategy=SelectionStrategy.MAP_ELITES,
        )
        
        # Seed population
        initial_code = "def solve(x): return x * 2"
        db.seed_population(initial_code, fitness=100.0)
        
        # Add some programs
        for i in range(5):
            program = Program(code=f"def solve(x): return x * {i}", fitness=float(i * 10))
            db.add_program(program)
        
        # Select parents
        parents = db.select_parents(3)
        print(f"✓ Selected {len(parents)} parents")
        
        # Get stats
        stats = db.get_population_stats()
        print(f"✓ Population stats: {stats['population_size']} programs, best fitness: {stats['best_fitness']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed ProgramDatabase test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_sampler():
    """Test PromptSampler."""
    print("\nTesting PromptSampler...")
    
    try:
        from alphaevolve import PromptSampler, PromptStyle, Program
        
        sampler = PromptSampler(
            prompt_style=PromptStyle.STANDARD,
            use_dynamic_formatting=False,
        )
        
        # Create sample programs
        parent = Program(code="def solve(x): return x * 2", fitness=100.0)
        prior_programs = [
            Program(code="def solve(x): return x * 3", fitness=150.0),
            Program(code="def solve(x): return x * 4", fitness=120.0),
        ]
        
        # Construct prompt
        prompt = sampler.construct_prompt(
            current_program=parent,
            prior_programs=prior_programs,
            task_description="Optimize the function",
        )
        
        print(f"✓ Prompt constructed (length: {len(prompt)} chars)")
        print(f"  Prompt preview: {prompt[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Failed PromptSampler test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_task_loader():
    """Test TaskLoader."""
    print("\nTesting TaskLoader...")
    
    try:
        from alphaevolve import TaskLoader
        
        # Test with example_task.py
        loader = TaskLoader("example_task.py")
        task_spec = loader.parse()
        
        print(f"✓ Task parsed successfully")
        print(f"  Skeleton length: {len(task_spec.skeleton_code)} chars")
        print(f"  Evolve blocks: {len(task_spec.evolve_blocks)}")
        
        if task_spec.evolve_blocks:
            print(f"  First evolve block length: {len(task_spec.evolve_blocks[0])} chars")
        
        return True
    except FileNotFoundError:
        print("⚠ example_task.py not found (skipping TaskLoader test)")
        return True
    except Exception as e:
        print(f"✗ Failed TaskLoader test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numerical_evaluator():
    """Test NumericalEvaluator."""
    print("\nTesting NumericalEvaluator...")
    
    try:
        from alphaevolve import NumericalEvaluator
        import numpy as np
        
        evaluator = NumericalEvaluator(
            test_inputs=[1, 2, 3, 4, 5],
            test_targets=[2, 4, 6, 8, 10],
        )
        
        # Test with correct function
        code = """
def solve(x):
    return x * 2
"""
        fitness = evaluator.evaluate(code)
        print(f"✓ Correct function fitness: {fitness:.4f}")
        
        # Test with incorrect function
        code2 = """
def solve(x):
    return x + 1
"""
        fitness2 = evaluator.evaluate(code2)
        print(f"✓ Incorrect function fitness: {fitness2:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Failed NumericalEvaluator test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print("AlphaEvolve: Testing Implementation")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("SearchConfig", test_config()))
    results.append(("ProgramDatabase", test_program_database()))
    results.append(("PromptSampler", test_prompt_sampler()))
    results.append(("TaskLoader", test_task_loader()))
    results.append(("NumericalEvaluator", test_numerical_evaluator()))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<50} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("="*70)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
        print("\nNote: LLM-related tests (LLMEnsemble, AlphaEvolveAgent) are not included")
        print("      as they require a GPU and HuggingFace token.")
        print("="*70)
