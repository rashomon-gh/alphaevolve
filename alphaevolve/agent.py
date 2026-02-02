"""
AlphaEvolve Agent module.

Implements the main evolutionary loop using all components.
"""
import torch
from typing import List, Optional, Dict, Any
from alphaevolve.config import SearchConfig
from alphaevolve.program_database import ProgramDatabase, Program, SelectionStrategy
from alphaevolve.prompt_sampler import PromptSampler, PromptStyle
from alphaevolve.llm_ensemble import LLMEnsemble, ModelConfig, ModelTier
from alphaevolve.evaluation_engine import EvaluationEngine
from alphaevolve.task_loader import TaskLoader, TaskSpecification


class AlphaEvolveAgent:
    """
    Main agent for LLM-guided evolutionary coding.
    
    Integrates all components:
    - Program Database with MAP-Elites selection
    - Prompt Sampler with rich context
    - LLM Ensemble with model tiering
    - Evaluation Engine with cascading and parallelization
    """
    
    def __init__(self, config: SearchConfig):
        """
        Initialize the AlphaEvolve agent.
        
        Args:
            config: Configuration for the search
        """
        self.config = config
        
        # Initialize Program Database
        self.database = ProgramDatabase(
            population_size=config.population_size,
            selection_strategy=SelectionStrategy(config.selection_strategy.value),
            diversity_weight=config.diversity_weight,
        )
        
        # Initialize Prompt Sampler
        self.prompt_sampler = PromptSampler(
            prompt_style=PromptStyle(config.prompt_style.value),
            use_dynamic_formatting=config.use_dynamic_formatting,
            num_context_programs=config.num_context_programs,
            include_evaluation_feedback=config.include_evaluation_feedback,
        )
        
        # Initialize LLM Ensemble
        if config.use_ensemble:
            # Create ensemble with fast and strong models
            fast_config = ModelConfig(
                model_id=config.model_id,
                tier=ModelTier.FAST,
                use_diff=config.use_diff_format,
            )
            strong_config = ModelConfig(
                model_id=config.strong_model_id or "google/gemma-2-9b-it",
                tier=ModelTier.STRONG,
                use_diff=config.use_diff_format,
            )
            self.llm_ensemble = LLMEnsemble([fast_config, strong_config])
        else:
            # Single model ensemble
            single_config = ModelConfig(
                model_id=config.model_id,
                tier=ModelTier.FAST,
                use_diff=config.use_diff_format,
            )
            self.llm_ensemble = LLMEnsemble([single_config])
        
        # Initialize Task Loader if using evolve blocks
        self.task_spec: Optional[TaskSpecification] = None
        if config.use_evolve_blocks and config.task_file:
            self.task_loader = TaskLoader(config.task_file)
            self.task_spec = self.task_loader.parse()
        
        # Track generations without improvement
        self.generations_without_improvement = 0
        self.best_fitness = -float("inf")
    
    def set_evaluator(self, evaluator: Any) -> None:
        """
        Set the evaluator function.
        
        Args:
            evaluator: Evaluator function or object
        """
        # If evaluator has an evaluate method, use it
        if hasattr(evaluator, 'evaluate'):
            self.evaluator = evaluator.evaluate
        else:
            self.evaluator = evaluator
        
        # Initialize Evaluation Engine
        self.evaluation_engine = EvaluationEngine(
            base_evaluator=self.evaluator,
            use_cascaded=self.config.use_cascaded_evaluation,
            use_parallel=self.config.use_parallel_evaluation,
            max_workers=self.config.max_workers,
        )
    
    def seed_population(self, initial_code: str) -> None:
        """
        Initialize the database with a user-provided starting point.
        
        Args:
            initial_code: Initial code to seed with
        """
        fitness = self.evaluation_engine.evaluate(initial_code)
        self.database.seed_population(initial_code, fitness)
        
        print(f"Seeded population with fitness: {fitness}")
        if fitness > self.best_fitness:
            self.best_fitness = fitness
    
    def construct_prompt(
        self,
        parent: Program,
        inspirations: List[Program],
        evaluation_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct prompt with rich context.
        
        Args:
            parent: Parent program to mutate
            inspirations: High-performing programs for context
            evaluation_feedback: Evaluation results for the parent
            
        Returns:
            Constructed prompt
        """
        task_description = ""
        if self.task_spec:
            task_description = "Optimize the code within EVOLVE-BLOCK markers."
        
        if self.config.use_diff_format:
            # Use diff prompt format
            return self.prompt_sampler.construct_diff_prompt(
                current_program=parent,
                prior_programs=inspirations,
                task_description=task_description,
            )
        else:
            # Use standard prompt format
            return self.prompt_sampler.construct_prompt(
                current_program=parent,
                prior_programs=inspirations,
                task_description=task_description,
                evaluation_feedback=evaluation_feedback,
            )
    
    def llm_mutate(
        self,
        parent: Program,
        inspirations: List[Program],
        generation: int,
    ) -> str:
        """
        Use LLM to propose a mutation.
        
        Args:
            parent: Parent program to mutate
            inspirations: High-performing programs for context
            generation: Current generation number
            
        Returns:
            Mutated code
        """
        # Construct prompt
        prompt = self.construct_prompt(parent, inspirations)
        
        # Generate mutation
        if self.config.use_diff_format:
            # Use diff format
            new_code = self.llm_ensemble.mutate_with_diff(
                original_code=parent.code,
                prompt=prompt,
                generation=generation,
                num_generations_without_improvement=self.generations_without_improvement,
            )
        else:
            # Generate full code
            response = self.llm_ensemble.mutate(
                prompt=prompt,
                generation=generation,
                num_generations_without_improvement=self.generations_without_improvement,
            )
            
            # Extract code from response
            new_code = self.llm_ensemble.extract_code(response)
        
        return new_code
    
    def step(self, generation_idx: int) -> bool:
        """
        Run one iteration of the evolutionary loop.
        
        Args:
            generation_idx: Current generation index
            
        Returns:
            True to continue, False to stop
        """
        print(f"\n{'='*60}")
        print(f"Generation {generation_idx}")
        print(f"{'='*60}")
        
        # Advance generation in database
        self.database.advance_generation()
        
        # Select parents using MAP-Elites or configured strategy
        num_parents = min(self.config.num_parent_context, len(self.database.population))
        parents = self.database.select_parents(num_parents)
        
        print(f"Selected {len(parents)} parents for mutation")
        for i, parent in enumerate(parents):
            print(f"  Parent {i+1}: fitness={parent.fitness:.4f}, gen={parent.generation}")
        
        # Sample from archive for additional context
        context_programs = self.database.sample_for_context(self.config.num_context_programs)
        
        # Generate offspring
        new_programs = []
        
        # Use the best parent for mutation
        parent = parents[0]
        
        # Get evaluation feedback for parent
        evaluation_feedback = parent.metadata if parent.metadata else {}
        
        print(f"\nGenerating {self.config.population_size} offspring...")
        
        for i in range(self.config.population_size):
            print(f"  > Offspring {i+1}/{self.config.population_size}...", end=" ")
            
            try:
                # LLM Mutation
                mutated_code = self.llm_mutate(
                    parent=parent,
                    inspirations=context_programs,
                    generation=generation_idx,
                )
                
                # Evaluation
                fitness = self.evaluation_engine.evaluate(mutated_code)
                
                print(f"fitness={fitness:.4f}")
                
                # Add to new programs
                new_program = Program(code=mutated_code, fitness=fitness)
                new_programs.append(new_program)
                
            except Exception as e:
                print(f"FAILED: {e}")
        
        # Add new programs to database
        for program in new_programs:
            self.database.add_program(program)
        
        # Prune population to maintain size
        self.database.prune_population()
        
        # Get statistics
        stats = self.database.get_population_stats()
        current_best = self.database.get_best_program()
        
        print(f"\nGeneration {generation_idx} Statistics:")
        print(f"  Population size: {stats['population_size']}")
        print(f"  Best fitness: {stats['best_fitness']:.4f}")
        print(f"  Mean fitness: {stats['mean_fitness']:.4f}")
        print(f"  Std fitness: {stats['std_fitness']:.4f}")
        
        # Check for improvement
        if current_best and current_best.fitness > self.best_fitness:
            improvement = current_best.fitness - self.best_fitness
            self.best_fitness = current_best.fitness
            self.generations_without_improvement = 0
            print(f"  New best! Improvement: +{improvement:.4f}")
        else:
            self.generations_without_improvement += 1
            print(f"  No improvement for {self.generations_without_improvement} generation(s)")
        
        # Check early stopping
        if self.generations_without_improvement >= self.config.early_stopping_threshold:
            print(f"\nEarly stopping: No improvement for {self.generations_without_improvement} generations")
            return False
        
        return True
    
    def get_best_program(self) -> Optional[Program]:
        """Get the best program found."""
        return self.database.get_best_program()
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        return self.database.get_population_stats()
