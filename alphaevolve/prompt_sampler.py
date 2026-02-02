"""
Prompt Sampler module for AlphaEvolve.

Constructs rich context prompts for LLM-guided mutation.
"""
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from alphaevolve.program_database import Program


class PromptStyle(Enum):
    """Different prompt formatting styles for diversity."""
    STANDARD = "standard"
    CONCISE = "concise"
    VERBOSE = "verbose"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"


@dataclass
class PromptComponents:
    """
    Components that make up a prompt.
    
    Attributes:
        system_instructions: Role-playing instructions
        prior_programs: High-performing past solutions
        current_program: The parent program to mutate
        evaluation_feedback: Scores and outputs from evaluation
        task_description: Description of the optimization task
    """
    system_instructions: str
    prior_programs: List[Program]
    current_program: Program
    evaluation_feedback: Dict[str, Any]
    task_description: str


class PromptSampler:
    """
    Constructs prompts with rich context for LLM-guided mutation.
    
    Features:
    - Dynamic formatting with multiple prompt styles
    - System instructions for role-playing
    - Prior programs as few-shot examples
    - Evaluation feedback for guidance
    - Stochastic formatting for diversity
    """
    
    # Predefined system instructions for different roles
    SYSTEM_INSTRUCTIONS = {
        "expert_developer": (
            "You are an expert software developer with deep knowledge of algorithms "
            "and optimization. Your task is to improve the given code to maximize "
            "its performance on a specific metric."
        ),
        "scientist": (
            "You are a research scientist specializing in computational methods. "
            "Approach the problem analytically and propose improvements based on "
            "mathematical and algorithmic principles."
        ),
        "optimization_specialist": (
            "You are a specialist in program optimization and code synthesis. "
            "Focus on finding efficient and elegant solutions that maximize the "
            "evaluation metrics."
        ),
        "creative_coder": (
            "You are a creative programmer who thinks outside the box. "
            "Explore novel approaches and innovative solutions while maintaining "
            "code correctness."
        ),
    }
    
    def __init__(
        self,
        prompt_style: PromptStyle = PromptStyle.STANDARD,
        use_dynamic_formatting: bool = True,
        num_context_programs: int = 3,
        include_evaluation_feedback: bool = True,
    ):
        """
        Initialize the prompt sampler.
        
        Args:
            prompt_style: Default prompt style to use
            use_dynamic_formatting: Whether to vary prompt style stochastically
            num_context_programs: Number of prior programs to include
            include_evaluation_feedback: Whether to include evaluation feedback
        """
        self.prompt_style = prompt_style
        self.use_dynamic_formatting = use_dynamic_formatting
        self.num_context_programs = num_context_programs
        self.include_evaluation_feedback = include_evaluation_feedback
    
    def construct_prompt(
        self,
        current_program: Program,
        prior_programs: List[Program],
        task_description: str = "",
        evaluation_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct a prompt with rich context for LLM mutation.
        
        Args:
            current_program: The parent program to mutate
            prior_programs: High-performing programs to use as context
            task_description: Description of the optimization task
            evaluation_feedback: Evaluation results and metrics
            
        Returns:
            Constructed prompt string
        """
        # Select prompt style (may vary if dynamic formatting is enabled)
        style = self._select_prompt_style()
        
        # Build prompt components
        components = PromptComponents(
            system_instructions=self._get_system_instructions(),
            prior_programs=prior_programs[:self.num_context_programs],
            current_program=current_program,
            evaluation_feedback=evaluation_feedback or {},
            task_description=task_description,
        )
        
        # Construct prompt based on style
        if style == PromptStyle.STANDARD:
            return self._construct_standard_prompt(components)
        elif style == PromptStyle.CONCISE:
            return self._construct_concise_prompt(components)
        elif style == PromptStyle.VERBOSE:
            return self._construct_verbose_prompt(components)
        elif style == PromptStyle.ANALYTICAL:
            return self._construct_analytical_prompt(components)
        elif style == PromptStyle.CREATIVE:
            return self._construct_creative_prompt(components)
        else:
            return self._construct_standard_prompt(components)
    
    def _select_prompt_style(self) -> PromptStyle:
        """
        Select a prompt style, potentially with stochastic variation.
        
        Returns:
            Selected PromptStyle
        """
        if not self.use_dynamic_formatting:
            return self.prompt_style
        
        # Randomly select from available styles
        styles = list(PromptStyle)
        weights = [0.4, 0.2, 0.2, 0.1, 0.1]  # Bias towards standard
        return random.choices(styles, weights=weights)[0]
    
    def _get_system_instructions(self) -> str:
        """
        Get system instructions, potentially with variation.
        
        Returns:
            System instructions string
        """
        if self.use_dynamic_formatting:
            # Randomly select from predefined instructions
            instruction_keys = list(self.SYSTEM_INSTRUCTIONS.keys())
            selected_key = random.choice(instruction_keys)
            return self.SYSTEM_INSTRUCTIONS[selected_key]
        else:
            return self.SYSTEM_INSTRUCTIONS["expert_developer"]
    
    def _construct_standard_prompt(self, components: PromptComponents) -> str:
        """Construct a standard prompt with all components."""
        prompt = f"{components.system_instructions}\n\n"
        
        # Task description
        if components.task_description:
            prompt += f"## Task Description\n{components.task_description}\n\n"
        
        # Prior programs (few-shot examples)
        if components.prior_programs:
            prompt += "## Prior Best Solutions\n"
            for i, program in enumerate(components.prior_programs, 1):
                prompt += f"\n### Example {i} (Score: {program.fitness:.4f})\n"
                prompt += f"```python\n{program.code}\n```\n"
        
        # Current program to improve
        prompt += "\n## Current Code to Improve\n"
        prompt += f"```python\n{components.current_program.code}\n```\n"
        
        # Evaluation feedback
        if self.include_evaluation_feedback and components.evaluation_feedback:
            prompt += "\n## Evaluation Feedback\n"
            prompt += f"Current Score: {components.current_program.fitness:.4f}\n"
            for metric, value in components.evaluation_feedback.items():
                prompt += f"{metric}: {value}\n"
        
        # Task instruction
        prompt += "\n## Task\n"
        prompt += "Analyze the current code and the prior solutions. "
        prompt += "Propose an improved version that achieves a higher score. "
        prompt += "Focus on the patterns and strategies in the prior solutions. "
        prompt += "Output ONLY the improved code (no explanations, no markdown formatting)."
        
        return prompt
    
    def _construct_concise_prompt(self, components: PromptComponents) -> str:
        """Construct a concise prompt with minimal context."""
        prompt = "Improve the following code:\n\n"
        
        # Show only the best prior program
        if components.prior_programs:
            best_prior = components.prior_programs[0]
            prompt += f"Best score so far: {best_prior.fitness:.4f}\n"
            prompt += f"```python\n{best_prior.code}\n```\n\n"
        
        # Current program
        prompt += f"Current score: {components.current_program.fitness:.4f}\n"
        prompt += f"```python\n{components.current_program.code}\n```\n\n"
        
        prompt += "Output improved code only:"
        
        return prompt
    
    def _construct_verbose_prompt(self, components: PromptComponents) -> str:
        """Construct a verbose prompt with detailed explanations."""
        prompt = f"{components.system_instructions}\n\n"
        prompt += "You are working on an evolutionary programming task. "
        prompt += "Your goal is to iteratively improve code solutions through mutation.\n\n"
        
        # Detailed task description
        prompt += "## Task Context\n"
        prompt += components.task_description or "Optimize the code to maximize performance metrics.\n"
        
        # Evolutionary progress
        prompt += "\n## Evolutionary Progress\n"
        prompt += f"Current generation: {components.current_program.generation}\n"
        prompt += f"Current best score: {max(p.fitness for p in components.prior_programs + [components.current_program]):.4f}\n"
        
        # Prior programs with analysis
        if components.prior_programs:
            prompt += "\n## Analysis of Prior Solutions\n"
            for i, program in enumerate(components.prior_programs, 1):
                prompt += f"\n### Solution {i}\n"
                prompt += f"Score: {program.fitness:.4f}\n"
                prompt += f"Generation: {program.generation}\n"
                prompt += f"```python\n{program.code}\n```\n"
        
        # Current program with analysis
        prompt += "\n## Current Solution Analysis\n"
        prompt += f"Score: {components.current_program.fitness:.4f}\n"
        prompt += f"```python\n{components.current_program.code}\n```\n"
        
        # Detailed evaluation feedback
        if self.include_evaluation_feedback and components.evaluation_feedback:
            prompt += "\n## Detailed Evaluation Metrics\n"
            for metric, value in components.evaluation_feedback.items():
                prompt += f"- {metric}: {value}\n"
        
        # Detailed task
        prompt += "\n## Mutation Task\n"
        prompt += "Based on the analysis above, propose a mutation to the current solution. "
        prompt += "Consider what made the prior solutions successful. "
        prompt += "Output ONLY the mutated code (no explanations, no markdown)."
        
        return prompt
    
    def _construct_analytical_prompt(self, components: PromptComponents) -> str:
        """Construct an analytical prompt focused on reasoning."""
        prompt = f"{components.system_instructions}\n\n"
        
        # Ask for analytical approach
        prompt += "## Analytical Task\n"
        prompt += "You will analyze code improvements from an algorithmic perspective.\n\n"
        
        # Prior solutions with algorithmic analysis
        if components.prior_programs:
            prompt += "## Algorithmic Patterns in Prior Solutions\n"
            for i, program in enumerate(components.prior_programs, 1):
                prompt += f"\nSolution {i} (Score: {program.fitness:.4f}):\n"
                prompt += f"```python\n{program.code}\n```\n"
                prompt += "Key algorithmic approach to consider:\n"
        
        # Current solution
        prompt += "\n## Current Algorithm\n"
        prompt += f"```python\n{components.current_program.code}\n```\n"
        prompt += f"Current performance: {components.current_program.fitness:.4f}\n"
        
        # Analytical task
        prompt += "\n## Algorithmic Improvement Task\n"
        prompt += "Analyze the algorithmic patterns above. "
        prompt += "Propose a mathematically or algorithmically justified improvement. "
        prompt += "Output ONLY the improved code (no explanations)."
        
        return prompt
    
    def _construct_creative_prompt(self, components: PromptComponents) -> str:
        """Construct a creative prompt encouraging novel approaches."""
        prompt = f"{components.system_instructions}\n\n"
        
        # Creative framing
        prompt += "## Creative Challenge\n"
        prompt += "Explore novel and innovative approaches to solve this problem. "
        prompt += "Think outside conventional solutions.\n\n"
        
        # Prior solutions as inspiration (not examples to copy)
        if components.prior_programs:
            prompt += "## Inspirations (High-Performing Solutions)\n"
            prompt += "These solutions achieved high scores, but we want you to innovate:\n"
            for i, program in enumerate(components.prior_programs, 1):
                prompt += f"\nInspiration {i} (Score: {program.fitness:.4f}):\n"
                prompt += f"```python\n{program.code}\n```\n"
        
        # Current solution
        prompt += "\n## Starting Point\n"
        prompt += f"```python\n{components.current_program.code}\n```\n"
        
        # Creative task
        prompt += "\n## Innovation Task\n"
        prompt += "Propose a novel approach that could significantly improve performance. "
        prompt += "Consider unconventional patterns, new algorithmic ideas, or creative optimizations. "
        prompt += "Output ONLY your novel solution (no explanations)."
        
        return prompt
    
    def construct_diff_prompt(
        self,
        current_program: Program,
        prior_programs: List[Program],
        task_description: str = "",
    ) -> str:
        """
        Construct a prompt for generating Search/Replace diffs instead of full code.
        
        Args:
            current_program: The parent program to mutate
            prior_programs: High-performing programs to use as context
            task_description: Description of the optimization task
            
        Returns:
            Prompt string formatted for diff generation
        """
        prompt = f"{self._get_system_instructions()}\n\n"
        prompt += "Generate a SEARCH/REPLACE diff to improve the code.\n\n"
        
        # Show current code
        prompt += "## Current Code\n"
        prompt += f"```python\n{current_program.code}\n```\n\n"
        
        # Show prior solutions for context
        if prior_programs:
            prompt += "## Reference Solutions (for inspiration)\n"
            for program in prior_programs[:2]:
                prompt += f"Score: {program.fitness:.4f}\n"
                prompt += f"```python\n{program.code}\n```\n\n"
        
        # Explain diff format
        prompt += "## Diff Format\n"
        prompt += "Use the following format to specify changes:\n\n"
        prompt += "<<<<<< SEARCH\n"
        prompt += "# Code to replace\n"
        prompt += "=======\n"
        prompt += "# New code\n"
        prompt += ">>>>>> REPLACE\n\n"
        
        # Task
        prompt += "## Task\n"
        prompt += "Generate a SEARCH/REPLACE diff to improve the code. "
        prompt += "Be precise with the SEARCH section - it must match exactly."
        
        return prompt
