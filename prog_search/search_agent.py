"""
Search Agent - Ephemeral agent for one evolutionary step.

Responsibilities:
- Selection
- Context Construction
- Delegation to Mutation and Scoring Agents
- Commit to Database
"""

from typing import List, Optional, Dict, Any
from alphaevolve.database import Database, Program
from alphaevolve.mutation_agent import MutationAgent
from alphaevolve.scoring_agent import ScoringAgent


class SearchAgent:
    """
    Ephemeral Search Agent.

    Runs for ONE evolutionary step, then terminates.
    """

    def __init__(
        self,
        database: Database,
        mutation_agent: MutationAgent,
        scoring_agent: ScoringAgent,
        task_description: str = "",
        use_diff: bool = False,
        skeleton_code: str = "",
        sample_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Search Agent.

        Args:
            database: Program database (shared state)
            mutation_agent: Mutation agent for code generation
            scoring_agent: Scoring agent for evaluation
            task_description: Description of the optimization task
            use_diff: Whether to use Search/Replace diff format
            skeleton_code: Full skeleton code including helper functions
            sample_data: Sample inputs and expected outputs for the task
        """
        self.database = database
        self.mutation_agent = mutation_agent
        self.scoring_agent = scoring_agent
        self.task_description = task_description
        self.use_diff = use_diff
        self.skeleton_code = skeleton_code
        self.sample_data = sample_data or {}

    def run(self, num_offspring: int = 1) -> List[int]:
        """
        Run one evolutionary step.

        Args:
            num_offspring: Number of offspring to generate

        Returns:
            List of program IDs added to database
        """
        # Select parent
        parent = self.database.select_parent()

        # Sample context programs
        context_programs = self.database.sample_context(3)

        # Generate offspring
        program_ids = []
        for i in range(num_offspring):
            print(f"  > Offspring {i + 1}/{num_offspring}...", end=" ")

            try:
                # Build prompt
                prompt = self._build_prompt(parent, context_programs)

                # Delegate to Mutation Agent
                child_code = self.mutation_agent.run(parent.code, prompt)

                # Delegate to Scoring Agent
                score_result = self.scoring_agent.run(child_code)

                print(f"fitness={score_result.fitness:.4f}")

                # Commit to database
                program_id = self.database.add_program(
                    code=child_code,
                    fitness=score_result.fitness,
                    metadata=score_result.metrics,
                    parent_id=parent.id,
                )
                program_ids.append(program_id)

            except Exception as e:
                print(f"FAILED: {e}")

        return program_ids

    def _build_prompt(
        self,
        parent: Program,
        context_programs: List[Program],
    ) -> str:
        """
        Build prompt with context.

        Args:
            parent: Parent program to mutate
            context_programs: Context programs for few-shot examples

        Returns:
            Constructed prompt string
        """
        prompt = "You are an expert software developer. Your task is to improve the given code.\n\n"

        # Task description
        if self.task_description:
            prompt += f"## Task\n{self.task_description}\n\n"

        # Sample data (critical for understanding the problem)
        if self.sample_data:
            prompt += "## Sample Data\n"
            prompt += "The function should transform inputs to produce the expected outputs:\n\n"
            sample_inputs = self.sample_data.get("inputs", [])[
                :5
            ]  # Show first 5 examples
            sample_outputs = self.sample_data.get("outputs", [])[:5]
            for i, (inp, out) in enumerate(zip(sample_inputs, sample_outputs)):
                prompt += f"  Input: {inp}\n"
                prompt += f"  Expected Output: {out}\n\n"

        # Helper functions context
        if self.skeleton_code:
            prompt += "## Available Helper Functions\n"
            prompt += "The following functions are available in the environment:\n\n"
            prompt += f"```python\n{self.skeleton_code}\n```\n"

        # Prior programs (few-shot examples)
        if context_programs:
            prompt += "## Prior Best Solutions\n"
            for i, program in enumerate(context_programs, 1):
                prompt += f"\n### Example {i} (Score: {program.fitness:.4f})\n"
                prompt += f"```python\n{program.code}\n```\n"

        # Current program to improve
        prompt += "\n## Current Code to Improve\n"
        prompt += f"```python\n{parent.code}\n```\n"

        # Get predictions for current code (show what it's doing wrong)
        predictions = self._get_code_predictions(parent.code)
        if predictions and self.sample_data.get("outputs"):
            prompt += "\n## Current Code Predictions vs Expected\n"
            prompt += "The current code produces these predictions:\n\n"
            sample_outputs = self.sample_data.get("outputs", [])[:5]
            for i, (pred, expected) in enumerate(zip(predictions[:5], sample_outputs)):
                prompt += f"  Input {i+1}: Predicted={pred:.4f}, Expected={expected:.4f}, Error={abs(pred-expected):.4f}\n"

        # Evaluation feedback
        if parent.metadata:
            prompt += "\n## Evaluation Feedback\n"
            prompt += f"Current Score: {parent.fitness:.4f}\n"
            for metric, value in parent.metadata.items():
                prompt += f"{metric}: {value}\n"

        # Task instruction
        prompt += "\n## Task\n"
        prompt += "Analyze the current code, the sample data, and the prior solutions. "
        prompt += "Propose an improved version that correctly transforms inputs to expected outputs. "
        prompt += "Pay attention to the error pattern in the predictions.\n"

        if self.use_diff:
            prompt += "Generate a SEARCH/REPLACE diff to improve the code.\n\n"
            prompt += "Use the following format:\n\n"
            prompt += "<<<<<< SEARCH\n"
            prompt += "# Code to replace\n"
            prompt += "=======\n"
            prompt += "# New code\n"
            prompt += ">>>>>> REPLACE\n\n"
            prompt += "Be precise with the SEARCH section - it must match exactly."
        else:
            prompt += "Output ONLY the improved `solve` function code (no explanations, no markdown formatting)."

        return prompt

    def _get_code_predictions(self, code: str) -> Optional[List[float]]:
        """
        Get predictions from code on sample data.

        Args:
            code: Code to execute

        Returns:
            List of predictions or None if execution fails
        """
        if not self.sample_data or not self.skeleton_code:
            return None

        try:
            # Reconstruct full code

            # Create a temporary file with the skeleton and code
            # This is a bit hacky but works for the EVOLVE-BLOCK format
            full_code = self.skeleton_code
            # Find the evolve block and replace it
            import re

            evolve_pattern = re.compile(
                r"# EVOLVE-BLOCK-START.*?# EVOLVE-BLOCK-END", re.DOTALL
            )
            replacement = f"# EVOLVE-BLOCK-START\n{code}\n# EVOLVE-BLOCK-END"
            full_code = evolve_pattern.sub(replacement, full_code)

            # Execute and get predictions
            namespace = {}
            exec(full_code, namespace, namespace)

            if "solve" not in namespace:
                return None

            sample_inputs = self.sample_data.get("inputs", [])[:5]
            predictions = []
            for inp in sample_inputs:
                result = namespace["solve"](inp)
                predictions.append(float(result))

            return predictions

        except Exception:
            return None
