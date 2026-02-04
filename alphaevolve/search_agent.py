"""
Search Agent - Ephemeral agent for one evolutionary step.

Responsibilities:
- Selection
- Context Construction
- Delegation to Mutation and Scoring Agents
- Commit to Database
"""

from typing import List, Optional, Callable
from alphaevolve.database import Database, Program
from alphaevolve.mutation_agent import MutationAgent
from alphaevolve.scoring_agent import ScoringAgent, ScoreResult


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
    ):
        """
        Initialize Search Agent.

        Args:
            database: Program database (shared state)
            mutation_agent: Mutation agent for code generation
            scoring_agent: Scoring agent for evaluation
            task_description: Description of the optimization task
            use_diff: Whether to use Search/Replace diff format
        """
        self.database = database
        self.mutation_agent = mutation_agent
        self.scoring_agent = scoring_agent
        self.task_description = task_description
        self.use_diff = use_diff

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

        # Prior programs (few-shot examples)
        if context_programs:
            prompt += "## Prior Best Solutions\n"
            for i, program in enumerate(context_programs, 1):
                prompt += f"\n### Example {i} (Score: {program.fitness:.4f})\n"
                prompt += f"```python\n{program.code}\n```\n"

        # Current program to improve
        prompt += "\n## Current Code to Improve\n"
        prompt += f"```python\n{parent.code}\n```\n"

        # Evaluation feedback
        if parent.metadata:
            prompt += "\n## Evaluation Feedback\n"
            prompt += f"Current Score: {parent.fitness:.4f}\n"
            for metric, value in parent.metadata.items():
                prompt += f"{metric}: {value}\n"

        # Task instruction
        prompt += "\n## Task\n"
        prompt += "Analyze the current code and the prior solutions. "
        prompt += "Propose an improved version that achieves a higher score. "
        prompt += "Focus on the patterns and strategies in the prior solutions. "

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
            prompt += "Output ONLY the improved code (no explanations, no markdown formatting)."

        return prompt
