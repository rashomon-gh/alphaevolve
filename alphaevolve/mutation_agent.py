"""
Mutation Agent - Ephemeral agent for code mutation.

Responsibilities:
- LLM Querying
- Diff Generation
- Patching
- Syntax Check
"""

import ast
from typing import Optional
from alphaevolve.llm_client import LLMClient, DiffParser
from alphaevolve.program_validator import ProgramValidator


class MutationAgent:
    """
    Ephemeral Mutation Agent.

    Spawned by Search Agent, generates mutated code using LLM,
    then terminates.
    """

    def __init__(self, llm_client: LLMClient, use_diff: bool = False):
        """
        Initialize Mutation Agent.

        Args:
            llm_client: LLM client for generation
            use_diff: Whether to use Search/Replace diff format
        """
        self.llm_client = llm_client
        self.use_diff = use_diff
        self.validator = ProgramValidator()

    def run(self, parent_code: str, prompt: str, max_retries: int = 3) -> str:
        """
        Generate a mutation.

        Args:
            parent_code: Parent code to mutate
            prompt: Prompt for LLM
            max_retries: Maximum retries on syntax errors

        Returns:
            Mutated code

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                if self.use_diff:
                    # Generate diff and apply
                    mutated_code = self.llm_client.generate_diff(parent_code, prompt)
                else:
                    # Generate full code
                    response = self.llm_client.generate(prompt)
                    mutated_code = self._extract_code(response)

                # Check syntax
                is_valid, error_msg = self.validator.validate_syntax(mutated_code)

                if is_valid:
                    return mutated_code
                else:
                    print(f"  Syntax error (attempt {attempt + 1}): {error_msg}")
                    # Retry with error feedback in prompt
                    prompt = f"{prompt}\n\nPrevious attempt had syntax error: {error_msg}\nPlease fix the syntax."

            except Exception as e:
                print(f"  Mutation error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    # Retry with error feedback
                    prompt = f"{prompt}\n\nPrevious attempt failed with error: {e}\nPlease try again."

        raise RuntimeError(f"Failed to generate valid code after {max_retries} attempts")

    @staticmethod
    def _extract_code(llm_response: str) -> str:
        """
        Extract Python code from LLM response.

        Args:
            llm_response: LLM response text

        Returns:
            Extracted code
        """
        import re

        # Try to find python code block
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        # Try generic code block
        match = re.search(r"```\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: return as-is
        return llm_response.strip()
