"""
Task Loader module for AlphaEvolve.

Handles parsing of user-provided code with EVOLVE-BLOCK markers
and extraction of evaluation functions.
"""
import re
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskSpecification:
    """
    Represents a user-provided optimization task.
    
    Attributes:
        skeleton_code: Code outside evolve blocks (fixed scaffolding)
        evolve_blocks: List of code blocks that can be modified
        evaluate_function: User-provided function to evaluate solutions
        original_code: The complete original code
    """
    skeleton_code: str
    evolve_blocks: list[str]
    evaluate_function: Optional[Callable] = None
    original_code: str = ""


class TaskLoader:
    """
    Parses user-provided code files and extracts task specifications.
    
    Expected format:
    - Code blocks between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END
      are eligible for modification
    - Code outside these blocks is treated as static skeleton
    - Users can provide an 'evaluate' function for custom evaluation
    """
    
    EVOLVE_START_MARKER = "# EVOLVE-BLOCK-START"
    EVOLVE_END_MARKER = "# EVOLVE-BLOCK-END"
    
    def __init__(self, code_file_path: str):
        """
        Initialize the task loader with a code file.
        
        Args:
            code_file_path: Path to the Python file containing the task
        """
        self.code_file_path = Path(code_file_path)
        self._code_content = self._read_code_file()
    
    def _read_code_file(self) -> str:
        """Read the content of the code file."""
        if not self.code_file_path.exists():
            raise FileNotFoundError(f"Code file not found: {self.code_file_path}")
        
        with open(self.code_file_path, 'r') as f:
            return f.read()
    
    def parse(self) -> TaskSpecification:
        """
        Parse the code file and extract task specification.
        
        Returns:
            TaskSpecification containing skeleton, evolve blocks, and evaluate function
        """
        # Find all evolve blocks
        evolve_blocks = []
        evolve_pattern = re.compile(
            rf'{self.EVOLVE_START_MARKER}(.*?){self.EVOLVE_END_MARKER}',
            re.DOTALL
        )
        
        for match in evolve_pattern.finditer(self._code_content):
            block_content = match.group(1).strip()
            evolve_blocks.append(block_content)
        
        # Extract skeleton (code outside evolve blocks)
        skeleton = re.sub(
            rf'{self.EVOLVE_START_MARKER}.*?{self.EVOLVE_END_MARKER}\n?',
            '',
            self._code_content,
            flags=re.DOTALL
        )
        
        # Try to extract evaluate function
        evaluate_function = self._extract_evaluate_function()
        
        return TaskSpecification(
            skeleton_code=skeleton,
            evolve_blocks=evolve_blocks,
            evaluate_function=evaluate_function,
            original_code=self._code_content
        )
    
    def _extract_evaluate_function(self) -> Optional[Callable]:
        """
        Extract the evaluate function from the code if present.
        
        Returns:
            The evaluate function if found, None otherwise
        """
        try:
            local_scope = {}
            exec(self._code_content, {}, local_scope)
            
            if "evaluate" in local_scope and callable(local_scope["evaluate"]):
                return local_scope["evaluate"]
            
            return None
        except Exception:
            # If we can't extract the function, it's okay - user can provide it separately
            return None
    
    def reconstruct_code(self, evolve_blocks: list[str]) -> str:
        """
        Reconstruct the full code by inserting evolved blocks into the skeleton.
        
        Args:
            evolve_blocks: List of evolved code blocks to insert
            
        Returns:
            Complete code with evolved blocks
        """
        # Create a pattern that matches evolve blocks
        evolve_pattern = re.compile(
            rf'{self.EVOLVE_START_MARKER}.*?{self.EVOLVE_END_MARKER}',
            re.DOTALL
        )
        
        # Count how many blocks we need to replace
        num_blocks = len(evolve_pattern.findall(self._code_content))
        
        if num_blocks != len(evolve_blocks):
            raise ValueError(
                f"Number of evolve blocks mismatch: expected {num_blocks}, "
                f"got {len(evolve_blocks)}"
            )
        
        # Replace each evolve block
        result_code = self._code_content
        for i, new_block in enumerate(evolve_blocks):
            # Replace the i-th occurrence
            def replace_nth(match, index=i, content=new_block):
                # Check if this is the index-th match
                replace_nth.count = getattr(replace_nth, 'count', 0)
                replace_nth.count += 1
                if replace_nth.count == index + 1:
                    return f"{self.EVOLVE_START_MARKER}\n{content}\n{self.EVOLVE_END_MARKER}"
                return match.group(0)
            
            result_code = evolve_pattern.sub(replace_nth, result_code, count=1)
        
        return result_code
    
    @staticmethod
    def create_example_task_file(output_path: str = "example_task.py") -> None:
        """
        Create an example task file with EVOLVE-BLOCK markers.
        
        Args:
            output_path: Path where to save the example file
        """
        example_code = '''"""
Example task for AlphaEvolve.
This file contains code with EVOLVE-BLOCK markers that delineate
the code segments eligible for modification.
"""

import numpy as np

# Static helper functions (not evolved)
def load_data():
    """Load training data."""
    return np.array([1, 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10])

def normalize(x):
    """Normalize input data."""
    return (x - x.mean()) / x.std()

# EVOLVE-BLOCK-START
def solve(x):
    """
    Transform input x to produce the correct output.
    This function will be evolved by AlphaEvolve.
    """
    # Initial implementation - needs improvement
    return x * 2
# EVOLVE-BLOCK-END

def evaluate():
    """
    Evaluate the current solution.
    Returns a dictionary of scalar metrics (higher is better).
    """
    X, y = load_data()
    X_norm = normalize(X)
    
    predictions = solve(X_norm)
    
    # Calculate metrics
    mse = np.mean((predictions - y) ** 2)
    accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like score
    
    return {
        "accuracy": float(accuracy),
        "negative_mse": float(-mse),
    }

if __name__ == "__main__":
    metrics = evaluate()
    print(f"Evaluation metrics: {metrics}")
'''
        with open(output_path, 'w') as f:
            f.write(example_code)
        
        print(f"Example task file created at: {output_path}")
