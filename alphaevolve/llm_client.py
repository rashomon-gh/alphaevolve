"""
LLM Client - Simple wrapper for LLM interactions.

Handles LLM querying and Search/Replace diff format.
"""

import re
import torch
from typing import Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from alphaevolve.secrets import values


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_diff: bool = False


class DiffParser:
    """Parser for Search/Replace diff format."""

    SEARCH_START = "<<<<<< SEARCH"
    SEPARATOR = "======="
    REPLACE_END = ">>>>>> REPLACE"

    @staticmethod
    def parse(llm_response: str) -> Tuple[str, str]:
        """
        Parse a Search/Replace diff from LLM response.

        Args:
            llm_response: LLM response containing diff

        Returns:
            Tuple of (search_text, replace_text)

        Raises:
            ValueError: If diff format is invalid
        """
        pattern = rf"{DiffParser.SEARCH_START}\s*\n(.*?)\s*\n{DiffParser.SEPARATOR}\s*\n(.*?)\s*\n{DiffParser.REPLACE_END}"
        match = re.search(pattern, llm_response, re.DOTALL)

        if not match:
            raise ValueError("Invalid diff format: Could not find SEARCH/REPLACE block")

        search_text = match.group(1).strip()
        replace_text = match.group(2).strip()

        return search_text, replace_text

    @staticmethod
    def apply(original_code: str, search_text: str, replace_text: str) -> str:
        """
        Apply a Search/Replace diff to code.

        Args:
            original_code: Original code to modify
            search_text: Text to search for
            replace_text: Text to replace with

        Returns:
            Modified code

        Raises:
            ValueError: If search text not found
        """
        if search_text not in original_code:
            raise ValueError(f"Search text not found in code: {search_text[:50]}...")

        # Replace the first occurrence
        new_code = original_code.replace(search_text, replace_text, 1)
        return new_code


class LLMClient:
    """
    Simple LLM client wrapper.

    Handles LLM model loading and text generation.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config

        print(f"Loading model: {config.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, token=values.huggingface_token.get_secret_value()
        )

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Loading model on GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"No GPU available, loading on CPU: {self.device}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            token=values.huggingface_token.get_secret_value(),
        ).to(self.device)  # type: ignore

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on device: {self.device}")

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Format for chat models
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        response = generated_text[
            len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
        ]

        return response

    def generate_diff(self, original_code: str, prompt: str) -> str:
        """
        Generate a mutation using Search/Replace diff format.

        Args:
            original_code: Original code to modify
            prompt: Input prompt

        Returns:
            Modified code
        """
        # Generate diff
        response = self.generate(prompt)

        try:
            # Parse and apply diff
            search_text, replace_text = DiffParser.parse(response)
            new_code = DiffParser.apply(original_code, search_text, replace_text)
            return new_code
        except ValueError:
            # If diff parsing fails, fall back to using full response as code
            print("Warning: Diff parsing failed, using full response as code")
            return self._extract_code(response)

    @staticmethod
    def _extract_code(llm_response: str) -> str:
        """
        Extract Python code from LLM response.

        Args:
            llm_response: LLM response text

        Returns:
            Extracted code
        """
        # Try to find python code block
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        # Try generic code block
        match = re.search(r"```\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: clean up and return as code
        clean_code = llm_response.replace("```", "").strip()
        return clean_code
