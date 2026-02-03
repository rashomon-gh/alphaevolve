"""
LLM Ensemble module for AlphaEvolve.

Implements model tiering (fast vs strong models) and Search/Replace diff generation.
"""

import torch
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
from alphaevolve.secrets import values


class ModelTier(Enum):
    """Model tiers for the ensemble."""

    FAST = "fast"  # High-throughput, lower capability
    STRONG = "strong"  # Lower throughput, higher capability


@dataclass
class ModelConfig:
    """
    Configuration for a model in the ensemble.

    Attributes:
        model_id: HuggingFace model ID
        tier: Model tier (fast or strong)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        use_diff: Whether to use Search/Replace diff format
        dtype: Data type for model weights
    """

    model_id: str
    tier: ModelTier
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_diff: bool = True
    dtype: torch.dtype = torch.float16


class DiffParser:
    """Parser for Search/Replace diff format."""

    SEARCH_START = "<<<<<< SEARCH"
    SEPARATOR = "======="
    REPLACE_END = ">>>>>> REPLACE"

    @staticmethod
    def parse_diff(llm_response: str) -> Tuple[str, str]:
        """
        Parse a Search/Replace diff from LLM response.

        Args:
            llm_response: LLM response containing diff

        Returns:
            Tuple of (search_text, replace_text)

        Raises:
            ValueError: If diff format is invalid
        """
        # Find the diff block
        pattern = rf"{DiffParser.SEARCH_START}\s*\n(.*?)\s*\n{DiffParser.SEPARATOR}\s*\n(.*?)\s*\n{DiffParser.REPLACE_END}"
        match = re.search(pattern, llm_response, re.DOTALL)

        if not match:
            raise ValueError("Invalid diff format: Could not find SEARCH/REPLACE block")

        search_text = match.group(1).strip()
        replace_text = match.group(2).strip()

        return search_text, replace_text

    @staticmethod
    def apply_diff(original_code: str, search_text: str, replace_text: str) -> str:
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

    @staticmethod
    def apply_multiple_diffs(original_code: str, diffs: List[Tuple[str, str]]) -> str:
        """
        Apply multiple diffs to code.

        Args:
            original_code: Original code to modify
            diffs: List of (search_text, replace_text) tuples

        Returns:
            Modified code
        """
        code = original_code
        for search_text, replace_text in diffs:
            code = DiffParser.apply_diff(code, search_text, replace_text)
        return code


class LLMModel:
    """
    Wrapper for a single LLM model in the ensemble.

    Thread-safe model wrapper that loads on the first available GPU
    (respecting CUDA_VISIBLE_DEVICES).
    """

    _lock = threading.Lock()
    _initialized = set()  # Track initialized models by model_id

    def __init__(self, config: ModelConfig):
        """
        Initialize the model.

        Args:
            config: Model configuration
        """
        self.config = config

        # Use lock to prevent concurrent model loading
        with LLMModel._lock:
            if config.model_id in LLMModel._initialized:
                # Model already initialized, just create a reference
                print(f"Model {config.model_id} already initialized, creating reference...")
                # We need to still load the model - but we'll ensure it's on the same device
                pass

            print(f"Loading model: {config.model_id} ({config.tier.value})...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_id, token=values.huggingface_token.get_secret_value()
            )

            # Determine device - respect CUDA_VISIBLE_DEVICES
            if torch.cuda.is_available():
                # Get the first visible GPU (respects CUDA_VISIBLE_DEVICES)
                device = torch.device("cuda")
                print(f"Loading model on GPU (respects CUDA_VISIBLE_DEVICES): {device}")
            else:
                device = torch.device("cpu")
                print(f"No GPU available, loading on CPU: {device}")

            # Load model on specific device (not auto to avoid multi-GPU)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=config.dtype,
                token=values.huggingface_token.get_secret_value(),
            ).to(device) # type: ignore

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Store the device the model is on
            self.device = device
            print(f"Model loaded on device: {self.device}")

            # Mark as initialized
            LLMModel._initialized.add(config.model_id)

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Thread-safe generation that locks access to the model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Lock to ensure thread-safe model access
        with self._lock:
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


class LLMEnsemble:
    """
    Ensemble of LLMs with model tiering.

    Features:
    - Fast models for high-throughput exploration
    - Strong models for breakthrough mutations
    - Support for Search/Replace diff format
    - Automatic model selection based on context
    """

    def __init__(self, models: List[ModelConfig]):
        """
        Initialize the ensemble.

        Args:
            models: List of model configurations
        """
        self.models = {}

        # Load models
        for config in models:
            self.models[config.tier] = LLMModel(config)

    def select_model(
        self, generation: int, num_generations_without_improvement: int = 0
    ) -> ModelTier:
        """
        Select a model tier based on context.

        Strategy:
        - Early generations: Use fast models for exploration
        - Stuck generations: Use strong models for breakthroughs
        - Random sampling: Mix of both tiers

        Args:
            generation: Current generation number
            num_generations_without_improvement: Generations without improvement

        Returns:
            Selected model tier
        """
        # Check which tiers are available
        available_tiers = list(self.models.keys())

        # If we're stuck, try strong model
        if num_generations_without_improvement >= 3:
            if ModelTier.STRONG in available_tiers:
                return ModelTier.STRONG
            # Fall back to fast model if strong not available
            return ModelTier.FAST

        # Early exploration: mostly fast models
        if generation < 10:
            if ModelTier.FAST in available_tiers:
                return ModelTier.FAST
            # Fall back to strong if fast not available
            return ModelTier.STRONG

        # Later generations: more balanced
        if generation < 30:
            # 70% fast, 30% strong
            import random
            if random.random() < 0.7 and ModelTier.FAST in available_tiers:
                return ModelTier.FAST
            elif ModelTier.STRONG in available_tiers:
                return ModelTier.STRONG
            # Fall back to whichever is available
            return available_tiers[0]

        # Final stages: prioritize strong models
        if ModelTier.STRONG in available_tiers:
            return ModelTier.STRONG
        return ModelTier.FAST

    def mutate(
        self,
        prompt: str,
        generation: int,
        num_generations_without_improvement: int = 0,
        force_tier: Optional[ModelTier] = None,
    ) -> str:
        """
        Generate a mutation using the ensemble.

        Args:
            prompt: Input prompt
            generation: Current generation number
            num_generations_without_improvement: Generations without improvement
            force_tier: Force specific model tier

        Returns:
            Generated code or diff
        """
        # Select model
        if force_tier:
            tier = force_tier
        else:
            tier = self.select_model(generation, num_generations_without_improvement)

        # Get model
        model = self.models[tier]

        # Generate
        response = model.generate(prompt)

        return response

    def mutate_with_diff(
        self,
        original_code: str,
        prompt: str,
        generation: int,
        num_generations_without_improvement: int = 0,
        force_tier: Optional[ModelTier] = None,
    ) -> str:
        """
        Generate a mutation using Search/Replace diff format.

        Args:
            original_code: Original code to modify
            prompt: Input prompt
            generation: Current generation number
            num_generations_without_improvement: Generations without improvement
            force_tier: Force specific model tier

        Returns:
            Modified code
        """
        # Generate diff
        response = self.mutate(
            prompt, generation, num_generations_without_improvement, force_tier
        )

        try:
            # Parse and apply diff
            search_text, replace_text = DiffParser.parse_diff(response)
            new_code = DiffParser.apply_diff(original_code, search_text, replace_text)
            return new_code
        except ValueError:
            # If diff parsing fails, fall back to using full response as code
            print("Warning: Diff parsing failed, using full response as code")
            return response

    def extract_code(self, llm_response: str) -> str:
        """
        Extract Python code from LLM response.

        Handles various formats:
        - Markdown code blocks (```python)
        - Plain code
        - Mixed text and code

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

        # Check if it's a diff format
        if DiffParser.SEARCH_START in llm_response:
            # Return as-is, will be parsed as diff
            return llm_response

        # Fallback: clean up common artifacts and return as code
        clean_code = llm_response.replace("```", "").strip()

        # Remove common explanatory text
        clean_code = re.sub(
            r"^(Here is|The|Improved).*?:\s*", "", clean_code, flags=re.IGNORECASE
        )

        return clean_code

    @staticmethod
    def create_default_ensemble() -> "LLMEnsemble":
        """
        Create a default ensemble with recommended models.

        Returns:
            Configured LLMEnsemble instance
        """
        # Fast model: Gemma-3-12B or similar       
        fast_config = ModelConfig(
            model_id="google/gemma-3-12b-it",
            tier=ModelTier.FAST,
            max_tokens=256,
            temperature=0.8,
            use_diff=False,
        )

        # Strong model: Gemma-3-27B or similar
        strong_config = ModelConfig(
            model_id="google/gemma-3-27b-it",
            tier=ModelTier.STRONG,
            max_tokens=512,
            temperature=0.7,
            use_diff=True,
        )

        return LLMEnsemble([fast_config, strong_config])

    @staticmethod
    def create_single_model_ensemble(
        model_id: str = "google/gemma-2b-it",
    ) -> "LLMEnsemble":
        """
        Create an ensemble with a single model (for testing or low-resource scenarios).

        Args:
            model_id: HuggingFace model ID

        Returns:
            Configured LLMEnsemble instance
        """
        config = ModelConfig(
            model_id=model_id,
            tier=ModelTier.FAST,
            max_tokens=256,
            temperature=0.7,
            use_diff=False,
        )

        return LLMEnsemble([config])
