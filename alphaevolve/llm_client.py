"""
LLM Client - Wrapper for LLM interactions supporting multiple backends.

Supports:
- HuggingFace (local model loading)
- OpenAI-compatible APIs (Ollama, VLLM, etc.)
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple
from dataclasses import dataclass

from alphaevolve.secrets import values


class BackendType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model_id: str
    backend: BackendType = BackendType.HUGGINGFACE
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    use_diff: bool = False
    base_url: str | None = None
    api_key: str | None = None


class DiffParser:
    """Parser for Search/Replace diff format."""

    SEARCH_START = "<<<<<< SEARCH"
    SEPARATOR = "======="
    REPLACE_END = ">>>>>> REPLACE"

    @staticmethod
    def parse(llm_response: str) -> Tuple[str, str]:
        pattern = rf"{DiffParser.SEARCH_START}\s*\n(.*?)\s*\n{DiffParser.SEPARATOR}\s*\n(.*?)\s*\n{DiffParser.REPLACE_END}"
        match = re.search(pattern, llm_response, re.DOTALL)

        if not match:
            raise ValueError("Invalid diff format: Could not find SEARCH/REPLACE block")

        search_text = match.group(1).strip()
        replace_text = match.group(2).strip()

        return search_text, replace_text

    @staticmethod
    def apply(original_code: str, search_text: str, replace_text: str) -> str:
        if search_text not in original_code:
            raise ValueError(f"Search text not found in code: {search_text[:50]}...")

        new_code = original_code.replace(search_text, replace_text, 1)
        return new_code


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @staticmethod
    def _extract_code(llm_response: str) -> str:
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        match = re.search(r"```\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        clean_code = llm_response.replace("```", "").strip()
        return clean_code


class HuggingFaceBackend(BaseLLMBackend):
    """Backend for HuggingFace local model inference."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not values.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN is required for HuggingFace backend")

        assert values.huggingface_token is not None
        hf_token = values.huggingface_token.get_secret_value()

        print(f"Loading model: {config.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=hf_token)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Loading model on GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            print(f"No GPU available, loading on CPU: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            token=hf_token,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on device: {self.device}")

    def generate(self, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response = generated_text[
            len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
        ]

        return response


class OpenAIBackend(BaseLLMBackend):
    """Backend for OpenAI-compatible APIs (Ollama, VLLM, etc.)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import OpenAI

        base_url = config.base_url or values.openai_base_url
        api_key = config.api_key
        if not api_key and values.openai_api_key:
            api_key = values.openai_api_key.get_secret_value()
        if not api_key:
            api_key = "ollama"

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = config.model_id

        print(
            f"Using OpenAI-compatible backend: {base_url or 'https://api.openai.com/v1'}"
        )
        print(f"Model: {config.model_id}")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        return response.choices[0].message.content or ""


class LLMClient:
    """
    LLM client wrapper supporting multiple backends.

    Supports HuggingFace (local) and OpenAI-compatible APIs (Ollama, VLLM, etc.).
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        if config.backend == BackendType.HUGGINGFACE:
            self.backend = HuggingFaceBackend(config)
        elif config.backend == BackendType.OPENAI:
            self.backend = OpenAIBackend(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    def generate(self, prompt: str) -> str:
        return self.backend.generate(prompt)

    def generate_diff(self, original_code: str, prompt: str) -> str:
        response = self.generate(prompt)

        try:
            search_text, replace_text = DiffParser.parse(response)
            new_code = DiffParser.apply(original_code, search_text, replace_text)
            return new_code
        except ValueError:
            print("Warning: Diff parsing failed, using full response as code")
            return self._extract_code(response)

    @staticmethod
    def _extract_code(llm_response: str) -> str:
        return BaseLLMBackend._extract_code(llm_response)
