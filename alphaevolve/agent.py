import torch
import re
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from alphaevolve.config import SearchConfig
from alphaevolve.search import Program, Evaluator
from alphaevolve.secrets import values


class AlphaEvolveAgent:
    def __init__(self, config: SearchConfig):
        self.config = config

        print(f"Loading {config.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, token=values.huggingface_token.get_secret_value()
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            # load model on GPU
            # requires accelerate
            device_map="auto",
            # TODO: make this a param
            dtype=torch.float16,
            token=values.huggingface_token.get_secret_value(),
        )
        self.evaluator = Evaluator()
        self.population: List[Program] = []
        self.best_fitness = float("-inf")
        self.generations_without_improvement = 0

    def seed_population(self, initial_code: str):
        """Initialize the database with a user-provided starting point."""
        fitness = self.evaluator.evaluate(initial_code)
        self.population.append(Program(code=initial_code, fitness=fitness))
        print(f"Seeded with fitness: {fitness}")

    def construct_prompt(self, parent: Program, inspirations: List[Program]) -> str:
        """
        Builds the 'Rich Context' prompt.
        It includes 'Prior programs' (inspirations) and the 'Current program' (parent) to mutate.
        """

        # 1. Context: Show high-performing past solutions
        prompt_content = "You are an intelligent coding assistant. Your goal is to optimize a Python function to match a hidden mathematical pattern.\n\n"

        if inspirations:
            prompt_content += "--- Prior Best Solutions ---\n"
            for p in inspirations:
                prompt_content += f"Score: {p.fitness}\nCode:\n{p.code}\n\n"

        # 2. Task: Present the parent code to modify
        prompt_content += "--- Current Code to Improve ---\n"
        prompt_content += f"{parent.code}\n\n"

        prompt_content += "--- Task ---\n"
        prompt_content += "Rewrite the 'Current Code' to improve its accuracy. "
        prompt_content += "Think about the pattern in the Prior Solutions. "
        prompt_content += "Output ONLY the full Python code for the 'solve' function. No markdown, no explanation."

        # Format for Gemma (Chat Template)
        messages = [{"role": "user", "content": prompt_content}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def extract_code(self, llm_response: str) -> str:
        """Parses the LLM output to extract executable Python code."""
        # Simple regex to find python code blocks if the model uses markdown
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1)

        # If no markdown, assume the whole response is code (fallback)
        # Cleaning up common chat artifacts
        clean_code = llm_response.replace("```", "").strip()
        return clean_code

    @torch.no_grad()
    def llm_mutate(self, parent: Program, inspirations: List[Program]) -> str:
        """
        Uses the LLM to propose a 'diff' or rewrite of the parent code.
        """
        prompt = self.construct_prompt(parent, inspirations)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,  # High temp for diversity/exploration
            do_sample=True,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output to get just the response
        response_text = generated_text[
            len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :
        ]

        return self.extract_code(response_text)

    def step(self, generation_idx):
        """Runs one iteration of the evolutionary loop."""
        print(f"\n--- Generation {generation_idx} ---")

        # Sort population by fitness (descending)
        self.population.sort(key=lambda p: p.fitness, reverse=True)

        # Keep top K as "Inspirations" for the prompt (Elitism)
        inspirations = self.population[: self.config.num_parent_context]

        new_programs = []

        # Generate offspring
        # We take the best parent and try to mutate it multiple times
        parent = self.population[0]

        for i in range(self.config.population_size):
            print(f"  > Mutating parent (Fitness: {parent.fitness})...", end="")

            try:
                # 1. LLM Mutation
                mutated_code = self.llm_mutate(parent, inspirations)

                # 2. Evaluation
                fitness = self.evaluator.evaluate(mutated_code)
                print(f" Result Fitness: {fitness}")

                # 3. Add to pool
                new_programs.append(Program(code=mutated_code, fitness=fitness))

            except Exception as e:
                print(f" Failed: {e}")

        # Update Population (Join and Select)
        self.population.extend(new_programs)
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        # Prune to fixed size, keeping only the best ones in terms of fitness
        self.population = self.population[: self.config.population_size]

        current_best_fitness = self.population[0].fitness
        print(f"Best in Gen {generation_idx}: {current_best_fitness}")

        # Check for fitness improvement and update early stopping counter
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.generations_without_improvement = 0
            print(f"New best fitness: {self.best_fitness}")
        else:
            self.generations_without_improvement += 1
            print(
                f"No improvement for {self.generations_without_improvement} generation(s)"
            )

        # Check early stopping condition
        if self.generations_without_improvement >= self.config.early_stopping_threshold:
            print(
                f"\nEarly stopping triggered: No improvement for "
                f"{self.generations_without_improvement} generations (threshold: {self.config.early_stopping_threshold})"
            )
            return False  # Signal to stop the evolutionary loop

        return True  # Continue the evolutionary loop
