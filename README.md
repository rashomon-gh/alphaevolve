# AlphaEvolve

An LLM-guided evolutionary coding agent for scientific and algorithmic discovery, inspired by [AlphaEvolve: A coding agent for scientific and algorithmic discovery](http://arxiv.org/abs/2506.13131).

## Description

AlphaEvolve uses Large Language Models (LLMs) as mutation operators in an evolutionary algorithm to optimize code. Unlike traditional genetic algorithms with predefined mutation operators, AlphaEvolve uses LLMs to generate context-aware code modifications based on high-performing solutions.

## Installation

### Prerequisites

- `uv` to manage the python environment
- Python 3.12+
- One of the following:
  - **HuggingFace backend**: CUDA-capable GPU + HuggingFace API token
  - **OpenAI-compatible backend**: Access to Ollama, VLLM, OpenAI API, or similar

### Setup

1. Clone repository:
```bash
git clone <repository-url>
cd alphaevolve
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your credentials
```

For **HuggingFace backend**:
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

For **OpenAI-compatible backend** (Ollama, VLLM, etc.):
```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=http://localhost:11434/v1
```

## Running

### Basic Usage (HuggingFace)

Run with default settings using local HuggingFace model:

```bash
uv run main.py
```

### Using OpenAI-Compatible Backends

**Ollama** (local inference):
```bash
uv run main.py \
  --backend openai \
  --base-url http://localhost:11434/v1 \
  --model-id llama3
```

**VLLM** (high-throughput local inference):
```bash
uv run main.py \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model-id meta-llama/Llama-3-8b
```

**OpenAI API**:
```bash
uv run main.py \
  --backend openai \
  --model-id gpt-4 \
  --api-key $OPENAI_API_KEY
```

### With Task File (EVOLVE-BLOCK Markers)

Create a task file with evolvable code blocks (see `example_task.py`):

```bash
uv run main.py --task-file example_task.py --use-evolve-blocks
```

### Custom Configuration

```bash
uv run main.py \
  --backend openai \
  --base-url http://localhost:11434/v1 \
  --model-id llama3 \
  --population-size 10 \
  --num-generations 50 \
  --selection-strategy map_elites \
  --use-cascaded-evaluation
```

### Parallel Evaluation Configuration

Increase worker count for higher throughput:

```bash
uv run main.py \
  --parallel-slots 8 \
  --use-cascaded-evaluation \
  --population-size 20
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | LLM backend: `huggingface` or `openai` | `huggingface` |
| `--base-url` | Base URL for OpenAI-compatible API | from env |
| `--api-key` | API key for OpenAI-compatible API | from env |
| `--model-id` | Model ID (HF model or OpenAI model name) | `google/gemma-2b-it` |
| `--population-size` | Population size | 5 |
| `--num-generations` | Number of generations | 50 |
| `--parallel-slots` | Max parallel Search Agents | 50 |
| `--selection-strategy` | Selection strategy | `map_elites` |
| `--temperature` | LLM temperature | 0.7 |
| `--max-tokens` | Max tokens to generate | 512 |
| `--use-diff-format` | Use SEARCH/REPLACE diff format | false |
| `--task-file` | Path to task file | none |
| `--use-evolve-blocks` | Enable EVOLVE-BLOCK parsing | false |

## Development

### Running Tests

Test Python syntax of all modules:

```bash
python test_syntax.py
```

## Example Usages

### Task File with EVOLVE-BLOCK Markers

Create `my_task.py`:

```python
import numpy as np

# Static helpers (not evolved)
def load_data():
    np.random.seed(42)
    return np.linspace(0, 10, 20), np.linspace(0, 10, 20)**2

# EVOLVE-BLOCK-START
def solve(x):
    """This function will be evolved"""
    return x * 5
# EVOLVE-BLOCK-END

def evaluate():
    X, y = load_data()
    predictions = solve(X)
    mse = np.mean((predictions - y) ** 2)
    return {"accuracy": 1.0 / (1.0 + mse)}
```

Run with:

```bash
uv run main.py --task-file my_task.py --use-evolve-blocks
```

### Programmatic Usage

```python
from alphaevolve.llm_client import LLMClient, LLMConfig, BackendType
from alphaevolve.config import Config
from alphaevolve.database import Database, SelectionStrategy
from alphaevolve.orchestrator import Orchestrator

# Configure LLM client
llm_config = LLMConfig(
    model_id="llama3",
    backend=BackendType.OPENAI,
    base_url="http://localhost:11434/v1",
    max_tokens=512,
    temperature=0.7,
)

# Create database and evaluator
database = Database(
    population_size=10,
    selection_strategy=SelectionStrategy.MAP_ELITES,
)

def evaluator(code: str) -> float:
    # Your evaluation logic here
    namespace = {}
    exec(code, namespace)
    return namespace.get("fitness", 0.0)

# Initialize orchestrator
orchestrator = Orchestrator(
    config=llm_config,
    database=database,
    evaluator=evaluator,
    task_description="Optimize the function",
    parallel_slots=10,
)

# Seed initial population
database.seed("def solve(x): return x * 2", 0.5)

# Run evolutionary search
stats = orchestrator.run(
    num_generations=50,
    population_size=10,
    early_stopping_threshold=5,
)

# Get best solution
best = orchestrator.get_best_program()
print(best.code)
```
