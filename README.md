# AlphaEvolve

An LLM-guided evolutionary coding agent for scientific and algorithmic discovery, inspired by [AlphaEvolve: A coding agent for scientific and algorithmic discovery](http://arxiv.org/abs/2506.13131).

## Description

AlphaEvolve uses Large Language Models (LLMs) as mutation operators in an evolutionary algorithm to optimize code. Unlike traditional genetic algorithms with predefined mutation operators, AlphaEvolve uses LLMs to generate context-aware code modifications based on high-performing solutions.

## Installation

### Prerequisites

- `uv` to manage the python environment
- Python 3.12+
- CUDA-capable GPU (for LLM inference)
- HuggingFace API token (for accessing models)

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

3. Create a `.env` file with your HuggingFace token:
```bash
cp .env.example .env
# Edit .env and add your HUGGINGFACE_TOKEN
```

## Running

### Basic Usage

Run with default settings:

```bash
uv run main.py
```

### With Task File (EVOLVE-BLOCK Markers)

Create a task file with evolvable code blocks (see `example_task.py`):

```bash
uv run main.py --task-file example_task.py --use-evolve-blocks
```

### Custom Configuration

```bash
uv run main.py \
  --model-id "google/gemma-2b-it" \
  --population-size 10 \
  --num-generations 50 \
  --num-parent-context 3 \
  --selection-strategy map_elites \
  --prompt-style analytical
```

### Using LLM Ensemble

Enable fast and strong model ensemble:

```bash
uv run main.py \
  --use-ensemble \
  --strong-model-id "google/gemma-2-9b-it" \
  --use-diff-format
```

## Development

### Running Tests

Test Python syntax of all modules:

```bash
python test_syntax.py
```

## Example Usages

### Simple Numerical Optimization

```python
from alphaevolve import AlphaEvolveAgent, SearchConfig, NumericalEvaluator
import numpy as np

# Create evaluator for y = x^2 pattern
evaluator = NumericalEvaluator(
    test_inputs=list(np.linspace(0, 10, 20)),
    test_targets=list(np.linspace(0, 10, 20)**2),
)

# Configure agent
config = SearchConfig(
    model_id="google/gemma-2b-it",
    population_size=10,
    num_generations=50,
    num_parent_context=3,
)

# Initialize and run
agent = AlphaEvolveAgent(config)
agent.set_evaluator(evaluator)
agent.seed_population("def solve(x): return x * 2")

for gen in range(1, config.num_generations + 1):
    if not agent.step(gen):
        break

# Get best solution
best = agent.get_best_program()
print(best.code)  # Should approximate x^2
```

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

### Advanced Configuration with Ensemble

```python
from alphaevolve import AlphaEvolveAgent, SearchConfig, ProgramDatabase, SelectionStrategy

config = SearchConfig(
    model_id="google/gemma-2b-it",
    population_size=20,
    num_generations=100,
    selection_strategy=SelectionStrategy.MAP_ELITES,
    diversity_weight=0.3,
    use_ensemble=True,
    strong_model_id="google/gemma-2-9b-it",
    use_diff_format=True,
    use_cascaded_evaluation=True,
    use_parallel_evaluation=True,
    max_workers=4,
)

agent = AlphaEvolveAgent(config)
# ... set evaluator and run
```
