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


Install dependencies using uv:
```bash
uv sync
```

Create a `.env` file:
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
  --model-id gemma3
```

**VLLM** (high-throughput local inference):
```bash
uv run main.py \
  --backend openai \
  --base-url http://localhost:8000/v1 \
  --model-id gemma3
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

## Evaluator Types

AlphaEvolve supports multiple evaluator types for different problem domains:

### NumericalEvaluator

For numerical function fitting with concrete input/output pairs:

```python
from alphaevolve.search import NumericalEvaluator

evaluator = NumericalEvaluator(
    test_inputs=[1, 2, 3, 4, 5],
    test_targets=[2, 4, 6, 8, 10],
    optimization_strategy="minimize",
)
```

### SymbolicEvaluator

For symbolic mathematics problems using SymPy:

```python
from sympy import symbols, sin, cos
from alphaevolve.search import SymbolicEvaluator

x = symbols('x')
evaluator = SymbolicEvaluator(
    target_expression=sin(x)**2 + cos(x)**2,  # Target: 1
    symbols_dict={'x': x},
    complexity_weight=0.1,
    equivalence_bonus=100.0,
)
```

### SymbolicRegressionEvaluator

For discovering mathematical formulas from data points:

```python
from sympy import symbols
from alphaevolve.search import SymbolicRegressionEvaluator

x = symbols('x')
evaluator = SymbolicRegressionEvaluator(
    data_points=[(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)],
    symbols_dict={'x': x},
    parsimony_pressure=0.01,  # Penalize complex expressions
    max_complexity=20,
)
```

## Example Usages

### Numerical Task File with EVOLVE-BLOCK Markers

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

### Symbolic Task File with EVOLVE-BLOCK Markers

Create `symbolic_task.py` for symbolic mathematics problems:

```python
from sympy import symbols, sin, cos, simplify

def get_target():
    return 1  # sin²(x) + cos²(x) = 1

# EVOLVE-BLOCK-START
def solve(x):
    """Discover the trigonometric identity"""
    return sin(x)**2 + cos(x)**2
# EVOLVE-BLOCK-END

def evaluate():
    x = symbols('x')
    target = get_target()
    result_expr = solve(x)
    
    diff = simplify(result_expr - target)
    is_exact = diff == 0
    
    if is_exact:
        from sympy import count_ops
        complexity = count_ops(result_expr)
        fitness = 100.0 + 1.0 / (1.0 + complexity)
    else:
        fitness = 0.0
    
    return {"fitness": fitness}
```

Run with:

```bash
uv run main.py --task-file symbolic_task.py --use-evolve-blocks
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

## Built-in Example Tasks

AlphaEvolve includes example tasks in `alphaevolve/examples.py`:

**Numerical Tasks:**
- `logistic_function_evolve_block_task()` - Sigmoid function fitting
- `composite_function_no_block_task()` - Composite x²sin(x) + 2cos(x/2)
- `damped_sine_wave_task()` - Damped oscillation fitting
- `piecewise_function_task()` - Piecewise linear/quadratic

**Symbolic Tasks:**
- `symbolic_simplification_task()` - Find (x+1)² equivalent
- `symbolic_trig_identity_task()` - Discover sin²(x) + cos²(x) = 1
- `symbolic_derivative_task()` - Find derivative of x³sin(x)
- `symbolic_integral_task()` - Find integral expression
- `symbolic_regression_quadratic_task()` - Discover x² + 2x + 1 from data
- `symbolic_regression_trig_task()` - Discover 2sin(x) + 1 from data
- `symbolic_expression_rewrite_task()` - Rewrite sin(2x) as 2sin(x)cos(x)
- `symbolic_multi_variable_task()` - Multi-variable (x+y)²

Example task files are also available in `examples/`:
- `example_simple.py` - Basic linear function
- `example_composite.py` - Composite function
- `example_symbolic.py` - Symbolic regression
- `example_symbolic_identity.py` - Trigonometric identity discovery
