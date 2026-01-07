# alphaevolve

Replicating alphaevolve from [AlphaEvolve: A coding agent for scientific and algorithmic discovery](http://arxiv.org/abs/2506.13131).

WIP.

## local dev

Get a token from huggingface hub before you run, especially if you plan to use gated models. Then put the token in a `.env` file, following the format in `.env.example`.

```bash
uv sync
source .venv/bin/activate
```

Run `main.py`

```bash
# Run with default settings
uv run main.py

# Run with custom settings
uv run main.py --model-id "google/gemma-2b-it" --population-size 10 --num-generations 50 --num-parent-context 3
```

### Command-line arguments

- `--model-id`: HuggingFace model ID to use (default: "google/gemma-2b-it")
- `--population-size`: Number of candidate programs in the population (default: 5)
- `--num-generations`: Number of generations to run the evolutionary search (default: 50)
- `--num-parent-context`: Number of best programs to include in LLM context for generation (default: 2)
