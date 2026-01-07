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

```
uv run main.py
```
