# alphaevolve

An open reimplementation of **AlphaEvolve** (Novikov et al., Google DeepMind,
[arXiv:2506.13131](https://arxiv.org/abs/2506.13131)): an evolutionary coding
agent that improves programs by sampling rich prompts from a program database,
asking an ensemble of LLMs for code diffs, applying them, scoring the results
with a user-provided `evaluate()` function in a sandbox, and registering
promising programs back into a MAP-Elites × islands database.

This project reproduces the *system* (paper §2), not DeepMind's proprietary
results; it is validated on open benchmark tasks. See `CLAUDE.md` for design
invariants and `PLAN.md` for the phased roadmap.

## Install

```bash
uv sync
```

## Test (offline, scripted fake LLM — no network)

```bash
uv run pytest -q
```

## Configure models

Edit `configs/models.yaml`. Three providers are predefined; pick which one
backs each tier (or override via env vars / per-run config):

| Provider | Endpoint | Use |
| --- | --- | --- |
| `lmstudio` | `${LMSTUDIO_BASE_URL:-http://localhost:1234/v1}` | fast tier on the local machine |
| `spike` | `${SPIKE_BASE_URL}` (+ `SPIKE_API_KEY`) | strong tier, institutional endpoints |
| `local-debug` | `${LOCAL_DEBUG_BASE_URL:-http://127.0.0.1:2026/v1}` | debugging |

Providers are probed at startup (`/v1/models`); a missing endpoint or model
fails fast with a clear message.

## Run

```bash
uv run alphaevolve run configs/binpack.yaml          # smoke task (Phase 5)
uv run alphaevolve inspect runs/<run-dir> --lineage  # best programs + lineage
uv run alphaevolve ablate configs/ablations/*.yaml   # §4 ablation suite
```

Runs are killable (Ctrl-C) and resumable:

```bash
uv run alphaevolve run configs/binpack.yaml --resume runs/<run-dir>
```

Each run directory contains the resolved `config.json`, the SQLite program
database (full lineage + prompts), JSONL logs of every LLM request and
pipeline event, and periodic best-program dumps under `best/`.

## Status

Phases 0–5 of `PLAN.md` are implemented: task/marker API, SEARCH/REPLACE diff
engine, sandboxed cascade evaluation, MAP-Elites × islands database, prompt
sampler (context, stochastic templates, meta-prompt evolution), LLM ensemble,
async pipeline, CLI, and the bin-packing smoke task. Phases 6–9 (matmul tensor
decomposition, math constructions, hardening, ablation compute) are next.
