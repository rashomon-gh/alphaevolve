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

Edit `configs/models.yaml`. All tiers run on the spike.tue.nl gateway
(`SPIKE_GATEWAY` + `SPIKE_API_KEY` env vars); swap any tier's model via env
var or a per-run `models:` override:

| Tier | Default model | Weight | Override |
| --- | --- | --- | --- |
| `fast` | `Gemma-4-31B-IT-NVFP4` | 4 | `FAST_MODEL` |
| `strong` | `DeepSeek-V4-Flash-0731` | 1 | `STRONG_MODEL` |
| `debug` | `gemma-4-12B-it` | — | `DEBUG_MODEL`, or `ensemble: [debug]` |

Providers are probed at startup (`/v1/models`); a missing endpoint or model
fails fast with a clear message. Gateway models scale to zero — the first
request may wait out a multi-minute cold start. Per-tier `max_tokens` values
are the probed server maximums; reasoning models need ≥16k output budget or
completions truncate mid-thought (visible as a 100% malformed-diff rate).

## Run

```bash
uv run alphaevolve run configs/binpack.yaml          # smoke task (Phase 5)
uv run alphaevolve run configs/matmul_222.yaml       # ⟨2,2,2⟩ → rediscover Strassen (Phase 6)
uv run alphaevolve run configs/matmul_444.yaml       # ⟨4,4,4⟩ stretch goal (rank ≤ 49)
uv run alphaevolve run configs/kissing_3.yaml        # kissing d=3 → rediscover 12 (Phase 7)
uv run alphaevolve run configs/packing_26.yaml       # circle packing n=26 (Phase 7)
uv run alphaevolve inspect runs/<run-dir> --lineage  # best programs + lineage
uv run alphaevolve report runs/<run-dir>             # curves, cost, failure modes (Phase 8)
uv run alphaevolve ablate configs/ablations/matmul_222_*.yaml   # §4 ablations (Phase 9)
uv run alphaevolve compare runs/<a> runs/<b> --out ablations/report.md
```

The construction tasks (kissing, packing) evolve a *search heuristic* that
receives the best known construction and a time budget (paper §3.2); the best
construction persists in the run's `state/` directory across generations.

Runs are killable (Ctrl-C) and resumable:

```bash
uv run alphaevolve run configs/binpack.yaml --resume runs/<run-dir>
```

Each run directory contains the resolved `config.json`, the SQLite program
database (full lineage + prompts), JSONL logs of every LLM request and
pipeline event, and periodic best-program dumps under `best/`.


