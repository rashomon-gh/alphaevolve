# CLAUDE.md — AlphaEvolve Reimplementation

## What this project is

An open reimplementation of **AlphaEvolve** (Novikov et al., Google DeepMind, arXiv:2506.13131): an evolutionary coding agent that improves programs by (1) sampling rich prompts from a program database, (2) asking an ensemble of LLMs to propose code diffs, (3) applying the diffs, (4) scoring the resulting programs with a user-provided `evaluate()` function, and (5) registering promising programs back into the database. The loop runs asynchronously and optimizes for throughput of evaluated ideas, not per-sample latency.

We are NOT reproducing DeepMind's proprietary results (Borg, TPU Verilog, XLA IR). We reproduce the **system** faithfully and validate it on open benchmark tasks: matrix-multiplication tensor decomposition, kissing numbers / packing problems, and 2D vector bin-packing heuristics.

Paper reference: https://arxiv.org/abs/2506.13131 — always consult PLAN.md for the phased roadmap and the section mapping before starting work.

## Architecture (mirrors paper Sections 2.1–2.6)

```
alphaevolve/
├── task/            # Task spec API (§2.1)
│   ├── markers.py       # EVOLVE-BLOCK-START/END parsing; skeleton vs evolvable regions
│   ├── spec.py          # TaskSpec: initial program, evaluate entrypoint, cascade config
│   └── sandbox.py       # Subprocess execution of evaluate() with timeouts + process-group
│                        #   kill; passes $AE_TASK_PARAMS (task sizes/budgets) and
│                        #   $AE_STATE_DIR (persistent cross-generation state, §3.2)
├── prompting/       # Prompt sampler (§2.2)
│   ├── sampler.py       # Assembles: prior programs + scores, current program, instructions
│   ├── context.py       # Explicit context (problem text, equations, literature snippets)
│   ├── stochastic.py    # Template placeholders with configured probability distributions
│   └── meta.py          # Meta-prompt evolution (co-evolved prompt DB)
├── generation/      # Creative generation (§2.3)
│   ├── llm.py           # Provider-agnostic async client (OpenAI-compatible)
│   ├── ensemble.py      # Fast-model / strong-model mixture with sampling ratios
│   └── diff.py          # SEARCH/REPLACE block parsing + application; full-rewrite mode
├── evaluation/      # Evaluation (§2.4)
│   ├── executor.py      # Async evaluation pool; parallel seeds
│   ├── cascade.py       # Staged evaluation with promotion thresholds
│   └── llm_feedback.py  # Optional LLM-graded auxiliary scores (e.g. simplicity)
├── database/        # Evolution (§2.5)
│   ├── programs.py      # Program record: code, scores dict, parent id, prompt id, gen
│   ├── map_elites.py    # Feature-grid elite archive
│   └── islands.py       # Island populations with periodic migration
├── pipeline/        # Distributed pipeline (§2.6)
│   └── controller.py    # asyncio orchestrator: sample → generate → apply → evaluate → register
├── tasks/           # Benchmark tasks (validation targets)
│   ├── matmul/          # Tensor decomposition for ⟨m,n,p⟩ (§3.1); exact half-integer
│   │                    #   verification (real + complex); *_lossonly.py = restricted
│   │                    #   evolve region for the function-only ablation
│   ├── kissing/         # Kissing number lower bounds (§3.2); evolves the search heuristic
│   ├── packing/         # Circle packing, max sum of radii (§3.2, App. B.12)
│   └── binpack/         # 2D vector bin-packing heuristic + simulator (§3.3.1 analog)
├── reporting.py     # Phase 8/9: run reports (curves, cost, failure modes) + `compare`
├── config.py        # YAML loading, ${ENV:-default} interpolation, `base:` inheritance
├── cli.py           # run / inspect / report / compare / ablate
└── configs/         # YAML per-task run configs; configs/ablations/ = §4 harness
                     #   (no-evolution, no-context, no-meta, fast-only, loss-only)
```

## Non-negotiable design invariants

1. **Ground truth is execution, never the LLM.** A program's scores come only from running `evaluate()` in the sandbox. LLM-generated feedback scores are auxiliary and must be clearly namespaced (`llm_*`) in the scores dict.
2. **Scores are `dict[str, float]`, maximized by convention.** Multi-objective throughout — never collapse to a scalar inside the database; selection/archiving logic decides how to combine.
3. **Skeleton code is immutable.** Only text inside `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` may be modified. Diff application must reject any SEARCH/REPLACE whose match lies outside an evolve block, and reject ambiguous (0 or ≥2) matches — same semantics as the paper's diff format.
4. **Failed candidates are data.** Programs that crash, time out, or fail cascade stage 0 are recorded with a failure reason (they inform meta-prompting), but are never parents.
5. **Everything is resumable.** The program database persists to SQLite; a killed run restarts from the DB without losing lineage. Every program stores its parent id and the prompt that produced it (full provenance).
6. **Throughput over latency.** All LLM calls and evaluations are async and concurrent; the controller must never serialize on a single slow evaluation. Backpressure via bounded queues.
7. **Determinism where possible.** Evaluators take explicit seeds; a config + DB snapshot + seed must reproduce selection decisions (LLM outputs are the only nondeterminism, and raw completions are logged).

## LLM backends

All model access goes through the OpenAI-compatible async client in `generation/llm.py`. Endpoints are configured in `configs/models.yaml`, never hardcoded. All tiers currently run on the institutional spike.tue.nl gateway (`SPIKE_GATEWAY` + `SPIKE_API_KEY` env vars), mirroring the paper's Flash/Pro ensemble:

- **Fast tier** (high sample rate, weight 4): `Gemma-4-31B-IT-NVFP4` (override with `FAST_MODEL`).
- **Strong tier** (occasional, higher quality, weight 1): `DeepSeek-V4-Flash-0731` (override with `STRONG_MODEL`).
- **Debug tier**: `gemma-4-12B-it` — cheap and fast; select with `ensemble: [debug]` in a run config or `DEBUG_MODEL`.

Gateway models scale to zero: the first request may sit through a cold start of several minutes (provider `timeout_s` accommodates this). Per-tier `max_tokens` is set to probed server maximums — gemma-4-12B has a hard 16384-token TOTAL context; reasoning models (Qwen3-class) need ≥16k output budget or every completion truncates mid-thought and shows up as a 100% malformed-diff rate.

Never assume a specific model is available; probe `/v1/models` at startup and fail with a clear message. All prompts and raw completions are logged to the run directory (JSONL) for debugging and for the ablation analysis.

## Coding conventions

- Python 3.12+, `uv` for env/deps, `ruff` + `pyright` clean before commit.
- Fully typed public APIs; dataclasses (frozen where possible) for records like `Program`, `EvalResult`, `PromptBundle`.
- `asyncio` end-to-end in the pipeline; no threads except inside the sandboxed evaluator subprocesses.
- Numerical work in tasks uses NumPy (CPU — chosen over JAX for Apple Silicon portability; the einsum sizes in scope don't need an accelerator); for matmul decomposition, exactness check = round entries to nearest integer/half-integer (real or complex) and verify the tensor equation exactly in integer arithmetic (paper §3.1).
- Tests: `pytest`, with a fake deterministic LLM (`tests/fakes/scripted_llm.py`) so the whole loop is testable offline. Every diff-application edge case gets a unit test.
- Keep modules small; the controller should read as a transcription of Figure 2 of the paper.

## Commands

```bash
uv sync                                   # install
uv run pytest -q                          # tests (offline, fake LLM)
uv run alphaevolve run configs/binpack.yaml         # smoke task (fast eval, good first target)
uv run alphaevolve run configs/matmul_222.yaml      # ⟨2,2,2⟩ → rediscover Strassen rank 7
uv run alphaevolve run configs/matmul_444.yaml      # ⟨4,4,4⟩ tensor decomposition (stretch)
uv run alphaevolve run configs/kissing_3.yaml       # kissing d=3 → rediscover 12
uv run alphaevolve run configs/packing_26.yaml      # circle packing n=26 (paper ref 2.635)
uv run alphaevolve run <config> --resume runs/<id>  # resume a killed run
uv run alphaevolve inspect runs/<id> --lineage      # lineage browser / best-program dump
uv run alphaevolve report runs/<id>                 # curves, cost accounting, failure modes
uv run alphaevolve ablate configs/ablations/matmul_222_*.yaml   # §4 ablation suite
uv run alphaevolve compare runs/<a> runs/<b> --out ablations/report.md
```

## What "done" means for a change

- The evolutionary loop on the bin-packing smoke task improves best score monotonically-in-archive over a 30-minute local run with the fast model only.
- Unit tests cover: marker parsing, diff apply/reject, cascade promotion, MAP-Elites cell assignment, island migration, DB resume.
- No API keys, endpoints, or absolute local paths committed; config via YAML + env vars only.

## Things Claude should NOT do

- Don't let the LLM rewrite the evaluator, the skeleton, or the task spec — only evolve-blocks evolve.
- Don't "helpfully" score programs with heuristics when an evaluation fails — record the failure.
- Don't merge multi-objective scores into one number anywhere upstream of selection.
- Don't add per-sample retries that hide malformed diffs; malformed output is a measurable signal (log and count it).
- Don't start a long evolutionary run inside CI or tests; anything model-dependent is behind an explicit CLI command.
