# PLAN.md — AlphaEvolve End-to-End Implementation Plan

Target paper: *AlphaEvolve: A coding agent for scientific and algorithmic discovery* (arXiv:2506.13131).
Each phase ends with a runnable milestone and tests. Phases 0–5 build the system (paper §2), phases 6–8 validate it on open tasks (paper §3), phase 9 runs the ablations (paper §4).

Paper → code map:
| Paper section | Component | Phase |
|---|---|---|
| §2.1 Task specification | `task/` (markers, spec, sandbox) | 1 |
| §2.3 Output format (diffs) | `generation/diff.py` | 1 |
| §2.4 Evaluation (cascade, parallel, LLM feedback) | `evaluation/` | 2 |
| §2.5 Evolution (MAP-Elites + islands) | `database/` | 3 |
| §2.2 Prompt sampling (context, stochastic, meta) | `prompting/` | 4 |
| §2.3 Models used (ensemble) | `generation/ensemble.py` | 4 |
| §2.6 Distributed pipeline | `pipeline/controller.py` | 5 |
| §3.1 Matmul tensor decomposition | `tasks/matmul/` | 6 |
| §3.2 Math constructions (kissing, packing) | `tasks/kissing/`, `tasks/packing/` | 7 |
| §3.3.1 Scheduling heuristic (open analog) | `tasks/binpack/` | 5 (smoke) |
| §4 Ablations | `ablations/` | 9 |

---

## Phase 0 — Repo scaffold and offline harness (½ day)

- `uv` project, `ruff`/`pyright`/`pytest` config, CI that runs offline tests only.
- Core dataclasses: `Program` (id, code, scores: dict[str, float], parent_id, prompt_id, generation, status, failure_reason), `EvalResult`, `TaskSpec`.
- SQLite persistence layer for programs + prompts + runs (schema migration via a single `schema.sql`).
- `tests/fakes/scripted_llm.py`: a fake async LLM that returns scripted diffs, so the entire loop is testable with no network.
- **Milestone:** `pytest` green; a `Program` round-trips through SQLite.

## Phase 1 — Task API, markers, and diff engine (1–2 days)

- `task/markers.py`: parse a source file into alternating skeleton / evolve-block segments (`# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`). Reassembly must be byte-exact.
- `generation/diff.py`: parse LLM output into ordered SEARCH/REPLACE blocks (`<<<<<<< SEARCH` / `=======` / `>>>>>>> REPLACE`); apply sequentially against the current program.
  - Reject: search text not found; search text found more than once; match spanning or outside evolve blocks; empty search on non-empty file.
  - Full-rewrite mode: config flag per task for short programs — LLM outputs the whole evolve block instead of diffs (paper §2.3).
- `task/sandbox.py`: run `evaluate(program_path, seed, stage) -> dict[str, float]` in a subprocess with CPU-time and wall-clock limits, memory cap, no network; capture stdout/stderr as program output artifacts (needed later for prompt rendering).
- **Milestone:** golden tests: a scripted diff transforms a toy program; every rejection case unit-tested; sandbox kills an infinite loop.

## Phase 2 — Evaluation subsystem (1–2 days)

- `evaluation/executor.py`: async pool evaluating a candidate over N seeds in parallel; aggregate to scores (e.g. best rank + fraction-of-seeds-achieving-it, matching the paper's matmul signal in §3.1).
- `evaluation/cascade.py`: ordered stages with per-stage budgets and promotion predicates ("proceed only if all earlier stages sufficiently promising", §2.4). Stage 0 is always a cheap smoke run to filter faulty programs.
- `evaluation/llm_feedback.py` (optional, off by default): grade properties like simplicity via a separate LLM call; write into scores as `llm_*` keys; support "discard if criterion unmet".
- **Milestone:** cascade correctly prunes a deliberately slow-but-bad candidate at stage 0 in tests.

## Phase 3 — Program database: MAP-Elites × islands (2 days)

- `database/map_elites.py`: user-defined feature descriptors per task (e.g. code length, runtime, diversity hash of scores) mapping programs to grid cells; cell keeps its elite per objective profile.
- `database/islands.py`: K independent populations; sampling for prompts happens within an island; periodic migration of elites between islands (interval + count configurable). This is the paper's stated combination (§2.5) — parameters are ours to tune since the paper gives no exact algorithm.
- Parent/inspiration sampling policy: sample 1 "current program" (to be improved) + M "prior programs" (diverse elites, possibly from different cells) for the prompt (§2.2, Fig. 3b).
- Multi-objective handling: archive admission compares score dicts per-cell without scalarizing; ties broken by recency then simplicity.
- **Milestone:** property tests — archive never regresses per cell; migration preserves lineage; DB resume reproduces identical archive state.

## Phase 4 — Prompt sampler + LLM ensemble (2 days)

- `prompting/sampler.py`: assemble the evolution prompt exactly in the paper's shape (Fig. 3b): system instructions → prior programs with rendered scores → current program with scores → diff-format rules → task instruction.
- `prompting/context.py`: per-task explicit context (problem statement, equations, references; support attaching text extracted from PDFs).
- `prompting/stochastic.py`: template placeholders with alternatives sampled from configured distributions (diversity knob, §2.2).
- `prompting/meta.py`: meta-prompt evolution — a second, smaller evolutionary DB of instruction snippets proposed by the LLM itself; sampled into prompts; credit-assigned by the improvement of children they produced.
- `generation/llm.py` + `ensemble.py`: async OpenAI-compatible client; fast/strong tier mixture with configurable ratio; per-tier temperature; retry-with-jitter on transport errors only (never on malformed content — that's signal); raw request/response logged to JSONL.
- Backends: LM Studio local endpoint for the fast tier; remote institutional endpoints for the strong tier (see CLAUDE.md; all in `configs/models.yaml`).
- **Milestone:** with the fake LLM, prompts snapshot-tested; with a real local model, one manual smoke generation produces a parseable diff.

## Phase 5 — Async pipeline + first end-to-end run (2–3 days)

- `pipeline/controller.py` (asyncio): the Figure-2 loop — sample prompt → generate → apply diff → evaluate (cascade) → register. Bounded queues between stages; configurable concurrency per stage; graceful shutdown + resume from DB.
- Run directory layout: config copy, DB, JSONL logs (prompts, completions, evals), periodic best-program dumps.
- CLI: `run`, `inspect` (lineage tree, best-per-cell table, score curves), `ablate`.
- **Smoke task — `tasks/binpack/`** (open analog of §3.3.1): a small trace-driven simulator of 2-resource (CPU, mem) online bin packing over synthetic workloads; evolve the priority-scoring heuristic function; score = utilization / stranded-resource recovery on held-out traces. Include the paper's published `alpha_evolve_score` heuristic (Fig. 6) as a reference baseline the system should rediscover or beat.
- **Milestone (system works end-to-end):** 30-minute local run with fast tier only strictly improves archive-best over the naive initial heuristic; run is killable and resumable.

## Phase 6 — Matrix multiplication tensor decomposition (3–5 days)

- Skeleton (fixed): tensor definition for ⟨m,n,p⟩, `evaluate()` that runs the evolved optimizer from R random seeds, rounds factor entries to nearest integer/half-integer, verifies exact reconstruction, and reports `best_rank_achieved` (negated for maximization) + `fraction_of_seeds` (§3.1).
- Evolve-block (initial program, deliberately simple, as in the paper): random init + reconstruction loss + Adam, in JAX (MLX fallback for Apple Silicon if JAX-Metal is problematic — decide in a spike).
- Cascade: stage 0 = tiny target ⟨2,2,2⟩ few steps (sanity), stage 1 = target sizes with short budget, stage 2 = full seed count/steps.
- Targets: start ⟨2,2,2⟩ (must find rank 7 = Strassen), then ⟨3,3,3⟩ (goal ≤ 23), then ⟨4,4,4⟩ (goal ≤ 49; the paper's 48 uses complex-valued factors — support `complex64` factor mode + half-integer rounding).
- **Milestone:** rank-7 ⟨2,2,2⟩ rediscovered from the simple initial program; ⟨3,3,3⟩ ≤ 23 reached. (⟨4,4,4⟩ = 48 is a stretch goal; the paper spent serious compute — document whatever we reach.)

## Phase 7 — Mathematical constructions: evolve the *search algorithm* (3–5 days)

- Implement the paper's key methodological device (§3.2): each generation evolves a **search heuristic program** that receives (a) the best known construction so far and (b) a fixed time budget (e.g. 100–1000 s), and must return an improved construction; `evaluate()` runs it and scores the returned construction.
- Tasks:
  - `tasks/kissing/`: dim-n kissing configurations — validity check (pairwise angular constraint) + count; small dims first (d=3 should hit 12) before attempting d=11 (paper: 593).
  - `tasks/packing/`: circles in a unit square maximizing sum of radii for fixed N (App. B.12) — easy exact validity check, fast objective, great for iteration.
- Persist best-construction state between generations (the "iterative refinement" chain of specialized heuristics the paper describes).
- **Milestone:** kissing d=3 → 12 rediscovered; packing task shows the early-heuristic/late-heuristic specialization in the lineage (verify via `inspect`).

## Phase 8 — Hardening + evaluation report (2 days)

- Score-curve plotting (best-so-far vs. LLM samples used and vs. wall clock) per task — the paper's Figure-8-style axes.
- Cost accounting: samples, tokens, eval-hours per run; per-tier breakdown.
- Failure-mode dashboard: malformed-diff rate, sandbox timeout rate, cascade stage-0 kill rate over time.
- README with reproduction commands for every milestone.

## Phase 9 — Ablations (§4) (2–3 days, mostly compute)

Run on the two paper-matching ablation tasks (matmul decomposition + kissing numbers), 3 seeds each, fixed sample budget; compare best-so-far curves:

1. **No evolution** — always prompt from the initial program (no DB sampling).
2. **No context in prompt** — strip explicit problem context.
3. **No meta-prompt evolution** — disable `prompting/meta.py`.
4. **No full-file evolution** — restrict evolve region to the loss function only (matmul task).
5. **Small base LLM only** — fast tier only, no strong model.

Deliverable: `ablations/report.md` with curves and a table; expected qualitative result = every component contributes (paper Fig. 8) — if one doesn't, investigate before concluding.

---

## Risks / open decisions

- **Compute reality check:** the paper's headline results used large evaluation budgets. Scope success as "system faithfully reproduces the *mechanism* and rediscovers known optima on small instances," with SOTA-matching as stretch.
- **JAX on Apple Silicon** for the matmul task: spike JAX-Metal vs. MLX vs. CPU-JAX early in Phase 6; the evolved code's framework is part of the task spec, so pick once.
- **MAP-Elites feature descriptors are unspecified in the paper** — treat them as a tunable design choice per task and log which grid was used with every run.
- **Sandboxing strength:** subprocess + rlimits first; move to containers only if evolved code starts doing surprising things.
- **LLM-feedback scores** (§2.4) are optional in the paper and off by default here — enable only after the base system is validated, so they can't mask evaluator bugs.
