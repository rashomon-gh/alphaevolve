"""Asyncio orchestrator (paper §2.6, Figure 2).

The loop: sample prompt → generate diff → apply → evaluate (cascade) →
register. Stages are decoupled by bounded queues so no single slow evaluation
serializes the pipeline (CLAUDE.md invariant 6); everything lands in SQLite
so a killed run resumes without losing lineage (invariant 5).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from alphaevolve.database.islands import IslandManager
from alphaevolve.database.programs import (
    EVALUATED,
    FAILED,
    Program,
    ProgramDB,
    PromptRecord,
    new_id,
)
from alphaevolve.evaluation.cascade import run_cascade
from alphaevolve.evaluation.llm_feedback import grade
from alphaevolve.generation.diff import (
    DiffError,
    apply_diff,
    apply_full_rewrite,
    extract_code_block,
    parse_diff,
)
from alphaevolve.generation.ensemble import TRANSPORT_ERRORS, Ensemble
from alphaevolve.generation.llm import Completion, LLMClient, ModelRegistry
from alphaevolve.prompting.context import TaskContext
from alphaevolve.prompting.meta import MetaPromptDB
from alphaevolve.prompting.sampler import PromptBundle, build_prompt
from alphaevolve.prompting.stochastic import StochasticSlot
from alphaevolve.task.markers import MarkerError
from alphaevolve.task.spec import TaskSpec


class Generator(Protocol):
    """What the controller needs from an LLM ensemble (real or scripted)."""

    async def generate(
        self, prompt: str, *, system: str | None = None, purpose: str = "evolve"
    ) -> Completion: ...


@dataclass
class RunStats:
    samples_issued: int = 0
    meta_proposals: int = 0
    malformed_diffs: int = 0
    transport_failures: int = 0
    eval_failures: int = 0
    registered: int = 0
    admitted: int = 0
    worker_errors: int = 0


@dataclass(frozen=True)
class _GenJob:
    kind: str  # "evolve" | "meta"
    bundle: PromptBundle | None = None
    parent: Program | None = None


@dataclass(frozen=True)
class _EvalJob:
    code: str
    parent: Program
    bundle: PromptBundle
    completion: Completion


@dataclass
class Controller:
    config: dict[str, Any]
    config_dir: Path
    run_dir: Path
    ensemble: Generator | None = None  # injected in tests; built from configs otherwise
    stats: RunStats = field(default_factory=RunStats)

    def __post_init__(self) -> None:
        cfg = self.config
        self.spec = TaskSpec.from_config(cfg["task"], self.config_dir)
        self.context = TaskContext.from_config(
            cfg["task"].get("context_cfg", cfg["task"]), self.config_dir
        )

        evo = cfg.get("evolution", {})
        self.seed = int(evo.get("seed", 0))
        self.num_islands = int(evo.get("islands", 1))
        self.num_inspirations = int(evo.get("inspirations", 2))
        self.islands = IslandManager(
            self.num_islands,
            self.spec.features,
            migration_interval=int(evo.get("migration_interval", 50)),
            migration_count=int(evo.get("migration_count", 2)),
            migration_seed=self.seed,
        )
        self.sample_rng = random.Random(self.seed + 1)

        prompting = cfg.get("prompting", {})
        self.slots = [StochasticSlot.from_config(s) for s in prompting.get("slots", [])]
        meta_cfg = prompting.get("meta", {})
        self.meta_enabled = bool(meta_cfg.get("enabled", False))
        self.meta_propose_every = int(meta_cfg.get("propose_every", 0))
        self.num_meta_snippets = int(meta_cfg.get("num_snippets", 2))
        self.meta_db = (
            MetaPromptDB.from_seed_texts([str(t) for t in meta_cfg.get("seeds", [])])
            if self.meta_enabled
            else None
        )

        ablation = cfg.get("ablation", {})
        self.no_evolution = bool(ablation.get("no_evolution", False))
        self.no_context = bool(ablation.get("no_context", False))
        if bool(ablation.get("no_meta", False)):
            self.meta_enabled = False
            self.meta_db = None

        limits = cfg.get("limits", {})
        self.max_samples = int(limits.get("max_samples", 100))
        self.max_seconds = float(limits.get("max_seconds", 3600))
        self.n_generators = int(limits.get("concurrent_generations", 2))
        self.n_evaluators = int(limits.get("concurrent_evaluations", 2))
        self.queue_size = int(limits.get("queue_size", 8))
        self.max_worker_errors = int(limits.get("max_worker_errors", 20))

        feedback_cfg = cfg.get("llm_feedback", {})
        self.feedback_criteria: list[str] = (
            [str(c) for c in feedback_cfg.get("criteria", [])]
            if bool(feedback_cfg.get("enabled", False))
            else []
        )
        self._active_generator: Generator | None = None

        self.objective = str(cfg.get("objective", ""))
        self.dump_every = int(cfg.get("dump_every", 10))

        self.db = ProgramDB(self.run_dir / "programs.sqlite")
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "best").mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "logs" / "events.jsonl"
        self._meta_path = self.run_dir / "meta_prompts.json"
        self._state_dir = self.run_dir / "state" if self.spec.uses_state else None
        self._stop = asyncio.Event()
        self._genesis: Program | None = None

    # -- setup ---------------------------------------------------------------

    def build_ensemble(self) -> Ensemble:
        models_path = self.config_dir / self.config.get("models_config", "models.yaml")
        from alphaevolve.config import deep_merge, load_yaml

        models_cfg = load_yaml(models_path)
        if "models" in self.config:  # per-run tier overrides
            models_cfg = deep_merge(models_cfg, self.config["models"])
        registry = ModelRegistry.from_config(models_cfg)
        tier_names = self.config.get("ensemble", list(registry.tiers))
        tiers = [registry.tiers[name] for name in tier_names]
        clients = {
            t.provider: LLMClient(
                registry.providers[t.provider], log_path=self.run_dir / "logs" / "llm.jsonl"
            )
            for t in tiers
        }
        return Ensemble(tiers, clients, rng=random.Random(self.seed + 2))

    async def preflight(self, ensemble: Ensemble) -> None:
        """Probe every provider; fail fast if an endpoint or model is missing."""
        for tier in ensemble.tiers:
            served = await ensemble.clients[tier.provider].probe()
            if tier.model not in served:
                raise RuntimeError(
                    f"tier '{tier.name}': model '{tier.model}' not served by provider "
                    f"'{tier.provider}'. Available: {', '.join(served) or '(none)'}"
                )

    async def _ensure_genesis(self, run_id: str) -> None:
        """Evaluate + register the initial program (generation 0), or replay
        the database on resume. Generation-0 elites seed every island."""
        evaluated = list(self.db.iter_programs(run_id, status=EVALUATED))
        if evaluated:
            genesis = next((p for p in evaluated if p.generation == 0), None)
            if genesis is None:
                raise RuntimeError(
                    "cannot resume: database has evaluated programs but no "
                    "generation-0 program (corrupt or foreign run directory?)"
                )
            self._genesis = genesis
            for program in evaluated:
                self._register_in_islands(program)
            self._log_event("resumed", programs=len(evaluated))
            return
        result = await run_cascade(
            self.spec, self.spec.initial_program, base_seed=self.seed, state_dir=self._state_dir
        )
        if not result.ok:
            raise RuntimeError(
                f"initial program failed evaluation: {result.failure_reason}\n"
                f"stderr: {result.artifacts.get('stderr', '')[-2000:]}"
            )
        genesis = Program(
            id=new_id(),
            code=self.spec.initial_program,
            scores=result.scores,
            parent_id=None,
            prompt_id=None,
            generation=0,
            island=0,
            artifacts=result.artifacts,
        )
        self.db.add_program(run_id, genesis)
        self._genesis = genesis
        self._register_in_islands(genesis)
        self._log_event("genesis", scores=result.scores)

    def _register_in_islands(self, program: Program) -> None:
        if program.generation == 0:
            for i in range(self.num_islands):
                self.islands.islands[i].admit(program.with_island(i))
        else:
            self.islands.register(program)

    # -- pipeline stages -----------------------------------------------------

    async def _sampler(self, run_id: str, prompt_q: asyncio.Queue, deadline: float) -> None:
        assert self._genesis is not None
        while (
            not self._stop.is_set()
            and self.stats.samples_issued < self.max_samples
            and time.monotonic() < deadline
        ):
            if (
                self.meta_enabled
                and self.meta_propose_every > 0
                and self.stats.samples_issued > 0
                and self.stats.samples_issued % self.meta_propose_every == 0
                and self.stats.meta_proposals < self.stats.samples_issued // self.meta_propose_every
            ):
                self.stats.meta_proposals += 1
                await prompt_q.put(_GenJob(kind="meta"))

            if self.no_evolution:
                parent, inspirations = self._genesis, []
            else:
                sample = self.islands.sample(self.num_inspirations, self.sample_rng)
                if sample is None:
                    await asyncio.sleep(0.1)
                    continue
                parent, inspirations = sample.parent, sample.inspirations
            bundle = build_prompt(
                parent,
                inspirations,
                context=self.context,
                full_rewrite=self.spec.full_rewrite,
                slots=self.slots,
                meta_db=self.meta_db,
                num_meta_snippets=self.num_meta_snippets,
                rng=self.sample_rng,
                include_context=not self.no_context,
            )
            self.db.add_prompt(
                run_id, PromptRecord(id=bundle.id, text=bundle.text, meta=bundle.meta)
            )
            self.stats.samples_issued += 1
            await prompt_q.put(_GenJob(kind="evolve", bundle=bundle, parent=parent))

    async def _generator(
        self, run_id: str, generator: Generator, prompt_q: asyncio.Queue, eval_q: asyncio.Queue
    ) -> None:
        # Workers must never die: an unhandled exception in a worker would
        # leave the bounded queues undrained and deadlock the sampler. Any
        # unexpected error is recorded loudly and the loop continues (a
        # repeated-error safety valve stops the whole run).
        while (job := await prompt_q.get()) is not None:
            try:
                await self._handle_gen_job(run_id, generator, job, eval_q)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._record_worker_error("generator")

    async def _handle_gen_job(
        self, run_id: str, generator: Generator, job: _GenJob, eval_q: asyncio.Queue
    ) -> None:
        if job.kind == "meta":
            assert self.meta_db is not None
            try:
                completion = await generator.generate(self.meta_db.propose_prompt(), purpose="meta")
            except TRANSPORT_ERRORS as exc:  # retries already exhausted in the ensemble
                self.stats.transport_failures += 1
                self._log_event("transport_failure", purpose="meta", error=str(exc)[:300])
                return
            text = completion.text.strip().strip('"')
            if text:
                snippet = self.meta_db.add(text)
                self._log_event("meta_added", snippet=snippet.text[:200])
            return

        assert job.bundle is not None and job.parent is not None
        try:
            completion = await generator.generate(job.bundle.text, purpose="evolve")
        except TRANSPORT_ERRORS as exc:
            self.stats.transport_failures += 1
            self._log_event("transport_failure", purpose="evolve", error=str(exc)[:300])
            return
        try:
            if self.spec.full_rewrite:
                code = apply_full_rewrite(job.parent.code, extract_code_block(completion.text))
            else:
                code = apply_diff(job.parent.code, parse_diff(completion.text))
        except (DiffError, MarkerError) as exc:
            # Malformed output is a measurable signal, never retried
            # (CLAUDE.md): record it as a failed candidate. MarkerError is
            # defensive — it means a corrupt parent slipped into the archive.
            reason = f"diff_{exc.reason}" if isinstance(exc, DiffError) else "marker_error"
            self.stats.malformed_diffs += 1
            failed = Program(
                id=new_id(),
                code="",
                scores={},
                parent_id=job.parent.id,
                prompt_id=job.bundle.id,
                generation=job.parent.generation + 1,
                island=job.parent.island,
                status=FAILED,
                failure_reason=reason,
                artifacts={"completion": completion.text[-8000:]},
            )
            self.db.add_program(run_id, failed)
            self._log_event("malformed_diff", reason=reason, tier=completion.tier)
            return
        await eval_q.put(
            _EvalJob(code=code, parent=job.parent, bundle=job.bundle, completion=completion)
        )

    async def _evaluator(self, run_id: str, eval_q: asyncio.Queue) -> None:
        while (job := await eval_q.get()) is not None:
            try:
                await self._handle_eval_job(run_id, job)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._record_worker_error("evaluator")

    async def _handle_eval_job(self, run_id: str, job: _EvalJob) -> None:
        result = await run_cascade(
            self.spec, job.code, base_seed=self.seed, state_dir=self._state_dir
        )
        status = EVALUATED if result.ok else FAILED
        program = Program(
            id=new_id(),
            code=job.code,
            scores=result.scores,
            parent_id=job.parent.id,
            prompt_id=job.bundle.id,
            generation=job.parent.generation + 1,
            island=job.parent.island,
            status=status,
            failure_reason=result.failure_reason,
            artifacts=result.artifacts,
        )
        if result.ok and self.feedback_criteria:
            program = replace(program, scores=program.scores | await self._llm_feedback(job.code))
        self.db.add_program(run_id, program)
        if result.ok:
            admitted = self.islands.register(program)
            self.stats.registered += 1
            self.stats.admitted += int(admitted)
            self._credit_meta(job, program)
            self._log_event(
                "registered",
                program=program.id,
                scores=program.scores,
                stage=result.stage_reached,
                admitted=admitted,
                tier=job.completion.tier,
                eval_seconds=result.seconds,
            )
            if self.dump_every and self.stats.registered % self.dump_every == 0:
                self._dump_best()
        else:
            self.stats.eval_failures += 1
            self._log_event(
                "eval_failed",
                reason=(result.failure_reason or "")[:300],
                tier=job.completion.tier,
                eval_seconds=result.seconds,
            )

    async def _llm_feedback(self, code: str) -> dict[str, float]:
        """Optional LLM-graded auxiliary scores (paper §2.4), keys ``llm_*``.
        Grading failures produce no keys — never a fabricated score."""
        generator = self._active_generator
        if generator is None:
            return {}

        class _Completer:
            async def complete(self, prompt: str) -> str:
                completion = await generator.generate(prompt, purpose="llm_feedback")
                return completion.text

        try:
            return await grade(code, self.feedback_criteria, _Completer())
        except TRANSPORT_ERRORS as exc:
            self._log_event("transport_failure", purpose="llm_feedback", error=str(exc)[:300])
            return {}

    def _record_worker_error(self, worker: str) -> None:
        self.stats.worker_errors += 1
        trace = traceback.format_exc()
        self._log_event("worker_error", worker=worker, traceback=trace[-2000:])
        print(f"[alphaevolve] {worker} worker error:\n{trace}", file=sys.stderr)
        if self.stats.worker_errors >= self.max_worker_errors:
            self._log_event("aborting", reason=f"{self.stats.worker_errors} worker errors")
            self._stop.set()

    def _credit_meta(self, job: _EvalJob, child: Program) -> None:
        if self.meta_db is None:
            return
        raw_ids = job.bundle.meta.get("meta_snippet_ids", [])
        snippet_ids = [str(s) for s in raw_ids] if isinstance(raw_ids, list) else []
        if not snippet_ids:
            return
        shared = [k for k in child.scores if k in job.parent.scores and not k.startswith("llm_")]
        if not shared:
            return
        improvement = sum(child.scores[k] - job.parent.scores[k] for k in shared) / len(shared)
        self.meta_db.credit_children(snippet_ids, improvement)

    # -- run -----------------------------------------------------------------

    async def run(self, generator: Generator | None = None) -> RunStats:
        run_id = self.db.latest_run_id()
        if run_id is None:
            run_id = self.db.create_run(self.spec.name, self.config)
            (self.run_dir / "config.json").write_text(json.dumps(self.config, indent=2))

        ensemble = generator or self.ensemble
        if ensemble is None:
            built = self.build_ensemble()
            await self.preflight(built)
            ensemble = built
        self._active_generator = ensemble

        if self.meta_db is not None and self._meta_path.exists():
            self.meta_db = MetaPromptDB.load(self._meta_path)

        await self._ensure_genesis(run_id)

        prompt_q: asyncio.Queue = asyncio.Queue(self.queue_size)
        eval_q: asyncio.Queue = asyncio.Queue(self.queue_size)
        deadline = time.monotonic() + self.max_seconds

        loop = asyncio.get_running_loop()
        # Signal handlers are unavailable off the main thread (tests) and on
        # some platforms; graceful Ctrl-C is best effort.
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, self._stop.set)

        try:
            sampler = asyncio.create_task(self._sampler(run_id, prompt_q, deadline))
            generators = [
                asyncio.create_task(self._generator(run_id, ensemble, prompt_q, eval_q))
                for _ in range(self.n_generators)
            ]
            evaluators = [
                asyncio.create_task(self._evaluator(run_id, eval_q))
                for _ in range(self.n_evaluators)
            ]
            await sampler
            for _ in generators:
                await prompt_q.put(None)
            await asyncio.gather(*generators)
            for _ in evaluators:
                await eval_q.put(None)
            await asyncio.gather(*evaluators)
        finally:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signal.SIGINT)
            if self.meta_db is not None:
                self.meta_db.save(self._meta_path)
            self._dump_best()
            self._log_event("finished", **vars(self.stats))
            self.db.close()
        return self.stats

    # -- reporting -----------------------------------------------------------

    def _objective_key(self) -> str | None:
        if self.objective:
            return self.objective
        if self._genesis and self._genesis.scores:
            return sorted(self._genesis.scores)[0]
        return None

    def _dump_best(self) -> None:
        key = self._objective_key()
        if key is None:
            return
        best = self.islands.best(key)
        if best is None:
            return
        (self.run_dir / "best" / "best.py").write_text(best.code)
        (self.run_dir / "best" / "best.json").write_text(
            json.dumps(
                {
                    "id": best.id,
                    "scores": best.scores,
                    "generation": best.generation,
                    "island": best.island,
                    "parent_id": best.parent_id,
                    "objective": key,
                },
                indent=2,
            )
        )

    def _log_event(self, event_type: str, **payload: object) -> None:
        record = {"ts": time.time(), "type": event_type, **payload}
        with self._events_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def make_run_dir(base: Path, task_name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{task_name}-{stamp}-{new_id()[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
