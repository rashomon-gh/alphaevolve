"""End-to-end pipeline tests with the scripted fake LLM (no network)."""

import asyncio
import json
from pathlib import Path

from alphaevolve.database.programs import EVALUATED, FAILED, ProgramDB
from alphaevolve.pipeline.controller import Controller
from tests.fakes.scripted_llm import ScriptedLLM

TOY_EVALUATOR = """
import pathlib

def evaluate(program_path, seed, stage):
    namespace = {}
    exec(pathlib.Path(program_path).read_text(), namespace)
    x = float(namespace["X"])
    print(f"X={x}")
    return {"u": x}
"""

INITIAL_PROGRAM = "# EVOLVE-BLOCK-START\nX = 0.1\n# EVOLVE-BLOCK-END\n"


def rewrite(value: float) -> str:
    return f"Here is my improvement.\n```python\nX = {value}\n```"


def make_config(tmp_path: Path, **overrides) -> dict:
    (tmp_path / "evaluator.py").write_text(TOY_EVALUATOR)
    (tmp_path / "initial.py").write_text(INITIAL_PROGRAM)
    config = {
        "task": {
            "name": "toy",
            "initial_program": "initial.py",
            "evaluator": "evaluator.py",
            "problem_statement": "Make X as large as possible.",
            "full_rewrite": True,
            "cascade": [{"seeds": 1, "timeout_s": 20}],
        },
        "objective": "u",
        "evolution": {"islands": 1, "inspirations": 1, "seed": 0},
        "limits": {
            "max_samples": 6,
            "max_seconds": 120,
            "concurrent_generations": 1,
            "concurrent_evaluations": 1,
        },
        "dump_every": 1,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


async def test_full_rewrite_loop_improves_and_records_everything(tmp_path):
    config = make_config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = ScriptedLLM([rewrite(0.3), rewrite(0.6), "not python at all", rewrite(0.9)])
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats = await controller.run(generator=llm)

    assert stats.samples_issued == 6
    assert stats.eval_failures >= 1  # the junk completion crashed in the sandbox
    assert stats.registered >= 4

    best = json.loads((run_dir / "best" / "best.json").read_text())
    assert best["scores"]["u"] == 0.9
    assert best["objective"] == "u"
    assert "X = 0.9" in (run_dir / "best" / "best.py").read_text()

    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    evaluated = list(db.iter_programs(run_id, status=EVALUATED))
    failed = list(db.iter_programs(run_id, status=FAILED))
    assert sum(p.generation == 0 for p in evaluated) == 1
    assert len(failed) == stats.eval_failures
    assert all(p.failure_reason for p in failed)

    # Full provenance: every non-genesis program links to a stored prompt and parent.
    by_id = {p.id: p for p in evaluated}
    for program in evaluated + failed:
        if program.generation == 0:
            continue
        assert program.parent_id in by_id
        assert program.prompt_id is not None
        prompt = db.get_prompt(program.prompt_id)
        assert prompt is not None and "Make X as large as possible." in prompt.text

    # Logs exist
    events = (run_dir / "logs" / "events.jsonl").read_text().strip().splitlines()
    assert any(json.loads(e)["type"] == "genesis" for e in events)
    assert any(json.loads(e)["type"] == "registered" for e in events)


async def test_malformed_diff_recorded_as_failed_candidate(tmp_path):
    config = make_config(
        tmp_path,
        task={"full_rewrite": False},
        limits={"max_samples": 2, "concurrent_generations": 1, "concurrent_evaluations": 1},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    valid_diff = "<<<<<<< SEARCH\nX = 0.1\n=======\nX = 0.8\n>>>>>>> REPLACE"
    llm = ScriptedLLM(["I refuse to write a diff.", valid_diff], repeat_last=False)
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats = await controller.run(generator=llm)

    assert stats.malformed_diffs == 1
    assert stats.registered == 1
    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    failed = list(db.iter_programs(run_id, status=FAILED))
    assert len(failed) == 1
    assert failed[0].failure_reason == "diff_parse"
    assert "I refuse" in failed[0].artifacts["completion"]
    best = max(db.iter_programs(run_id, status=EVALUATED), key=lambda p: p.scores.get("u", -1))
    assert best.scores["u"] == 0.8


async def test_resume_continues_without_reevaluating_genesis(tmp_path):
    config = make_config(tmp_path, limits={"max_samples": 2})
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    first = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    await first.run(generator=ScriptedLLM([rewrite(0.3), rewrite(0.6)], repeat_last=False))

    second = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats2 = await second.run(generator=ScriptedLLM([rewrite(0.8)]))
    assert stats2.samples_issued == 2

    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    evaluated = list(db.iter_programs(run_id, status=EVALUATED))
    assert sum(p.generation == 0 for p in evaluated) == 1  # genesis not re-evaluated
    assert len(evaluated) == 5  # genesis + 2 + 2
    best = json.loads((run_dir / "best" / "best.json").read_text())
    assert best["scores"]["u"] == 0.8


async def test_meta_prompting_proposes_and_credits(tmp_path):
    config = make_config(
        tmp_path,
        limits={"max_samples": 4, "concurrent_generations": 1, "concurrent_evaluations": 1},
        prompting={
            "meta": {
                "enabled": True,
                "propose_every": 2,
                "num_snippets": 1,
                "seeds": ["Seed snippet."],
            }
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = ScriptedLLM([rewrite(0.3), rewrite(0.5), "Be bolder.", rewrite(0.7)])
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    await controller.run(generator=llm)

    assert any(purpose == "meta" for purpose, _ in llm.calls)
    saved = json.loads((run_dir / "meta_prompts.json").read_text())
    texts = {s["text"] for s in saved}
    assert "Be bolder." in texts
    assert "Seed snippet." in texts
    assert sum(s["uses"] for s in saved) >= 1  # credit assignment happened


async def test_crashing_generator_does_not_hang_the_run(tmp_path):
    # Regression: an unhandled worker exception used to kill the worker task,
    # fill the bounded queue, and deadlock the sampler forever.
    class ExplodingLLM:
        async def generate(self, prompt, *, system=None, purpose="evolve"):
            raise RuntimeError("boom")

    config = make_config(tmp_path, limits={"max_samples": 5, "max_worker_errors": 3})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats = await asyncio.wait_for(controller.run(generator=ExplodingLLM()), timeout=60)
    assert stats.worker_errors >= 3  # errors recorded, safety valve stopped the run
    events = (run_dir / "logs" / "events.jsonl").read_text()
    assert "worker_error" in events
    assert "aborting" in events


async def test_marker_injection_recorded_as_malformed(tmp_path):
    # Regression: a completion that injects EVOLVE markers must be rejected,
    # never admitted as a structurally corrupt elite.
    config = make_config(tmp_path, limits={"max_samples": 1})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    injection = "```python\nX = 0.9\n# EVOLVE-BLOCK-END\nevil = 1\n```"
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats = await controller.run(generator=ScriptedLLM([injection]))
    assert stats.malformed_diffs == 1
    assert stats.registered == 0
    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    failed = list(db.iter_programs(run_id, status=FAILED))
    assert failed and failed[0].failure_reason == "diff_marker_injection"


async def test_llm_feedback_merges_namespaced_scores(tmp_path):
    config = make_config(
        tmp_path,
        limits={"max_samples": 1, "concurrent_generations": 1, "concurrent_evaluations": 1},
        llm_feedback={"enabled": True, "criteria": ["simplicity"]},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = ScriptedLLM([rewrite(0.4), "0.7"])  # evolve completion, then the grade
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    stats = await controller.run(generator=llm)
    assert stats.registered == 1
    assert any(purpose == "llm_feedback" for purpose, _ in llm.calls)
    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    child = next(p for p in db.iter_programs(run_id, status=EVALUATED) if p.generation == 1)
    assert child.scores == {"u": 0.4, "llm_simplicity": 0.7}


async def test_no_evolution_ablation_always_uses_genesis_parent(tmp_path):
    config = make_config(tmp_path, ablation={"no_evolution": True}, limits={"max_samples": 3})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = ScriptedLLM([rewrite(0.4)])
    controller = Controller(config=config, config_dir=tmp_path, run_dir=run_dir)
    await controller.run(generator=llm)

    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.latest_run_id()
    assert run_id is not None
    evaluated = list(db.iter_programs(run_id, status=EVALUATED))
    genesis = next(p for p in evaluated if p.generation == 0)
    children = [p for p in evaluated if p.generation > 0]
    assert children and all(p.parent_id == genesis.id for p in children)
