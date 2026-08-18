import json
import time
from pathlib import Path

from alphaevolve.database.programs import FAILED, Program, ProgramDB, new_id
from alphaevolve.reporting import summarize_run, write_comparison, write_report


def fabricate_run(run_dir: Path, best_values: list[float], objective: str = "u") -> None:
    (run_dir / "logs").mkdir(parents=True)
    db = ProgramDB(run_dir / "programs.sqlite")
    run_id = db.create_run("toytask", {"objective": objective})
    t0 = time.time()
    for i, value in enumerate(best_values):
        db.add_program(
            run_id,
            Program(
                id=new_id(),
                code=f"X = {value}\n",
                scores={objective: value},
                parent_id=None,
                prompt_id=None,
                generation=i,
                created_at=t0 + i,
            ),
        )
    db.add_program(
        run_id,
        Program(
            id=new_id(),
            code="",
            scores={},
            parent_id=None,
            prompt_id=None,
            generation=1,
            status=FAILED,
            failure_reason="stage 0: seed 0: timeout after 20s",
            created_at=t0 + len(best_values),
        ),
    )
    db.close()
    llm_records = [
        {"tier": "fast", "prompt_tokens": 100, "completion_tokens": 50, "seconds": 1.5},
        {"tier": "fast", "prompt_tokens": 120, "completion_tokens": 60, "seconds": 2.0},
        {"tier": "strong", "prompt_tokens": 200, "completion_tokens": 90, "seconds": 4.0},
    ]
    with (run_dir / "logs" / "llm.jsonl").open("w") as f:
        for r in llm_records:
            f.write(json.dumps(r) + "\n")
    events = [
        {"ts": t0 + 1, "type": "registered", "eval_seconds": 3.0},
        {"ts": t0 + 2, "type": "malformed_diff", "reason": "parse"},
        {"ts": t0 + 3, "type": "eval_failed", "reason": "stage 0: timeout", "eval_seconds": 20.0},
    ]
    with (run_dir / "logs" / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_summarize_run(tmp_path):
    fabricate_run(tmp_path, [0.1, 0.5, 0.3, 0.8])
    summary = summarize_run(tmp_path)
    assert summary.task_name == "toytask"
    assert summary.best == 0.8
    assert [v for _, _, v in summary.curve] == [0.1, 0.5, 0.5, 0.8]  # monotone best-so-far
    assert summary.evaluated == 4
    assert summary.failed == 1
    assert summary.failure_reasons == {"stage 0": 1}
    assert summary.tier_costs["fast"]["calls"] == 2
    assert summary.tier_costs["strong"]["prompt_tokens"] == 200
    assert summary.eval_seconds == 23.0
    assert summary.time_buckets  # dashboard data present


def test_write_report(tmp_path):
    fabricate_run(tmp_path, [0.1, 0.4, 0.9])
    out = write_report(tmp_path)
    text = out.read_text()
    assert "best: **0.9**" in text
    assert "Cost accounting" in text
    assert "| fast | 2 |" in text
    assert "Failure modes" in text
    assert "stage 0" in text


def test_write_comparison(tmp_path):
    a, b = tmp_path / "runA", tmp_path / "runB"
    fabricate_run(a, [0.1, 0.6])
    fabricate_run(b, [0.1, 0.2, 0.9])
    out = write_comparison([a, b], tmp_path / "ablations" / "report.md")
    text = out.read_text()
    assert "| runA | toytask | 0.6 |" in text
    assert "| runB | toytask | 0.9 |" in text
    assert text.count("## run") == 2
