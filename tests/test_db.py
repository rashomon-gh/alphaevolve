from typing import Any

from alphaevolve.database.programs import (
    EVALUATED,
    FAILED,
    Program,
    ProgramDB,
    PromptRecord,
    new_id,
)


def make_program(**kwargs) -> Program:
    defaults: dict[str, Any] = dict(
        id=new_id(),
        code="x = 1\n",
        scores={"utilization": 0.5},
        parent_id=None,
        prompt_id=None,
        generation=0,
    )
    defaults.update(kwargs)
    return Program(**defaults)


def test_program_roundtrip(tmp_path):
    db = ProgramDB(tmp_path / "db.sqlite")
    run_id = db.create_run("toy", {"a": 1})
    program = make_program(artifacts={"stdout": "hello"})
    db.add_program(run_id, program)
    loaded = db.get_program(program.id)
    assert loaded == program


def test_failed_program_roundtrip(tmp_path):
    db = ProgramDB(tmp_path / "db.sqlite")
    run_id = db.create_run("toy", {})
    failed = make_program(scores={}, status=FAILED, failure_reason="stage 0: crash")
    db.add_program(run_id, failed)
    loaded = db.get_program(failed.id)
    assert loaded is not None
    assert loaded.status == FAILED
    assert loaded.failure_reason == "stage 0: crash"
    assert not loaded.is_parent_eligible


def test_prompt_roundtrip(tmp_path):
    db = ProgramDB(tmp_path / "db.sqlite")
    run_id = db.create_run("toy", {})
    prompt = PromptRecord(id=new_id(), text="improve this", meta={"parent_id": "abc"})
    db.add_prompt(run_id, prompt)
    assert db.get_prompt(prompt.id) == prompt


def test_iter_programs_insertion_order_and_status_filter(tmp_path):
    db = ProgramDB(tmp_path / "db.sqlite")
    run_id = db.create_run("toy", {})
    programs = [make_program(created_at=float(i)) for i in range(5)]
    programs[2] = make_program(created_at=2.0, scores={}, status=FAILED, failure_reason="boom")
    for p in programs:
        db.add_program(run_id, p)
    evaluated = list(db.iter_programs(run_id, status=EVALUATED))
    assert [p.created_at for p in evaluated] == [0.0, 1.0, 3.0, 4.0]
    assert db.count_programs(run_id) == 5
    assert db.count_programs(run_id, status=FAILED) == 1


def test_reopen_persists(tmp_path):
    path = tmp_path / "db.sqlite"
    db = ProgramDB(path)
    run_id = db.create_run("toy", {"cfg": True})
    program = make_program()
    db.add_program(run_id, program)
    db.close()

    db2 = ProgramDB(path)
    assert db2.latest_run_id() == run_id
    assert db2.get_run(run_id) == ("toy", {"cfg": True})
    assert db2.get_program(program.id) == program
