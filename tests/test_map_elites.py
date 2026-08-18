from typing import Any

from alphaevolve.database.map_elites import MapElitesArchive, dominates
from alphaevolve.database.programs import FAILED, Program, new_id
from alphaevolve.task.spec import FeatureSpec


def make_program(scores, code="x = 1\n", created_at=0.0, **kwargs) -> Program:
    defaults: dict[str, Any] = dict(
        id=new_id(),
        code=code,
        scores=scores,
        parent_id=None,
        prompt_id=None,
        generation=1,
        created_at=created_at,
    )
    defaults.update(kwargs)
    return Program(**defaults)


# -- dominance -------------------------------------------------------------


def test_dominates_strict():
    assert dominates({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 1.0})
    assert not dominates({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 1.0})


def test_incomparable_neither_dominates():
    a, b = {"a": 2.0, "b": 0.0}, {"a": 0.0, "b": 2.0}
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_equal_no_domination():
    assert not dominates({"a": 1.0}, {"a": 1.0})


def test_llm_scores_ignored(caplog):
    # llm_* keys are auxiliary (invariant 1): they never drive admission.
    assert dominates({"a": 2.0, "llm_style": 0.0}, {"a": 1.0, "llm_style": 1.0})
    assert not dominates({"a": 1.0, "llm_style": 1.0}, {"a": 1.0, "llm_style": 0.0})


def test_missing_key_is_worst():
    assert dominates({"a": 1.0, "b": 1.0}, {"a": 1.0})


# -- cell assignment -------------------------------------------------------


def test_cell_assignment_bins():
    archive = MapElitesArchive(
        [FeatureSpec("code_length", bins=[10.0, 20.0]), FeatureSpec("score:u", bins=[0.5])]
    )
    p = make_program({"u": 0.7}, code="x" * 15)
    assert archive.cell_of(p) == (1, 1)
    assert archive.cell_of(make_program({"u": 0.2}, code="x" * 5)) == (0, 0)
    assert archive.cell_of(make_program({"u": 0.9}, code="x" * 25)) == (2, 1)


def test_no_features_single_cell():
    archive = MapElitesArchive([])
    archive.admit(make_program({"u": 0.1}))
    archive.admit(make_program({"u": 0.5}))
    assert len(archive.cells) == 1
    assert archive.elites[0].scores["u"] == 0.5


# -- admission -------------------------------------------------------------


def test_archive_never_regresses_per_cell():
    archive = MapElitesArchive([])
    import random

    rng = random.Random(42)
    best = float("-inf")
    for i in range(200):
        value = rng.random()
        archive.admit(make_program({"u": value}, created_at=float(i)))
        best = max(best, value)
        assert archive.elites[0].scores["u"] == best


def test_tie_broken_by_recency_then_simplicity():
    archive = MapElitesArchive([])
    old = make_program({"u": 0.5}, code="x = 111\n", created_at=0.0)
    new_same_len = make_program({"u": 0.5}, code="y = 222\n", created_at=1.0)
    new_longer = make_program({"u": 0.5}, code="z = 3333333\n", created_at=2.0)
    assert archive.admit(old)
    assert archive.admit(new_same_len)  # same scores, same length -> recency wins
    assert archive.elites[0].id == new_same_len.id
    assert not archive.admit(new_longer)  # same scores but longer -> rejected
    assert archive.elites[0].id == new_same_len.id


def test_incomparable_keeps_incumbent():
    archive = MapElitesArchive([])
    a = make_program({"u": 0.9, "w": 0.1})
    b = make_program({"u": 0.1, "w": 0.9})
    assert archive.admit(a)
    assert not archive.admit(b)
    assert archive.elites[0].id == a.id


def test_failed_program_never_admitted():
    archive = MapElitesArchive([])
    failed = make_program({}, status=FAILED, failure_reason="crash")
    assert not archive.admit(failed)
    assert archive.cells == {}


def test_best_by_key():
    archive = MapElitesArchive([FeatureSpec("code_length", bins=[10.0])])
    a = make_program({"u": 0.3}, code="short")
    b = make_program({"u": 0.8}, code="a much longer program body")
    archive.admit(a)
    archive.admit(b)
    best = archive.best("u")
    assert best is not None and best.id == b.id
    assert archive.best("nonexistent") is None
