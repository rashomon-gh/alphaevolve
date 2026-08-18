import random
from typing import cast

from alphaevolve.database.programs import Program, new_id
from alphaevolve.prompting.context import TaskContext
from alphaevolve.prompting.meta import MetaPromptDB
from alphaevolve.prompting.sampler import build_prompt
from alphaevolve.prompting.stochastic import StochasticSlot, sample_slots


def make_program(code: str, scores: dict[str, float], artifacts=None) -> Program:
    return Program(
        id=new_id(),
        code=code,
        scores=scores,
        parent_id=None,
        prompt_id=None,
        generation=1,
        artifacts=artifacts or {},
    )


def test_prompt_shape_follows_fig3b():
    parent = make_program("X = 1\n", {"u": 0.5}, artifacts={"stdout": "eval says hi"})
    inspiration = make_program("X = 2\n", {"u": 0.7})
    context = TaskContext(problem_statement="Pack the bins tightly.")
    bundle = build_prompt(parent, [inspiration], context=context, rng=random.Random(0))

    text = bundle.text
    # Ordered sections: context -> prior programs -> current program -> format -> task.
    positions = [
        text.index("Problem context"),
        text.index("Pack the bins tightly."),
        text.index("Prior program 1"),
        text.index("X = 2"),
        text.index("Current program"),
        text.index("X = 1"),
        text.index("eval says hi"),
        text.index("<<<<<<< SEARCH"),
        text.index("## Task"),
    ]
    assert positions == sorted(positions)
    assert "u=0.5" in text and "u=0.7" in text
    assert bundle.meta["parent_id"] == parent.id
    assert bundle.meta["inspiration_ids"] == [inspiration.id]


def test_prompt_full_rewrite_rules():
    parent = make_program("X = 1\n", {"u": 0.5})
    bundle = build_prompt(parent, [], context=TaskContext(), full_rewrite=True)
    assert "complete new content of the evolve block" in bundle.text
    assert "<<<<<<< SEARCH" not in bundle.text


def test_no_context_ablation_strips_context():
    parent = make_program("X = 1\n", {"u": 0.5})
    context = TaskContext(problem_statement="SECRET CONTEXT")
    bundle = build_prompt(parent, [], context=context, include_context=False)
    assert "SECRET CONTEXT" not in bundle.text
    assert bundle.meta["include_context"] is False


def test_stochastic_slots_deterministic_given_rng():
    slots = [StochasticSlot(name="s", options=["A", "B"], weights=[1.0, 1.0])]
    values1 = [sample_slots(slots, random.Random(5))["s"] for _ in range(10)]
    values2 = [sample_slots(slots, random.Random(5))["s"] for _ in range(10)]
    assert values1 == values2
    parent = make_program("X = 1\n", {"u": 0.5})
    bundle = build_prompt(parent, [], context=TaskContext(), slots=slots, rng=random.Random(1))
    slot_values = cast(dict[str, str], bundle.meta["slot_values"])
    assert slot_values["s"] in {"A", "B"}
    assert slot_values["s"] in bundle.text


def test_meta_snippets_sampled_and_credited(tmp_path):
    db = MetaPromptDB.from_seed_texts(["Snippet one.", "Snippet two."])
    parent = make_program("X = 1\n", {"u": 0.5})
    bundle = build_prompt(
        parent, [], context=TaskContext(), meta_db=db, num_meta_snippets=2, rng=random.Random(0)
    )
    ids = [str(s) for s in cast(list[str], bundle.meta["meta_snippet_ids"])]
    assert len(ids) == 2
    for sid in ids:
        assert db.snippets[sid].text in bundle.text

    db.credit_children(ids, improvement=0.25)
    assert all(db.snippets[sid].credit == 0.25 for sid in ids)

    # credited snippets are favored in future samples
    path = tmp_path / "meta.json"
    db.save(path)
    loaded = MetaPromptDB.load(path)
    assert {s.text for s in loaded.snippets.values()} == {"Snippet one.", "Snippet two."}
    assert all(loaded.snippets[sid].uses == 1 for sid in ids)


def test_meta_propose_prompt_lists_credit():
    db = MetaPromptDB.from_seed_texts(["Do the thing."])
    prompt = db.propose_prompt()
    assert "Do the thing." in prompt
    assert "credit" in prompt
