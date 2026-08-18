import random

from alphaevolve.database.islands import IslandManager
from alphaevolve.database.programs import Program, new_id
from alphaevolve.task.spec import FeatureSpec

FEATURES = [FeatureSpec("score:u", bins=[0.25, 0.5, 0.75])]


def make_program(u: float, island: int = 0, created_at: float = 0.0) -> Program:
    return Program(
        id=new_id(),
        code=f"x = {u}\n",
        scores={"u": u},
        parent_id="parent-0",
        prompt_id=None,
        generation=1,
        island=island,
        created_at=created_at,
    )


def test_register_routes_to_island_archive():
    mgr = IslandManager(3, FEATURES, migration_interval=0)
    mgr.register(make_program(0.9, island=2))
    assert not mgr.islands[0].cells
    assert len(mgr.islands[2].cells) == 1


def test_migration_copies_elites_and_preserves_lineage():
    mgr = IslandManager(2, FEATURES, migration_interval=2, migration_count=1)
    a = make_program(0.9, island=0)
    b = make_program(0.1, island=1)
    mgr.register(a)
    mgr.register(b)  # second registration triggers migration
    # island 0's elite a migrated to island 1 (same id, island updated).
    migrated = [p for p in mgr.islands[1].elites if p.id == a.id]
    assert migrated and migrated[0].island == 1
    assert migrated[0].parent_id == a.parent_id  # lineage intact
    # the original stays on island 0
    assert any(p.id == a.id for p in mgr.islands[0].elites)


def test_migration_admission_still_by_dominance():
    mgr = IslandManager(2, [], migration_interval=2, migration_count=1)
    weak = make_program(0.2, island=0)
    strong = make_program(0.9, island=1)
    mgr.register(strong)
    mgr.register(weak)  # migration: weak -> island 1, strong -> island 0
    assert mgr.islands[1].elites[0].scores["u"] == 0.9  # weak migrant rejected
    assert mgr.islands[0].elites[0].scores["u"] == 0.9  # strong migrant replaces weak


def test_sample_within_island():
    mgr = IslandManager(2, FEATURES, migration_interval=0)
    for i, u in enumerate([0.1, 0.3, 0.6, 0.9]):
        mgr.register(make_program(u, island=0, created_at=float(i)))
    mgr.register(make_program(0.5, island=1))
    rng = random.Random(7)
    for _ in range(20):
        sample = mgr.sample(2, rng)
        assert sample is not None
        island_ids = {p.id for p in mgr.islands[sample.island].elites}
        assert sample.parent.id in island_ids
        assert all(p.id in island_ids for p in sample.inspirations)
        assert sample.parent.id not in {p.id for p in sample.inspirations}


def test_sample_empty_returns_none():
    mgr = IslandManager(2, FEATURES)
    assert mgr.sample(2, random.Random(0)) is None


def test_rebuild_reproduces_archive_state_including_migrations():
    def build() -> IslandManager:
        return IslandManager(
            3, FEATURES, migration_interval=5, migration_count=2, migration_seed=13
        )

    programs = [
        make_program(round(0.05 * i, 2) % 1.0, island=i % 3, created_at=float(i)) for i in range(37)
    ]
    live = build()
    for p in programs:
        live.register(p)

    replayed = build()
    replayed.rebuild(programs)  # replay in the same insertion order

    for live_island, replay_island in zip(live.islands, replayed.islands, strict=True):
        live_cells = {cell: p.id for cell, p in live_island.cells.items()}
        replay_cells = {cell: p.id for cell, p in replay_island.cells.items()}
        assert live_cells == replay_cells


def test_best_across_islands():
    mgr = IslandManager(2, FEATURES, migration_interval=0)
    mgr.register(make_program(0.4, island=0))
    mgr.register(make_program(0.8, island=1))
    best = mgr.best("u")
    assert best is not None and best.scores["u"] == 0.8
