"""Island populations with periodic migration (paper §2.5).

K independent MAP-Elites archives; prompt sampling happens within one island;
every `migration_interval` registrations each island sends copies of
`migration_count` random elites to the next island on the ring, which admits
them under the normal dominance rules. Lineage is preserved: a migrant keeps
its id/parent, only its island field changes.

Determinism (CLAUDE.md invariant 7): migration draws come from a dedicated
rng seeded from config, and only register() consumes it — so replaying the
evaluated programs from SQLite in insertion order rebuilds the exact archive
state, including migrations. Prompt sampling uses a separate rng.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from alphaevolve.database.map_elites import MapElitesArchive
from alphaevolve.database.programs import Program
from alphaevolve.task.spec import FeatureSpec


@dataclass(frozen=True)
class ParentSample:
    parent: Program
    inspirations: list[Program]
    island: int


class IslandManager:
    def __init__(
        self,
        num_islands: int,
        features: list[FeatureSpec],
        *,
        migration_interval: int = 50,
        migration_count: int = 2,
        migration_seed: int = 0,
    ) -> None:
        if num_islands < 1:
            raise ValueError("need at least one island")
        self.islands = [MapElitesArchive(features) for _ in range(num_islands)]
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self._migration_rng = random.Random(migration_seed)
        self.registrations = 0

    def register(self, program: Program) -> bool:
        """Admit an evaluated program into its island's archive. Failed
        programs are stored in SQLite only (invariant 4) — never here."""
        admitted = self.islands[program.island % len(self.islands)].admit(program)
        self.registrations += 1
        if (
            len(self.islands) > 1
            and self.migration_interval > 0
            and self.registrations % self.migration_interval == 0
        ):
            self.migrate()
        return admitted

    def migrate(self) -> None:
        # Snapshot first so a migrant admitted this round doesn't re-migrate.
        outgoing: list[tuple[int, list[Program]]] = []
        for idx, island in enumerate(self.islands):
            elites = island.elites
            count = min(self.migration_count, len(elites))
            outgoing.append((idx, self._migration_rng.sample(elites, count) if count else []))
        for idx, migrants in outgoing:
            target = (idx + 1) % len(self.islands)
            for program in migrants:
                self.islands[target].admit(program.with_island(target))

    def sample(self, num_inspirations: int, rng: random.Random) -> ParentSample | None:
        """Paper §2.2 / Fig. 3b: 1 current program to improve + M diverse
        prior programs, all drawn within one (non-empty) island."""
        populated = [i for i, isl in enumerate(self.islands) if isl.cells]
        if not populated:
            return None
        island_idx = rng.choice(populated)
        elites = self.islands[island_idx].elites
        parent = rng.choice(elites)
        others = [p for p in elites if p.id != parent.id]
        count = min(num_inspirations, len(others))
        inspirations = rng.sample(others, count) if count else []
        return ParentSample(parent=parent, inspirations=inspirations, island=island_idx)

    def rebuild(self, programs: Iterable[Program]) -> None:
        """Resume (invariant 5): replay evaluated programs in insertion order
        through register(), reproducing admissions and migrations exactly."""
        for program in programs:
            self.register(program)

    def best(self, key: str) -> Program | None:
        candidates = [b for isl in self.islands if (b := isl.best(key)) is not None]
        return max(candidates, key=lambda p: p.scores[key], default=None)
