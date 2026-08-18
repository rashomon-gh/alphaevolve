"""MAP-Elites archive (paper §2.5).

Feature descriptors map a program to a grid cell; each cell keeps one elite.
Admission compares score dicts without scalarizing (CLAUDE.md invariant 2):
the challenger wins only if it weakly dominates the incumbent; exact score
ties break by recency, then simplicity (shorter code).
"""

from __future__ import annotations

from bisect import bisect_right

from alphaevolve.database.programs import Program
from alphaevolve.task.spec import FeatureSpec

Cell = tuple[int, ...]

_IGNORED_PREFIX = "llm_"  # auxiliary scores never drive admission (invariant 1)


def _objective_keys(scores: dict[str, float]) -> set[str]:
    return {k for k in scores if not k.startswith(_IGNORED_PREFIX)}


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Weak Pareto dominance on the union of objective keys (missing = -inf)."""
    keys = _objective_keys(a) | _objective_keys(b)
    if not keys:
        return False
    strictly_better = False
    for key in keys:
        va, vb = a.get(key, float("-inf")), b.get(key, float("-inf"))
        if va < vb:
            return False
        if va > vb:
            strictly_better = True
    return strictly_better


def _tied(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = _objective_keys(a) | _objective_keys(b)
    return all(a.get(k, float("-inf")) == b.get(k, float("-inf")) for k in keys)


def feature_value(program: Program, feature: FeatureSpec) -> float:
    if feature.name == "code_length":
        return float(len(program.code))
    if feature.name.startswith("score:"):
        return program.scores.get(feature.name.removeprefix("score:"), float("-inf"))
    raise ValueError(f"unknown feature descriptor: {feature.name}")


class MapElitesArchive:
    def __init__(self, features: list[FeatureSpec]) -> None:
        self.features = features
        self.cells: dict[Cell, Program] = {}

    def cell_of(self, program: Program) -> Cell:
        # No features configured -> single-cell archive (plain hill climb).
        return tuple(bisect_right(f.bins, feature_value(program, f)) for f in self.features)

    def admit(self, program: Program) -> bool:
        """Try to install program as its cell's elite. Deterministic given
        insertion order, so DB replay reproduces the archive exactly."""
        if not program.is_parent_eligible:
            return False
        cell = self.cell_of(program)
        incumbent = self.cells.get(cell)
        if incumbent is None or dominates(program.scores, incumbent.scores):
            self.cells[cell] = program
            return True
        # Exact score tie: the newer program takes the cell unless it is
        # longer — recency for freshness, simplicity as the veto.
        if _tied(program.scores, incumbent.scores) and len(program.code) <= len(incumbent.code):
            self.cells[cell] = program
            return True
        return False

    @property
    def elites(self) -> list[Program]:
        return list(self.cells.values())

    def best(self, key: str) -> Program | None:
        scored = [p for p in self.cells.values() if key in p.scores]
        return max(scored, key=lambda p: p.scores[key], default=None)
