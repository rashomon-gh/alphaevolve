"""Program records and SQLite persistence (paper §2.5, CLAUDE.md invariants 4/5)."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

_SCHEMA = (Path(__file__).parent / "schema.sql").read_text()

EVALUATED = "evaluated"
FAILED = "failed"


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Program:
    """One program in the database. Scores are multi-objective and maximized by
    convention; failed programs carry empty scores and a failure_reason."""

    id: str
    code: str
    scores: dict[str, float]
    parent_id: str | None
    prompt_id: str | None
    generation: int
    island: int = 0
    status: str = EVALUATED
    failure_reason: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def is_parent_eligible(self) -> bool:
        # Invariant 4: failed candidates are data, never parents.
        return self.status == EVALUATED

    def with_island(self, island: int) -> Program:
        return replace(self, island=island)


@dataclass(frozen=True)
class PromptRecord:
    id: str
    text: str
    meta: dict[str, object]
    created_at: float = field(default_factory=time.time)


class ProgramDB:
    """SQLite-backed store for runs, prompts, and programs.

    Synchronous by design: every operation is a point read/write on a local
    WAL-mode database, cheap enough to call from the asyncio controller.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- runs ---------------------------------------------------------------

    def create_run(
        self, task_name: str, config: dict[str, object], run_id: str | None = None
    ) -> str:
        rid = run_id or new_id()
        self._conn.execute(
            "INSERT INTO runs (id, task_name, config, created_at) VALUES (?, ?, ?, ?)",
            (rid, task_name, json.dumps(config), time.time()),
        )
        self._conn.commit()
        return rid

    def get_run(self, run_id: str) -> tuple[str, dict[str, object]] | None:
        row = self._conn.execute(
            "SELECT task_name, config FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def latest_run_id(self) -> str | None:
        row = self._conn.execute("SELECT id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    # -- prompts ------------------------------------------------------------

    def add_prompt(self, run_id: str, prompt: PromptRecord) -> None:
        self._conn.execute(
            "INSERT INTO prompts (id, run_id, text, meta, created_at) VALUES (?, ?, ?, ?, ?)",
            (prompt.id, run_id, prompt.text, json.dumps(prompt.meta), prompt.created_at),
        )
        self._conn.commit()

    def get_prompt(self, prompt_id: str) -> PromptRecord | None:
        row = self._conn.execute(
            "SELECT id, text, meta, created_at FROM prompts WHERE id = ?", (prompt_id,)
        ).fetchone()
        if row is None:
            return None
        return PromptRecord(id=row[0], text=row[1], meta=json.loads(row[2]), created_at=row[3])

    # -- programs -----------------------------------------------------------

    def add_program(self, run_id: str, program: Program) -> None:
        self._conn.execute(
            "INSERT INTO programs (id, run_id, code, scores, parent_id, prompt_id, generation,"
            " island, status, failure_reason, artifacts, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                program.id,
                run_id,
                program.code,
                json.dumps(program.scores),
                program.parent_id,
                program.prompt_id,
                program.generation,
                program.island,
                program.status,
                program.failure_reason,
                json.dumps(program.artifacts),
                program.created_at,
            ),
        )
        self._conn.commit()

    def get_program(self, program_id: str) -> Program | None:
        row = self._conn.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
        return self._row_to_program(row) if row else None

    def iter_programs(self, run_id: str, status: str | None = None) -> Iterator[Program]:
        """Programs in exact insertion order (rowid) so replay is deterministic
        even when two programs share a created_at clock quantum."""
        query = "SELECT * FROM programs WHERE run_id = ?"
        params: list[object] = [run_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY rowid"
        for row in self._conn.execute(query, params):
            yield self._row_to_program(row)

    def count_programs(self, run_id: str, status: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM programs WHERE run_id = ?"
        params: list[object] = [run_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        return int(self._conn.execute(query, params).fetchone()[0])

    @staticmethod
    def _row_to_program(row: sqlite3.Row | tuple) -> Program:
        (
            pid,
            _run_id,
            code,
            scores,
            parent_id,
            prompt_id,
            generation,
            island,
            status,
            failure_reason,
            artifacts,
            created_at,
        ) = row
        return Program(
            id=pid,
            code=code,
            scores=json.loads(scores),
            parent_id=parent_id,
            prompt_id=prompt_id,
            generation=generation,
            island=island,
            status=status,
            failure_reason=failure_reason,
            artifacts=json.loads(artifacts),
            created_at=created_at,
        )
