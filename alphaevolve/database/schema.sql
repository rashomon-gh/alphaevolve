-- AlphaEvolve program database schema (single-file migration).
-- Invariant 5 (CLAUDE.md): everything is resumable; full provenance per program.

CREATE TABLE IF NOT EXISTS runs (
    id         TEXT PRIMARY KEY,
    task_name  TEXT NOT NULL,
    config     TEXT NOT NULL,          -- JSON copy of the resolved run config
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    text        TEXT NOT NULL,          -- fully rendered prompt sent to the LLM
    meta        TEXT NOT NULL,          -- JSON: parent id, inspiration ids, template/meta choices, tier
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS programs (
    id             TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES runs(id),
    code           TEXT NOT NULL,       -- full program source (skeleton + evolve blocks)
    scores         TEXT NOT NULL,       -- JSON dict[str, float]; {} for failures
    parent_id      TEXT REFERENCES programs(id),
    prompt_id      TEXT REFERENCES prompts(id),
    generation     INTEGER NOT NULL,
    island         INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL,       -- 'evaluated' | 'failed'
    failure_reason TEXT,                -- non-null iff status = 'failed'
    artifacts      TEXT NOT NULL DEFAULT '{}',  -- JSON: stdout/stderr excerpts etc.
    created_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_programs_run ON programs(run_id, status);
CREATE INDEX IF NOT EXISTS idx_programs_island ON programs(run_id, island);
