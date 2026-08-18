"""Run reports and ablation comparisons (PLAN.md Phase 8/9).

Everything is computed from the run directory's SQLite database and JSONL
logs — no state of its own. Reports are markdown; if matplotlib is
installed, PNG score curves are written next to the report.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

from alphaevolve.database.programs import EVALUATED, FAILED, ProgramDB


@dataclass
class RunSummary:
    run_dir: Path
    run_id: str
    task_name: str
    objective: str
    # (sample_index, wall_seconds_since_start, best_so_far) per evaluated program
    curve: list[tuple[int, float, float]]
    evaluated: int
    failed: int
    failure_reasons: dict[str, int]
    tier_costs: dict[str, dict[str, float]]  # tier -> calls/prompt_tokens/completion_tokens/seconds
    eval_seconds: float
    time_buckets: list[dict[str, float]] = field(default_factory=list)

    @property
    def best(self) -> float | None:
        return self.curve[-1][2] if self.curve else None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def summarize_run(run_dir: Path) -> RunSummary:
    db = ProgramDB(run_dir / "programs.sqlite")
    try:
        run_id = db.latest_run_id()
        if run_id is None:
            raise ValueError(f"no run recorded in {run_dir}")
        got = db.get_run(run_id)
        assert got is not None
        task_name, config = got
        objective = str(config.get("objective", ""))

        evaluated = list(db.iter_programs(run_id, status=EVALUATED))
        failed = list(db.iter_programs(run_id, status=FAILED))
        if not objective and evaluated:
            objective = sorted(evaluated[0].scores)[0]

        curve: list[tuple[int, float, float]] = []
        best = float("-inf")
        start_ts = evaluated[0].created_at if evaluated else 0.0
        for index, program in enumerate(evaluated):
            if objective in program.scores:
                best = max(best, program.scores[objective])
            if best > float("-inf"):
                curve.append((index, program.created_at - start_ts, best))

        reasons: dict[str, int] = {}
        for program in failed:
            key = (program.failure_reason or "unknown").split(":")[0].split(" seed")[0]
            reasons[key] = reasons.get(key, 0) + 1

        tier_costs: dict[str, dict[str, float]] = {}
        for record in _read_jsonl(run_dir / "logs" / "llm.jsonl"):
            tier = record.get("tier", "?")
            cost = tier_costs.setdefault(
                tier, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
            )
            cost["calls"] += 1
            cost["prompt_tokens"] += record.get("prompt_tokens", 0)
            cost["completion_tokens"] += record.get("completion_tokens", 0)
            cost["seconds"] += record.get("seconds", 0.0)

        events = _read_jsonl(run_dir / "logs" / "events.jsonl")
        eval_seconds = sum(
            e.get("eval_seconds", 0.0) for e in events if e["type"] in ("registered", "eval_failed")
        )
        time_buckets = _failure_buckets(events)

        return RunSummary(
            run_dir=run_dir,
            run_id=run_id,
            task_name=task_name,
            objective=objective,
            curve=curve,
            evaluated=len(evaluated),
            failed=len(failed),
            failure_reasons=reasons,
            tier_costs=tier_costs,
            eval_seconds=eval_seconds,
            time_buckets=time_buckets,
        )
    finally:
        db.close()


def _failure_buckets(events: list[dict], n_buckets: int = 8) -> list[dict[str, float]]:
    """Failure-mode dashboard data: rates over wall-clock buckets."""
    tracked = [e for e in events if e["type"] in ("registered", "eval_failed", "malformed_diff")]
    if not tracked:
        return []
    t0 = min(e["ts"] for e in tracked)
    t1 = max(e["ts"] for e in tracked)
    width = max((t1 - t0) / n_buckets, 1e-9)
    buckets: list[dict[str, float]] = []
    for b in range(n_buckets):
        lo, hi = t0 + b * width, t0 + (b + 1) * width
        window = [e for e in tracked if lo <= e["ts"] <= (hi if b == n_buckets - 1 else hi)]
        total = len(window)
        if total == 0:
            continue
        malformed = sum(e["type"] == "malformed_diff" for e in window)
        eval_failed = sum(e["type"] == "eval_failed" for e in window)
        timeouts = sum(
            e["type"] == "eval_failed" and "timeout" in str(e.get("reason", "")) for e in window
        )
        buckets.append(
            {
                "start_s": lo - t0,
                "samples": float(total),
                "malformed_rate": malformed / total,
                "eval_failure_rate": eval_failed / total,
                "timeout_rate": timeouts / total,
            }
        )
    return buckets


def _ascii_curve(curve: list[tuple[int, float, float]], width: int = 60, height: int = 12) -> str:
    """Best-so-far vs. samples as a plain-text plot (always available)."""
    if len(curve) < 2:
        return "(not enough data for a curve)"
    values = [v for _, _, v in curve]
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    columns = [values[int(i * (len(values) - 1) / (width - 1))] for i in range(width)]
    rows = []
    for level in range(height - 1, -1, -1):
        threshold = lo + span * level / (height - 1)
        row = "".join("█" if v >= threshold else " " for v in columns)
        rows.append(f"{threshold:>12.6g} |{row}")
    rows.append(" " * 13 + "+" + "-" * width)
    rows.append(" " * 13 + f" 0 … {curve[-1][0]} evaluated programs")
    return "\n".join(rows)


def _maybe_png(summary: RunSummary, out_dir: Path) -> str | None:
    if importlib.util.find_spec("matplotlib") is None:
        return None
    import matplotlib  # type: ignore[import-not-found]  # optional, guarded above

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = [i for i, _, _ in summary.curve]
    ts = [t for _, t, _ in summary.curve]
    ys = [v for _, _, v in summary.curve]
    axes[0].plot(xs, ys)
    axes[0].set_xlabel("evaluated programs")
    axes[0].set_ylabel(summary.objective)
    axes[1].plot(ts, ys)
    axes[1].set_xlabel("wall clock (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{summary.task_name}: best {summary.objective} so far")
    path = out_dir / "best_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def render_report(summary: RunSummary, out_dir: Path) -> str:
    lines = [
        f"# Run report: {summary.task_name} ({summary.run_id[:8]})",
        "",
        f"- objective: **{summary.objective}**",
        f"- best: **{summary.best:.6g}**" if summary.best is not None else "- best: (none)",
        f"- programs: {summary.evaluated} evaluated, {summary.failed} failed",
        f"- evaluation compute: {summary.eval_seconds / 3600:.3f} eval-hours",
        "",
        "## Best-so-far curve",
        "",
        "```",
        _ascii_curve(summary.curve),
        "```",
        "",
    ]
    png = _maybe_png(summary, out_dir)
    if png:
        lines += [f"![best curve]({png})", ""]

    lines += [
        "## Cost accounting",
        "",
        "| tier | calls | prompt tokens | completion tokens | LLM seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for tier, cost in sorted(summary.tier_costs.items()):
        lines.append(
            f"| {tier} | {cost['calls']:.0f} | {cost['prompt_tokens']:.0f} "
            f"| {cost['completion_tokens']:.0f} | {cost['seconds']:.1f} |"
        )

    lines += ["", "## Failure modes", ""]
    if summary.failure_reasons:
        lines += ["| reason | count |", "| --- | --- |"]
        for reason, count in sorted(summary.failure_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("No failed candidates.")
    if summary.time_buckets:
        lines += [
            "",
            "| t (s) | samples | malformed rate | eval-failure rate | timeout rate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for b in summary.time_buckets:
            lines.append(
                f"| {b['start_s']:.0f} | {b['samples']:.0f} | {b['malformed_rate']:.2f} "
                f"| {b['eval_failure_rate']:.2f} | {b['timeout_rate']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, out_path: Path | None = None) -> Path:
    summary = summarize_run(run_dir)
    out_path = out_path or run_dir / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(summary, out_path.parent))
    return out_path


def write_comparison(run_dirs: list[Path], out_path: Path) -> Path:
    """Ablation-style comparison (PLAN.md Phase 9 deliverable)."""
    summaries = [summarize_run(d) for d in run_dirs]
    lines = [
        "# Run comparison",
        "",
        "| run | task | best | evaluated | failed | LLM calls | eval-hours |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        calls = sum(c["calls"] for c in s.tier_costs.values())
        best = f"{s.best:.6g}" if s.best is not None else "—"
        lines.append(
            f"| {s.run_dir.name} | {s.task_name} | {best} | {s.evaluated} "
            f"| {s.failed} | {calls:.0f} | {s.eval_seconds / 3600:.3f} |"
        )
    for s in summaries:
        lines += ["", f"## {s.run_dir.name}", "", "```", _ascii_curve(s.curve), "```"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
