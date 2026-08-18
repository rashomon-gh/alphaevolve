"""CLI: run / inspect / ablate (PLAN.md Phase 5)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from alphaevolve.config import load_run_config
from alphaevolve.database.programs import EVALUATED, FAILED, ProgramDB
from alphaevolve.pipeline.controller import Controller, make_run_dir


def _cmd_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    if args.resume:
        run_dir = Path(args.resume).resolve()
        config = json.loads((run_dir / "config.json").read_text())
        _, config_dir = load_run_config(config_path)  # task paths resolve like the original run
    else:
        config, config_dir = load_run_config(config_path)
        run_dir = make_run_dir(Path(config.get("run_dir", "runs")), config["task"]["name"])
    if args.max_samples is not None:
        config.setdefault("limits", {})["max_samples"] = args.max_samples
    controller = Controller(config=config, config_dir=config_dir, run_dir=run_dir)
    print(f"run dir: {run_dir}")
    stats = asyncio.run(controller.run())
    print(
        f"done: {stats.samples_issued} samples, {stats.registered} registered, "
        f"{stats.admitted} admitted, {stats.malformed_diffs} malformed diffs, "
        f"{stats.eval_failures} eval failures, {stats.transport_failures} transport failures"
    )
    best_json = run_dir / "best" / "best.json"
    if best_json.exists():
        best = json.loads(best_json.read_text())
        print(f"best ({best['objective']}): {best['scores']}  [gen {best['generation']}]")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    db_path = run_dir / "programs.sqlite"
    if not db_path.exists():  # opening would create an empty DB in a wrong dir
        print(f"no program database at {db_path}", file=sys.stderr)
        return 1
    db = ProgramDB(db_path)
    run_id = db.latest_run_id()
    if run_id is None:
        print("no run found in this directory", file=sys.stderr)
        return 1
    task_name, _config = db.get_run(run_id) or ("?", {})
    evaluated = list(db.iter_programs(run_id, status=EVALUATED))
    failed_count = db.count_programs(run_id, status=FAILED)
    print(f"run {run_id} (task {task_name}): {len(evaluated)} evaluated, {failed_count} failed")
    if not evaluated:
        return 0

    keys = sorted({k for p in evaluated for k in p.scores if not k.startswith("llm_")})
    print("\nbest per objective:")
    by_id = {p.id: p for p in evaluated}
    for key in keys:
        best = max((p for p in evaluated if key in p.scores), key=lambda p: p.scores[key])
        print(f"  {key:30s} {best.scores[key]:.6g}  (gen {best.generation}, id {best.id[:8]})")
        if args.lineage and key == keys[0]:
            print("\nlineage of best program (newest first):")
            node = best
            while node is not None:
                scores = ", ".join(f"{k}={v:.4g}" for k, v in sorted(node.scores.items()))
                print(f"  gen {node.generation:3d}  {node.id[:8]}  [{scores}]")
                node = by_id.get(node.parent_id) if node.parent_id else None
    if args.dump_best:
        best = max((p for p in evaluated if keys[0] in p.scores), key=lambda p: p.scores[keys[0]])
        print(f"\n--- best program ({keys[0]}) ---\n{best.code}")
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    for config in args.configs:
        print(f"=== ablation config: {config} ===")
        ns = argparse.Namespace(config=config, resume=None, max_samples=args.max_samples)
        code = _cmd_run(ns)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphaevolve")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the evolutionary loop for a task config")
    p_run.add_argument("config")
    p_run.add_argument("--resume", metavar="RUN_DIR", help="resume a previous run directory")
    p_run.add_argument("--max-samples", type=int, default=None)
    p_run.set_defaults(func=_cmd_run)

    p_inspect = sub.add_parser("inspect", help="summarize a run directory")
    p_inspect.add_argument("run_dir")
    p_inspect.add_argument("--lineage", action="store_true")
    p_inspect.add_argument("--dump-best", action="store_true")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_ablate = sub.add_parser("ablate", help="run ablation configs sequentially")
    p_ablate.add_argument("configs", nargs="+")
    p_ablate.add_argument("--max-samples", type=int, default=None)
    p_ablate.set_defaults(func=_cmd_ablate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
