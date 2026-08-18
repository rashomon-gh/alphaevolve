"""Subprocess entry point: load an evaluator file and run evaluate().

Invoked as: python -m alphaevolve.task._runner <evaluator> <program> <seed> <stage> <out_json>
Scores are written to out_json so the program's own stdout/stderr stay clean
artifacts for prompt rendering.
"""

from __future__ import annotations

import importlib.util
import json
import sys


def main() -> int:
    evaluator_path, program_path, seed, stage, out_path = sys.argv[1:6]
    spec = importlib.util.spec_from_file_location("ae_evaluator", evaluator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = module.evaluate(program_path, seed=int(seed), stage=int(stage))
    if not isinstance(scores, dict):
        raise TypeError(f"evaluate() must return dict[str, float], got {type(scores).__name__}")
    scores = {str(k): float(v) for k, v in scores.items()}
    with open(out_path, "w") as f:
        json.dump(scores, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
