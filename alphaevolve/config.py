"""YAML config loading with ${ENV_VAR:-default} interpolation and deep-merge.

Endpoints and keys live in YAML + environment variables only (CLAUDE.md):
nothing here is ever hardcoded in the pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _interpolate(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise KeyError(f"config references undefined environment variable ${{{var}}}")

    return _ENV_RE.sub(repl, value)


def _walk(node: Any) -> Any:
    if isinstance(node, str):
        return _interpolate(node)
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v) for v in node]
    return node


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return _walk(data)


def load_run_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a run config, following top-level ``base:`` inheritance (used by
    ablation configs). Returns (config, anchor_dir): relative paths inside the
    config resolve against the *deepest base* file's directory, so an ablation
    override never has to repeat the task paths."""
    path = Path(path).resolve()
    data = load_yaml(path)
    if "base" in data:
        base_cfg, anchor = load_run_config(path.parent / str(data.pop("base")))
        return deep_merge(base_cfg, data), anchor
    return data, path.parent


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
