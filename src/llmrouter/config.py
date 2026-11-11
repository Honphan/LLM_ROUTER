"""Experiment configuration: YAML files composed with ``include:`` and ``--set`` overrides.

``${VAR}`` placeholders are kept verbatim in the loaded config (and therefore in
``config.resolved.json``) so secrets never reach disk; call :func:`expand_env` at the
point of use.
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MISSING = object()


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``extra``; nested dicts merge, everything else is replaced."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_file(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"circular include: {chain}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    includes = content.pop("include", None) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        merged = deep_merge(merged, _load_file(path.parent / include, (*stack, path)))
    return deep_merge(merged, content)


def apply_override(config: dict[str, Any], assignment: str) -> None:
    """Apply one ``a.b.c=value`` override in place; the value is parsed as YAML."""
    key, sep, raw = assignment.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"override must look like key.path=value, got '{assignment}'")
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise ValueError(f"cannot set '{key}': '{part}' is not a mapping")
        node = child
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict[str, Any]:
    """Load a YAML config, resolve ``include:`` recursively, then apply ``--set`` overrides."""
    config = _load_file(Path(path), ())
    for assignment in overrides:
        apply_override(config, assignment)
    return config


def get(config: dict[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    """Read ``a.b.c`` from a nested config; raise ``KeyError`` when missing and no default."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"missing config key '{dotted}'")
            return default
        node = node[part]
    return node


def expand_env(value: str) -> str:
    """Replace ``${VAR}`` with the environment value; raise if a variable is unset."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"environment variable '{name}' is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def save_resolved(config: dict[str, Any], path: str | Path) -> None:
    """Write the merged config as JSON; ``${VAR}`` placeholders stay unexpanded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
