"""Turn downloaded raw files into processed battle and benchmark tables.

Each entry of ``data.prepare.battles`` / ``data.prepare.benchmarks`` names a ``kind``
(which converter to use), the raw files it reads and the processed file it writes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from llmrouter.data.sources.arena import ARENA_RAW_COLUMNS, convert_arena
from llmrouter.data.sources.routellm_bench import convert_correctness_table, convert_mt_bench

log = logging.getLogger(__name__)

Spec = Mapping[str, Any]


def _arena_55k(raw_dir: Path, spec: Spec) -> pd.DataFrame:
    raw = pd.read_csv(raw_dir / spec["raw"], usecols=list(ARENA_RAW_COLUMNS))
    return convert_arena(raw, source=str(spec.get("source", "arena")))


def _correctness_csv(raw_dir: Path, spec: Spec) -> pd.DataFrame:
    """One CSV file, or every ``*.csv`` of a directory, with a boolean column per model."""
    path = raw_dir / spec["raw"]
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")
    raw = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return convert_correctness_table(raw, spec["strong_column"], spec["weak_column"])


def _mt_bench(raw_dir: Path, spec: Spec) -> pd.DataFrame:
    questions = pd.read_json(raw_dir / spec["questions"], lines=True)
    judgements = pd.read_json(raw_dir / spec["judgements"], lines=True)
    return convert_mt_bench(questions, judgements, spec["strong_column"], spec["weak_column"])


CONVERTERS: dict[str, Callable[[Path, Spec], pd.DataFrame]] = {
    "arena_55k": _arena_55k,
    "correctness_csv": _correctness_csv,
    "mt_bench": _mt_bench,
}


def convert_all(prepare: Spec, raw_dir: str | Path) -> list[dict[str, Any]]:
    """Run every configured conversion and write the processed parquet files."""
    root = Path(raw_dir)
    written: list[dict[str, Any]] = []
    for group in ("battles", "benchmarks"):
        for spec in prepare.get(group) or []:
            kind = spec["kind"]
            if kind not in CONVERTERS:
                raise ValueError(f"unknown data kind '{kind}'; have {sorted(CONVERTERS)}")
            frame = CONVERTERS[kind](root, spec)
            output = Path(spec["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output, index=False)
            entry: dict[str, Any] = {"kind": kind, "output": str(output), "rows": len(frame)}
            if group == "benchmarks":
                entry["strong_mean"] = float(frame["strong_score"].mean())
                entry["weak_mean"] = float(frame["weak_score"].mean())
            log.info("wrote %s (%d rows)", output, len(frame))
            written.append(entry)
    return written
