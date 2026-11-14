"""Read battle sources and benchmarks that are already in the standard schema."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from llmrouter.data.schema import validate_battles, validate_benchmark

log = logging.getLogger(__name__)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file type '{suffix}' for {path}")


def load_battles(sources: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Concatenate battle sources.

    Each source is ``{path, weight?, source?}``. ``weight`` multiplies the per-row weight;
    ``source`` overrides the ``source`` column of that file.
    """
    if not sources:
        raise ValueError("data.sources is empty")
    frames = []
    for spec in sources:
        path = Path(spec["path"])
        frame = _read_table(path)
        if "weight" not in frame.columns:
            frame["weight"] = 1.0
        if "source" in spec:
            frame["source"] = spec["source"]
        elif "source" not in frame.columns:
            frame["source"] = path.stem
        frame = validate_battles(frame)
        frame["weight"] = frame["weight"] * float(spec.get("weight", 1.0))
        log.info("loaded %d battles from %s", len(frame), path)
        frames.append(frame)
    return validate_battles(pd.concat(frames, ignore_index=True))


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """Read a benchmark with columns ``prompt, strong_score, weak_score``."""
    return validate_benchmark(_read_table(Path(path)))
