"""The single battle schema every data source is converted to."""

from __future__ import annotations

import numpy as np
import pandas as pd

BATTLE_COLUMNS: tuple[str, ...] = ("prompt", "model_a", "model_b", "winner", "source", "weight")
WINNERS: tuple[str, ...] = ("model_a", "model_b", "tie")
BENCHMARK_COLUMNS: tuple[str, ...] = ("prompt", "strong_score", "weak_score")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing columns: {missing}")


def validate_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Check the battle schema and return a copy with normalised dtypes.

    Extra columns (for example ``tier_a``/``tier_b``) are kept.
    """
    _require_columns(battles, BATTLE_COLUMNS, "battles")
    out = battles.reset_index(drop=True).copy()
    for column in ("prompt", "model_a", "model_b", "winner", "source"):
        if out[column].isna().any():
            raise ValueError(f"battles column '{column}' contains missing values")
        out[column] = out[column].astype(str)
    bad = ~out["winner"].isin(WINNERS)
    if bad.any():
        values = sorted(out.loc[bad, "winner"].unique())[:5]
        raise ValueError(f"winner must be one of {WINNERS}, got {values}")
    if (out["model_a"] == out["model_b"]).any():
        raise ValueError("model_a and model_b must differ in every battle")
    weight = pd.to_numeric(out["weight"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("weight must be finite and > 0")
    out["weight"] = weight
    return out


def validate_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Check the three-column benchmark format and return a copy with float scores."""
    _require_columns(benchmark, BENCHMARK_COLUMNS, "benchmark")
    out = benchmark.reset_index(drop=True).copy()
    if out["prompt"].isna().any():
        raise ValueError("benchmark column 'prompt' contains missing values")
    out["prompt"] = out["prompt"].astype(str)
    for column in ("strong_score", "weak_score"):
        scores = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"benchmark column '{column}' must be finite numbers")
        out[column] = scores
    return out
