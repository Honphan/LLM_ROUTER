"""Threshold calibration: pick alpha so a target share of queries goes to the strong model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def threshold_for_rate(scores: ArrayLike, strong_rate: float) -> float:
    """Return alpha such that about ``strong_rate`` of ``scores`` satisfy ``score >= alpha``.

    Tied scores can make the achieved rate larger than requested; see :func:`achieved_rate`.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot calibrate on an empty score array")
    if not 0.0 <= strong_rate <= 1.0:
        raise ValueError(f"strong_rate must be in [0, 1], got {strong_rate}")
    n_strong = int(np.floor(strong_rate * values.size + 0.5))
    if n_strong <= 0:
        return float(np.nextafter(values.max(), np.inf))
    return float(np.sort(values)[::-1][n_strong - 1])


def achieved_rate(scores: ArrayLike, alpha: float) -> float:
    """Share of scores routed to the strong model at threshold ``alpha``."""
    return float(np.mean(np.asarray(scores, dtype=np.float64) >= alpha))


def calibrate(scores: ArrayLike, strong_rates: Sequence[float]) -> dict[str, dict[str, float]]:
    """Thresholds for several target rates, keyed by the rate formatted like ``"0.5"``."""
    result: dict[str, dict[str, float]] = {}
    for rate in strong_rates:
        alpha = threshold_for_rate(scores, rate)
        result[f"{rate:g}"] = {
            "target_rate": float(rate),
            "alpha": alpha,
            "achieved_rate": achieved_rate(scores, alpha),
        }
    return result
