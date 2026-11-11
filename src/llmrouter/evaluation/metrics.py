"""Routing metrics from RouteLLM section 3.2.

A router gives every question a score ``P(strong)``. Sending the ``k`` highest-scoring
questions to the strong model and the rest to the weak one gives one point of the
call-performance curve. Ties in the score are broken by input order (stable sort).

Difference from the paper: equation 8 writes APGR as a mean over ten cost levels. Taken
literally over 10%..100% that gives a random router 0.55, while the paper reports 0.50.
We integrate equation 7 with the trapezoid rule over 0%, 10%, ..., 100%, which gives a
random router 0.50 in expectation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_arrays(
    scores: ArrayLike, strong: ArrayLike, weak: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    p = np.asarray(scores, dtype=np.float64)
    s = np.asarray(strong, dtype=np.float64)
    w = np.asarray(weak, dtype=np.float64)
    if not (p.ndim == s.ndim == w.ndim == 1 and len(p) == len(s) == len(w)):
        raise ValueError("scores, strong and weak must be 1-D arrays of equal length")
    if len(p) == 0:
        raise ValueError("cannot evaluate an empty benchmark")
    if np.isnan(p).any():
        raise ValueError("router scores contain NaN")
    return p, s, w


def quality_curve(scores: ArrayLike, strong: ArrayLike, weak: ArrayLike) -> FloatArray:
    """Mean quality when the top-k questions go to the strong model, for k = 0..n."""
    p, s, w = _as_arrays(scores, strong, weak)
    order = np.argsort(-p, kind="stable")
    gain = (s - w)[order]
    return (w.sum() + np.concatenate([[0.0], np.cumsum(gain)])) / len(p)


def pgr_curve(scores: ArrayLike, strong: ArrayLike, weak: ArrayLike) -> FloatArray:
    """Performance gap recovered (equation 6) for k = 0..n strong calls; shape ``(n + 1,)``."""
    _, s, w = _as_arrays(scores, strong, weak)
    gap = s.mean() - w.mean()
    if gap == 0:
        raise ValueError("strong and weak models have the same mean score; PGR is undefined")
    return (quality_curve(scores, strong, weak) - w.mean()) / gap


def _calls_for(fraction: float | FloatArray, n: int) -> NDArray[np.int64]:
    """Number of strong calls for a call fraction, rounding half up."""
    return np.clip(np.floor(np.asarray(fraction) * n + 0.5), 0, n).astype(np.int64)


def pgr_at(scores: ArrayLike, strong: ArrayLike, weak: ArrayLike, call_fraction: float) -> float:
    """PGR when ``call_fraction`` of the questions go to the strong model."""
    curve = pgr_curve(scores, strong, weak)
    return float(curve[_calls_for(call_fraction, len(curve) - 1)])


def call_performance(
    scores: ArrayLike, strong: ArrayLike, weak: ArrayLike, n_levels: int = 10
) -> tuple[FloatArray, FloatArray]:
    """(call fractions, PGR) at ``n_levels + 1`` evenly spaced cost levels from 0 to 1."""
    curve = pgr_curve(scores, strong, weak)
    fractions = np.linspace(0.0, 1.0, n_levels + 1)
    return fractions, curve[_calls_for(fractions, len(curve) - 1)]


def apgr(scores: ArrayLike, strong: ArrayLike, weak: ArrayLike, n_levels: int = 10) -> float:
    """Average PGR: trapezoid integral of PGR over the strong-call fraction (equation 7)."""
    _, pgr = call_performance(scores, strong, weak, n_levels)
    return float(np.mean((pgr[:-1] + pgr[1:]) / 2.0))


def cpt(scores: ArrayLike, strong: ArrayLike, weak: ArrayLike, target: float) -> float:
    """Call-performance threshold: smallest strong-call fraction with PGR >= ``target``."""
    curve = pgr_curve(scores, strong, weak)
    reached = curve >= target - 1e-12
    if not reached.any():
        return float("nan")
    return float(np.argmax(reached) / (len(curve) - 1))


def random_baseline(
    strong: ArrayLike,
    weak: ArrayLike,
    n_seeds: int = 20,
    seed: int = 0,
    n_levels: int = 10,
) -> dict[str, float | int]:
    """Mean and 95% confidence half-width of APGR / CPT for uniformly random scores."""
    if n_seeds < 2:
        raise ValueError("need at least 2 seeds for a confidence interval")
    s = np.asarray(strong, dtype=np.float64)
    rows = []
    for offset in range(n_seeds):
        scores = np.random.default_rng(seed + offset).random(len(s))
        rows.append(
            (
                apgr(scores, strong, weak, n_levels),
                cpt(scores, strong, weak, 0.5),
                cpt(scores, strong, weak, 0.8),
            )
        )
    values = np.array(rows)
    mean = values.mean(axis=0)
    half_width = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    result: dict[str, float | int] = {"n_seeds": n_seeds, "first_seed": seed}
    for column, key in enumerate(("apgr", "cpt_50", "cpt_80")):
        result[f"{key}_mean"] = float(mean[column])
        result[f"{key}_ci95"] = float(half_width[column])
    return result
