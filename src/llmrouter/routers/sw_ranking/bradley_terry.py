"""Pure math for similarity-weighted ranking: weights, tier-level counts, Bradley-Terry.

Everything works on a batch of queries at once. No file or network access here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MIN_STRENGTH = 1e-12


def similarity_weights(
    similarities: NDArray[np.floating], gamma: float = 10.0, normalize_by_max: bool = True
) -> NDArray[np.float32]:
    """Battle weights for each query, shape ``(n_queries, n_battles)``.

    ``omega = gamma ** (1 + s')`` (paper eq. 9) with ``s' = s / max_j s_j`` per query when
    ``normalize_by_max`` (what the reference code does), else ``s' = s``.
    """
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    sims = np.asarray(similarities, dtype=np.float32)
    if normalize_by_max and sims.shape[1] > 0:
        peak = sims.max(axis=1, keepdims=True)
        sims = sims / np.where(peak > 1e-6, peak, 1.0)
    return np.exp(np.float32(np.log(gamma)) * (1.0 + sims))


def group_battles(
    tier_a: NDArray[np.integer], tier_b: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Sort battles by ordered tier pair.

    Returns ``(order, starts, group_a, group_b)``: ``order`` sorts the battles so equal
    ``(tier_a, tier_b)`` pairs are contiguous, ``starts`` are the first positions of each
    run, and ``group_a[g]``, ``group_b[g]`` are the tiers of run ``g``.
    """
    a = np.asarray(tier_a, dtype=np.int64)
    b = np.asarray(tier_b, dtype=np.int64)
    width = int(max(a.max(initial=0), b.max(initial=0))) + 1
    key = a * width + b
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]]).astype(np.int64)
    if len(sorted_key) == 0:
        starts = np.empty(0, dtype=np.int64)
    return order, starts, sorted_key[starts] // width, sorted_key[starts] % width


def aggregate_tier_wins(
    weights: NDArray[np.floating],
    a_wins: NDArray[np.floating],
    b_wins: NDArray[np.floating],
    starts: NDArray[np.int64],
    group_a: NDArray[np.int64],
    group_b: NDArray[np.int64],
    n_tiers: int,
) -> NDArray[np.float64]:
    """Weighted win matrices, shape ``(n_queries, n_tiers, n_tiers)``.

    ``weights`` is ``(n_queries, n_battles)`` with battles in grouped order; ``a_wins[i]``
    and ``b_wins[i]`` are the (fractional) wins battle ``i`` gives each side.
    ``out[q, t, u]`` is the weighted number of wins of tier ``t`` over tier ``u``.
    """
    n_queries = weights.shape[0]
    wins = np.zeros((n_queries, n_tiers, n_tiers), dtype=np.float64)
    if len(starts) == 0:
        return wins
    for_a = np.add.reduceat(weights * a_wins, starts, axis=1, dtype=np.float64)
    for_b = np.add.reduceat(weights * b_wins, starts, axis=1, dtype=np.float64)
    flat = wins.reshape(n_queries, n_tiers * n_tiers)
    # (tier_a, tier_b) runs are unique, so each column is hit once per assignment.
    flat[:, group_a * n_tiers + group_b] += for_a
    flat[:, group_b * n_tiers + group_a] += for_b
    return wins


def fit_bradley_terry_batch(
    wins: NDArray[np.floating],
    prior: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> NDArray[np.float64]:
    """Bradley-Terry strengths for a batch of win matrices, shape ``(n_queries, n_tiers)``.

    Maximum-likelihood fit with the MM algorithm (Hunter, 2004); ties must already be
    split as half a win each. This is the same optimum as the reference implementation's
    unpenalised logistic regression on +/-1 indicator rows. ``prior`` adds that many wins
    and losses against a virtual player of strength 1 (0 = pure MLE, like the paper).
    Tiers without games keep strength 1. Stops when no log-strength moves by more than
    ``tol``.
    """
    wins = np.asarray(wins, dtype=np.float64)
    games = wins + np.swapaxes(wins, 1, 2)
    total_wins = wins.sum(axis=2) + prior
    played = games.sum(axis=2) > 0
    strength = np.ones(wins.shape[:2], dtype=np.float64)
    for _ in range(max_iter):
        pair_sum = strength[:, :, None] + strength[:, None, :]
        denominator = (games / pair_sum).sum(axis=2) + 2.0 * prior / (strength + 1.0)
        updated = np.where(played, total_wins / np.maximum(denominator, 1e-300), 1.0)
        updated = np.maximum(updated, _MIN_STRENGTH)
        if prior == 0.0:
            # Without a prior the scale is free: pin the geometric mean of played tiers to 1.
            log_mean = (np.log(updated) * played).sum(axis=1) / np.maximum(played.sum(axis=1), 1)
            updated = np.where(played, updated / np.exp(log_mean)[:, None], 1.0)
        change = np.max(np.abs(np.log(updated) - np.log(strength)))
        strength = updated
        if change < tol:
            break
    return strength


def win_probability(
    strength: NDArray[np.float64], strong_tier: int, weak_tier: int
) -> NDArray[np.float64]:
    """``P(strong tier beats weak tier)`` per query: ``p_s / (p_s + p_w)``."""
    strong = strength[:, strong_tier]
    weak = strength[:, weak_tier]
    return strong / (strong + weak)
