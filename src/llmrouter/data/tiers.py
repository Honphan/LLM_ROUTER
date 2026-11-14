"""Global model ratings and tier assignment (RouteLLM section 4.1 and appendix A).

Models are rated with a Bradley-Terry fit over all battles, then split into contiguous
tiers that minimise within-tier rating variance using dynamic programming.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def compute_ratings(
    battles: pd.DataFrame,
    prior: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Bradley-Terry ratings on an Elo-like scale (1000 + 400 log10 strength).

    Ties count as half a win for each side. ``prior`` adds that many wins and losses
    against a virtual model of strength 1, which keeps unbeaten or winless models finite
    and fixes the scale. Solved with the MM algorithm (Hunter, 2004).
    """
    models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
    index = {model: i for i, model in enumerate(models)}
    a = battles["model_a"].map(index).to_numpy()
    b = battles["model_b"].map(index).to_numpy()
    weight = battles["weight"].to_numpy(dtype=np.float64)
    winner = battles["winner"].to_numpy()
    a_share = np.where(winner == "model_a", 1.0, np.where(winner == "tie", 0.5, 0.0))

    n = len(models)
    wins = np.zeros((n, n))
    np.add.at(wins, (a, b), weight * a_share)
    np.add.at(wins, (b, a), weight * (1.0 - a_share))
    games = wins + wins.T
    total_wins = wins.sum(axis=1) + prior

    strength = np.ones(n)
    for _ in range(max_iter):
        pair_sum = strength[:, None] + strength[None, :]
        denominator = (games / pair_sum).sum(axis=1) + 2.0 * prior / (strength + 1.0)
        updated = total_wins / denominator
        converged = np.max(np.abs(np.log(updated) - np.log(strength))) < tol
        strength = updated
        if converged:
            break
    ratings = 1000.0 + 400.0 * np.log10(strength)
    return {model: float(rating) for model, rating in zip(models, ratings, strict=True)}


def _segment_costs(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """cost[i, j] = sum of squared deviations of values[i:j] from their mean."""
    n = len(values)
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(values**2)])
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    length = np.maximum(j - i, 1)
    total = prefix[j] - prefix[i]
    cost = prefix_sq[j] - prefix_sq[i] - total**2 / length
    return np.where(j > i, np.maximum(cost, 0.0), np.inf)


def segment_tiers(ratings: Mapping[str, float], n_tiers: int = 10) -> dict[str, int]:
    """Assign each model a tier (0 = best) by minimum within-tier variance segmentation."""
    if not ratings:
        raise ValueError("no ratings to segment")
    if n_tiers < 1:
        raise ValueError("n_tiers must be >= 1")
    models = sorted(ratings, key=lambda model: (-ratings[model], model))
    values = np.array([ratings[model] for model in models], dtype=np.float64)
    n = len(values)
    k = min(n_tiers, n)
    cost = _segment_costs(values)

    best = np.full((k + 1, n + 1), np.inf)
    cut = np.zeros((k + 1, n + 1), dtype=np.int64)
    best[0, 0] = 0.0
    for tier in range(1, k + 1):
        # candidates[i, j] = best[tier - 1, i] + cost[i, j]: last tier covers values[i:j]
        candidates = best[tier - 1][:, None] + cost
        cut[tier] = np.argmin(candidates, axis=0)
        best[tier] = np.min(candidates, axis=0)

    assignment: dict[str, int] = {}
    end = n
    for tier in range(k, 0, -1):
        start = int(cut[tier, end])
        for model in models[start:end]:
            assignment[model] = tier - 1
        end = start
    return assignment


def resolve_model(model: str, aliases: Mapping[str, str] | None) -> str:
    """Map a model that is absent from the battle data onto the model it stands in for."""
    return (aliases or {}).get(model, model)


def check_pair(
    tiers: Mapping[str, int],
    strong_model: str,
    weak_model: str,
    aliases: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Return (strong tier, weak tier); raise if unknown, equal, or in the wrong order."""
    resolved = [resolve_model(model, aliases) for model in (strong_model, weak_model)]
    for original, model in zip((strong_model, weak_model), resolved, strict=True):
        if model not in tiers:
            raise ValueError(
                f"model '{original}' has no tier; add it to data.model_aliases or the battles"
            )
    strong_tier, weak_tier = tiers[resolved[0]], tiers[resolved[1]]
    if strong_tier >= weak_tier:
        raise ValueError(
            f"strong model '{strong_model}' (tier {strong_tier}) must be in a better tier "
            f"than weak model '{weak_model}' (tier {weak_tier})"
        )
    return strong_tier, weak_tier


def assign_tiers(battles: pd.DataFrame, tiers: Mapping[str, int]) -> pd.DataFrame:
    """Return battles with integer ``tier_a`` and ``tier_b`` columns."""
    out = battles.copy()
    for side in ("a", "b"):
        mapped = out[f"model_{side}"].map(tiers)
        if mapped.isna().any():
            unknown = sorted(out.loc[mapped.isna(), f"model_{side}"].unique())[:5]
            raise ValueError(f"models without a tier: {unknown}")
        out[f"tier_{side}"] = mapped.astype(np.int64)
    return out
