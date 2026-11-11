from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from llmrouter.data.tiers import (
    assign_tiers,
    check_pair,
    compute_ratings,
    resolve_model,
    segment_tiers,
)
from llmrouter.data.toy import toy_model_strengths


def _battle(a: str, b: str, winner: str, weight: float = 1.0) -> dict[str, object]:
    return {
        "prompt": "p",
        "model_a": a,
        "model_b": b,
        "winner": winner,
        "source": "t",
        "weight": weight,
    }


def test_two_models_match_closed_form() -> None:
    # A beats B in 3 of 4 games. With prior c the MM fixed point has strength ratio
    # (wins_a + c) / (wins_b + c) only approximately, so compare against the no-prior limit.
    battles = pd.DataFrame([_battle("A", "B", "model_a")] * 3 + [_battle("A", "B", "model_b")])
    ratings = compute_ratings(battles, prior=1e-9)
    assert ratings["A"] - ratings["B"] == pytest.approx(400 * np.log10(3.0), abs=1e-3)


def test_tie_counts_as_half_a_win_and_weight_scales() -> None:
    ties = pd.DataFrame([_battle("A", "B", "tie")] * 4)
    ratings = compute_ratings(ties)
    assert ratings["A"] == pytest.approx(ratings["B"], abs=1e-6)

    repeated = pd.DataFrame([_battle("A", "B", "model_a")] * 3 + [_battle("A", "B", "model_b")])
    weighted = pd.DataFrame([_battle("A", "B", "model_a", 3.0), _battle("A", "B", "model_b")])
    assert compute_ratings(repeated) == pytest.approx(compute_ratings(weighted))


def test_ratings_match_logistic_regression_reference() -> None:
    """Bradley-Terry MLE equals an unregularised logistic regression on +1/-1 indicators."""
    rng = np.random.default_rng(0)
    models = ["m0", "m1", "m2", "m3"]
    strength = np.array([1.5, 0.5, 0.0, -1.0])
    a = rng.integers(0, 4, size=4000)
    b = (a + rng.integers(1, 4, size=4000)) % 4
    a_wins = rng.random(4000) < 1 / (1 + np.exp(-(strength[a] - strength[b])))
    battles = pd.DataFrame(
        {
            "prompt": "p",
            "model_a": np.array(models)[a],
            "model_b": np.array(models)[b],
            "winner": np.where(a_wins, "model_a", "model_b"),
            "source": "t",
            "weight": 1.0,
        }
    )
    ratings = compute_ratings(battles, prior=1e-9)
    ours = np.array([ratings[m] for m in models]) * np.log(10) / 400
    ours -= ours.mean()

    design = np.zeros((4000, 4))
    design[np.arange(4000), a] = 1.0
    design[np.arange(4000), b] = -1.0
    reference = LogisticRegression(C=1e8, fit_intercept=False, tol=1e-10, max_iter=1000)
    reference.fit(design, a_wins)
    expected = reference.coef_[0] - reference.coef_[0].mean()
    assert ours == pytest.approx(expected, abs=1e-3)


def test_unbeaten_model_stays_finite() -> None:
    battles = pd.DataFrame([_battle("A", "B", "model_a")] * 5)
    ratings = compute_ratings(battles)
    assert np.isfinite(list(ratings.values())).all()
    assert ratings["A"] > ratings["B"]


def test_segment_tiers_finds_obvious_clusters() -> None:
    ratings = {"a": 1300, "b": 1290, "c": 1100, "d": 1095, "e": 1090, "f": 800}
    assert segment_tiers(ratings, n_tiers=3) == {"a": 0, "b": 0, "c": 1, "d": 1, "e": 1, "f": 2}


def test_segment_tiers_minimises_within_tier_variance_brute_force() -> None:
    values = [10.0, 9.0, 7.5, 7.0, 4.0, 3.9, 1.0]
    ratings = {f"m{i}": v for i, v in enumerate(values)}
    tiers = segment_tiers(ratings, n_tiers=3)

    def sse(cuts: tuple[int, int]) -> float:
        parts = [values[: cuts[0]], values[cuts[0] : cuts[1]], values[cuts[1] :]]
        return sum(float(np.sum((np.array(p) - np.mean(p)) ** 2)) for p in parts)

    best = min(((i, j) for i in range(1, 6) for j in range(i + 1, 7)), key=sse)
    expected = [0] * best[0] + [1] * (best[1] - best[0]) + [2] * (7 - best[1])
    assert [tiers[f"m{i}"] for i in range(7)] == expected


def test_segment_tiers_with_fewer_models_than_tiers() -> None:
    assert segment_tiers({"a": 2.0, "b": 1.0}, n_tiers=10) == {"a": 0, "b": 1}
    with pytest.raises(ValueError):
        segment_tiers({}, n_tiers=3)


def test_toy_strengths_are_recovered_in_order(toy_battles: pd.DataFrame) -> None:
    tiers = segment_tiers(compute_ratings(toy_battles), n_tiers=10)
    by_strength = sorted(toy_model_strengths(), key=lambda m: -toy_model_strengths()[m])
    assert sorted(tiers, key=tiers.get) == by_strength
    assert check_pair(tiers, "toy-strong", "toy-weak") == (0, 3)


def test_check_pair_and_aliases() -> None:
    tiers = {"gpt": 0, "mix": 2, "other": 2}
    assert check_pair(tiers, "gpt", "mix") == (0, 2)
    assert check_pair(tiers, "new-strong", "mix", {"new-strong": "gpt"}) == (0, 2)
    assert resolve_model("new-strong", {"new-strong": "gpt"}) == "gpt"
    with pytest.raises(ValueError, match="better tier"):
        check_pair(tiers, "mix", "other")
    with pytest.raises(ValueError, match="no tier"):
        check_pair(tiers, "unknown", "mix")


def test_assign_tiers_adds_columns() -> None:
    battles = pd.DataFrame([_battle("A", "B", "model_a")])
    out = assign_tiers(battles, {"A": 0, "B": 4})
    assert (out["tier_a"].item(), out["tier_b"].item()) == (0, 4)
    with pytest.raises(ValueError, match="without a tier"):
        assign_tiers(battles, {"A": 0})
