"""Tests for the SW ranking router (routers/sw_ranking): math first, then the router."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from llmrouter.data import tiers as tiers_mod
from llmrouter.embeddings.providers import HashingEmbedder
from llmrouter.evaluation import metrics
from llmrouter.registry import ROUTERS
from llmrouter.routers import Router
from llmrouter.routers.sw_ranking import bradley_terry as bt
from llmrouter.routers.sw_ranking.router import SWRankingRouter

HARD = "prove theorem integral matrix eigenvalue lemma polynomial induction"
EASY = "hello thanks friend weather weekend coffee morning greeting"


# ------------------------------------------------------------------------- math
def test_similarity_weights_formula() -> None:
    sims = np.array([[0.5, 0.25, 0.0]])
    normalised = bt.similarity_weights(sims, gamma=10.0, normalize_by_max=True)
    assert normalised[0] == pytest.approx([100.0, 10.0**1.5, 10.0], rel=1e-5)
    raw = bt.similarity_weights(sims, gamma=10.0, normalize_by_max=False)
    assert raw[0] == pytest.approx([10.0**1.5, 10.0**1.25, 10.0], rel=1e-5)
    with pytest.raises(ValueError, match="gamma"):
        bt.similarity_weights(sims, gamma=0.0)


def test_group_and_aggregate_match_a_plain_loop() -> None:
    rng = np.random.default_rng(0)
    n, n_tiers = 200, 4
    tier_a = rng.integers(0, n_tiers, n)
    tier_b = (tier_a + rng.integers(1, n_tiers, n)) % n_tiers
    a_share = rng.choice([1.0, 0.5, 0.0], n)
    weights = rng.random((3, n))

    order, starts, group_a, group_b = bt.group_battles(tier_a, tier_b)
    wins = bt.aggregate_tier_wins(
        weights[:, order], a_share[order], 1.0 - a_share[order], starts, group_a, group_b, n_tiers
    )
    expected = np.zeros((3, n_tiers, n_tiers))
    for q in range(3):
        for i in range(n):
            expected[q, tier_a[i], tier_b[i]] += weights[q, i] * a_share[i]
            expected[q, tier_b[i], tier_a[i]] += weights[q, i] * (1.0 - a_share[i])
    assert wins == pytest.approx(expected, rel=1e-9)


def test_bradley_terry_two_tiers_by_hand() -> None:
    # Tier 0 beats tier 1 three times out of four -> MLE P(0 beats 1) = 0.75 exactly.
    wins = np.array([[[0.0, 3.0], [1.0, 0.0]]])
    strength = bt.fit_bradley_terry_batch(wins)
    assert bt.win_probability(strength, 0, 1)[0] == pytest.approx(0.75, abs=1e-8)
    assert bt.win_probability(strength, 1, 0)[0] == pytest.approx(0.25, abs=1e-8)


def test_bradley_terry_matches_logistic_regression() -> None:
    rng = np.random.default_rng(1)
    n_tiers = 5
    wins = rng.uniform(0.5, 6.0, size=(4, n_tiers, n_tiers))
    wins[:, np.arange(n_tiers), np.arange(n_tiers)] = 0.0
    strength = bt.fit_bradley_terry_batch(wins, tol=1e-12)

    pairs = [(i, j) for i in range(n_tiers) for j in range(n_tiers) if i != j]
    design = np.zeros((2 * len(pairs), n_tiers))
    target = np.zeros(2 * len(pairs))
    for row, (i, j) in enumerate(pairs):  # row: "i beat j"; mirrored row: "j lost to i"
        design[2 * row, [i, j]] = [1.0, -1.0]
        target[2 * row] = 1.0
        design[2 * row + 1, [j, i]] = [1.0, -1.0]
    for q in range(len(wins)):
        sample_weight = np.repeat([wins[q, i, j] for i, j in pairs], 2)
        model = LogisticRegression(fit_intercept=False, C=np.inf, tol=1e-10, max_iter=5000)
        model.fit(design, target, sample_weight=sample_weight)
        theta = model.coef_[0]
        expected = 1.0 / (1.0 + np.exp(-(theta[0] - theta[3])))
        assert bt.win_probability(strength, 0, 3)[q] == pytest.approx(expected, abs=1e-4)


def test_bradley_terry_handles_unplayed_and_winless_tiers() -> None:
    wins = np.zeros((1, 3, 3))
    wins[0, 0, 1] = 5.0  # tier 1 never wins, tier 2 never plays
    strength = bt.fit_bradley_terry_batch(wins, max_iter=200)
    assert np.all(np.isfinite(strength)) and np.all(strength > 0)
    assert bt.win_probability(strength, 0, 1)[0] > 0.99
    with_prior = bt.fit_bradley_terry_batch(wins, prior=0.5)
    assert 0.5 < bt.win_probability(with_prior, 0, 1)[0] < 0.99


# ----------------------------------------------------------------------- router
def _battles(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Battles between models S (tier 0) and W (tier 1) from (prompt, winner) rows."""
    frame = pd.DataFrame(rows, columns=["prompt", "winner"])
    frame["model_a"], frame["model_b"] = "S", "W"
    frame["source"], frame["weight"] = "test", 1.0
    frame["tier_a"], frame["tier_b"] = 0, 1
    return frame


def _fit(battles: pd.DataFrame, embedder: HashingEmbedder, **params: object) -> Router:
    router = ROUTERS.create("sw_ranking", strong_model="S", weak_model="W", **params)
    router.fit(battles, embedder.embed(battles["prompt"].tolist()), embedder)
    return router


def test_router_prefers_strong_where_strong_wins(toy_embedder: HashingEmbedder) -> None:
    rows = [(f"{HARD} case {i}", "model_a") for i in range(8)]
    rows += [(f"{HARD} case {i} again", "model_b") for i in range(2)]
    rows += [(f"{EASY} case {i}", "model_a") for i in range(5)]
    rows += [(f"{EASY} case {i} again", "model_b") for i in range(5)]
    router = _fit(_battles(rows), toy_embedder)
    p = router.strong_win_rate([f"{HARD} new", f"{EASY} new"])
    assert p.shape == (2,) and p.dtype == np.float64
    assert np.all((p > 0.5 - 1e-9) & (p < 1.0))
    assert p[0] > p[1] + 0.05
    assert router.strong_win_rate([]).shape == (0,)
    details = router.explain([f"{HARD} new"])
    assert details[0]["p_strong"] == pytest.approx(p[0]) and details[0]["max_similarity"] > 0.5


def test_router_is_monotone_in_strong_wins(toy_embedder: HashingEmbedder) -> None:
    def p_with(n_strong_wins: int) -> float:
        rows = [(f"{HARD} case {i}", "model_a" if i < n_strong_wins else "model_b")
                for i in range(10)]  # fmt: skip
        return float(_fit(_battles(rows), toy_embedder).strong_win_rate([HARD])[0])

    values = [p_with(n) for n in (2, 5, 8)]
    assert values[0] < values[1] < values[2]


def test_router_uniform_weights_equal_global_win_rate(toy_embedder: HashingEmbedder) -> None:
    rows = [(f"{HARD} case {i}", "model_a") for i in range(6)]
    rows += [(f"{EASY} case {i}", "model_b") for i in range(3)]
    rows += [(f"{EASY} tie {i}", "tie") for i in range(2)]
    # gamma = 1 makes every weight 1, so P(strong) is the plain win share (6 + 1) / 11.
    router = _fit(_battles(rows), toy_embedder, gamma=1.0)
    assert router.strong_win_rate([HARD, EASY]) == pytest.approx([7 / 11, 7 / 11], abs=1e-6)


def test_router_ignores_same_tier_battles(toy_embedder: HashingEmbedder) -> None:
    base = _battles([(f"{HARD} case {i}", "model_a" if i < 3 else "model_b") for i in range(5)])
    extra = base.head(3).assign(model_b="S2", tier_b=0, winner="model_b")
    both = pd.concat([base, extra], ignore_index=True)
    assert _fit(both, toy_embedder).strong_win_rate([HARD]) == pytest.approx(
        _fit(base, toy_embedder).strong_win_rate([HARD])
    )


def test_router_save_load_and_embedder_check(tmp_path: Path, toy_embedder: HashingEmbedder) -> None:
    rows = [(f"{HARD} case {i}", "model_a" if i % 3 else "model_b") for i in range(9)]
    router = _fit(_battles(rows), toy_embedder, gamma=5.0, store_dtype="float16")
    router.save(tmp_path / "router")
    loaded = Router.load(tmp_path / "router", toy_embedder)
    assert isinstance(loaded, SWRankingRouter) and loaded.gamma == 5.0
    prompts = [HARD, EASY]
    assert np.array_equal(loaded.strong_win_rate(prompts), router.strong_win_rate(prompts))
    with pytest.raises(ValueError, match="fit with embedder"):
        Router.load(tmp_path / "router", HashingEmbedder(dim=128))


def test_router_rejects_bad_input(toy_embedder: HashingEmbedder) -> None:
    battles = _battles([(f"{HARD} case {i}", "model_a") for i in range(4)])
    embeddings = toy_embedder.embed(battles["prompt"].tolist())
    router = ROUTERS.create("sw_ranking", strong_model="S", weak_model="missing")
    with pytest.raises(ValueError, match="does not appear"):
        router.fit(battles, embeddings, toy_embedder)
    router = ROUTERS.create("sw_ranking", strong_model="S", weak_model="W")
    with pytest.raises(ValueError, match="same number of rows"):
        router.fit(battles, embeddings[:2], toy_embedder)
    with pytest.raises(ValueError, match="not fitted"):
        router.strong_win_rate([HARD])
    with pytest.raises(ValueError, match="store_dtype"):
        ROUTERS.create("sw_ranking", strong_model="S", weak_model="W", store_dtype="int8")


def test_router_beats_random_on_toy(
    toy_battles: pd.DataFrame, toy_benchmark: pd.DataFrame, toy_embedder: HashingEmbedder
) -> None:
    tiers = tiers_mod.segment_tiers(tiers_mod.compute_ratings(toy_battles))
    battles = tiers_mod.assign_tiers(toy_battles, tiers)
    router = ROUTERS.create("sw_ranking", strong_model="toy-strong", weak_model="toy-weak")
    router.fit(battles, toy_embedder.embed(battles["prompt"].tolist()), toy_embedder)
    scores = router.strong_win_rate(toy_benchmark["prompt"].tolist())
    apgr = metrics.apgr(
        scores, toy_benchmark["strong_score"].to_numpy(), toy_benchmark["weak_score"].to_numpy()
    )
    assert apgr > 0.6
