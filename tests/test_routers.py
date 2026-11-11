"""Tests for the Router interface (routers/base.py) and the random baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from llmrouter.embeddings import Embedder
from llmrouter.embeddings.providers import HashingEmbedder
from llmrouter.evaluation import metrics
from llmrouter.registry import ROUTERS
from llmrouter.routers import Router
from llmrouter.routers.random.router import RandomRouter


@ROUTERS.register("_test_mean_embedding")
class MeanEmbeddingRouter(Router):
    """Minimal embedding router used to exercise the base-class contract."""

    def __init__(self, strong_model: str, weak_model: str, seed: int = 0, scale: float = 1.0):
        super().__init__(strong_model, weak_model, seed)
        self.scale = scale
        self.direction: NDArray[np.float64] | None = None

    def fit(self, battles: pd.DataFrame, embeddings: Any, embedder: Embedder | None) -> None:
        super().fit(battles, embeddings, embedder)
        self.direction = np.asarray(embeddings, dtype=np.float64).mean(axis=0)

    def strong_win_rate(self, prompts: Sequence[str]) -> NDArray[np.float64]:
        if len(prompts) == 0:
            return np.empty(0)
        assert self._embedder is not None and self.direction is not None
        logits = self.scale * (self._embedder.embed(prompts) @ self.direction)
        return self.check_probabilities(1.0 / (1.0 + np.exp(-logits)), len(prompts))

    def get_params(self) -> dict[str, Any]:
        return {**super().get_params(), "scale": self.scale}

    def _save_state(self, path: Path) -> None:
        np.save(path / "direction.npy", self.direction)

    def _load_state(self, path: Path) -> None:
        self.direction = np.load(path / "direction.npy")


PROMPTS = ["prove the theorem by induction", "recipe for garlic pasta", "hello my friend today"]


def _random(seed: int = 0) -> Router:
    return ROUTERS.create("random", strong_model="s", weak_model="w", seed=seed)


def test_random_output_contract() -> None:
    router = _random()
    p = router.strong_win_rate(PROMPTS)
    assert p.shape == (3,) and p.dtype == np.float64
    assert np.all((p >= 0) & (p < 1))
    assert router.strong_win_rate([]).shape == (0,)


def test_random_depends_on_seed_and_prompt_not_order() -> None:
    router = _random(seed=1)
    forward = router.strong_win_rate(PROMPTS)
    backward = router.strong_win_rate(PROMPTS[::-1])
    assert np.array_equal(forward, backward[::-1])
    assert not np.array_equal(forward, _random(seed=2).strong_win_rate(PROMPTS))


def test_random_scores_are_uniform() -> None:
    p = _random().strong_win_rate([f"prompt number {i}" for i in range(5000)])
    assert p.mean() == pytest.approx(0.5, abs=0.02)
    assert np.histogram(p, bins=10, range=(0, 1))[0].min() > 400


def test_random_router_apgr_is_about_half(toy_benchmark: pd.DataFrame) -> None:
    strong = toy_benchmark["strong_score"].to_numpy()
    weak = toy_benchmark["weak_score"].to_numpy()
    values = [
        metrics.apgr(_random(seed).strong_win_rate(toy_benchmark["prompt"].tolist()), strong, weak)
        for seed in range(30)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.03)


def test_route_uses_greater_or_equal_alpha() -> None:
    router = _random()
    p = router.strong_win_rate(PROMPTS)
    alpha = float(np.sort(p)[1])
    assert router.route(PROMPTS, alpha).tolist() == (p >= alpha).tolist()
    assert router.route(PROMPTS, alpha).sum() == 2
    assert router.route(PROMPTS, 0.0).all() and not router.route(PROMPTS, 1.0).any()


def test_explain_reports_probability() -> None:
    router = _random()
    details = router.explain(PROMPTS)
    assert [d["p_strong"] for d in details] == router.strong_win_rate(PROMPTS).tolist()


def test_random_save_load_roundtrip_without_embedder(tmp_path: Path) -> None:
    router = _random(seed=7)
    router.fit(pd.DataFrame(), None, None)
    router.save(tmp_path / "router")
    loaded = Router.load(tmp_path / "router")
    assert isinstance(loaded, RandomRouter)
    assert (loaded.strong_model, loaded.weak_model, loaded.seed) == ("s", "w", 7)
    assert np.array_equal(loaded.strong_win_rate(PROMPTS), router.strong_win_rate(PROMPTS))


def _fitted(toy_battles: pd.DataFrame, embedder: HashingEmbedder) -> Router:
    router = ROUTERS.create("_test_mean_embedding", strong_model="s", weak_model="w", scale=2.0)
    router.fit(toy_battles, embedder.embed(toy_battles["prompt"].tolist()), embedder)
    return router


def test_embedding_router_save_load_roundtrip(
    tmp_path: Path, toy_battles: pd.DataFrame, toy_embedder: HashingEmbedder
) -> None:
    router = _fitted(toy_battles, toy_embedder)
    router.save(tmp_path / "router")
    loaded = Router.load(tmp_path / "router", toy_embedder)
    assert loaded.scale == 2.0  # type: ignore[attr-defined]
    assert np.allclose(loaded.strong_win_rate(PROMPTS), router.strong_win_rate(PROMPTS))


def test_load_rejects_different_or_missing_embedder(
    tmp_path: Path, toy_battles: pd.DataFrame, toy_embedder: HashingEmbedder
) -> None:
    _fitted(toy_battles, toy_embedder).save(tmp_path / "router")
    with pytest.raises(ValueError, match="fit with embedder"):
        Router.load(tmp_path / "router", HashingEmbedder(dim=128))
    with pytest.raises(ValueError, match="fit with embedder"):
        Router.load(tmp_path / "router")


def test_load_rejects_wrong_class_and_schema_version(tmp_path: Path) -> None:
    _random().save(tmp_path / "router")
    with pytest.raises(TypeError, match="not a MeanEmbeddingRouter"):
        MeanEmbeddingRouter.load(tmp_path / "router")
    meta = tmp_path / "router" / "meta.json"
    meta.write_text(
        meta.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 99')
    )
    with pytest.raises(ValueError, match="schema version"):
        Router.load(tmp_path / "router")


def test_fit_requires_embeddings_when_router_uses_them(toy_battles: pd.DataFrame) -> None:
    router = ROUTERS.create("_test_mean_embedding", strong_model="s", weak_model="w")
    with pytest.raises(ValueError, match="needs embeddings"):
        router.fit(toy_battles, None, None)


def test_check_probabilities_rejects_bad_output() -> None:
    with pytest.raises(ValueError, match="shape"):
        Router.check_probabilities(np.array([0.5]), 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Router.check_probabilities(np.array([0.5, np.nan]), 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Router.check_probabilities(np.array([0.5, 1.2]), 2)
