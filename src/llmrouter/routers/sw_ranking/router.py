"""Similarity-weighted (SW) ranking router (RouteLLM section 4.2, eq. 9-10).

For each query, every training battle is weighted by how similar its prompt is to the
query; a Bradley-Terry model over model tiers is fit on those weighted battles and
``P(strong) = sigmoid(xi_strong - xi_weak)``. Nothing is trained ahead of time: ``fit``
only stores the battle embeddings and outcomes.

Differences from the paper / reference code:

* Battles are first summed into weighted tier-vs-tier win counts and the Bradley-Terry
  problems of a whole batch of queries are solved together (MM algorithm). The optimum is
  the one the reference gets from one logistic regression per query; it is just faster.
* Battles between two models of the same tier are dropped: they carry no information for
  a tier-level Bradley-Terry model (the reference keeps them through an indexing quirk).
* The embedder is whatever the config names, not necessarily ``text-embedding-3-small``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from llmrouter.embeddings.base import Embedder
from llmrouter.registry import ROUTERS
from llmrouter.routers.base import Router
from llmrouter.routers.sw_ranking import bradley_terry as bt

log = logging.getLogger(__name__)

_STATE_FILE = "state.npz"
_TIERS_FILE = "tiers.json"


def _unit_rows(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    out = np.asarray(matrix, dtype=np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


@ROUTERS.register("sw_ranking")
class SWRankingRouter(Router):
    def __init__(
        self,
        strong_model: str,
        weak_model: str,
        seed: int = 0,
        gamma: float = 10.0,
        normalize_by_max: bool = True,
        prior: float = 0.0,
        max_iter: int = 5000,
        tol: float = 1e-9,
        batch_size: int = 256,
        store_dtype: str = "float32",
    ) -> None:
        super().__init__(strong_model, weak_model, seed)
        if store_dtype not in {"float16", "float32"}:
            raise ValueError("store_dtype must be float16 or float32")
        self.gamma = gamma
        self.normalize_by_max = normalize_by_max
        self.prior = prior
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.store_dtype = store_dtype
        # State set by fit / _load_state. Battles are kept in tier-pair grouped order.
        self._embeddings: NDArray[np.floating] | None = None
        self._a_wins: NDArray[np.float32] | None = None
        self._b_wins: NDArray[np.float32] | None = None
        self._starts: NDArray[np.int64] | None = None
        self._group_a: NDArray[np.int64] | None = None
        self._group_b: NDArray[np.int64] | None = None
        self._n_tiers = 0
        self._strong_tier = -1
        self._weak_tier = -1

    # ------------------------------------------------------------------ learning
    def fit(
        self,
        battles: pd.DataFrame,
        embeddings: NDArray[np.floating] | None,
        embedder: Embedder | None,
    ) -> None:
        super().fit(battles, embeddings, embedder)
        assert embeddings is not None
        if len(embeddings) != len(battles):
            raise ValueError("embeddings and battles must have the same number of rows")
        for column in ("tier_a", "tier_b"):
            if column not in battles.columns:
                raise ValueError("sw_ranking needs battles with tier_a / tier_b columns")

        tier_a = battles["tier_a"].to_numpy(dtype=np.int64)
        tier_b = battles["tier_b"].to_numpy(dtype=np.int64)
        model_tier = dict(zip(battles["model_a"], tier_a.tolist(), strict=True))
        model_tier.update(zip(battles["model_b"], tier_b.tolist(), strict=True))
        for model in (self.strong_model, self.weak_model):
            if model not in model_tier:
                raise ValueError(f"model '{model}' does not appear in the fit battles")
        self._strong_tier = int(model_tier[self.strong_model])
        self._weak_tier = int(model_tier[self.weak_model])
        if self._strong_tier == self._weak_tier:
            raise ValueError("strong and weak model are in the same tier")
        self._n_tiers = int(max(tier_a.max(), tier_b.max())) + 1

        keep = np.flatnonzero(tier_a != tier_b)
        log.info("sw_ranking: %d battles, %d across different tiers", len(battles), len(keep))
        order, starts, group_a, group_b = bt.group_battles(tier_a[keep], tier_b[keep])
        rows = keep[order]
        winner = battles["winner"].to_numpy()[rows]
        a_share = np.where(winner == "model_a", 1.0, np.where(winner == "tie", 0.5, 0.0))
        weight = battles["weight"].to_numpy(dtype=np.float64)[rows]
        self._a_wins = (weight * a_share).astype(np.float32)
        self._b_wins = (weight * (1.0 - a_share)).astype(np.float32)
        self._starts, self._group_a, self._group_b = starts, group_a, group_b
        self._embeddings = _unit_rows(np.asarray(embeddings)[rows]).astype(self.store_dtype)

    # ----------------------------------------------------------------- inference
    def _scores(self, prompts: Sequence[str]) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Return ``(P(strong), highest similarity to a training battle)`` per prompt."""
        if self._embeddings is None or self._embedder is None:
            raise ValueError("sw_ranking router is not fitted")
        assert self._a_wins is not None and self._b_wins is not None
        assert self._starts is not None and self._group_a is not None
        assert self._group_b is not None
        queries = _unit_rows(self._embedder.embed(prompts))
        train = self._embeddings.astype(np.float32, copy=False)
        p = np.empty(len(prompts), dtype=np.float64)
        peak = np.empty(len(prompts), dtype=np.float32)
        for start in range(0, len(prompts), self.batch_size):
            stop = start + self.batch_size
            sims = queries[start:stop] @ train.T
            peak[start:stop] = sims.max(axis=1)
            weights = bt.similarity_weights(sims, self.gamma, self.normalize_by_max)
            wins = bt.aggregate_tier_wins(
                weights,
                self._a_wins,
                self._b_wins,
                self._starts,
                self._group_a,
                self._group_b,
                self._n_tiers,
            )
            strength = bt.fit_bradley_terry_batch(wins, self.prior, self.max_iter, self.tol)
            p[start:stop] = bt.win_probability(strength, self._strong_tier, self._weak_tier)
        return p, peak

    def strong_win_rate(self, prompts: Sequence[str]) -> NDArray[np.float64]:
        if len(prompts) == 0:
            return np.empty(0, dtype=np.float64)
        p, _ = self._scores(prompts)
        return self.check_probabilities(p, len(prompts))

    def explain(self, prompts: Sequence[str]) -> list[dict[str, Any]]:
        if len(prompts) == 0:
            return []
        p, peak = self._scores(prompts)
        return [
            {"p_strong": float(prob), "max_similarity": float(sim)}
            for prob, sim in zip(p, peak, strict=True)
        ]

    # --------------------------------------------------------------- persistence
    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "gamma": self.gamma,
            "normalize_by_max": self.normalize_by_max,
            "prior": self.prior,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "batch_size": self.batch_size,
            "store_dtype": self.store_dtype,
        }

    def _save_state(self, path: Path) -> None:
        if self._embeddings is None:
            raise ValueError("sw_ranking router is not fitted")
        np.savez(
            path / _STATE_FILE,
            embeddings=self._embeddings,
            a_wins=self._a_wins,
            b_wins=self._b_wins,
            starts=self._starts,
            group_a=self._group_a,
            group_b=self._group_b,
        )
        tiers = {
            "n_tiers": self._n_tiers,
            "strong_tier": self._strong_tier,
            "weak_tier": self._weak_tier,
        }
        (path / _TIERS_FILE).write_text(json.dumps(tiers, indent=2) + "\n", encoding="utf-8")

    def _load_state(self, path: Path) -> None:
        with np.load(path / _STATE_FILE) as state:
            self._embeddings = state["embeddings"]
            self._a_wins = state["a_wins"]
            self._b_wins = state["b_wins"]
            self._starts = state["starts"]
            self._group_a = state["group_a"]
            self._group_b = state["group_b"]
        tiers = json.loads((path / _TIERS_FILE).read_text(encoding="utf-8"))
        self._n_tiers = int(tiers["n_tiers"])
        self._strong_tier = int(tiers["strong_tier"])
        self._weak_tier = int(tiers["weak_tier"])
