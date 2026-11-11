"""The one interface every routing algorithm implements.

A router estimates ``P(strong wins | prompt)`` in [0, 1]; higher means the query needs the
strong model more. The decision rule is fixed: ``P >= alpha -> strong model``, and alpha
only ever comes from calibration.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from llmrouter.embeddings.base import Embedder
from llmrouter.registry import ROUTERS

SCHEMA_VERSION = 1
_META_FILE = "meta.json"


class Router(ABC):
    """Base class. Subclasses implement ``strong_win_rate`` and, if they learn, ``fit``."""

    registry_name: ClassVar[str]
    #: Set to False for routers that never look at embeddings (skips embedding at fit time).
    uses_embeddings: ClassVar[bool] = True

    def __init__(self, strong_model: str, weak_model: str, seed: int = 0) -> None:
        self.strong_model = strong_model
        self.weak_model = weak_model
        self.seed = seed
        self._embedder: Embedder | None = None
        self._embedder_name: str | None = None

    # ------------------------------------------------------------------ learning
    def fit(
        self,
        battles: pd.DataFrame,
        embeddings: NDArray[np.floating] | None,
        embedder: Embedder | None,
    ) -> None:
        """Learn from battles (standard schema plus ``tier_a``/``tier_b``).

        ``embeddings[i]`` is the embedding of ``battles.prompt[i]``. Subclasses call
        ``super().fit(...)`` first so the embedder is recorded.
        """
        if self.uses_embeddings:
            if embedder is None or embeddings is None:
                raise ValueError(f"router '{self.registry_name}' needs embeddings to fit")
            self._embedder = embedder
            self._embedder_name = embedder.name

    # ----------------------------------------------------------------- inference
    @abstractmethod
    def strong_win_rate(self, prompts: Sequence[str]) -> NDArray[np.float64]:
        """Return ``P(strong wins)`` per prompt: shape ``(n,)``, values in [0, 1], no NaN."""

    def route(self, prompts: Sequence[str], alpha: float) -> NDArray[np.bool_]:
        """True where the prompt goes to the strong model (``P(strong) >= alpha``)."""
        return self.strong_win_rate(prompts) >= alpha

    def explain(self, prompts: Sequence[str]) -> list[dict[str, Any]]:
        """Per-prompt details for ``route --explain``; routers may add their own fields."""
        return [{"p_strong": float(p)} for p in self.strong_win_rate(prompts)]

    @staticmethod
    def check_probabilities(p: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Validate the ``strong_win_rate`` contract and return ``p``."""
        if p.shape != (n,):
            raise ValueError(f"expected shape ({n},), got {p.shape}")
        if n and not np.all((p >= 0.0) & (p <= 1.0)):
            raise ValueError("strong_win_rate must be in [0, 1] with no NaN")
        return p

    # --------------------------------------------------------------- persistence
    def get_params(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this router; subclasses extend this."""
        return {
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
            "seed": self.seed,
        }

    def _save_state(self, path: Path) -> None:  # noqa: B027 - optional hook
        """Write learned state into directory ``path``."""

    def _load_state(self, path: Path) -> None:  # noqa: B027 - optional hook
        """Read learned state from directory ``path``."""

    def save(self, path: str | Path) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "router": self.registry_name,
            "embedder": self._embedder_name,
            "params": self.get_params(),
        }
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self._save_state(directory)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> Router:
        """Rebuild a saved router; refuses an embedder different from the one used to fit."""
        directory = Path(path)
        meta = json.loads((directory / _META_FILE).read_text(encoding="utf-8"))
        if meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"router state has schema version {meta['schema_version']}, "
                f"this code expects {SCHEMA_VERSION}; fit again"
            )
        router_cls = ROUTERS.get(meta["router"])
        if not issubclass(router_cls, cls):
            raise TypeError(f"saved router is '{meta['router']}', not a {cls.__name__}")
        saved_embedder = meta["embedder"]
        if saved_embedder is not None:
            current = embedder.name if embedder is not None else None
            if current != saved_embedder:
                raise ValueError(
                    f"router was fit with embedder '{saved_embedder}' "
                    f"but is being loaded with '{current}'"
                )
        router = router_cls(**meta["params"])
        if saved_embedder is not None:
            router._embedder = embedder
            router._embedder_name = saved_embedder
        router._load_state(directory)
        return router
