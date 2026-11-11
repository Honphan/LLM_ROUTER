"""Embedder interface shared by every provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class Embedder(ABC):
    """Maps texts to fixed-size vectors.

    ``name`` must identify the embedding space (provider, model, dimension): routers store
    it at fit time and refuse to load with a different embedder, and caches are keyed by it.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return an array of shape ``(len(texts), dim)``."""
