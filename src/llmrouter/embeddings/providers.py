"""Embedding providers. Offline ``hashing`` only for now; API providers come with real data."""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from llmrouter.embeddings.base import Embedder
from llmrouter.registry import EMBEDDERS


@EMBEDDERS.register("hashing")
class HashingEmbedder(Embedder):
    """Deterministic bag of words + character n-grams hashed into ``dim`` signed buckets.

    No network, no model download. Good enough to give lexically similar prompts a high
    cosine similarity, which is all the offline toy pipeline needs.
    """

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        if dim < 1 or ngram < 1:
            raise ValueError("dim and ngram must be >= 1")
        self._dim = dim
        self._ngram = ngram

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}-n{self._ngram}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        grams = [lowered[i : i + self._ngram] for i in range(len(lowered) - self._ngram + 1)]
        return lowered.split() + grams

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            features = self._features(text)
            if not features:
                continue
            hashes = np.fromiter(
                (zlib.crc32(feature.encode("utf-8")) for feature in features),
                dtype=np.uint32,
                count=len(features),
            )
            buckets = hashes % self._dim
            signs = np.where((hashes >> 31) & 1, 1.0, -1.0).astype(np.float32)
            np.add.at(out[row], buckets, signs)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)
