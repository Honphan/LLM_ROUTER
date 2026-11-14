"""Embedding providers: offline ``hashing`` and local ``sentence_transformers`` models."""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from llmrouter.embeddings.base import Embedder
from llmrouter.registry import EMBEDDERS

log = logging.getLogger(__name__)


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


@EMBEDDERS.register("sentence_transformers")
class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers model: free, no API key, GPU if available.

    The model is loaded on first use. Give ``dim`` in the config to let cache lookups
    work without loading the model; it is checked against the model once loaded.
    Needs the ``hf`` extra (``uv pip install -e ".[hf]"``).
    """

    def __init__(
        self,
        model: str,
        dim: int | None = None,
        max_seq_length: int = 512,
        batch_size: int = 64,
        device: str = "auto",
        fp16: bool = True,
    ) -> None:
        self._model_name = model
        self._dim = dim
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._device = device
        self._fp16 = fp16
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"st-{self._model_name}-len{self._max_seq_length}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ValueError(
                "embedder 'sentence_transformers' needs the 'hf' extra: uv pip install -e \".[hf]\""
            ) from error
        device = None if self._device == "auto" else self._device
        model = SentenceTransformer(self._model_name, device=device)
        model.max_seq_length = self._max_seq_length
        if self._fp16 and str(model.device).startswith("cuda"):
            model.half()
        get_dim = getattr(model, "get_embedding_dimension", None)
        actual = int((get_dim or model.get_sentence_embedding_dimension)())
        if self._dim is not None and self._dim != actual:
            raise ValueError(
                f"embedder dim is configured as {self._dim} but '{self._model_name}' "
                f"produces {actual}"
            )
        self._dim = actual
        self._model = model
        log.info("loaded %s on %s", self._model_name, model.device)
        return model

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if len(texts) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 2000,
        )
        return np.asarray(vectors, dtype=np.float32)
