"""On-disk embedding cache keyed by (text content, embedder name, dimension, dtype)."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from llmrouter.embeddings.base import Embedder

log = logging.getLogger(__name__)


def cache_key(embedder: Embedder, texts: Sequence[str], dtype: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{embedder.name}|{embedder.dim}|{dtype}|{len(texts)}".encode())
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()[:32]


def embed_cached(
    embedder: Embedder,
    texts: Sequence[str],
    cache_dir: str | Path | None,
    dtype: str = "float32",
) -> NDArray[np.floating]:
    """Embed ``texts``, reusing a cached array when content, embedder and dtype all match."""
    if dtype not in {"float32", "float16"}:
        raise ValueError(f"dtype must be float32 or float16, got {dtype}")
    if cache_dir is None:
        return embedder.embed(texts).astype(dtype)

    directory = Path(cache_dir)
    key = cache_key(embedder, texts, dtype)
    array_path = directory / f"{key}.npy"
    if array_path.exists():
        log.info("embedding cache hit: %s (%d texts)", array_path.name, len(texts))
        return np.load(array_path)

    embeddings = embedder.embed(texts).astype(dtype)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(array_path, embeddings)
    metadata = {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "dtype": dtype,
        "n_texts": len(texts),
    }
    (directory / f"{key}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log.info("embedding cache write: %s (%d texts)", array_path.name, len(texts))
    return embeddings
