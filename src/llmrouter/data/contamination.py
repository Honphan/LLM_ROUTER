"""Cross-contamination check between benchmark and training prompts (RouteLLM appendix B)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalise(embeddings: NDArray[np.floating]) -> NDArray[np.float32]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def max_similarity(
    benchmark_embeddings: NDArray[np.floating],
    train_embeddings: NDArray[np.floating],
    batch_size: int = 1024,
) -> NDArray[np.float32]:
    """For each benchmark row, the highest cosine similarity to any training row."""
    bench = _normalise(benchmark_embeddings)
    if len(train_embeddings) == 0:
        return np.full(len(bench), -1.0, dtype=np.float32)
    train = _normalise(train_embeddings)
    out = np.empty(len(bench), dtype=np.float32)
    for start in range(0, len(bench), batch_size):
        chunk = bench[start : start + batch_size]
        out[start : start + batch_size] = (chunk @ train.T).max(axis=1)
    return out


def find_contaminated(
    benchmark_embeddings: NDArray[np.floating],
    train_embeddings: NDArray[np.floating],
    threshold: float = 0.95,
    batch_size: int = 1024,
) -> NDArray[np.bool_]:
    """Boolean mask of benchmark rows whose cosine to some training row is >= threshold."""
    return max_similarity(benchmark_embeddings, train_embeddings, batch_size) >= threshold
