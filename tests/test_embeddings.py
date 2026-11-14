from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from llmrouter.embeddings import Embedder, embed_cached
from llmrouter.embeddings.providers import HashingEmbedder
from llmrouter.registry import EMBEDDERS


class CountingEmbedder(Embedder):
    def __init__(self, name: str = "counting") -> None:
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += 1
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_hashing_is_registered_deterministic_and_normalised() -> None:
    embedder = EMBEDDERS.create("hashing", dim=32)
    texts = ["prove the theorem by induction", "bake the pasta with garlic", ""]
    out = embedder.embed(texts)
    assert out.shape == (3, 32) and out.dtype == np.float32
    assert np.array_equal(out, HashingEmbedder(dim=32).embed(texts))
    assert np.linalg.norm(out[:2], axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    assert not out[2].any()
    assert embedder.name == "hashing-32-n3" and embedder.dim == 32


def test_hashing_similar_texts_are_closer() -> None:
    embedder = HashingEmbedder(dim=256)
    a, b, c = embedder.embed(
        [
            "prove the theorem about matrix eigenvalue",
            "prove a lemma about matrix eigenvalue",
            "recipe for garlic butter pasta soup",
        ]
    )
    assert a @ b > a @ c + 0.2


def test_hashing_empty_batch() -> None:
    assert HashingEmbedder(dim=8).embed([]).shape == (0, 8)


def test_cache_hits_on_same_content_and_misses_on_any_change(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    texts = ["one", "three"]
    first = embed_cached(embedder, texts, tmp_path)
    second = embed_cached(embedder, texts, tmp_path)
    assert embedder.calls == 1
    assert np.array_equal(first, second)

    embed_cached(embedder, ["one", "four!"], tmp_path)  # different content
    embed_cached(embedder, texts, tmp_path, dtype="float16")  # different dtype
    assert embedder.calls == 3
    other = CountingEmbedder(name="other")  # different embedder
    embed_cached(other, texts, tmp_path)
    assert other.calls == 1

    metadata = [json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.glob("*.json")]
    assert {m["dtype"] for m in metadata} == {"float32", "float16"}
    assert {m["embedder"] for m in metadata} == {"counting", "other"}


def test_cache_disabled_and_bad_dtype(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    embed_cached(embedder, ["a"], None)
    embed_cached(embedder, ["a"], None)
    assert embedder.calls == 2
    with pytest.raises(ValueError, match="dtype"):
        embed_cached(embedder, ["a"], tmp_path, dtype="int8")


def test_caching_embedder_shares_one_entry_per_text_set(tmp_path: Path) -> None:
    from llmrouter.embeddings import CachingEmbedder

    inner = CountingEmbedder()
    cached = CachingEmbedder(inner, tmp_path)
    assert (cached.name, cached.dim) == (inner.name, inner.dim)
    first = cached.embed(["bb", "a", "bb", "ccc"])
    assert first.shape == (4, 2) and np.array_equal(first[0], first[2])
    assert first[:, 0].tolist() == [2.0, 1.0, 2.0, 3.0]
    again = cached.embed(["ccc", "a", "bb"])  # same set, other order: no new embed call
    assert inner.calls == 1 and again[:, 0].tolist() == [3.0, 1.0, 2.0]
    assert cached.embed([]).shape == (0, 2) and inner.calls == 1


def test_sentence_transformer_embedder_with_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    class FakeModel:
        device = "cpu"

        def __init__(self, name: str, device: str | None = None) -> None:
            self.name, self.max_seq_length = name, 0

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            assert kwargs["normalize_embeddings"] is True
            return np.ones((len(texts), 3), dtype=np.float64)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    embedder = EMBEDDERS.create("sentence_transformers", model="org/tiny", max_seq_length=128)
    assert embedder.name == "st-org/tiny-len128"
    out = embedder.embed(["a", "b"])
    assert out.shape == (2, 3) and out.dtype == np.float32 and embedder.dim == 3
    assert embedder.embed([]).shape == (0, 3)
    wrong = EMBEDDERS.create("sentence_transformers", model="org/tiny", dim=5)
    assert wrong.dim == 5  # configured dim is trusted until the model is loaded
    with pytest.raises(ValueError, match="produces 3"):
        wrong.embed(["a"])
