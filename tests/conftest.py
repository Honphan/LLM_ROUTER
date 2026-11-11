"""Shared fixtures: synthetic data only, nothing touches ``data/`` or ``experiments/``."""

from __future__ import annotations

import pandas as pd
import pytest

from llmrouter.data.toy import make_toy_battles, make_toy_benchmark
from llmrouter.embeddings.providers import HashingEmbedder


@pytest.fixture(scope="session")
def toy_battles() -> pd.DataFrame:
    return make_toy_battles(n_battles=600, seed=0)


@pytest.fixture(scope="session")
def toy_benchmark() -> pd.DataFrame:
    return make_toy_benchmark(n_questions=200, seed=1)


@pytest.fixture(scope="session")
def toy_embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)
