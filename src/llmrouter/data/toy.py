"""Synthetic battles and benchmark for offline end-to-end runs and tests.

Prompts come from "hard" topics (where model strength matters a lot) and "easy" topics
(where every model does about equally well), so a sensible router can beat random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HARD_TOPICS: dict[str, tuple[str, ...]] = {
    "math": (
        "prove theorem integral derivative matrix eigenvalue topology lemma "
        "converge series manifold algebra polynomial induction bound"
    ).split(),
    "code": (
        "implement algorithm recursion compiler thread deadlock complexity pointer "
        "refactor async database index concurrency parser memory kernel"
    ).split(),
    "physics": (
        "quantum entropy relativity hamiltonian tensor photon momentum field "
        "boson thermodynamic lagrangian spin scattering wavefunction orbit"
    ).split(),
}
EASY_TOPICS: dict[str, tuple[str, ...]] = {
    "chat": (
        "hello thanks friend weather weekend smile coffee morning greeting "
        "birthday funny story joke evening holiday"
    ).split(),
    "cooking": (
        "recipe pasta salt butter oven simmer garlic onion breakfast "
        "soup noodle sugar bake fry rice"
    ).split(),
    "travel": (
        "trip hotel beach ticket museum luggage train island "
        "sunset market passport tour picnic map lake"
    ).split(),
}
GENERAL_WORDS: tuple[str, ...] = (
    "please explain what how the about with tell me can you describe give short answer".split()
)


def _sample_prompts(
    rng: np.random.Generator, n: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (prompts, is_hard, topic names)."""
    topics = [(name, words, True) for name, words in HARD_TOPICS.items()]
    topics += [(name, words, False) for name, words in EASY_TOPICS.items()]
    topic_ids = rng.integers(0, len(topics), size=n)
    prompts: list[str] = []
    for topic_id in topic_ids:
        _, words, _ = topics[topic_id]
        n_topic = int(rng.integers(7, 13))
        chosen = list(rng.choice(words, size=n_topic, replace=True))
        chosen += list(rng.choice(GENERAL_WORDS, size=3, replace=True))
        prompts.append(" ".join(str(word) for word in rng.permutation(chosen)))
    is_hard = np.array([topics[t][2] for t in topic_ids], dtype=bool)
    names = np.array([topics[t][0] for t in topic_ids])
    return prompts, is_hard, names


def toy_model_strengths(
    strong_model: str = "toy-strong", weak_model: str = "toy-weak"
) -> dict[str, float]:
    """Latent strengths: the strong model on top, the weak one in the middle of the pack."""
    return {
        strong_model: 2.4,
        "toy-large": 1.6,
        "toy-medium": 0.8,
        weak_model: 0.0,
        "toy-small": -0.8,
        "toy-tiny": -1.6,
    }


def make_toy_battles(
    n_battles: int = 2000,
    seed: int = 0,
    strong_model: str = "toy-strong",
    weak_model: str = "toy-weak",
    tie_rate: float = 0.1,
) -> pd.DataFrame:
    """Generate battles in the standard schema; stronger models win hard prompts more often."""
    rng = np.random.default_rng(seed)
    strengths = toy_model_strengths(strong_model, weak_model)
    names = np.array(list(strengths))
    values = np.array(list(strengths.values()))

    prompts, is_hard, _ = _sample_prompts(rng, n_battles)
    a = rng.integers(0, len(names), size=n_battles)
    b = (a + rng.integers(1, len(names), size=n_battles)) % len(names)
    sharpness = np.where(is_hard, 1.5, 0.2)
    p_a_wins = 1.0 / (1.0 + np.exp(-sharpness * (values[a] - values[b])))
    a_wins = rng.random(n_battles) < p_a_wins
    is_tie = rng.random(n_battles) < tie_rate
    winner = np.where(is_tie, "tie", np.where(a_wins, "model_a", "model_b"))
    return pd.DataFrame(
        {
            "prompt": prompts,
            "model_a": names[a],
            "model_b": names[b],
            "winner": winner,
            "source": "toy",
            "weight": 1.0,
        }
    )


def make_toy_benchmark(n_questions: int = 300, seed: int = 1) -> pd.DataFrame:
    """Generate a benchmark where the weak model fails mostly on hard prompts."""
    rng = np.random.default_rng(seed)
    prompts, is_hard, _ = _sample_prompts(rng, n_questions)
    p_strong = np.where(is_hard, 0.85, 0.92)
    p_weak = np.where(is_hard, 0.30, 0.88)
    return pd.DataFrame(
        {
            "prompt": prompts,
            "strong_score": (rng.random(n_questions) < p_strong).astype(np.float64),
            "weak_score": (rng.random(n_questions) < p_weak).astype(np.float64),
        }
    )
