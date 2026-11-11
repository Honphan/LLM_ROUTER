"""Tests for llmrouter.data: schema, io, filters, split, toy, contamination."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmrouter.data.contamination import find_contaminated, max_similarity
from llmrouter.data.filters import filter_battles, first_user_turn
from llmrouter.data.io import load_battles, load_benchmark
from llmrouter.data.schema import BATTLE_COLUMNS, validate_battles, validate_benchmark
from llmrouter.data.split import split_calibration
from llmrouter.data.toy import make_toy_battles, make_toy_benchmark


def _battles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt": ["explain recursion in python", "hi", "write a haiku about rain"],
            "model_a": ["m1", "m1", "m2"],
            "model_b": ["m2", "m3", "m3"],
            "winner": ["model_a", "tie", "model_b"],
            "source": ["t", "t", "t"],
            "weight": [1, 1, 2],
        }
    )


# ------------------------------------------------------------------------ schema
def test_validate_battles_accepts_and_normalises() -> None:
    out = validate_battles(_battles())
    assert list(out.columns[:6]) == list(BATTLE_COLUMNS)
    assert out["weight"].dtype == np.float64


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("winner", "strong", "winner must be one of"),
        ("weight", 0, "weight must be finite"),
        ("model_b", "m1", "must differ"),
    ],
)
def test_validate_battles_rejects_bad_rows(column: str, value: object, message: str) -> None:
    battles = _battles()
    battles.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_battles(battles)


def test_validate_battles_requires_all_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_battles(_battles().drop(columns=["source"]))


def test_validate_benchmark() -> None:
    bench = pd.DataFrame({"prompt": ["a"], "strong_score": [1], "weak_score": [0]})
    assert validate_benchmark(bench)["strong_score"].dtype == np.float64
    bench["weak_score"] = ["x"]
    with pytest.raises(ValueError, match="weak_score"):
        validate_benchmark(bench)


# ---------------------------------------------------------------------------- io
def test_load_battles_applies_source_weight_and_name(tmp_path: Path) -> None:
    parquet = tmp_path / "a.parquet"
    jsonl = tmp_path / "b.jsonl"
    _battles().to_parquet(parquet, index=False)
    _battles().drop(columns=["weight", "source"]).to_json(jsonl, orient="records", lines=True)
    out = load_battles([{"path": str(parquet), "weight": 3}, {"path": str(jsonl), "source": "vi"}])
    assert len(out) == 6
    assert out["weight"].tolist() == [3.0, 3.0, 6.0, 1.0, 1.0, 1.0]
    assert out["source"].tolist() == ["t"] * 3 + ["vi"] * 3


def test_load_battles_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_battles([])


def test_load_benchmark_roundtrip(tmp_path: Path, toy_benchmark: pd.DataFrame) -> None:
    path = tmp_path / "bench.parquet"
    toy_benchmark.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(load_benchmark(path), toy_benchmark)


# ----------------------------------------------------------------------- filters
def test_filter_drops_short_prompts() -> None:
    out = filter_battles(_battles(), min_prompt_chars=16)
    assert out["prompt"].tolist() == ["explain recursion in python", "write a haiku about rain"]


def test_first_user_turn() -> None:
    conversation = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question"},
    ]
    assert first_user_turn(conversation) == "first question"
    assert first_user_turn("plain prompt") == "plain prompt"
    assert first_user_turn([]) == ""


# ------------------------------------------------------------------------- split
def test_split_is_disjoint_by_prompt_and_deterministic(toy_battles: pd.DataFrame) -> None:
    doubled = pd.concat([toy_battles, toy_battles], ignore_index=True)
    fit, calibration = split_calibration(doubled, 0.2, seed=3)
    assert len(fit) + len(calibration) == len(doubled)
    assert not set(fit["prompt"]) & set(calibration["prompt"])
    share = calibration["prompt"].nunique() / doubled["prompt"].nunique()
    assert share == pytest.approx(0.2, abs=0.01)
    fit_again, _ = split_calibration(doubled, 0.2, seed=3)
    pd.testing.assert_frame_equal(fit, fit_again)


def test_split_rejects_bad_fraction(toy_battles: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fraction"):
        split_calibration(toy_battles, 1.0, seed=0)


# --------------------------------------------------------------------------- toy
def test_toy_battles_follow_schema_and_are_deterministic() -> None:
    battles = make_toy_battles(n_battles=300, seed=5)
    pd.testing.assert_frame_equal(validate_battles(battles), battles)
    pd.testing.assert_frame_equal(battles, make_toy_battles(n_battles=300, seed=5))
    assert battles["prompt"].str.len().min() >= 16
    assert {"toy-strong", "toy-weak"} <= set(battles["model_a"]) | set(battles["model_b"])
    assert set(battles["winner"]) == {"model_a", "model_b", "tie"}


def test_toy_benchmark_has_a_quality_gap() -> None:
    bench = validate_benchmark(make_toy_benchmark(n_questions=500, seed=2))
    assert bench["strong_score"].mean() - bench["weak_score"].mean() > 0.15


# ----------------------------------------------------------------- contamination
def test_contamination_flags_near_duplicates_only() -> None:
    train = np.array([[1.0, 0.0], [0.0, 1.0]])
    bench = np.array([[2.0, 0.01], [1.0, 1.0], [0.0, -3.0]])
    similarity = max_similarity(bench, train)
    assert similarity == pytest.approx([1.0, np.sqrt(0.5), 0.0], abs=1e-4)
    assert find_contaminated(bench, train, threshold=0.95).tolist() == [True, False, False]


def test_contamination_with_no_training_rows() -> None:
    assert not find_contaminated(np.ones((2, 3)), np.empty((0, 3))).any()
