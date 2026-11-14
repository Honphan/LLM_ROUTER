"""Tests for raw-data conversion and download (data/sources, data/prepare.py, data/download.py).

All rows are hand-written; no real dataset content and no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llmrouter.data.download import MANIFEST_FILE, download_files, expand_downloads
from llmrouter.data.filters import filter_battles
from llmrouter.data.prepare import convert_all
from llmrouter.data.schema import BATTLE_COLUMNS
from llmrouter.data.sources.arena import convert_arena, first_turn
from llmrouter.data.sources.routellm_bench import convert_correctness_table, convert_mt_bench


def _arena_raw() -> pd.DataFrame:
    rows = [
        ("m1", "m2", json.dumps(["  explain how a hash map works  ", "and then?"]), 1, 0, 0),
        ("m2", "m3", json.dumps(["write a haiku about the sea"]), 0, 1, 0),
        ("m1", "m3", json.dumps(["compare tcp and udp briefly"]), 0, 0, 1),
        ("m3", "m1", json.dumps(["hi"]), 1, 0, 0),
        ("m2", "m2", json.dumps(["a model against itself here"]), 1, 0, 0),
    ]
    columns = ["model_a", "model_b", "prompt", "winner_model_a", "winner_model_b", "winner_tie"]
    return pd.DataFrame(rows, columns=columns)


def test_first_turn_handles_odd_cells() -> None:
    assert first_turn(json.dumps([" first ", "second"])) == "first"
    assert first_turn("not json at all") == "not json at all"
    assert first_turn(json.dumps([None])) == "" and first_turn(json.dumps([])) == ""
    assert first_turn(float("nan")) == ""
    assert first_turn(json.dumps(["bad \ud800 char"])) == "bad ? char"


def test_convert_arena_schema_winners_and_filter() -> None:
    battles = convert_arena(_arena_raw())
    assert tuple(battles.columns) == BATTLE_COLUMNS
    assert len(battles) == 4  # the self-play row is dropped
    assert battles["winner"].tolist() == ["model_a", "model_b", "tie", "model_a"]
    assert battles["prompt"].iloc[0] == "explain how a hash map works"
    assert set(battles["source"]) == {"arena"} and set(battles["weight"]) == {1.0}
    assert len(filter_battles(battles, 16)) == 3  # "hi" is too short
    with pytest.raises(ValueError, match="missing columns"):
        convert_arena(_arena_raw().drop(columns=["winner_tie"]))


def test_convert_correctness_table() -> None:
    raw = pd.DataFrame(
        {
            "prompt": ["q one", "q two", "q three"],
            "big": [True, True, False],
            "small": ["True", "False", "False"],
        }  # noqa: E501
    )
    bench = convert_correctness_table(raw, "big", "small")
    assert bench["strong_score"].tolist() == [1.0, 1.0, 0.0]
    assert bench["weak_score"].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="missing columns"):
        convert_correctness_table(raw, "big", "absent")


def _mt_bench_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = pd.DataFrame(
        {
            "question_id": [1, 2, 3],
            "turns": [["first a", "then a"], ["first b", "then b"], ["c", "c2"]],
        }  # noqa: E501
    )
    judgements = pd.DataFrame(
        [
            (1, "big", 1, 10), (1, "big", 2, 8), (1, "small", 1, 6), (1, "small", 2, 4),
            (2, "big", 1, 9), (2, "big", 2, 9), (2, "small", 1, 9), (2, "small", 2, 7),
            (3, "big", 1, 5), (3, "other", 1, 1),
        ],
        columns=["question_id", "model", "turn", "score"],
    )  # fmt: skip
    return questions, judgements


def test_convert_mt_bench_averages_turns_and_needs_both_models() -> None:
    bench = convert_mt_bench(*_mt_bench_raw(), "big", "small")
    assert bench["prompt"].tolist() == ["first a", "first b"]  # question 3 lacks "small"
    assert bench["strong_score"].tolist() == [9.0, 9.0]
    assert bench["weak_score"].tolist() == [5.0, 8.0]
    with pytest.raises(ValueError, match="no MT Bench judgements"):
        convert_mt_bench(*_mt_bench_raw(), "big", "absent")


def test_expand_downloads_both_forms() -> None:
    pairs = expand_downloads(
        [
            {"url": "http://x/a.csv", "path": "one/a.csv"},
            {"base_url": "http://y/dir/", "dir": "two", "files": ["b.csv", "c.csv"]},
        ]
    )
    assert pairs == [
        ("http://x/a.csv", "one/a.csv"),
        ("http://y/dir/b.csv", "two/b.csv"),
        ("http://y/dir/c.csv", "two/c.csv"),
    ]


def test_download_and_convert_from_local_files(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    _arena_raw().to_csv(remote / "train.csv", index=False)
    pd.DataFrame(
        {"prompt": ["q one", "q two"], "big": [True, False], "small": [False, False]}
    ).to_csv(  # noqa: E501
        remote / "part.csv", index=False
    )
    questions, judgements = _mt_bench_raw()
    questions.to_json(remote / "question.jsonl", orient="records", lines=True)
    judgements.to_json(remote / "judgements.jsonl", orient="records", lines=True)

    raw_dir = tmp_path / "raw"
    specs = [
        {"url": (remote / "train.csv").as_uri(), "path": "arena/train.csv"},
        {
            "base_url": remote.as_uri(),
            "dir": "bench",
            "files": ["part.csv", "question.jsonl", "judgements.jsonl"],
        },
    ]
    assert download_files(specs, raw_dir)["downloaded"] == 4
    manifest = json.loads((raw_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert set(manifest) == {
        "arena/train.csv",
        "bench/part.csv",
        "bench/question.jsonl",
        "bench/judgements.jsonl",
    }
    assert len(manifest["arena/train.csv"]["sha256"]) == 64
    again = download_files(specs, raw_dir)  # raw files are never fetched twice
    assert (again["downloaded"], again["already_present"]) == (0, 4)

    out = tmp_path / "processed"
    prepare = {
        "battles": [
            {"kind": "arena_55k", "raw": "arena/train.csv", "output": str(out / "arena.parquet")}
        ],
        "benchmarks": [
            {
                "kind": "correctness_csv",
                "raw": "bench",
                "strong_column": "big",
                "weak_column": "small",
                "output": str(out / "bench_a.parquet"),
            },
            {
                "kind": "mt_bench",
                "questions": "bench/question.jsonl",
                "judgements": "bench/judgements.jsonl",
                "strong_column": "big",
                "weak_column": "small",
                "output": str(out / "bench_b.parquet"),
            },
        ],
    }
    written = convert_all(prepare, raw_dir)
    assert [entry["rows"] for entry in written] == [4, 2, 2]
    assert written[1]["strong_mean"] == 0.5 and written[2]["weak_mean"] == 6.5
    assert len(pd.read_parquet(out / "arena.parquet")) == 4
    with pytest.raises(ValueError, match="unknown data kind"):
        convert_all({"battles": [{"kind": "nope", "output": "x"}]}, raw_dir)
