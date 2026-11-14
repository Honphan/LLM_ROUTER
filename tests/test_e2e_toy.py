"""End-to-end run of the CLI (cli.py + pipeline.py) on synthetic data, fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from llmrouter.cli import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    config = {
        "include": [
            str(REPO / "configs" / "embeddings" / "hashing.yaml"),
            str(REPO / "configs" / "models" / "toy.yaml"),
        ],
        "name": "random_toy",
        "seed": 0,
        "output_dir": str(tmp_path / "experiments"),
        "embedder": {"cache_dir": str(tmp_path / "cache")},
        "data": {
            "sources": [{"path": str(tmp_path / "data" / "toy_battles.parquet")}],
            "min_prompt_chars": 16,
            "n_tiers": 10,
            "calibration_fraction": 0.2,
            "toy": {
                "battles_path": str(tmp_path / "data" / "toy_battles.parquet"),
                "n_battles": 500,
                "n_benchmark": 150,
            },
        },
        "router": {"name": "random", "params": {}},
        "eval": {
            "benchmark_name": "toy",
            "benchmark_path": str(tmp_path / "data" / "bench_toy.parquet"),
            "random_seeds": 20,
        },
    }
    path = tmp_path / "random_toy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> object:
    capsys.readouterr()
    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_full_cycle(config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = ["--config", str(config_path)]
    experiment = tmp_path / "experiments" / "random_toy"

    toy = _run(capsys, "toy-data", *config)
    assert (toy["n_battles"], toy["n_benchmark"]) == (500, 150)

    info = _run(capsys, "info", *config)
    assert info["n_battles"] == 500 and info["n_models"] == 6
    assert info["strong"] == {"model": "toy-strong", "tier": 0}
    assert info["weak"]["tier"] > 0
    assert info["n_fit"] + info["n_calibration"] == 500

    _run(capsys, "fit", *config)
    assert (experiment / "router" / "meta.json").exists()
    resolved = json.loads((experiment / "config.resolved.json").read_text(encoding="utf-8"))
    assert resolved["embedder"]["name"] == "hashing" and "include" not in resolved

    calibrated = _run(capsys, "calibrate", *config, "--pct", "0.3", "--pct", "0.5", "--pct", "0.8")
    thresholds = calibrated["thresholds"]
    assert set(thresholds) == {"0.3", "0.5", "0.8"}
    assert thresholds["0.3"]["alpha"] > thresholds["0.5"]["alpha"] > thresholds["0.8"]["alpha"]
    for rate, entry in thresholds.items():
        assert entry["achieved_rate"] == pytest.approx(float(rate), abs=0.02)
    assert (experiment / "thresholds.json").exists()

    summary = _run(capsys, "evaluate", *config)
    assert summary["router"] == "random" and summary["benchmark"] == "toy"
    assert summary["n_questions"] + summary["n_contaminated_removed"] == 150
    assert summary["n_questions"] > 100
    assert summary["random"]["n_seeds"] == 20
    assert abs(summary["random"]["apgr_mean"] - 0.5) < 0.05
    assert abs(summary["apgr"] - 0.5) < 0.15
    assert summary["cost"]["saving_ratio_at_cpt_50"] > 1.0
    saved = json.loads((experiment / "eval" / "toy" / "summary.json").read_text(encoding="utf-8"))
    assert saved["apgr"] == summary["apgr"] and len(saved["curve"]["pgr"]) == 11
    per_question = pd.read_parquet(experiment / "eval" / "toy" / "per_question.parquet")
    assert len(per_question) == summary["n_questions"]

    routed = _run(
        capsys,
        "route",
        *config,
        "--pct",
        "0.5",
        "--explain",
        "--prompt",
        "prove the theorem about matrix eigenvalue",
        "--prompt",
        "recipe for garlic butter pasta soup",
    )
    assert len(routed) == 2
    for row in routed:
        assert row["model"] in {"toy-strong", "toy-weak"}
        assert (row["model"] == "toy-strong") == (row["p_strong"] >= row["alpha"])

    report = _run(capsys, "report", *config)
    assert report[0]["benchmark"] == "toy" and report[0]["apgr"] == summary["apgr"]


def test_set_override_changes_the_run(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    toy = _run(capsys, "toy-data", "--config", str(config_path), "--set", "data.toy.n_battles=120")
    assert toy["n_battles"] == 120


def test_errors_return_exit_code_1(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["toy-data", "--config", str(config_path)]) == 0
    assert main(["evaluate", "--config", str(config_path)]) == 1  # not fitted yet
    assert main(["fit", "--config", str(config_path), "--set", "router.name=missing"]) == 1
    assert main(["fit", "--config", str(config_path), "--set", "models.weak.name=nope"]) == 1


def test_sw_ranking_cycle_beats_random(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = ["--config", str(config_path), "--set", "name=sw_toy"]
    config += ["--set", "router.name=sw_ranking", "--set", "router.params.gamma=10"]
    _run(capsys, "toy-data", *config)
    _run(capsys, "fit", *config)
    assert (tmp_path / "experiments" / "sw_toy" / "router" / "state.npz").exists()
    _run(capsys, "calibrate", *config, "--pct", "0.5")
    summary = _run(capsys, "evaluate", *config)
    assert summary["router"] == "sw_ranking"
    assert summary["apgr"] > summary["random"]["apgr_mean"] + 0.15
    assert summary["cpt_50"] < summary["random"]["cpt_50_mean"]
    routed = _run(capsys, "route", *config, "--pct", "0.5", "--explain", "--prompt", "prove it")
    assert 0.0 <= routed[0]["max_similarity"] <= 1.0
