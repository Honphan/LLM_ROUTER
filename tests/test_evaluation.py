"""Tests for llmrouter.evaluation: metrics, calibrate, runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmrouter.evaluation import metrics, runner
from llmrouter.evaluation.calibrate import achieved_rate, calibrate, threshold_for_rate

# Hand-worked example, 4 questions. Ranked by score: q0, q2, q1, q3.
# Strong is always right; weak is right on q1 and q3 only.
#   strong calls k:  0     1     2     3     4
#   quality:         0.50  0.75  1.00  1.00  1.00
#   PGR:             0.00  0.50  1.00  1.00  1.00
SCORES = [0.9, 0.1, 0.8, 0.2]
STRONG = [1, 1, 1, 1]
WEAK = [0, 1, 0, 1]


def test_curves_match_hand_computation() -> None:
    assert metrics.quality_curve(SCORES, STRONG, WEAK) == pytest.approx([0.5, 0.75, 1, 1, 1])
    assert metrics.pgr_curve(SCORES, STRONG, WEAK) == pytest.approx([0, 0.5, 1, 1, 1])


def test_pgr_cpt_apgr_match_hand_computation() -> None:
    assert metrics.pgr_at(SCORES, STRONG, WEAK, 0.25) == pytest.approx(0.5)
    assert metrics.pgr_at(SCORES, STRONG, WEAK, 1.0) == pytest.approx(1.0)
    assert metrics.cpt(SCORES, STRONG, WEAK, 0.5) == pytest.approx(0.25)
    assert metrics.cpt(SCORES, STRONG, WEAK, 0.8) == pytest.approx(0.5)
    # Levels 0, 0.1, ..., 1.0 -> k = 0,0,1,1,2,2,2,3,3,4,4 -> PGR = 0,0,.5,.5,1,1,1,1,1,1,1
    # Trapezoids: 0 + .25 + .5 + .75 + 6 * 1 = 7.5, divided by 10 intervals.
    assert metrics.apgr(SCORES, STRONG, WEAK) == pytest.approx(0.75)


def test_perfect_and_worst_router_bracket_random() -> None:
    rng = np.random.default_rng(0)
    strong = rng.random(400)
    weak = strong - rng.random(400)
    gain = strong - weak
    best = metrics.apgr(gain, strong, weak)
    worst = metrics.apgr(-gain, strong, weak)
    assert worst < 0.5 < best
    assert best + worst == pytest.approx(1.0, abs=0.02)


def test_ties_break_by_input_order() -> None:
    flat = [0.5, 0.5, 0.5, 0.5]
    assert metrics.pgr_curve(flat, STRONG, WEAK) == pytest.approx([0, 0.5, 0.5, 1, 1])


def test_random_scores_give_apgr_half() -> None:
    rng = np.random.default_rng(1)
    strong = (rng.random(2000) < 0.85).astype(float)
    weak = (rng.random(2000) < 0.6).astype(float)
    baseline = metrics.random_baseline(strong, weak, n_seeds=30, seed=0)
    assert baseline["n_seeds"] == 30
    assert abs(baseline["apgr_mean"] - 0.5) <= max(baseline["apgr_ci95"], 0.01)
    assert baseline["cpt_50_mean"] == pytest.approx(0.5, abs=0.05)
    assert baseline["cpt_80_mean"] == pytest.approx(0.8, abs=0.05)
    assert 0 < baseline["apgr_ci95"] < 0.05


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="same mean"):
        metrics.pgr_curve([0.1, 0.2], [1, 0], [0, 1])
    with pytest.raises(ValueError, match="equal length"):
        metrics.pgr_curve([0.1], [1, 0], [0, 1])
    with pytest.raises(ValueError, match="empty"):
        metrics.pgr_curve([], [], [])
    with pytest.raises(ValueError, match="NaN"):
        metrics.pgr_curve([np.nan, 0.1], [1, 1], [0, 0])
    with pytest.raises(ValueError, match="at least 2"):
        metrics.random_baseline(STRONG, WEAK, n_seeds=1)


# --------------------------------------------------------------------- calibrate
def test_threshold_hits_requested_rate() -> None:
    scores = np.linspace(0.0, 0.99, 100)
    for rate in (0.1, 0.3, 0.5, 0.8, 1.0):
        alpha = threshold_for_rate(scores, rate)
        assert achieved_rate(scores, alpha) == pytest.approx(rate)
    assert achieved_rate(scores, threshold_for_rate(scores, 0.0)) == 0.0


def test_threshold_with_ties_reports_achieved_rate() -> None:
    result = calibrate([0.5, 0.5, 0.5, 0.1], [0.25])
    assert result["0.25"]["alpha"] == 0.5
    assert result["0.25"]["achieved_rate"] == 0.75


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        threshold_for_rate([], 0.5)
    with pytest.raises(ValueError, match="strong_rate"):
        threshold_for_rate([0.1], 1.5)


# ------------------------------------------------------------------------ runner
def test_runner_summary_and_files(tmp_path: Path) -> None:
    bench = pd.DataFrame(
        {"prompt": [f"question {i}" for i in range(4)], "strong_score": STRONG, "weak_score": WEAK}
    )
    summary = runner.summarise(SCORES, bench, random_seeds=5, seed=0)
    assert summary["apgr"] == pytest.approx(0.75)
    assert (summary["cpt_50"], summary["cpt_80"]) == (0.25, 0.5)
    assert (summary["strong_mean"], summary["weak_mean"]) == (1.0, 0.5)
    assert len(summary["curve"]["pgr"]) == 11
    assert summary["random"]["n_seeds"] == 5

    path = runner.write_results(tmp_path / "eval" / "toy", summary, SCORES, bench)
    assert json.loads(path.read_text(encoding="utf-8"))["apgr"] == pytest.approx(0.75)
    per_question = pd.read_parquet(path.parent / "per_question.parquet")
    assert per_question["p_strong"].tolist() == SCORES
    assert list(per_question.columns) == ["prompt", "strong_score", "weak_score", "p_strong"]
