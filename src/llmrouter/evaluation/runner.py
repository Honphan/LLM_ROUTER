"""Score a router's outputs on a benchmark and write the result files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from llmrouter.evaluation import metrics

log = logging.getLogger(__name__)


def summarise(
    scores: ArrayLike,
    benchmark: pd.DataFrame,
    random_seeds: int = 20,
    seed: int = 0,
    n_levels: int = 10,
) -> dict[str, Any]:
    """Compute APGR, CPT(50%), CPT(80%), the call-performance curve and the random baseline."""
    strong = benchmark["strong_score"].to_numpy(dtype=np.float64)
    weak = benchmark["weak_score"].to_numpy(dtype=np.float64)
    fractions, pgr = metrics.call_performance(scores, strong, weak, n_levels)
    return {
        "n_questions": int(len(benchmark)),
        "strong_mean": float(strong.mean()),
        "weak_mean": float(weak.mean()),
        "apgr": metrics.apgr(scores, strong, weak, n_levels),
        "cpt_50": metrics.cpt(scores, strong, weak, 0.5),
        "cpt_80": metrics.cpt(scores, strong, weak, 0.8),
        "curve": {"call_fraction": fractions.tolist(), "pgr": pgr.tolist()},
        "random": metrics.random_baseline(strong, weak, random_seeds, seed, n_levels),
    }


def write_results(
    out_dir: str | Path,
    summary: dict[str, Any],
    scores: ArrayLike,
    benchmark: pd.DataFrame,
) -> Path:
    """Write ``summary.json`` and ``per_question.parquet``; return the summary path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    per_question = benchmark[["prompt", "strong_score", "weak_score"]].copy()
    per_question["p_strong"] = np.asarray(scores, dtype=np.float64)
    per_question.to_parquet(directory / "per_question.parquet", index=False)
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote %s", summary_path)
    return summary_path
