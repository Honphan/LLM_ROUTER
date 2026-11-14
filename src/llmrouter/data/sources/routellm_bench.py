"""Graded benchmark responses published with the RouteLLM code -> benchmark schema.

Output columns are ``prompt, strong_score, weak_score``. The raw files hold one column
(MMLU, GSM8K) or one set of judgement rows (MT Bench) per model; which model
plays the strong and the weak role comes from the config.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from llmrouter.data.schema import BENCHMARK_COLUMNS, validate_benchmark

log = logging.getLogger(__name__)


def convert_correctness_table(
    raw: pd.DataFrame, strong_column: str, weak_column: str
) -> pd.DataFrame:
    """MMLU / GSM8K style table: one row per question, one boolean column per model."""
    missing = [c for c in ("prompt", strong_column, weak_column) if c not in raw.columns]
    if missing:
        raise ValueError(f"benchmark table is missing columns: {missing}")

    def as_score(column: str) -> np.ndarray:
        values = raw[column]
        if not (
            pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values)
        ):
            values = values.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
        return pd.to_numeric(values).to_numpy(dtype=np.float64)

    out = pd.DataFrame(
        {
            "prompt": raw["prompt"].astype(str).to_numpy(),
            "strong_score": as_score(strong_column),
            "weak_score": as_score(weak_column),
        }
    )
    return validate_benchmark(out[list(BENCHMARK_COLUMNS)])


def convert_mt_bench(
    questions: pd.DataFrame, judgements: pd.DataFrame, strong_model: str, weak_model: str
) -> pd.DataFrame:
    """MT Bench: prompt = first turn, score = mean judge score over the question's turns."""
    scores = (
        judgements.loc[judgements["model"].isin([strong_model, weak_model])]
        .groupby(["question_id", "model"])["score"]
        .mean()
        .unstack("model")
    )
    for model in (strong_model, weak_model):
        if model not in scores.columns:
            raise ValueError(f"no MT Bench judgements for model '{model}'")
    merged = questions.merge(scores, left_on="question_id", right_index=True, how="inner")
    merged = merged.dropna(subset=[strong_model, weak_model])
    log.info("mt_bench: %d questions, %d with both models judged", len(questions), len(merged))
    out = pd.DataFrame(
        {
            "prompt": merged["turns"].map(lambda turns: str(turns[0])).to_numpy(),
            "strong_score": merged[strong_model].to_numpy(dtype=np.float64),
            "weak_score": merged[weak_model].to_numpy(dtype=np.float64),
        }
    )
    return validate_benchmark(out[list(BENCHMARK_COLUMNS)])
