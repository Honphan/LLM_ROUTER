"""Chatbot Arena human-preference battles (the 55k Kaggle release) -> standard battle schema.

Raw columns: ``model_a, model_b, prompt, winner_model_a, winner_model_b, winner_tie`` where
``prompt`` is a JSON list with one string per user turn. Like the RouteLLM reference code,
only the first user turn is kept.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from llmrouter.data.schema import BATTLE_COLUMNS, validate_battles

log = logging.getLogger(__name__)

ARENA_RAW_COLUMNS: tuple[str, ...] = (
    "model_a",
    "model_b",
    "prompt",
    "winner_model_a",
    "winner_model_b",
    "winner_tie",
)


def first_turn(raw_prompt: object) -> str:
    """First user turn of a raw ``prompt`` cell (JSON list of turns), stripped.

    Text that cannot be encoded as UTF-8 (lone surrogates in the raw dump) is replaced so
    the result can always be written to parquet.
    """
    if not isinstance(raw_prompt, str):
        return ""
    try:
        turns = json.loads(raw_prompt)
    except json.JSONDecodeError:
        turns = raw_prompt
    if isinstance(turns, list):
        turns = turns[0] if turns else ""
    if not isinstance(turns, str):
        return ""
    return turns.encode("utf-8", "replace").decode("utf-8").strip()


def convert_arena(raw: pd.DataFrame, source: str = "arena") -> pd.DataFrame:
    """Convert raw Arena rows to ``prompt, model_a, model_b, winner, source, weight``."""
    missing = [column for column in ARENA_RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"arena data is missing columns: {missing}")
    a_wins = pd.to_numeric(raw["winner_model_a"]).to_numpy() == 1
    b_wins = pd.to_numeric(raw["winner_model_b"]).to_numpy() == 1
    out = pd.DataFrame(
        {
            "prompt": raw["prompt"].map(first_turn).to_numpy(),
            "model_a": raw["model_a"].astype(str).to_numpy(),
            "model_b": raw["model_b"].astype(str).to_numpy(),
            "winner": np.where(a_wins, "model_a", np.where(b_wins, "model_b", "tie")),
            "source": source,
            "weight": 1.0,
        }
    )
    keep = (out["model_a"] != out["model_b"]) & (out["prompt"].str.len() > 0)
    log.info("arena: %d raw rows -> %d battles (dropped self-play / empty)", len(out), keep.sum())
    return validate_battles(out.loc[keep, list(BATTLE_COLUMNS)])
