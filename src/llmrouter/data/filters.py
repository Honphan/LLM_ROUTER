"""Row filters applied to battles before fitting (RouteLLM section 5, training data)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def first_user_turn(conversation: str | Sequence[Mapping[str, Any]]) -> str:
    """Return the first user message of a conversation, or the string itself."""
    if isinstance(conversation, str):
        return conversation
    for turn in conversation:
        if turn.get("role") == "user":
            return str(turn.get("content", ""))
    return ""


def filter_battles(battles: pd.DataFrame, min_prompt_chars: int = 16) -> pd.DataFrame:
    """Drop battles whose prompt is shorter than ``min_prompt_chars`` characters."""
    before = len(battles)
    keep = battles["prompt"].str.strip().str.len() >= min_prompt_chars
    out = battles.loc[keep].reset_index(drop=True)
    log.info("filter min_prompt_chars=%d: %d -> %d battles", min_prompt_chars, before, len(out))
    return out
