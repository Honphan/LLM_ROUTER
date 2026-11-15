"""GPT-4-judge battles (``routellm/gpt4_judge_battles``) -> standard battle schema.

The RouteLLM paper (section 4.1.2) augments Arena with prompts answered by both models of
the routing pair and labelled by a GPT-4 judge. The raw layout is the same as the Arena
55k release (``model_a, model_b, prompt, winner_model_a, winner_model_b, winner_tie`` with
``prompt`` a JSON list of user turns), so the Arena converter is reused.

The labels do not mean the same thing as an Arena vote: ``model_b`` (the weak model) "wins"
when the judge finds its answer good enough, which is the majority of rows. Global ratings
fit on Arena plus this source therefore rank the weak model above the strong one.
"""

from __future__ import annotations

import pandas as pd

from llmrouter.data.sources.arena import convert_arena

GPT4_JUDGE_SOURCE = "gpt4_judge"


def convert_gpt4_judge(raw: pd.DataFrame, source: str = GPT4_JUDGE_SOURCE) -> pd.DataFrame:
    """Convert raw judge rows to ``prompt, model_a, model_b, winner, source, weight``."""
    return convert_arena(raw, source=source)
