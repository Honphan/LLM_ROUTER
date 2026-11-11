"""Token cost arithmetic (RouteLLM appendix D). Prices always come from ``configs/models/``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million input / output tokens."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float

    @classmethod
    def from_config(cls, model: Mapping[str, Any]) -> ModelPrice:
        return cls(
            input_usd_per_mtok=float(model["input_usd_per_mtok"]),
            output_usd_per_mtok=float(model["output_usd_per_mtok"]),
        )


def cost_usd(price: ModelPrice, tokens_in: float, tokens_out: float) -> float:
    """Cost of one call (or a batch) with the given token counts."""
    return (tokens_in * price.input_usd_per_mtok + tokens_out * price.output_usd_per_mtok) / 1e6


def blended_usd_per_mtok(price: ModelPrice, tokens_in: float, tokens_out: float) -> float:
    """Average price per million tokens for a given input/output token mix."""
    total = tokens_in + tokens_out
    if total <= 0:
        raise ValueError("token counts must sum to a positive number")
    return cost_usd(price, tokens_in, tokens_out) / total * 1e6


def routed_cost_usd(
    strong_fraction: float,
    strong: ModelPrice,
    weak: ModelPrice,
    tokens_in: float,
    tokens_out: float,
) -> float:
    """Expected cost per query when ``strong_fraction`` of queries go to the strong model."""
    if not 0.0 <= strong_fraction <= 1.0:
        raise ValueError("strong_fraction must be in [0, 1]")
    return strong_fraction * cost_usd(strong, tokens_in, tokens_out) + (
        1.0 - strong_fraction
    ) * cost_usd(weak, tokens_in, tokens_out)
