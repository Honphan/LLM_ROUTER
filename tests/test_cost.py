from __future__ import annotations

import pytest

from llmrouter.llm.cost import ModelPrice, blended_usd_per_mtok, cost_usd, routed_cost_usd

GPT4 = ModelPrice(10.0, 30.0)
MIXTRAL = ModelPrice(0.24, 0.24)


def test_paper_appendix_d_blended_prices() -> None:
    # (95 * 10 + 264 * 30) / (95 + 264) = 24.7 USD per million tokens
    assert blended_usd_per_mtok(GPT4, 95, 264) == pytest.approx(24.7, abs=0.05)
    assert blended_usd_per_mtok(MIXTRAL, 95, 264) == pytest.approx(0.24)


def test_cost_usd_and_routed_cost() -> None:
    assert cost_usd(GPT4, 1_000_000, 1_000_000) == pytest.approx(40.0)
    strong_only = cost_usd(GPT4, 95, 264)
    weak_only = cost_usd(MIXTRAL, 95, 264)
    assert routed_cost_usd(1.0, GPT4, MIXTRAL, 95, 264) == pytest.approx(strong_only)
    assert routed_cost_usd(0.0, GPT4, MIXTRAL, 95, 264) == pytest.approx(weak_only)
    assert routed_cost_usd(0.25, GPT4, MIXTRAL, 95, 264) == pytest.approx(
        0.25 * strong_only + 0.75 * weak_only
    )


def test_from_config_and_validation() -> None:
    price = ModelPrice.from_config({"name": "x", "input_usd_per_mtok": 1, "output_usd_per_mtok": 2})
    assert price == ModelPrice(1.0, 2.0)
    with pytest.raises(ValueError):
        blended_usd_per_mtok(price, 0, 0)
    with pytest.raises(ValueError):
        routed_cost_usd(1.2, price, price, 1, 1)
