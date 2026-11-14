"""Hold out part of the battles for threshold calibration."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def split_calibration(
    battles: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split battles into (fit, calibration) by unique prompt, so no prompt is in both."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"calibration fraction must be in [0, 1), got {fraction}")
    prompts = np.sort(battles["prompt"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(prompts)
    n_calibration = int(round(fraction * len(shuffled)))
    held_out = set(shuffled[:n_calibration].tolist())
    is_calibration = battles["prompt"].isin(held_out).to_numpy()
    fit = battles.loc[~is_calibration].reset_index(drop=True)
    calibration = battles.loc[is_calibration].reset_index(drop=True)
    log.info("split: %d fit battles, %d calibration battles", len(fit), len(calibration))
    return fit, calibration
