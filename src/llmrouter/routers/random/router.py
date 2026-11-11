"""Random router: the baseline every reported number is compared against.

Differences from the paper: none. Scores are uniform in [0, 1) and depend only on
(seed, prompt), so the same prompt always gets the same score regardless of batch order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from llmrouter.registry import ROUTERS
from llmrouter.routers.base import Router


@ROUTERS.register("random")
class RandomRouter(Router):
    uses_embeddings = False

    def strong_win_rate(self, prompts: Sequence[str]) -> NDArray[np.float64]:
        if len(prompts) == 0:
            return np.empty(0, dtype=np.float64)
        salt = f"{self.seed}\x00".encode()
        raw = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(salt + prompt.encode("utf-8"), digest_size=8).digest(),
                    "little",
                )
                for prompt in prompts
            ),
            dtype=np.uint64,
            count=len(prompts),
        )
        # Keep the top 53 bits so the result is an exact float64 in [0, 1).
        p = (raw >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        return self.check_probabilities(p, len(prompts))
