from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from lotto64.config import DEFAULT_WEIGHTS

def bayesian_style_weight_search(
    objective: Callable[[Dict[str, float]], float],
    trials: int = 20,
    seed: int = 20260720,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    외부 최적화 라이브러리 없이 재현 가능한 확률적 탐색을 수행합니다.
    초기 무작위 탐색 후 상위 가중치 주변을 점진적으로 좁혀 탐색합니다.
    """
    rng = np.random.default_rng(seed)
    best = dict(DEFAULT_WEIGHTS)
    best_score = -np.inf
    history = []

    center = np.array(list(DEFAULT_WEIGHTS.values()), dtype=float)
    keys = list(DEFAULT_WEIGHTS.keys())

    for trial in range(trials):
        scale = max(0.05, 0.25 * (1 - trial / max(1, trials)))
        proposal_vector = np.clip(
            rng.normal(center, scale * center + 0.005),
            0.001,
            None,
        )
        proposal = dict(zip(keys, proposal_vector))
        score = float(objective(proposal))
        history.append({"trial": trial + 1, "objective": score, **proposal})

        if score > best_score:
            best_score = score
            best = proposal
            center = proposal_vector

    history_df = pd.DataFrame(history).sort_values("objective", ascending=False).reset_index(drop=True)
    return best, history_df
