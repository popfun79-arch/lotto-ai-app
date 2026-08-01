from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from lotto64.analysis.gap import row_numbers
from lotto64.recommend.filters import hard_filter
from lotto64.utils.lotto_math import (
    ac_value, consecutive_pairs, end_digit_sum, low_count,
    neighbor_set, odd_count, prime_count, zone_counts,
)

def score_combo(combo: Sequence[int], scores: pd.DataFrame, df: pd.DataFrame) -> dict:
    numbers = tuple(sorted(map(int, combo)))
    score_map = dict(zip(scores["number"], scores["final_score"]))
    zones = zone_counts(numbers)
    latest = set(row_numbers(df.iloc[-1]))

    total = sum(numbers)
    end_sum = end_digit_sum(numbers)
    odd = odd_count(numbers)
    low = low_count(numbers)
    ac = ac_value(numbers)
    carry = len(set(numbers) & latest)
    neighbor = len(set(numbers) & neighbor_set(latest))
    missing = sum(v == 0 for v in zones)

    score = float(np.mean([score_map[n] for n in numbers]))
    score += 0.08 if 125 <= total <= 165 else 0.02
    score += 0.06 if 22 <= end_sum <= 36 else 0.01
    score += 0.06 if odd in (2, 3, 4) else 0.0
    score += 0.06 if low in (2, 3, 4) else 0.0
    score += 0.07 if ac in (7, 8, 9, 10) else 0.02
    score += 0.05 if missing in (0, 1) else 0.01
    score += 0.05 if carry in (0, 1, 2) else 0.01
    score += 0.05 if neighbor in (1, 2, 3) else 0.01

    return {
        "combination": numbers,
        "final_score": score,
        "sum": total,
        "end_digit_sum": end_sum,
        "odd_count": odd,
        "low_count": low,
        "ac": ac,
        "prime_count": prime_count(numbers),
        "consecutive_pairs": consecutive_pairs(numbers),
        "zone_pattern": "-".join(map(str, zones)),
        "missing_zones": missing,
        "carryover_count": carry,
        "neighbor_count": neighbor,
    }

def generate_ranked(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_count: int = 18,
    limit: int = 180000,
    seed: int = 20260720,
) -> pd.DataFrame:
    candidates = scores.head(candidate_count)["number"].astype(int).tolist()
    possible = math.comb(len(candidates), 6)

    if possible <= limit:
        combos = list(combinations(candidates, 6))
    else:
        rng = random.Random(seed)
        pool = set()
        while len(pool) < limit:
            pool.add(tuple(sorted(rng.sample(candidates, 6))))
        combos = list(pool)

    rows = [score_combo(c, scores, df) for c in combos if hard_filter(c)]
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["final_score", "combination"],
        ascending=[False, True],
    ).reset_index(drop=True)
