from __future__ import annotations

import numpy as np
import pandas as pd

from lotto64.analysis.gap import build_gap_tables, row_numbers
from lotto64.utils.lotto_math import (
    ac_value, consecutive_pairs, end_digit_sum, low_count,
    neighbor_set, odd_count, prime_count, zone_counts,
)

DNA_FEATURES = [
    "sum", "end_digit_sum", "odd_count", "low_count", "prime_count", "ac",
    "consecutive_pairs", "used_zones", "missing_zones",
    "zone_1_10", "zone_11_20", "zone_21_30", "zone_31_40", "zone_41_45",
    "carryover_count", "neighbor_count", "bonus_window_count",
    "gap_sum", "gap_mean", "gap_max", "gap_std",
    "gap_le_5", "gap_le_10", "gap_gt_10", "gap_ge_17",
]

def build_dna(df: pd.DataFrame) -> pd.DataFrame:
    _, gap_rounds = build_gap_tables(df)
    rows = []

    for i, row in df.iterrows():
        current = row_numbers(row)
        previous = row_numbers(df.iloc[i - 1]) if i > 0 else []
        previous_bonus = int(df.iloc[i - 1]["bonus"]) if i > 0 else 0
        zones = zone_counts(current)
        gap = gap_rounds.iloc[i]

        rows.append({
            "round": int(row["round"]),
            "sum": sum(current),
            "end_digit_sum": end_digit_sum(current),
            "odd_count": odd_count(current),
            "low_count": low_count(current),
            "prime_count": prime_count(current),
            "ac": ac_value(current),
            "consecutive_pairs": consecutive_pairs(current),
            "used_zones": sum(v > 0 for v in zones),
            "missing_zones": sum(v == 0 for v in zones),
            "zone_1_10": zones[0], "zone_11_20": zones[1], "zone_21_30": zones[2],
            "zone_31_40": zones[3], "zone_41_45": zones[4],
            "carryover_count": len(set(current) & set(previous)),
            "neighbor_count": len(set(current) & neighbor_set(previous)) if previous else 0,
            "bonus_window_count": (
                sum(abs(n - previous_bonus) <= 2 for n in current)
                if previous_bonus else 0
            ),
            "gap_sum": gap["gap_sum"], "gap_mean": gap["gap_mean"],
            "gap_max": gap["gap_max"], "gap_std": gap["gap_std"],
            "gap_le_5": gap["gap_le_5"], "gap_le_10": gap["gap_le_10"],
            "gap_gt_10": gap["gap_gt_10"], "gap_ge_17": gap["gap_ge_17"],
        })

    out = pd.DataFrame(rows)
    out["gap_sum_ma5"] = out["gap_sum"].rolling(5, min_periods=3).mean()
    out["gap_sum_ma10"] = out["gap_sum"].rolling(10, min_periods=5).mean()
    out["gap_sum_slope5"] = out["gap_sum"].rolling(5, min_periods=5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
        raw=False,
    )
    return out

def classify_states(dna: pd.DataFrame) -> pd.DataFrame:
    out = dna.copy()
    q25 = out["gap_sum"].quantile(0.25)
    q75 = out["gap_sum"].quantile(0.75)
    q90 = out["gap_sum"].quantile(0.90)

    def classify(row: pd.Series) -> str:
        if row["gap_sum"] <= q25 and row["gap_max"] <= 10:
            return "COMPRESSION"
        if row["gap_sum"] >= q90 or row["gap_max"] >= 17:
            return "EXTREME_EXPANSION"
        if row["gap_sum"] >= q75:
            return "EXPANSION"
        return "NORMAL"

    out["gap_state"] = out.apply(classify, axis=1)
    return out
