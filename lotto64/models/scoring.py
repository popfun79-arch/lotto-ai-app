from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from lotto64.analysis.dna import build_dna, classify_states
from lotto64.analysis.gap import current_gap_table, row_numbers
from lotto64.analysis.similarity import similar_rounds
from lotto64.config import DEFAULT_WEIGHTS, NUMBERS
from lotto64.utils.lotto_math import neighbor_set, zone_counts, zone_index

def _minmax(series: pd.Series) -> pd.Series:
    low, high = float(series.min()), float(series.max())
    if math.isclose(low, high):
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low)

def pair_counts(df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for _, row in df.iterrows():
        for pair in combinations(sorted(row_numbers(row)), 2):
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def number_scores(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    egr_threshold: int = 17,
    similarity_k: int = 15,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    recent30 = df.tail(min(30, len(df)))
    recent100 = df.tail(min(100, len(df)))

    long_counts = pd.Series(0.0, index=NUMBERS)
    short_counts = pd.Series(0.0, index=NUMBERS)

    for _, row in df.iterrows():
        for number in row_numbers(row):
            long_counts[number] += 1

    for _, row in recent30.iterrows():
        for number in row_numbers(row):
            short_counts[number] += 1

    long_norm, short_norm = _minmax(long_counts), _minmax(short_counts)
    gaps = current_gap_table(df).set_index("number")
    latest = set(row_numbers(df.iloc[-1]))
    latest_neighbors = neighbor_set(latest)
    latest_bonus = int(df.iloc[-1]["bonus"])

    pair_raw = {n: 0 for n in NUMBERS}
    for pair, count in pair_counts(recent100).items():
        for number in pair:
            pair_raw[number] += count
    max_pair = max(pair_raw.values()) or 1

    dna = build_dna(df)
    sim = similar_rounds(dna, similarity_k)
    follow = {n: 0.0 for n in NUMBERS}
    if not sim.empty:
        by_round = {int(r["round"]): row_numbers(r) for _, r in df.iterrows()}
        for _, item in sim.iterrows():
            weight = 1 / (1 + float(item["distance"]))
            for number in by_round.get(int(item["next_round"]), []):
                follow[number] += weight
    max_follow = max(follow.values()) or 1

    state = classify_states(dna).iloc[-1]["gap_state"]
    latest_zones = zone_counts(list(latest))

    rows = []
    for number in NUMBERS:
        current_gap = int(gaps.loc[number, "current_gap"])
        egr_score = 1.0 if current_gap >= egr_threshold else (
            0.6 if current_gap >= egr_threshold - 5 else 0.2
        )
        state_score = 0.5
        if state == "COMPRESSION" and current_gap > 10:
            state_score = 1.0
        elif state == "EXTREME_EXPANSION" and current_gap <= 10:
            state_score = 1.0

        components = {
            "long_frequency": float(long_norm.loc[number]),
            "short_frequency": float(short_norm.loc[number]),
            "gap": float(gaps.loc[number, "gap_percentile"]),
            "egr": egr_score,
            "carry": 1.0 if number in latest else 0.25,
            "neighbor": 1.0 if number in latest_neighbors else 0.25,
            "bonus_window": 1.0 if latest_bonus and abs(number - latest_bonus) <= 2 else 0.25,
            "pair": pair_raw[number] / max_pair,
            "zone_recovery": 1.0 if latest_zones[zone_index(number)] == 0 else 0.5,
            "dna_similarity": follow[number] / max_follow,
            "state": state_score,
            "hot_cold": 0.75 if short_counts[number] <= 3 else 0.45,
        }

        total_weight = sum(weights.values())
        final_score = sum(weights[key] * components[key] for key in weights) / total_weight
        rows.append({
            "number": number,
            "final_score": final_score,
            "current_gap": current_gap,
            **{f"score_{key}": value for key, value in components.items()},
        })

    out = pd.DataFrame(rows).sort_values(
        ["final_score", "number"],
        ascending=[False, True],
    ).reset_index(drop=True)

    shifted = out["final_score"] - out["final_score"].min() + 1e-9
    out["relative_probability"] = shifted / shifted.sum()
    out["relative_probability_pct"] = out["relative_probability"] * 100
    out["grade"] = pd.qcut(
        out["final_score"].rank(method="first"),
        5,
        labels=["E", "D", "C", "B", "A"],
    )
    return out

def explain_number(score_row: pd.Series, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for key, weight in weights.items():
        component = float(score_row[f"score_{key}"])
        rows.append({
            "feature": key,
            "component_score": component,
            "weight": weight,
            "contribution": component * weight,
        })
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)
