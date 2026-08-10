from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from lotto64.analysis.gap import build_gap_tables, current_gap_table
from lotto64.analysis.gap_sum_series import (
    forecast_next_gap_sum,
    gap_sum_pattern_score,
)
from lotto64.analysis.sum_series import forecast_next_sum, sum_pattern_score
from lotto64.models.pattern_master import (
    candidate_sets,
    gap_bucket,
    master_number_scores,
)
from lotto64.utils.lotto_math import (
    ac_value,
    low_count,
    max_consecutive_run,
    odd_count,
    zone_counts,
)


BUCKETS = ["0", "1-2", "3-5", "6-10", "11-16", "17+"]


def _feature_distributions(df: pd.DataFrame, window: int = 50) -> dict:
    recent = df.tail(min(window, len(df)))
    odd_values = Counter()
    low_values = Counter()
    zones_values = Counter()
    last_values = Counter()
    ac_values = Counter()

    for _, row in recent.iterrows():
        nums = [int(row[f"n{i}"]) for i in range(1, 7)]
        odd_values[odd_count(nums)] += 1
        low_values[low_count(nums)] += 1
        zones_values[sum(v > 0 for v in zone_counts(nums))] += 1
        last_values[len({n % 10 for n in nums})] += 1
        ac_values[ac_value(nums)] += 1

    def normalize(counter: Counter) -> dict:
        top = max(counter.values()) if counter else 1
        return {key: value / top for key, value in counter.items()}

    return {
        "odd": normalize(odd_values),
        "low": normalize(low_values),
        "zones": normalize(zones_values),
        "last": normalize(last_values),
        "ac": normalize(ac_values),
    }


def _gap_bucket_means(df: pd.DataFrame, window: int = 50) -> dict[str, float]:
    _, rounds = build_gap_tables(df)
    recent = rounds.tail(min(window, len(rounds)))
    rows = []

    for values in recent["gap_values"]:
        counts = Counter(
            gap_bucket(int(value))
            for value in values
            if pd.notna(value)
        )
        rows.append({bucket: counts[bucket] for bucket in BUCKETS})

    if not rows:
        return {bucket: 1.0 for bucket in BUCKETS}

    frame = pd.DataFrame(rows)
    return frame.mean().to_dict()


def _bucket_composition_score(
    combo: tuple[int, ...],
    gap_map: dict[int, int],
    target: dict[str, float],
) -> float:
    counts = Counter(gap_bucket(gap_map[n]) for n in combo)
    distance = sum(
        abs(float(counts[bucket]) - float(target[bucket]))
        for bucket in BUCKETS
    )
    return float(max(0.0, 1.0 - distance / 12.0))


def _candidate_pattern_ok(
    combo: tuple[int, ...],
    gap_map: dict[int, int],
    sum_low: float,
    sum_high: float,
    gap_low: float,
    gap_high: float,
) -> bool:
    total = sum(combo)
    gap_total = sum(gap_map[n] for n in combo)
    buckets = Counter(gap_bucket(gap_map[n]) for n in combo)

    return (
        sum_low <= total <= sum_high
        and gap_low <= gap_total <= gap_high
        and max_consecutive_run(combo) < 4
        and odd_count(combo) not in (0, 6)
        and low_count(combo) not in (0, 6)
        and sum(v > 0 for v in zone_counts(combo)) >= 3
        and buckets["17+"] <= 1
        and buckets["0"] <= 1
        and buckets["6-10"] >= 1
        and buckets["3-5"] <= 2
        and buckets["11-16"] <= 2
    )


def rank_final_combinations(
    df: pd.DataFrame,
    pool_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    master = master_number_scores(df)
    pool = master.head(pool_size)["number"].astype(int).tolist()

    master_map = dict(zip(master["number"], master["master_score"]))
    pool_scores = np.asarray([master_map[n] for n in pool], dtype=float)
    low_score, high_score = float(pool_scores.min()), float(pool_scores.max())

    def normalized_number_score(number: int) -> float:
        if np.isclose(low_score, high_score):
            return 0.5
        return (master_map[number] - low_score) / (high_score - low_score)

    gaps = current_gap_table(df)
    gap_map = dict(zip(
        gaps["number"].astype(int),
        gaps["current_gap"].astype(int),
    ))

    sum_forecast = forecast_next_sum(
        df,
        state_window=50,
        transition_lookback=100,
    )
    gap_forecast = forecast_next_gap_sum(
        df,
        state_window=50,
        transition_lookback=100,
    )
    target_buckets = _gap_bucket_means(df, 50)
    feature_dist = _feature_distributions(df, 50)

    latest = {
        int(df.iloc[-1][f"n{i}"])
        for i in range(1, 7)
    }

    rows = []
    for combo in combinations(pool, 6):
        combo = tuple(sorted(combo))
        total = sum(combo)
        gap_total = sum(gap_map[n] for n in combo)

        if not _candidate_pattern_ok(
            combo,
            gap_map,
            sum_forecast.target_low,
            sum_forecast.target_high,
            gap_forecast.target_low,
            gap_forecast.target_high + 2,
        ):
            continue

        number_score = float(np.mean([
            normalized_number_score(n)
            for n in combo
        ]))
        draw_sum_score = sum_pattern_score(total, sum_forecast)
        gap_total_score = gap_sum_pattern_score(gap_total, gap_forecast)
        bucket_score = _bucket_composition_score(
            combo,
            gap_map,
            target_buckets,
        )

        odd = odd_count(combo)
        low = low_count(combo)
        used_zones = sum(v > 0 for v in zone_counts(combo))
        unique_last = len({n % 10 for n in combo})
        ac = ac_value(combo)
        carry = len(set(combo) & latest)

        balance_score = (
            feature_dist["odd"].get(odd, 0.0)
            + feature_dist["low"].get(low, 0.0)
        ) / 2
        zone_score = feature_dist["zones"].get(used_zones, 0.0)
        last_score = feature_dist["last"].get(unique_last, 0.0)
        ac_score = feature_dist["ac"].get(ac, 0.0)

        # 최근 GAP=0 hazard도 고려하되 과도한 이월 집중은 막는다.
        if carry == 1:
            carry_score = 1.0
        elif carry in (0, 2):
            carry_score = 0.80
        else:
            carry_score = 0.35

        final_score = (
            0.34 * number_score
            + 0.20 * draw_sum_score
            + 0.20 * gap_total_score
            + 0.09 * bucket_score
            + 0.06 * balance_score
            + 0.04 * zone_score
            + 0.025 * last_score
            + 0.025 * ac_score
            + 0.04 * carry_score
        )

        bucket_counts = Counter(gap_bucket(gap_map[n]) for n in combo)

        rows.append({
            "combination": combo,
            "final_score": final_score,
            "number_score": number_score,
            "sum": total,
            "sum_pattern_score": draw_sum_score,
            "gap_sum": gap_total,
            "gap_sum_pattern_score": gap_total_score,
            "gap_bucket_score": bucket_score,
            "odd_count": odd,
            "low_count": low,
            "used_zones": used_zones,
            "unique_last_digits": unique_last,
            "ac": ac,
            "carryover_count": carry,
            "gap_pattern": "/".join(
                f"{bucket}:{bucket_counts[bucket]}"
                for bucket in BUCKETS
            ),
        })

    ranked = pd.DataFrame(rows)
    if not ranked.empty:
        ranked = ranked.sort_values(
            ["final_score", "combination"],
            ascending=[False, True],
        ).reset_index(drop=True)

    context = {
        "candidate_sets": candidate_sets(master),
        "sum_forecast": sum_forecast.to_dict(),
        "gap_sum_forecast": gap_forecast.to_dict(),
        "pool": sorted(pool),
        "gap_bucket_target_mean": target_buckets,
    }
    return ranked, master, context


def build_final_portfolio(
    ranked: pd.DataFrame,
    size: int = 20,
    max_overlap: int = 4,
    max_exposure: int = 10,
    exposure_penalty: float = 0.005,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked

    selected: list[dict] = []
    exposure: Counter = Counter()
    remaining = ranked.copy()

    while len(selected) < size and not remaining.empty:
        best = None
        best_adjusted = -1e9

        for _, row in remaining.iterrows():
            combo = tuple(row["combination"])

            if any(exposure[n] >= max_exposure for n in combo):
                continue
            if any(
                len(set(combo) & set(item["combination"])) > max_overlap
                for item in selected
            ):
                continue

            adjusted = float(row["final_score"]) - exposure_penalty * sum(
                exposure[n] for n in combo
            )
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best = row.to_dict()

        if best is None:
            max_exposure += 1
            if max_exposure > size:
                break
            continue

        selected.append(best)
        for number in best["combination"]:
            exposure[number] += 1

        chosen = tuple(best["combination"])
        remaining = remaining[
            remaining["combination"].apply(tuple) != chosen
        ]

    return pd.DataFrame(selected).reset_index(drop=True)


def final_recommendation_bundle(df: pd.DataFrame) -> dict:
    ranked, master, context = rank_final_combinations(df, pool_size=20)
    portfolio = build_final_portfolio(ranked, size=20)

    return {
        "master_scores": master,
        "candidate_sets": context["candidate_sets"],
        "ranked": ranked,
        "portfolio": portfolio,
        "context": context,
    }
