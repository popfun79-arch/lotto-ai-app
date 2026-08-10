from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from lotto64.analysis.gap import current_gap_table, row_numbers
from lotto64.config import MAIN_COLUMNS, NUMBERS
from lotto64.models.scoring import number_scores
from lotto64.utils.lotto_math import neighbor_set, zone_index


MASTER_WEIGHTS = {
    # Howard-inspired games-out / skips emphasis
    "skip_hit_recent50": 0.18,
    "skip_hit_recent100": 0.10,
    "drawings_since_hit": 0.10,
    "skips_due": 0.08,
    # Python rolling-frequency / regime analysis
    "hot_20": 0.10,
    "hot_50": 0.08,
    "frequency_trend": 0.08,
    # Howard-style groups / last digits / multiple-hit context
    "number_group_recovery": 0.06,
    "last_digit_recovery": 0.05,
    "multiple_hit_neighbor": 0.07,
    # Existing Python/DNA/GAP ensemble
    "python_base": 0.10,
}


def gap_bucket(gap: int) -> str:
    gap = int(gap)
    if gap == 0:
        return "0"
    if gap <= 2:
        return "1-2"
    if gap <= 5:
        return "3-5"
    if gap <= 10:
        return "6-10"
    if gap <= 16:
        return "11-16"
    return "17+"


def _minmax(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    low, high = float(arr.min()), float(arr.max())
    if np.isclose(low, high):
        return np.full(len(arr), 0.5)
    return (arr - low) / (high - low)


def _window_counts(df: pd.DataFrame) -> Counter:
    counts: Counter = Counter()
    for _, row in df.iterrows():
        counts.update(row_numbers(row))
    return counts


def _games_out_records(df: pd.DataFrame) -> pd.DataFrame:
    last_seen = {n: None for n in NUMBERS}
    records = []

    for _, row in df.iterrows():
        round_no = int(row["round"])
        hits = set(row_numbers(row))

        for number in NUMBERS:
            if last_seen[number] is None:
                continue
            gap = round_no - int(last_seen[number]) - 1
            records.append({
                "round": round_no,
                "number": number,
                "gap": gap,
                "gap_bucket": gap_bucket(gap),
                "hit": int(number in hits),
            })

        for number in hits:
            last_seen[number] = round_no

    return pd.DataFrame(records)


def _hazard_by_bucket(
    df: pd.DataFrame,
    lookback: int,
) -> dict[str, float]:
    records = _games_out_records(df)
    if records.empty:
        return {}

    latest = int(df["round"].max())
    records = records[records["round"] > latest - lookback]
    grouped = records.groupby("gap_bucket").agg(
        exposures=("hit", "size"),
        hits=("hit", "sum"),
    )
    grouped["rate"] = (grouped["hits"] + 1) / (grouped["exposures"] + 2)

    raw = grouped["rate"].to_dict()
    keys = list(raw)
    normalized = _minmax(raw.values())
    return dict(zip(keys, normalized))


def master_number_scores(
    df: pd.DataFrame,
    egr_threshold: int = 17,
    similarity_k: int = 15,
) -> pd.DataFrame:
    """
    최종 번호 점수.

    핵심:
    - Drawings Since Hit / Games Out
    - Skip-and-Hit empirical hazard
    - Skips Due (현재 GAP / 개인 평균 GAP)
    - 최근 20/50회 hot, 이전 50회 대비 trend
    - Number Groups / Last Digits
    - 직전회 이월·이웃수
    - 기존 Python/DNA ensemble

    Gail Howard의 상용 소프트웨어를 재현하는 것이 아니라,
    공개적으로 알려진 분석 주제를 한국 6/45 데이터에 맞춰
    재구성한 연구용 구현입니다.
    """
    if len(df) < 30:
        raise ValueError("Pattern Master에는 최소 30회 데이터가 필요합니다.")

    gaps = current_gap_table(df)
    gap_map = gaps.set_index("number")["current_gap"].to_dict()
    gap_pct = gaps.set_index("number")["gap_percentile"].to_dict()

    hazard50 = _hazard_by_bucket(df, 50)
    hazard100 = _hazard_by_bucket(df, 100)

    recent20 = _window_counts(df.tail(min(20, len(df))))
    recent50 = _window_counts(df.tail(min(50, len(df))))

    if len(df) >= 100:
        previous50 = _window_counts(df.iloc[-100:-50])
    elif len(df) > 50:
        previous50 = _window_counts(df.iloc[:-50])
    else:
        previous50 = Counter()

    hot20_norm = dict(zip(
        NUMBERS,
        _minmax(recent20[n] for n in NUMBERS),
    ))
    hot50_norm = dict(zip(
        NUMBERS,
        _minmax(recent50[n] for n in NUMBERS),
    ))
    trend_norm = dict(zip(
        NUMBERS,
        _minmax(recent50[n] - previous50[n] for n in NUMBERS),
    ))

    zone_counts: Counter = Counter()
    digit_counts: Counter = Counter()
    for _, row in df.tail(min(20, len(df))).iterrows():
        for number in row_numbers(row):
            zone_counts[zone_index(number)] += 1
            digit_counts[number % 10] += 1

    zone_values = [zone_counts[z] for z in range(5)]
    zmin, zmax = min(zone_values), max(zone_values)
    zone_recovery = {
        z: 1 - (zone_counts[z] - zmin) / (zmax - zmin or 1)
        for z in range(5)
    }

    digit_values = [digit_counts[d] for d in range(10)]
    dmin, dmax = min(digit_values), max(digit_values)
    digit_recovery = {
        d: 1 - (digit_counts[d] - dmin) / (dmax - dmin or 1)
        for d in range(10)
    }

    base = number_scores(
        df,
        egr_threshold=egr_threshold,
        similarity_k=similarity_k,
    )
    base_map = dict(zip(base["number"], base["final_score"]))
    base_norm = dict(zip(
        NUMBERS,
        _minmax(base_map[n] for n in NUMBERS),
    ))

    latest = set(row_numbers(df.iloc[-1]))
    latest_neighbors = neighbor_set(latest)

    rows = []
    for number in NUMBERS:
        gap = int(gap_map[number])
        bucket = gap_bucket(gap)
        gap_row = gaps[gaps["number"] == number].iloc[0]
        average_gap = float(gap_row["average_gap"]) if pd.notna(
            gap_row["average_gap"]
        ) else np.nan

        if pd.notna(average_gap) and average_gap > 0:
            due_ratio = gap / average_gap
            skips_due = min(1.0, due_ratio / 1.2)
        else:
            skips_due = 0.5

        if number in latest:
            multiple_context = 1.0
        elif number in latest_neighbors:
            multiple_context = 0.75
        else:
            multiple_context = 0.35

        components = {
            "skip_hit_recent50": float(hazard50.get(bucket, 0.5)),
            "skip_hit_recent100": float(hazard100.get(bucket, 0.5)),
            "drawings_since_hit": float(gap_pct[number]),
            "skips_due": float(skips_due),
            "hot_20": float(hot20_norm[number]),
            "hot_50": float(hot50_norm[number]),
            "frequency_trend": float(trend_norm[number]),
            "number_group_recovery": float(zone_recovery[zone_index(number)]),
            "last_digit_recovery": float(digit_recovery[number % 10]),
            "multiple_hit_neighbor": float(multiple_context),
            "python_base": float(base_norm[number]),
        }

        master_score = sum(
            MASTER_WEIGHTS[key] * components[key]
            for key in MASTER_WEIGHTS
        )

        rows.append({
            "number": number,
            "master_score": master_score,
            "current_gap": gap,
            "gap_bucket": bucket,
            "average_gap": average_gap,
            **{f"score_{key}": value for key, value in components.items()},
        })

    out = pd.DataFrame(rows).sort_values(
        ["master_score", "number"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_sets(scores: pd.DataFrame) -> dict[int, list[int]]:
    return {
        size: sorted(scores.head(size)["number"].astype(int).tolist())
        for size in (11, 13, 15)
    }
