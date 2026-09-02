from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from lotto64.config import MAIN_COLUMNS, NUMBERS

SKIP_BUCKETS = ("0", "1-2", "3-5", "6-10", "11-16", "17+")


def row_numbers(row: pd.Series) -> list[int]:
    return [int(row[c]) for c in MAIN_COLUMNS]


def skip_bucket(skip: int) -> str:
    value = int(skip)
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 16:
        return "11-16"
    return "17+"


def _regime(series: pd.Series) -> pd.Series:
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    q90 = float(series.quantile(0.90))

    def classify(value: float) -> str:
        if value <= q25:
            return "압축"
        if value >= q90:
            return "극단 확장"
        if value >= q75:
            return "확장"
        return "보통"

    return series.apply(classify)


def build_skip_period_history(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """
    각 회차 직전까지의 정보만 사용해 해당 회차 당첨번호 6개의
    '건너띔 기간(skip period)'을 계산합니다.

    skip=0은 직전 회차에 출현한 번호가 이번 회차에 다시 나온 경우입니다.
    skip=1은 한 회차를 건너뛴 뒤 출현한 경우입니다.
    """
    ordered = df.sort_values("round").reset_index(drop=True)
    if window is not None:
        ordered = ordered.tail(int(window)).reset_index(drop=True)

    last_seen: dict[int, int | None] = {n: None for n in NUMBERS}
    rows: list[dict] = []

    for _, row in ordered.iterrows():
        round_no = int(row["round"])
        numbers = row_numbers(row)
        skips = [
            round_no - last_seen[n] - 1
            if last_seen[n] is not None
            else np.nan
            for n in numbers
        ]
        valid = [int(x) for x in skips if pd.notna(x)]
        bucket_counts = Counter(skip_bucket(x) for x in valid)

        record = {
            "round": round_no,
            "date": row.get("date"),
            "actual_numbers": " ".join(map(str, numbers)),
            "skip_values": skips,
            "skip_sum": float(sum(valid)) if valid else np.nan,
            "skip_mean": float(np.mean(valid)) if valid else np.nan,
            "skip_median": float(np.median(valid)) if valid else np.nan,
            "skip_max": float(max(valid)) if valid else np.nan,
            "skip_std": float(np.std(valid)) if valid else np.nan,
            "skip_0_count": bucket_counts["0"],
            "skip_1_2_count": bucket_counts["1-2"],
            "skip_3_5_count": bucket_counts["3-5"],
            "skip_6_10_count": bucket_counts["6-10"],
            "skip_11_16_count": bucket_counts["11-16"],
            "skip_17plus_count": bucket_counts["17+"],
        }
        rows.append(record)

        for number in numbers:
            last_seen[number] = round_no

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["skip_sum_delta1"] = out["skip_sum"].diff()
    out["skip_mean_delta1"] = out["skip_mean"].diff()
    out["skip_sum_ma5"] = out["skip_sum"].rolling(5, min_periods=3).mean()
    out["skip_sum_ma10"] = out["skip_sum"].rolling(10, min_periods=5).mean()
    out["skip_regime"] = _regime(out["skip_sum"].fillna(0.0))

    # Explicit construction keeps the exported schema stable and readable.
    out["skip_pattern"] = out.apply(
        lambda r: " / ".join(
            [
                f"0:{int(r['skip_0_count'])}",
                f"1-2:{int(r['skip_1_2_count'])}",
                f"3-5:{int(r['skip_3_5_count'])}",
                f"6-10:{int(r['skip_6_10_count'])}",
                f"11-16:{int(r['skip_11_16_count'])}",
                f"17+:{int(r['skip_17plus_count'])}",
            ]
        ),
        axis=1,
    )
    return out


def current_skip_profile(df: pd.DataFrame) -> pd.DataFrame:
    """현재 회차 기준 번호별 건너띔 기간과 과거 실제 skip 분포를 계산합니다."""
    ordered = df.sort_values("round").reset_index(drop=True)
    latest = int(ordered["round"].max())
    last_seen: dict[int, int | None] = {n: None for n in NUMBERS}
    history: dict[int, list[int]] = {n: [] for n in NUMBERS}

    for _, row in ordered.iterrows():
        round_no = int(row["round"])
        for number in row_numbers(row):
            if last_seen[number] is not None:
                history[number].append(round_no - last_seen[number] - 1)
            last_seen[number] = round_no

    rows = []
    for number in NUMBERS:
        current = latest - last_seen[number] if last_seen[number] is not None else len(ordered)
        values = history[number]
        rows.append({
            "number": number,
            "current_skip": int(current),
            "average_skip": float(np.mean(values)) if values else np.nan,
            "median_skip": float(np.median(values)) if values else np.nan,
            "max_skip": int(max(values)) if values else np.nan,
            "skip_observations": len(values),
            "current_bucket": skip_bucket(current),
        })

    out = pd.DataFrame(rows)
    out["current_skip_percentile"] = out["current_skip"].rank(pct=True)
    return out


def build_empirical_hazard(df: pd.DataFrame) -> pd.DataFrame:
    """
    번호별 empirical hazard.

    hazard(g) = P(다음 회차 적중 | 직전 적중 후 g회 건너뜀)
             = count(exact skip=g) / count(skip>=g)
    """
    profile = current_skip_profile(df)
    last_seen: dict[int, int | None] = {n: None for n in NUMBERS}
    observed: dict[int, list[int]] = {n: [] for n in NUMBERS}
    ordered = df.sort_values("round")

    for _, row in ordered.iterrows():
        round_no = int(row["round"])
        for number in row_numbers(row):
            if last_seen[number] is not None:
                observed[number].append(round_no - last_seen[number] - 1)
            last_seen[number] = round_no

    rows = []
    for number in NUMBERS:
        values = observed[number]
        current = int(profile.loc[profile["number"] == number, "current_skip"].iloc[0])
        exact = sum(v == current for v in values)
        survived = sum(v >= current for v in values)
        hazard = exact / survived if survived else 0.0
        baseline = 1.0 / (float(np.mean(values)) + 1.0) if values else 0.0
        rows.append({
            "number": number,
            "current_skip": current,
            "current_bucket": skip_bucket(current),
            "exact_skip_count": exact,
            "survived_count": survived,
            "empirical_hazard": hazard,
            "baseline_hazard": baseline,
            "hazard_lift": hazard / baseline if baseline > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def skip_composition_target(df: pd.DataFrame, window: int = 50) -> dict[str, float]:
    history = build_skip_period_history(df)
    recent = history.tail(min(int(window), len(history)))
    if recent.empty:
        return {bucket: 1.0 for bucket in SKIP_BUCKETS}

    columns = {
        "0": "skip_0_count",
        "1-2": "skip_1_2_count",
        "3-5": "skip_3_5_count",
        "6-10": "skip_6_10_count",
        "11-16": "skip_11_16_count",
        "17+": "skip_17plus_count",
    }
    return {bucket: float(recent[column].mean()) for bucket, column in columns.items()}


def skip_composition_score(
    combo: tuple[int, ...] | list[int],
    df: pd.DataFrame,
    window: int = 50,
) -> float:
    """후보 조합의 현재 skip-age 구성이 최근 실제 당첨 구성과 얼마나 가까운지."""
    profile = current_skip_profile(df).set_index("number")
    target = skip_composition_target(df, window)
    counts = Counter(
        skip_bucket(int(profile.loc[int(number), "current_skip"]))
        for number in combo
    )
    distance = sum(
        abs(float(counts[bucket]) - float(target[bucket]))
        for bucket in SKIP_BUCKETS
    )
    return float(max(0.0, 1.0 - distance / 12.0))


def skip_hazard_score(combo: tuple[int, ...] | list[int], df: pd.DataFrame) -> float:
    """후보 번호들의 현재 skip-age empirical hazard 상대순위를 0~1로 반환합니다."""
    hazard = build_empirical_hazard(df).copy()
    hazard["hazard_rank_score"] = hazard["empirical_hazard"].rank(pct=True)
    values = [
        float(hazard.loc[hazard["number"] == int(number), "hazard_rank_score"].iloc[0])
        for number in combo
    ]
    return float(np.mean(values)) if values else 0.0
