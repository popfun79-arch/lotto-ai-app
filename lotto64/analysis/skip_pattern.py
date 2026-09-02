from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lotto64.config import MAIN_COLUMNS, NUMBERS

SKIP_BUCKETS = ("0", "1-2", "3-5", "6-10", "11-16", "17+")
SKIP_BUCKET_COLUMNS = {
    "0": "skip_0_count",
    "1-2": "skip_1_2_count",
    "3-5": "skip_3_5_count",
    "6-10": "skip_6_10_count",
    "11-16": "skip_11_16_count",
    "17+": "skip_17plus_count",
}
SKIP_SUM_BANDS = ("0-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")


@dataclass(frozen=True)
class SkipPatternForecast:
    current_skip_sum: float
    current_state: str
    current_direction: str
    target_center: float
    target_low: float
    target_high: float
    wide_low: float
    wide_high: float
    matched_transitions: int
    match_mode: str
    state_window: int
    transition_lookback: int
    bucket_target: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


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


def skip_sum_band(value: float) -> str:
    total = float(value)
    if total < 20:
        return "0-19"
    if total < 30:
        return "20-29"
    if total < 40:
        return "30-39"
    if total < 50:
        return "40-49"
    if total < 60:
        return "50-59"
    if total < 70:
        return "60-69"
    return "70+"


def _add_causal_skip_states(
    frame: pd.DataFrame,
    state_window: int = 50,
) -> pd.DataFrame:
    """각 행보다 앞선 회차만으로 skip 합계 상태를 분류합니다."""
    out = frame.copy()
    prior = out["skip_sum"].shift(1)
    rolling = prior.rolling(
        int(state_window),
        min_periods=max(12, int(state_window) // 3),
    )
    out["skip_prior_q25"] = rolling.quantile(0.25)
    out["skip_prior_q75"] = rolling.quantile(0.75)
    out["skip_prior_q90"] = rolling.quantile(0.90)

    def classify(row: pd.Series) -> str:
        if pd.isna(row["skip_sum"]) or pd.isna(row["skip_prior_q25"]):
            return "UNKNOWN"
        if row["skip_sum"] <= row["skip_prior_q25"]:
            return "LOW"
        if row["skip_sum"] >= row["skip_prior_q90"]:
            return "EXTREME"
        if row["skip_sum"] >= row["skip_prior_q75"]:
            return "HIGH"
        return "MID"

    out["skip_sum_state"] = out.apply(classify, axis=1)
    out["skip_direction"] = np.select(
        [out["skip_sum_delta1"] > 3, out["skip_sum_delta1"] < -3],
        ["UP", "DOWN"],
        default="FLAT",
    )
    out.loc[out["skip_sum_delta1"].isna(), "skip_direction"] = "UNKNOWN"
    out["skip_pattern_state"] = (
        out["skip_sum_state"].astype(str)
        + "|"
        + out["skip_direction"].astype(str)
    )
    regime_labels = {
        "UNKNOWN": "자료 부족",
        "LOW": "압축",
        "MID": "보통",
        "HIGH": "확장",
        "EXTREME": "극단 확장",
    }
    out["skip_regime"] = out["skip_sum_state"].map(regime_labels)
    return out


def build_skip_period_history(df: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """
    각 회차 직전까지의 정보만 사용해 해당 회차 당첨번호 6개의
    '건너띔 기간(skip period)'을 계산합니다.

    skip=0은 직전 회차에 출현한 번호가 이번 회차에 다시 나온 경우입니다.
    skip=1은 한 회차를 건너뛴 뒤 출현한 경우입니다.
    """
    ordered = df.sort_values("round").reset_index(drop=True)

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
        record.update({f"skip_n{i}": skips[i - 1] for i in range(1, 7)})
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
    out["skip_sum_ma20"] = out["skip_sum"].rolling(20, min_periods=10).mean()
    out["skip_sum_std20"] = out["skip_sum"].rolling(20, min_periods=10).std(ddof=0)
    out["skip_sum_band"] = out["skip_sum"].apply(
        lambda value: skip_sum_band(value) if pd.notna(value) else "UNKNOWN"
    )
    out = _add_causal_skip_states(out)

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
    if window is not None:
        return out.tail(int(window)).reset_index(drop=True)
    return out


def skip_period_distribution(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """실제 당첨번호의 건너띔 기간별 출현 빈도와 비율을 반환합니다."""
    history = build_skip_period_history(df)
    if window is not None:
        history = history.tail(min(int(window), len(history)))

    values = [
        int(value)
        for row in history.get("skip_values", pd.Series(dtype=object))
        for value in row
        if pd.notna(value)
    ]
    if not values:
        return pd.DataFrame(columns=["skip", "count", "rate"])

    counts = pd.Series(values, dtype=int).value_counts().sort_index()
    out = counts.rename_axis("skip").reset_index(name="count")
    out["rate"] = out["count"] / int(out["count"].sum())
    return out


def skip_sum_distribution(
    df: pd.DataFrame,
    window: int = 100,
) -> pd.DataFrame:
    """회차별 건너띔 합계를 고정 구간으로 묶어 빈도와 비율을 반환합니다."""
    history = build_skip_period_history(df).dropna(subset=["skip_sum"])
    recent = history.tail(min(int(window), len(history)))
    counts = Counter(recent["skip_sum"].map(skip_sum_band))
    total = max(1, int(sum(counts.values())))
    return pd.DataFrame([
        {
            "skip_sum_band": band,
            "count": int(counts[band]),
            "rate": float(counts[band] / total),
        }
        for band in SKIP_SUM_BANDS
    ])


def _forecast_from_rows(
    history: pd.DataFrame,
    next_rows: pd.DataFrame,
    current_state: str,
    current_direction: str,
    match_mode: str,
    state_window: int,
    transition_lookback: int,
) -> SkipPatternForecast:
    values = next_rows["skip_sum"].astype(float)
    bucket_target = {
        bucket: float(next_rows[column].mean())
        for bucket, column in SKIP_BUCKET_COLUMNS.items()
    }
    return SkipPatternForecast(
        current_skip_sum=float(history.iloc[-1]["skip_sum"]),
        current_state=current_state,
        current_direction=current_direction,
        target_center=float(values.quantile(0.50)),
        target_low=float(values.quantile(0.25)),
        target_high=float(values.quantile(0.75)),
        wide_low=float(values.quantile(0.10)),
        wide_high=float(values.quantile(0.90)),
        matched_transitions=int(len(next_rows)),
        match_mode=match_mode,
        state_window=int(state_window),
        transition_lookback=int(transition_lookback),
        bucket_target=bucket_target,
    )


def forecast_next_skip_pattern(
    df: pd.DataFrame,
    state_window: int = 50,
    transition_lookback: int = 120,
    min_matches: int = 8,
) -> SkipPatternForecast:
    """
    현재 skip 합계 상태와 직전 방향이 같았던 과거 회차의 다음 회차를
    우선 사용해 합계 구간과 6개 구간 구성을 예측합니다.

    표본이 부족하면 동일 상태, 최근 전이 순으로 자동 후퇴합니다.
    """
    history = build_skip_period_history(df).dropna(
        subset=["skip_sum"]
    ).reset_index(drop=True)
    if len(history) < 2:
        raise ValueError("건너띔 패턴 예측에는 유효 회차가 2개 이상 필요합니다.")

    history = _add_causal_skip_states(history, state_window=state_window)
    latest = history.iloc[-1]
    current_state = str(latest["skip_sum_state"])
    current_direction = str(latest["skip_direction"])
    start = max(0, len(history) - int(transition_lookback) - 1)

    exact_indices = [
        i + 1
        for i in range(start, len(history) - 1)
        if str(history.iloc[i]["skip_sum_state"]) == current_state
        and str(history.iloc[i]["skip_direction"]) == current_direction
    ]
    if current_state != "UNKNOWN" and len(exact_indices) >= int(min_matches):
        return _forecast_from_rows(
            history,
            history.iloc[exact_indices],
            current_state,
            current_direction,
            "state+direction",
            state_window,
            transition_lookback,
        )

    state_indices = [
        i + 1
        for i in range(start, len(history) - 1)
        if str(history.iloc[i]["skip_sum_state"]) == current_state
    ]
    if current_state != "UNKNOWN" and len(state_indices) >= int(min_matches):
        return _forecast_from_rows(
            history,
            history.iloc[state_indices],
            current_state,
            current_direction,
            "state",
            state_window,
            transition_lookback,
        )

    fallback_start = max(1, len(history) - int(transition_lookback))
    return _forecast_from_rows(
        history,
        history.iloc[fallback_start:],
        current_state,
        current_direction,
        "recent",
        state_window,
        transition_lookback,
    )


def skip_sum_pattern_score(
    value: float,
    forecast: SkipPatternForecast,
) -> float:
    """후보 조합의 다음 회차 skip 합계를 전이 기반 예측구간으로 평가합니다."""
    total = float(value)
    if forecast.target_low <= total <= forecast.target_high:
        half = max(4.0, (forecast.target_high - forecast.target_low) / 2)
        return float(max(
            0.85,
            1.0 - 0.15 * abs(total - forecast.target_center) / half,
        ))

    if forecast.wide_low <= total <= forecast.wide_high:
        if total < forecast.target_low:
            distance = forecast.target_low - total
            span = max(4.0, forecast.target_low - forecast.wide_low)
        else:
            distance = total - forecast.target_high
            span = max(4.0, forecast.wide_high - forecast.target_high)
        return float(max(0.45, 0.80 - 0.35 * distance / span))

    distance = (
        forecast.wide_low - total
        if total < forecast.wide_low
        else total - forecast.wide_high
    )
    return float(max(0.05, 0.40 * np.exp(-distance / 12.0)))


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
