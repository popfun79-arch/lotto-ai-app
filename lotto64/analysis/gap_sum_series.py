from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lotto64.analysis.gap import build_gap_tables


@dataclass(frozen=True)
class GapSumForecast:
    current_gap_sum: float
    current_state: str
    target_center: float
    target_low: float
    target_high: float
    wide_low: float
    wide_high: float
    matched_transitions: int
    state_window: int
    transition_lookback: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_gap_sum_series(
    df: pd.DataFrame,
    state_window: int = 50,
) -> pd.DataFrame:
    _, rounds = build_gap_tables(df)
    if rounds.empty:
        return rounds

    out = rounds[["round", "gap_sum", "gap_mean", "gap_max",
                  "gap_le_5", "gap_le_10", "gap_gt_10", "gap_ge_17"]].copy()

    out["gap_sum_ma5"] = out["gap_sum"].rolling(5, min_periods=3).mean()
    out["gap_sum_ma10"] = out["gap_sum"].rolling(10, min_periods=5).mean()
    out["gap_sum_ma20"] = out["gap_sum"].rolling(20, min_periods=10).mean()
    out["gap_sum_std20"] = out["gap_sum"].rolling(20, min_periods=10).std(ddof=0)

    prior = out["gap_sum"].shift(1)
    rolling = prior.rolling(
        state_window,
        min_periods=max(12, state_window // 3),
    )
    out["prior_q25"] = rolling.quantile(0.25)
    out["prior_q50"] = rolling.quantile(0.50)
    out["prior_q75"] = rolling.quantile(0.75)

    def classify(row: pd.Series) -> str:
        if pd.isna(row["gap_sum"]) or pd.isna(row["prior_q25"]):
            return "UNKNOWN"
        if row["gap_sum"] <= row["prior_q25"]:
            return "LOW"
        if row["gap_sum"] >= row["prior_q75"]:
            return "HIGH"
        return "MID"

    out["gap_sum_state"] = out.apply(classify, axis=1)
    return out


def forecast_next_gap_sum(
    df: pd.DataFrame,
    state_window: int = 50,
    transition_lookback: int = 100,
    min_matches: int = 8,
) -> GapSumForecast:
    series = build_gap_sum_series(df, state_window=state_window).dropna(
        subset=["gap_sum"]
    ).reset_index(drop=True)

    if len(series) < 20:
        values = series["gap_sum"].astype(float)
        return GapSumForecast(
            current_gap_sum=float(values.iloc[-1]),
            current_state="UNKNOWN",
            target_center=float(values.median()),
            target_low=float(values.quantile(0.25)),
            target_high=float(values.quantile(0.75)),
            wide_low=float(values.quantile(0.10)),
            wide_high=float(values.quantile(0.90)),
            matched_transitions=max(0, len(values) - 1),
            state_window=state_window,
            transition_lookback=transition_lookback,
        )

    current_state = str(series.iloc[-1]["gap_sum_state"])
    start = max(0, len(series) - transition_lookback - 1)
    next_values: list[float] = []

    for i in range(start, len(series) - 1):
        if current_state != "UNKNOWN":
            if str(series.iloc[i]["gap_sum_state"]) != current_state:
                continue
        next_values.append(float(series.iloc[i + 1]["gap_sum"]))

    if len(next_values) < min_matches:
        recent = series["gap_sum"].tail(
            min(transition_lookback, len(series))
        ).astype(float)
        next_values = recent.iloc[1:].tolist()

    values = pd.Series(next_values, dtype=float)
    return GapSumForecast(
        current_gap_sum=float(series.iloc[-1]["gap_sum"]),
        current_state=current_state,
        target_center=float(values.quantile(0.50)),
        target_low=float(values.quantile(0.25)),
        target_high=float(values.quantile(0.75)),
        wide_low=float(values.quantile(0.10)),
        wide_high=float(values.quantile(0.90)),
        matched_transitions=int(len(values)),
        state_window=state_window,
        transition_lookback=transition_lookback,
    )


def gap_sum_pattern_score(
    value: float,
    forecast: GapSumForecast,
) -> float:
    value = float(value)

    if forecast.target_low <= value <= forecast.target_high:
        half = max(5.0, (forecast.target_high - forecast.target_low) / 2)
        return float(
            max(
                0.85,
                1.0 - 0.15 * abs(value - forecast.target_center) / half,
            )
        )

    if forecast.wide_low <= value <= forecast.wide_high:
        if value < forecast.target_low:
            distance = forecast.target_low - value
            span = max(5.0, forecast.target_low - forecast.wide_low)
        else:
            distance = value - forecast.target_high
            span = max(5.0, forecast.wide_high - forecast.target_high)
        return float(max(0.45, 0.80 - 0.35 * distance / span))

    if value < forecast.wide_low:
        distance = forecast.wide_low - value
    else:
        distance = value - forecast.wide_high

    return float(max(0.05, 0.40 * np.exp(-distance / 15.0)))
