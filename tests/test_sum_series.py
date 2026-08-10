from __future__ import annotations

import pandas as pd

from lotto64.analysis.sum_series import (
    build_sum_series,
    compare_sum_windows,
    forecast_next_sum,
    sum_pattern_score,
)


def make_df(rows: int = 120) -> pd.DataFrame:
    data = []
    for i in range(rows):
        base = 1 + (i % 20)
        numbers = [
            base,
            base + 3,
            base + 6,
            base + 9,
            base + 12,
            min(45, base + 15),
        ]
        # guarantee six unique valid values
        numbers = sorted(set(numbers))
        while len(numbers) < 6:
            candidate = numbers[-1] + 1
            if candidate > 45:
                candidate = numbers[0] - 1
            if 1 <= candidate <= 45:
                numbers.append(candidate)
                numbers = sorted(set(numbers))
        numbers = numbers[:6]

        data.append({
            "round": 1000 + i,
            "date": "2026-01-01",
            **{f"n{j+1}": numbers[j] for j in range(6)},
            "bonus": 45 if 45 not in numbers else 44,
        })
    return pd.DataFrame(data)


def test_sum_series_has_core_features():
    df = make_df()
    series = build_sum_series(df, state_window=30)
    assert len(series) == len(df)
    for col in ["sum", "ma5", "ma10", "ma20", "std20", "sum_state"]:
        assert col in series.columns


def test_forecast_ranges_are_ordered():
    forecast = forecast_next_sum(make_df(), state_window=30, transition_lookback=80)
    assert forecast.wide_low <= forecast.target_low
    assert forecast.target_low <= forecast.target_center
    assert forecast.target_center <= forecast.target_high
    assert forecast.target_high <= forecast.wide_high


def test_sum_pattern_score_prefers_center():
    forecast = forecast_next_sum(make_df(), state_window=30, transition_lookback=80)
    center = int(round(forecast.target_center))
    far = 40 if center < 100 else 230
    assert sum_pattern_score(center, forecast) >= sum_pattern_score(far, forecast)


def test_compare_windows():
    compare = compare_sum_windows(make_df(), window=50)
    assert len(compare) == 2
    assert {"평균", "중앙값", "표준편차", "Q25", "Q75"}.issubset(compare.columns)
