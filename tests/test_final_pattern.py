from __future__ import annotations

import pandas as pd
import pytest

from lotto64.analysis.gap_sum_series import (
    build_gap_sum_series,
    forecast_next_gap_sum,
)
from lotto64.models.pattern_master import (
    candidate_sets,
    master_number_scores,
)
from lotto64.recommend.final_pattern import final_recommendation_bundle


def sample_df(rows: int = 140) -> pd.DataFrame:
    out = []
    for i in range(rows):
        start = 1 + (i % 25)
        nums = sorted({
            start,
            min(45, start + 4),
            min(45, start + 8),
            min(45, start + 12),
            min(45, start + 16),
            min(45, start + 20),
        })
        while len(nums) < 6:
            for value in range(1, 46):
                if value not in nums:
                    nums.append(value)
                if len(nums) == 6:
                    break
        nums = sorted(nums[:6])

        bonus = next(v for v in range(45, 0, -1) if v not in nums)
        out.append({
            "round": 1000 + i,
            "date": "2026-01-01",
            **{f"n{j+1}": nums[j] for j in range(6)},
            "bonus": bonus,
        })
    return pd.DataFrame(out)


def test_gap_sum_series():
    df = sample_df()
    series = build_gap_sum_series(df, state_window=30)
    assert "gap_sum_state" in series.columns
    assert len(series) == len(df)


def test_gap_sum_forecast_order():
    forecast = forecast_next_gap_sum(
        sample_df(),
        state_window=30,
        transition_lookback=80,
    )
    assert forecast.wide_low <= forecast.target_low
    assert forecast.target_low <= forecast.target_center
    assert forecast.target_center <= forecast.target_high
    assert forecast.target_high <= forecast.wide_high


def test_master_candidate_sizes():
    scores = master_number_scores(sample_df())
    sets = candidate_sets(scores)
    assert len(sets[11]) == 11
    assert len(sets[13]) == 13
    assert len(sets[15]) == 15


def test_final_bundle_shapes():
    bundle = final_recommendation_bundle(sample_df())
    assert len(bundle["master_scores"]) == 45
    assert set(bundle["candidate_sets"]) == {11, 13, 15}
    assert "sum_forecast" in bundle["context"]
    assert "gap_sum_forecast" in bundle["context"]
    assert "skip_pattern_forecast" in bundle["context"]
    assert sum(bundle["context"]["final_score_weights"].values()) == pytest.approx(1.0)
    assert bundle["context"]["final_score_weights"]["skip_sum_transition"] == 0.14
    assert "skip_sum" in bundle["ranked"].columns
    assert "skip_sum_pattern_score" in bundle["ranked"].columns
    assert (
        bundle["ranked"]["gap_sum"]
        == bundle["ranked"]["skip_sum"] + 6
    ).all()
