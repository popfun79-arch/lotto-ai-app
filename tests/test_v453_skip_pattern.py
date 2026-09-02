import pandas as pd

from lotto64.analysis.skip_pattern import (
    build_empirical_hazard,
    build_skip_period_history,
    current_skip_profile,
    skip_bucket,
)


def _df():
    rows = []
    draws = {
        1: [1, 3, 10, 20, 30, 40],
        2: [2, 4, 11, 21, 31, 41],
        3: [1, 5, 12, 22, 32, 42],
        4: [6, 13, 23, 33, 43, 44],
        5: [2, 7, 14, 24, 34, 45],
        6: [1, 8, 15, 25, 35, 44],
    }
    for round_no, nums in draws.items():
        rows.append({"round": round_no, "date": f"2026-01-{round_no:02d}", **{f"n{i}": n for i, n in enumerate(nums, 1)}, "bonus": 9})
    return pd.DataFrame(rows)


def test_skip_bucket_boundaries():
    assert skip_bucket(0) == "0"
    assert skip_bucket(2) == "1-2"
    assert skip_bucket(5) == "3-5"
    assert skip_bucket(10) == "6-10"
    assert skip_bucket(16) == "11-16"
    assert skip_bucket(17) == "17+"


def test_round_skip_periods_are_previous_hit_based():
    history = build_skip_period_history(_df())
    row3 = history.loc[history["round"] == 3].iloc[0]
    # Number 1 appeared in round 1 and round 3: one skipped round.
    assert row3["skip_values"][0] == 1
    # Number 5 is first seen in round 3, so its skip value is NaN.
    assert pd.isna(row3["skip_values"][1])


def test_current_skip_profile_and_hazard_exist():
    df = _df()
    profile = current_skip_profile(df)
    n1 = profile.loc[profile["number"] == 1].iloc[0]
    # Number 1 appeared in round 6, so its current skip is zero.
    assert n1["current_skip"] == 0

    hazard = build_empirical_hazard(df)
    assert len(hazard) == 45
    assert ((hazard["empirical_hazard"] >= 0) & (hazard["empirical_hazard"] <= 1)).all()
