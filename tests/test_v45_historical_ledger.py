from __future__ import annotations

import pandas as pd
import pytest

from lotto64.backtest.historical_ledger import (
    classify_failure,
    compare_previous_recent,
    max_strict_validation_rounds,
    required_history_start,
)


def test_required_history_start_1236():
    assert required_history_start(
        1236,
        validation_rounds=100,
        train_window=200,
    ) == 937


def test_current_213_rows_allow_13_strict_rounds():
    df = pd.DataFrame({"round": range(1024, 1237)})
    assert max_strict_validation_rounds(df, 200) == 13


def test_failure_candidate_stage_has_priority():
    record = {
        "candidate_15_hits": 2,
        "sum_core_hit": 1,
        "gap_sum_core_hit": 1,
        "best20_max_hit": 2,
        "master_top20_hits": 4,
    }
    primary, reasons = classify_failure(record)
    assert primary == "후보번호 단계 미포착"
    assert "후보번호 단계 미포착" in reasons


def test_failure_combination_stage():
    record = {
        "candidate_15_hits": 4,
        "sum_core_hit": 1,
        "gap_sum_core_hit": 1,
        "best20_max_hit": 2,
        "master_top20_hits": 4,
    }
    primary, _ = classify_failure(record)
    assert primary == "조합 구성/포트폴리오 단계 손실"


def test_previous_recent_50_comparison():
    rows = []
    for i in range(100):
        rows.append({
            "round": 1100 + i,
            "candidate_11_hits": 1,
            "candidate_13_hits": 2,
            "candidate_15_hits": 2 if i < 50 else 3,
            "best5_max_hit": 1,
            "best10_max_hit": 2,
            "best15_max_hit": 2,
            "best20_max_hit": 2 if i < 50 else 3,
            "sum_core_hit": 0 if i < 50 else 1,
            "sum_abs_error": 20 if i < 50 else 10,
            "gap_sum_core_hit": 0 if i < 50 else 1,
            "gap_sum_abs_error": 15 if i < 50 else 8,
            "master_top20_hits": 2 if i < 50 else 3,
        })
    result = compare_previous_recent(
        pd.DataFrame(rows),
        window=50,
    )
    row = result[
        result["metric"] == "후보15 평균 적중"
    ].iloc[0]
    assert row["previous_50"] == pytest.approx(2.0)
    assert row["recent_50"] == pytest.approx(3.0)
    assert row["change"] == pytest.approx(1.0)
