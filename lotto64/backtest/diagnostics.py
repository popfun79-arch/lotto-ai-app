from __future__ import annotations

import pandas as pd

from lotto64.analysis.gap import build_gap_tables, row_numbers
from lotto64.recommend.final_pattern import final_recommendation_bundle


def diagnose_target_round(
    df: pd.DataFrame,
    target_index: int,
) -> dict:
    if target_index <= 0 or target_index >= len(df):
        raise IndexError("target_index 범위 오류")

    train = df.iloc[:target_index].reset_index(drop=True)
    actual_row = df.iloc[target_index]
    actual = set(row_numbers(actual_row))
    actual_sum = int(sum(actual))

    bundle = final_recommendation_bundle(train)
    portfolio = bundle["portfolio"]
    candidates = bundle["candidate_sets"]
    context = bundle["context"]

    actual_gap_sum = None
    _, gap_rounds = build_gap_tables(df.iloc[:target_index + 1])
    target_round = int(actual_row["round"])
    match = gap_rounds[gap_rounds["round"] == target_round]
    if not match.empty:
        actual_gap_sum = float(match.iloc[0]["gap_sum"])

    hits = {}
    for size, numbers in candidates.items():
        hits[f"candidate_{size}_hits"] = len(actual & set(numbers))

    combo_hits = []
    for combo in portfolio.get("combination", []):
        combo_hits.append(len(actual & set(combo)))

    max_combo_hit = max(combo_hits) if combo_hits else 0

    sum_fc = context["sum_forecast"]
    gap_fc = context["gap_sum_forecast"]

    sum_core = (
        float(sum_fc["target_low"])
        <= actual_sum
        <= float(sum_fc["target_high"])
    )
    gap_core = (
        actual_gap_sum is not None
        and float(gap_fc["target_low"])
        <= actual_gap_sum
        <= float(gap_fc["target_high"])
    )

    reasons = []
    if hits["candidate_15_hits"] < 3:
        reasons.append("후보번호 단계 미포착")
    if not sum_core:
        reasons.append("조합 합계 상태 예측 이탈")
    if not gap_core:
        reasons.append("GAP 합계 상태 예측 이탈")
    if hits["candidate_15_hits"] >= 3 and max_combo_hit < 3:
        reasons.append("조합 구성/포트폴리오 단계 손실")
    if not reasons:
        reasons.append("주요 상태는 적합했으나 무작위 변동 영향")

    return {
        "round": target_round,
        "actual_numbers": sorted(actual),
        "actual_sum": actual_sum,
        "actual_gap_sum": actual_gap_sum,
        **hits,
        "portfolio_max_hit": max_combo_hit,
        "sum_core_hit": int(sum_core),
        "gap_sum_core_hit": int(gap_core),
        "reasons": " | ".join(reasons),
    }


def diagnostic_backtest(
    df: pd.DataFrame,
    rounds: int = 30,
) -> pd.DataFrame:
    start = max(110, len(df) - rounds)
    rows = [
        diagnose_target_round(df, idx)
        for idx in range(start, len(df))
    ]
    return pd.DataFrame(rows)
