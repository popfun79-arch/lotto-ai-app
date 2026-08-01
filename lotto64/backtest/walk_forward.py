from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd

from lotto64.analysis.gap import row_numbers
from lotto64.models.scoring import number_scores
from lotto64.recommend.combination import generate_ranked
from lotto64.recommend.portfolio import build_portfolio

def walk_forward(
    df: pd.DataFrame,
    rounds: int = 50,
    train_window: int = 300,
    candidate_count: int = 18,
    top_combos: int = 10,
    seed: int = 20260720,
    weights: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    if len(df) < 80:
        raise ValueError("Walk-forward 백테스트에는 최소 80회 데이터가 필요합니다.")

    start = max(60, len(df) - rounds)
    targets = list(range(start, len(df)))
    rows = []

    for position, idx in enumerate(targets):
        train = df.iloc[max(0, idx - train_window): idx].reset_index(drop=True)
        actual = set(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx]["round"])

        scores = number_scores(
            train,
            weights=weights,
            similarity_k=min(10, max(3, len(train) // 20)),
        )

        record = {"round": round_no}
        for count in (11, 13, 15):
            predicted = set(scores.head(count)["number"].astype(int))
            record[f"candidate_{count}_hits"] = len(actual & predicted)

        ranked = generate_ranked(
            train,
            scores,
            candidate_count=min(candidate_count, 18),
            limit=12000,
            seed=seed + round_no,
        )
        top = build_portfolio(ranked, top_combos, 0.55)
        hits = [len(actual & set(combo)) for combo in top.get("combination", [])]
        max_hit = max(hits) if hits else 0

        record.update({
            "top_combo_max_hit": max_hit,
            "top_combo_3plus": int(max_hit >= 3),
            "top_combo_4plus": int(max_hit >= 4),
            "top_combo_5plus": int(max_hit >= 5),
            "top_combo_6": int(max_hit >= 6),
        })
        rows.append(record)

        if progress_callback:
            progress_callback(
                (position + 1) / len(targets),
                f"{round_no}회 검증 중...",
            )

    return pd.DataFrame(rows)

def summarize(result: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "검증 회차 수": len(result),
        "후보 11수 평균 적중": result["candidate_11_hits"].mean(),
        "후보 13수 평균 적중": result["candidate_13_hits"].mean(),
        "후보 15수 평균 적중": result["candidate_15_hits"].mean(),
        "TOP조합 평균 최고 적중": result["top_combo_max_hit"].mean(),
        "TOP조합 3개 이상 비율": result["top_combo_3plus"].mean(),
        "TOP조합 4개 이상 비율": result["top_combo_4plus"].mean(),
        "TOP조합 5개 이상 비율": result["top_combo_5plus"].mean(),
        "TOP조합 6개 비율": result["top_combo_6"].mean(),
    }
    return pd.DataFrame([{"지표": key, "값": value} for key, value in metrics.items()])
