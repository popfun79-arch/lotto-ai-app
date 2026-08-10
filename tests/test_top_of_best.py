from __future__ import annotations

import pandas as pd

from lotto64.recommend.top_of_best import (
    build_top_of_best_sets,
    validate_nested_sets,
)


def sample_portfolio(rows: int = 20) -> pd.DataFrame:
    # 테스트 목적상 서로 다른 20개의 조합을 확실히 만듭니다.
    data = []
    for i in range(rows):
        combo = (
            1 + i,
            21 + (i % 5),
            26 + (i % 5),
            31 + (i % 5),
            36 + (i % 5),
            41 + (i % 5),
        )
        data.append({
            "combination": combo,
            "final_score": 1.0 - i * 0.01,
            "sum": sum(combo),
            "gap_sum": 40 + (i % 5),
        })
    return pd.DataFrame(data)


def test_top_of_best_sizes():
    sets = build_top_of_best_sets(sample_portfolio())
    assert len(sets[5]) == 5
    assert len(sets[10]) == 10
    assert len(sets[15]) == 15
    assert len(sets[20]) == 20


def test_top_of_best_is_nested():
    sets = build_top_of_best_sets(sample_portfolio())
    assert validate_nested_sets(sets)

    top5 = sets[5]["combination"].tolist()
    top10 = sets[10]["combination"].tolist()
    top15 = sets[15]["combination"].tolist()
    top20 = sets[20]["combination"].tolist()

    assert top10[:5] == top5
    assert top15[:10] == top10
    assert top20[:15] == top15


def test_top_of_best_has_only_twenty_unique_recommendations():
    sets = build_top_of_best_sets(sample_portfolio())
    all_unique = {
        tuple(combo)
        for frame in sets.values()
        for combo in frame["combination"].tolist()
    }
    assert len(all_unique) == 20
