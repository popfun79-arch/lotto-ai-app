from __future__ import annotations

from typing import Sequence

import pandas as pd

def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

def build_portfolio(ranked: pd.DataFrame, size: int = 20, max_jaccard: float = 0.50) -> pd.DataFrame:
    selected = []

    for _, row in ranked.iterrows():
        combo = tuple(row["combination"])
        if all(jaccard(combo, item["combination"]) <= max_jaccard for item in selected):
            selected.append(row.to_dict())
        if len(selected) >= size:
            break

    if len(selected) < size:
        seen = {tuple(item["combination"]) for item in selected}
        for _, row in ranked.iterrows():
            combo = tuple(row["combination"])
            if combo not in seen:
                selected.append(row.to_dict())
                seen.add(combo)
            if len(selected) >= size:
                break

    return pd.DataFrame(selected).reset_index(drop=True)

def strategy_sets(ranked: pd.DataFrame, size: int = 20) -> dict[str, pd.DataFrame]:
    if ranked.empty:
        return {"안정형": ranked, "균형형": ranked, "공격형": ranked}

    stable = ranked[
        ranked["sum"].between(125, 160)
        & ranked["ac"].between(7, 10)
        & ranked["odd_count"].between(2, 4)
        & ranked["low_count"].between(2, 4)
    ]

    aggressive = ranked.sort_values(
        ["neighbor_count", "carryover_count", "final_score"],
        ascending=[False, True, False],
    )

    return {
        "안정형": build_portfolio(stable if not stable.empty else ranked, size, 0.55),
        "균형형": build_portfolio(ranked, size, 0.50),
        "공격형": build_portfolio(aggressive, size, 0.45),
    }
