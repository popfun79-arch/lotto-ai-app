from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TOP_OF_BEST_SIZES = (5, 10, 15, 20)


def build_top_of_best_sets(
    portfolio: pd.DataFrame,
    sizes: Iterable[int] = TOP_OF_BEST_SIZES,
) -> dict[int, pd.DataFrame]:
    """
    하나의 최종 통합 포트폴리오에서 누적형 추천을 만듭니다.

    BEST 5 ⊂ BEST 10 ⊂ BEST 15 ⊂ BEST 20

    즉 네 화면을 합쳐 50개의 서로 다른 조합을 만드는 것이 아니라,
    최대 20개의 동일한 최종 추천 목록을 강도별로 잘라 보여줍니다.
    """
    normalized_sizes = tuple(
        sorted({int(size) for size in sizes if int(size) > 0})
    )

    if portfolio is None or portfolio.empty:
        return {
            size: pd.DataFrame()
            for size in normalized_sizes
        }

    ranked = portfolio.copy().reset_index(drop=True)

    if "rank" in ranked.columns:
        ranked = ranked.drop(columns=["rank"])

    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return {
        size: ranked.head(min(size, len(ranked))).copy()
        for size in normalized_sizes
    }


def validate_nested_sets(
    sets: dict[int, pd.DataFrame],
) -> bool:
    previous: list[tuple[int, ...]] = []

    for size in sorted(sets):
        frame = sets[size]
        if frame.empty:
            current: list[tuple[int, ...]] = []
        else:
            current = [
                tuple(values)
                for values in frame["combination"].tolist()
            ]

        if previous and current[: len(previous)] != previous:
            return False

        previous = current

    return True

