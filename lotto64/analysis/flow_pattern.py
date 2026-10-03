from __future__ import annotations

"""
[모듈 설명: 로또 대각선 흐름 및 이웃·폭포·역폭포 패턴 분석]
이 모듈은 로또 6/45 추첨에서 빈번하게 관찰되는 대각선 흐름(Diagonal Flow) 패턴을 분석합니다.
초보자 및 교육자도 쉽게 이해할 수 있도록 각 개념을 명확한 수학적 규칙으로 정의합니다.

1. 이웃수 (Neighbor Numbers):
   - 직전 회차 당첨번호 n을 기준으로 바로 옆의 번호인 n - 1 과 n + 1 을 의미합니다.
   - 예: 지난 회차에 15가 나왔다면, 14와 16이 이웃수가 됩니다. (1의 왼쪽은 없으므로 2만, 45의 오른쪽은 없으므로 44만 해당)

2. 폭포 패턴 (Cascade / Waterfall Pattern):
   - 회차가 거듭되면서 번호가 +1씩 우하향으로 떨어지는 계단형 흐름입니다.
   - 2연속 폭포 (2-Step): 직전(R-1) 회차에 n 출현 -> 이번(R) 회차에 n + 1 출현
   - 3연속 폭포 (3-Step): (R-2)회차에 n -> (R-1)회차에 n + 1 -> 이번(R) 회차에 n + 2 출현

3. 역폭포 패턴 (Reverse Cascade Pattern):
   - 회차가 거듭되면서 번호가 -1씩 좌하향으로 떨어지는 역계단형 흐름입니다.
   - 2연속 역폭포 (2-Step): 직전(R-1) 회차에 n -> 이번(R) 회차에 n - 1 출현
   - 3연속 역폭포 (3-Step): (R-2)회차에 n -> (R-1)회차에 n - 1 -> 이번(R) 회차에 n - 2 출현

4. 이월수 (Carryover / Repeat):
   - 직전(R-1) 회차에 나온 번호가 이번(R) 회차에 그대로 다시 출현하는 현상 (n -> n)
"""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from lotto64.analysis.gap import row_numbers
from lotto64.config import NUMBERS


def get_neighbors(numbers: Sequence[int]) -> set[int]:
    """
    주어진 번호들의 인접한 이웃 번호(n - 1, n + 1) 집합을 구합니다.
    로또 유효 범위인 1 ~ 45를 벗어나지 않도록 안전하게 필터링합니다.
    """
    neighbors: set[int] = set()
    for num in numbers:
        val = int(num)
        if val > 1:
            neighbors.add(val - 1)
        if val < 45:
            neighbors.add(val + 1)
    return neighbors


def get_cascade_candidates(numbers: Sequence[int]) -> set[int]:
    """
    폭포(우하향 +1) 흐름 후보 번호 집합을 구합니다.
    직전 회차 번호 n에 대해 다음 회차에 1이 증가한 n + 1 후보입니다.
    """
    candidates: set[int] = set()
    for num in numbers:
        val = int(num)
        if val < 45:
            candidates.add(val + 1)
    return candidates


def get_reverse_cascade_candidates(numbers: Sequence[int]) -> set[int]:
    """
    역폭포(좌하향 -1) 흐름 후보 번호 집합을 구합니다.
    직전 회차 번호 n에 대해 다음 회차에 1이 감소한 n - 1 후보입니다.
    """
    candidates: set[int] = set()
    for num in numbers:
        val = int(num)
        if val > 1:
            candidates.add(val - 1)
    return candidates


@dataclass(frozen=True)
class FlowTargets:
    """
    다음 회차를 위해 도출된 대각선 흐름 타깃 번호 모음 데이터 클래스
    """
    target_round: int                      # 예측 대상 회차 번호 (예: 1243회)
    previous_numbers: list[int]            # 직전(R-1) 회차 당첨번호 6개
    prev2_numbers: list[int]               # 전전(R-2) 회차 당첨번호 6개
    carryover_candidates: list[int]        # 이월수 후보 (직전 회차 당첨번호와 동일)
    neighbor_candidates: list[int]         # 이웃수 후보 (±1 번호 전체)
    cascade_step2_candidates: list[int]    # 2단계 폭포 후보 (+1 번호)
    cascade_step3_candidates: list[int]    # 3단계 폭포 유력 후보 (R-2회차 n -> R-1회차 n+1 이후 이번 회차 n+2)
    reverse_step2_candidates: list[int]    # 2단계 역폭포 후보 (-1 번호)
    reverse_step3_candidates: list[int]    # 3단계 역폭포 유력 후보 (R-2회차 n -> R-1회차 n-1 이후 이번 회차 n-2)
    number_flow_scores: dict[int, float]   # 번호별(1~45) 대각선 흐름 가중치 점수 (0.0 ~ 1.0)


def analyze_flow_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 또는 선택된 회차 이력을 순회하며, 각 회차에서
    이월수, 이웃수, 폭포수(+1), 역폭포수(-1), 3연속 폭포/역폭포가
    실제로 각각 몇 개씩 발생했는지 정밀하게 계산하여 데이터프레임으로 반환합니다.
    """
    if len(df) < 2:
        return pd.DataFrame()

    records = []
    # 이전 회차들의 당첨번호를 순서대로 추적
    history_numbers: list[tuple[int, set[int]]] = []

    for _, row in df.iterrows():
        round_no = int(row["round"])
        current_set = set(row_numbers(row))

        if len(history_numbers) >= 1:
            prev_round, prev_set = history_numbers[-1]
            # 1. 이월수: 직전 회차와 겹친 번호
            carries = current_set & prev_set

            # 2. 이웃수: 직전 회차 번호들의 ±1 중 이번 회차에 당첨된 번호
            neighbors_pool = get_neighbors(list(prev_set))
            neighbor_hits = current_set & neighbors_pool

            # 3. 2단계 폭포 (+1 대각선): 직전 회차 번호 + 1 중 이번 회차 당첨
            cascade_pool = get_cascade_candidates(list(prev_set))
            cascade_hits = current_set & cascade_pool

            # 4. 2단계 역폭포 (-1 대각선): 직전 회차 번호 - 1 중 이번 회차 당첨
            reverse_pool = get_reverse_cascade_candidates(list(prev_set))
            reverse_hits = current_set & reverse_pool

            # 5. 3연속 폭포 / 역폭포 추적 (R-2 회차가 있는 경우)
            cascade_3step_hits = set()
            reverse_3step_hits = set()

            if len(history_numbers) >= 2:
                _, prev2_set = history_numbers[-2]
                # 3연속 폭포: n in prev2 -> n+1 in prev -> n+2 in current
                for n in prev2_set:
                    if (n + 1) in prev_set and (n + 2) in current_set:
                        cascade_3step_hits.add(n + 2)

                # 3연속 역폭포: n in prev2 -> n-1 in prev -> n-2 in current
                for n in prev2_set:
                    if (n - 1) in prev_set and (n - 2) in current_set:
                        reverse_3step_hits.add(n - 2)

            records.append({
                "round": round_no,
                "carryover_count": len(carries),
                "carryover_numbers": sorted(list(carries)),
                "neighbor_count": len(neighbor_hits),
                "neighbor_numbers": sorted(list(neighbor_hits)),
                "cascade_count": len(cascade_hits),
                "cascade_numbers": sorted(list(cascade_hits)),
                "reverse_cascade_count": len(reverse_hits),
                "reverse_cascade_numbers": sorted(list(reverse_hits)),
                "cascade_3step_count": len(cascade_3step_hits),
                "cascade_3step_numbers": sorted(list(cascade_3step_hits)),
                "reverse_cascade_3step_count": len(reverse_3step_hits),
                "reverse_cascade_3step_numbers": sorted(list(reverse_3step_hits)),
            })

        history_numbers.append((round_no, current_set))

    return pd.DataFrame(records)


def summarize_flow_statistics(flow_history_df: pd.DataFrame) -> dict[str, float]:
    """
    이웃·폭포·역폭포 이력 데이터프레임으로부터 통계적 기대치 및 평균 출현율을 산출합니다.
    """
    if flow_history_df.empty:
        return {
            "avg_neighbor_count": 0.0,
            "avg_carryover_count": 0.0,
            "avg_cascade_count": 0.0,
            "avg_reverse_cascade_count": 0.0,
            "rate_neighbor_at_least_1": 0.0,
            "rate_cascade_at_least_1": 0.0,
            "rate_reverse_at_least_1": 0.0,
        }

    total_rounds = len(flow_history_df)
    return {
        "avg_neighbor_count": float(flow_history_df["neighbor_count"].mean()),
        "avg_carryover_count": float(flow_history_df["carryover_count"].mean()),
        "avg_cascade_count": float(flow_history_df["cascade_count"].mean()),
        "avg_reverse_cascade_count": float(flow_history_df["reverse_cascade_count"].mean()),
        "rate_neighbor_at_least_1": float((flow_history_df["neighbor_count"] >= 1).mean()),
        "rate_cascade_at_least_1": float((flow_history_df["cascade_count"] >= 1).mean()),
        "rate_reverse_at_least_1": float((flow_history_df["reverse_cascade_count"] >= 1).mean()),
    }


def next_round_flow_targets(df: pd.DataFrame) -> FlowTargets:
    """
    현재까지의 추첨 데이터를 바탕으로, 다음 회차에서 주목해야 할
    이웃수, 폭포수, 역폭포수, 3연속 대각선 유력 후보 및 번호별 흐름 점수를 도출합니다.
    """
    if len(df) < 1:
        raise ValueError("흐름 분석을 위해 최소 1회 이상의 데이터가 필요합니다.")

    last_row = df.iloc[-1]
    target_round = int(last_row["round"]) + 1
    prev_numbers = sorted(list(row_numbers(last_row)))

    prev2_numbers: list[int] = []
    if len(df) >= 2:
        prev2_numbers = sorted(list(row_numbers(df.iloc[-2])))

    prev_set = set(prev_numbers)
    prev2_set = set(prev2_numbers)

    # 1. 기본 후보군 산출
    carryover_candidates = sorted(list(prev_set))
    neighbor_candidates = sorted(list(get_neighbors(prev_numbers)))
    cascade_step2_candidates = sorted(list(get_cascade_candidates(prev_numbers)))
    reverse_step2_candidates = sorted(list(get_reverse_cascade_candidates(prev_numbers)))

    # 2. 3연속 대각선 후보 산출
    # 폭포 3단계 완성 후보: n in prev2 -> n+1 in prev 이면 다음 번호는 n+2
    cascade_step3_candidates = []
    for n in prev2_set:
        if (n + 1) in prev_set and (n + 2) <= 45:
            cascade_step3_candidates.append(n + 2)
    cascade_step3_candidates = sorted(list(set(cascade_step3_candidates)))

    # 역폭포 3단계 완성 후보: n in prev2 -> n-1 in prev 이면 다음 번호는 n-2
    reverse_step3_candidates = []
    for n in prev2_set:
        if (n - 1) in prev_set and (n - 2) >= 1:
            reverse_step3_candidates.append(n - 2)
    reverse_step3_candidates = sorted(list(set(reverse_step3_candidates)))

    # 3. 1~45 각 번호의 대각선 흐름 가중치 점수(0.0 ~ 1.0) 계산
    # 가중치 설계:
    # - 3단계 폭포/역폭포 유력 번호: 강한 대각선 연속성이 형성되어 있으므로 보너스 부여
    # - 2단계 폭포/역폭포 번호: 일반 이웃수 대비 방향성(+1 또는 -1) 모멘텀 반영
    # - 일반 이웃수 / 이월수: 안정적 기본 흐름 반영
    number_scores: dict[int, float] = {}
    for num in NUMBERS:
        score = 0.20  # 기본 점수

        # 이월수 여부
        if num in prev_set:
            score += 0.25

        # 이웃수 여부
        if num in neighbor_candidates:
            score += 0.30

        # 폭포 2단계(+1) 여부
        if num in cascade_step2_candidates:
            score += 0.15

        # 역폭포 2단계(-1) 여부
        if num in reverse_step2_candidates:
            score += 0.15

        # 3연속 폭포 완성 후보
        if num in cascade_step3_candidates:
            score += 0.25

        # 3연속 역폭포 완성 후보
        if num in reverse_step3_candidates:
            score += 0.25

        # 0.0 ~ 1.0 범위로 정규화
        number_scores[num] = min(1.0, score)

    return FlowTargets(
        target_round=target_round,
        previous_numbers=prev_numbers,
        prev2_numbers=prev2_numbers,
        carryover_candidates=carryover_candidates,
        neighbor_candidates=neighbor_candidates,
        cascade_step2_candidates=cascade_step2_candidates,
        cascade_step3_candidates=cascade_step3_candidates,
        reverse_step2_candidates=reverse_step2_candidates,
        reverse_step3_candidates=reverse_step3_candidates,
        number_flow_scores=number_scores,
    )
