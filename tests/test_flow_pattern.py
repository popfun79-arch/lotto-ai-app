from __future__ import annotations

import pandas as pd
import pytest

from lotto64.analysis.flow_pattern import (
    analyze_flow_history,
    get_cascade_candidates,
    get_neighbors,
    get_reverse_cascade_candidates,
    next_round_flow_targets,
    summarize_flow_statistics,
)


def test_neighbor_and_cascade_candidates():
    # 번호 10에 대해:
    # 이웃수: 9, 11
    # 폭포 후보(+1): 11
    # 역폭포 후보(-1): 9
    assert get_neighbors([10]) == {9, 11}
    assert get_cascade_candidates([10]) == {11}
    assert get_reverse_cascade_candidates([10]) == {9}

    # 경계값 테스트: 1과 45
    assert get_neighbors([1]) == {2}
    assert get_neighbors([45]) == {44}
    assert get_cascade_candidates([45]) == set()
    assert get_reverse_cascade_candidates([1]) == set()


def test_flow_history_cascade_and_reverse_cascade():
    # 회차 시나리오 구성:
    # 1회차: 1, 10, 20, 30, 40, 44
    # 2회차: 2 (1의 폭포수 +1), 10 (이월수), 19 (20의 역폭포수 -1), 31 (30의 폭포수 +1), 41 (40의 폭포수 +1), 45 (44의 폭포수 +1)
    # 3회차: 3 (1->2->3: 3연속 폭포!), 18 (20->19->18: 3연속 역폭포!), 11, 25, 35, 42
    df = pd.DataFrame([
        {"round": 1, "n1": 1, "n2": 10, "n3": 20, "n4": 30, "n5": 40, "n6": 44, "bonus": 7},
        {"round": 2, "n1": 2, "n2": 10, "n3": 19, "n4": 31, "n5": 41, "n6": 45, "bonus": 8},
        {"round": 3, "n1": 3, "n2": 11, "n3": 18, "n4": 25, "n5": 35, "n6": 42, "bonus": 9},
    ])

    history = analyze_flow_history(df)
    assert len(history) == 2

    # 2회차 결과 검증
    r2 = history[history["round"] == 2].iloc[0]
    assert r2["carryover_count"] == 1  # 번호 10 이월
    assert 10 in r2["carryover_numbers"]
    assert r2["cascade_count"] == 4    # 2, 31, 41, 45 (각각 1, 30, 40, 44의 +1)
    assert r2["reverse_cascade_count"] == 1  # 19 (20의 -1)

    # 3회차 결과 검증 (3연속 폭포 및 3연속 역폭포)
    r3 = history[history["round"] == 3].iloc[0]
    assert 3 in r3["cascade_3step_numbers"]   # 1 -> 2 -> 3
    assert 18 in r3["reverse_cascade_3step_numbers"]  # 20 -> 19 -> 18

    # 요약 통계 검증
    summary = summarize_flow_statistics(history)
    assert summary["avg_carryover_count"] >= 0.0
    assert summary["rate_cascade_at_least_1"] == 1.0


def test_next_round_flow_targets():
    df = pd.DataFrame([
        {"round": 100, "n1": 10, "n2": 20, "n3": 25, "n4": 30, "n5": 35, "n6": 40, "bonus": 5},
        {"round": 101, "n1": 11, "n2": 19, "n3": 26, "n4": 31, "n5": 36, "n6": 41, "bonus": 6},
    ])

    targets = next_round_flow_targets(df)
    assert targets.target_round == 102
    assert 11 in targets.previous_numbers

    # 3단계 폭포 후보: 100회차 10 -> 101회차 11 이므로 다음 회차 12!
    assert 12 in targets.cascade_step3_candidates
    # 3단계 역폭포 후보: 100회차 20 -> 101회차 19 이므로 다음 회차 18!
    assert 18 in targets.reverse_step3_candidates

    # 점수 부여 확인 (12와 18은 3단계 보너스로 점수가 높아야 함)
    assert targets.number_flow_scores[12] > targets.number_flow_scores[5]
    assert targets.number_flow_scores[18] > targets.number_flow_scores[5]
