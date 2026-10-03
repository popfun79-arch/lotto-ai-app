from __future__ import annotations

"""
[모듈 설명: Gail Howard 《Lottery Master Guide》 7대 핵심 분석 및 종합 진단기]
이 모듈은 세계적인 로또 복권 통계 분석가 게일 하워드(Gail Howard)의 분석 이론과
현대 Python 시계열·통계 분석 기법을 결합하여 다음 7대 핵심 지표를 종합 산출합니다.

1. Games Out / Skips (미출현 간격):
   - 각 번호가 몇 회 동안 출현하지 않았는지(건너띔)를 파악하고, 단기(0~5), 중기(6~10), 장기(11+) 구간으로 분류합니다.
2. Number Groups (번호대 5분할 분석):
   - 1~9(단번대), 10~19(10번대), 20~29(20번대), 30~39(30번대), 40~45(40번대)의 밸런스 및 멸구간(출현하지 않은 번호대)을 진단합니다.
3. Last Digits (끝수 분석):
   - 끝자리 숫자(0~9)의 출현 경향, 동끝수(동일한 끝수를 가진 번호 쌍)의 발생 빈도를 추적합니다.
4. Odd-Even (홀짝 밸런스):
   - 6개 당첨번호 중 홀수와 짝수의 비율(통계적으로 3:3, 4:2, 2:4가 전체의 약 80%를 차지)을 분석합니다.
5. High-Low (고저 밸런스):
   - 저번호(1~22)와 고번호(23~45)의 황금 비율을 진단합니다.
6. Hot / Cold (다출수와 희소수):
   - 최근 20회/50회 출현 빈도를 기준으로 과열수(Hot), 보온수(Warm), 냉각수(Cold)를 정밀 분류합니다.
7. Sum Balance (합계 밸런스 시계열):
   - 6개 당첨번호 총합의 이동평균 및 통계적 표준 구간(일반적으로 100 ~ 175)의 위치를 진단합니다.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from lotto64.analysis.flow_pattern import (
    FlowTargets,
    analyze_flow_history,
    next_round_flow_targets,
    summarize_flow_statistics,
)
from lotto64.analysis.gap import current_gap_table, row_numbers
from lotto64.analysis.skip_pattern import current_skip_profile
from lotto64.analysis.sum_series import forecast_next_sum
from lotto64.config import NUMBERS
from lotto64.utils.lotto_math import (
    NUMBER_GROUPS,
    NUMBER_GROUP_LABELS,
    end_digit_sum,
    low_count,
    odd_count,
    zone_counts,
    zone_index,
)


@dataclass(frozen=True)
class HowardSummaryReport:
    """
    게일 하워드 7대 분석 종합 진단 결과 보고서 데이터 클래스
    """
    target_round: int                      # 다음 추첨 회차 (예: 1243)
    latest_round: int                      # 최근 추첨 회차 (예: 1242)
    latest_numbers: list[int]              # 최근 당첨 번호 6개
    
    # 1. Games Out / Skips
    short_skips_count: int                 # 단기 미출수(0~5회) 개수
    mid_skips_count: int                   # 중기 미출수(6~10회) 개수
    long_skips_count: int                  # 장기 미출수(11회 이상) 개수
    cold_numbers_15plus: list[int]         # 15회 이상 장기 미출수 목록
    
    # 2. Number Groups (5대 구간)
    group_counts_latest: dict[str, int]    # 최근 회차 번호대별 출현 개수
    missing_groups_latest: list[str]       # 최근 회차 멸(출현 없는) 구간
    group_frequency_recent20: dict[str, int] # 최근 20회 번호대별 누적 출현 수
    
    # 3. Last Digits (끝수)
    last_digit_counts_latest: dict[int, int] # 최근 회차 끝수(0~9)별 개수
    same_digit_pairs_latest: list[int]     # 최근 회차에 2개 이상 나온 동끝수
    digit_frequency_recent20: dict[int, int] # 최근 20회 끝수별 누적 출현 수
    
    # 4. Odd-Even (홀짝)
    latest_odd_even: str                   # 최근 회차 홀짝 비율 (예: "3:3")
    odd_even_distribution_recent50: dict[str, int] # 최근 50회 홀짝 비율 분포
    
    # 5. High-Low (고저)
    latest_high_low: str                   # 최근 회차 고저 비율 (예: "3:3")
    high_low_distribution_recent50: dict[str, int] # 최근 50회 고저 비율 분포
    
    # 6. Hot / Warm / Cold
    hot_numbers: list[int]                 # 최근 20회 4회 이상 다출(Hot) 번호
    warm_numbers: list[int]                # 최근 20회 2~3회 출현(Warm) 번호
    cold_numbers: list[int]                # 최근 20회 0~1회 출현(Cold) 번호
    
    # 7. Sum Balance (합계)
    latest_sum: int                        # 최근 회차 당첨 번호 합계
    sum_target_center: float               # 다음 회차 합계 중심값 예측
    sum_target_low: float                  # 다음 회차 합계 핵심 하한선
    sum_target_high: float                 # 다음 회차 합계 핵심 상한선
    
    # + 대각선 흐름 분석 연동
    flow_targets: FlowTargets              # 다음 회차 이웃·폭포·역폭포 타깃
    flow_summary: dict[str, float]         # 흐름 평균 통계


def build_howard_summary(df: pd.DataFrame, window: int = 50) -> HowardSummaryReport:
    """
    주어진 로또 데이터를 바탕으로 게일 하워드 7대 분석 종합 보고서를 생성합니다.
    모든 분석은 미래 데이터를 참조하지 않고 과거 회차 기준으로만 안전하게 수행됩니다.
    """
    if len(df) < 10:
        raise ValueError("Gail Howard 분석을 위해 최소 10회 이상의 데이터가 필요합니다.")

    recent_df = df.tail(min(window, len(df))).reset_index(drop=True)
    last_row = df.iloc[-1]
    latest_round = int(last_row["round"])
    target_round = latest_round + 1
    latest_numbers = sorted(list(row_numbers(last_row)))

    # 1. Games Out / Skips
    skip_profile = current_skip_profile(df)
    short_skips = skip_profile[skip_profile["current_skip"] <= 5]
    mid_skips = skip_profile[(skip_profile["current_skip"] >= 6) & (skip_profile["current_skip"] <= 10)]
    long_skips = skip_profile[skip_profile["current_skip"] >= 11]
    cold_15plus = skip_profile[skip_profile["current_skip"] >= 15]["number"].astype(int).tolist()

    # 2. Number Groups (번호대 분석)
    latest_zones = zone_counts(latest_numbers)
    group_counts_latest = {
        NUMBER_GROUP_LABELS[i]: latest_zones[i]
        for i in range(len(NUMBER_GROUPS))
    }
    missing_groups = [
        NUMBER_GROUP_LABELS[i]
        for i in range(len(NUMBER_GROUPS))
        if latest_zones[i] == 0
    ]

    # 최근 20회 번호대 누적
    recent20 = df.tail(min(20, len(df)))
    group_freq20 = Counter()
    for _, r in recent20.iterrows():
        for n in row_numbers(r):
            group_freq20[zone_index(n)] += 1
    group_frequency_recent20 = {
        NUMBER_GROUP_LABELS[i]: group_freq20[i]
        for i in range(len(NUMBER_GROUPS))
    }

    # 3. Last Digits (끝수 분석)
    last_digits_latest = [n % 10 for n in latest_numbers]
    digit_counter_latest = Counter(last_digits_latest)
    last_digit_counts_latest = {d: digit_counter_latest[d] for d in range(10)}
    same_digit_pairs_latest = sorted([d for d, c in digit_counter_latest.items() if c >= 2])

    digit_freq20 = Counter()
    for _, r in recent20.iterrows():
        for n in row_numbers(r):
            digit_freq20[n % 10] += 1
    digit_frequency_recent20 = {d: digit_freq20[d] for d in range(10)}

    # 4. Odd-Even (홀짝)
    latest_odd = odd_count(latest_numbers)
    latest_odd_even = f"{latest_odd}:{6 - latest_odd}"

    odd_dist = Counter()
    for _, r in recent_df.iterrows():
        o = odd_count(row_numbers(r))
        odd_dist[f"{o}:{6 - o}"] += 1
    odd_even_distribution_recent50 = dict(odd_dist)

    # 5. High-Low (고저)
    latest_low = low_count(latest_numbers)
    latest_high_low = f"{latest_low}:{6 - latest_low}"

    high_low_dist = Counter()
    for _, r in recent_df.iterrows():
        lo = low_count(row_numbers(r))
        high_low_dist[f"{lo}:{6 - lo}"] += 1
    high_low_distribution_recent50 = dict(high_low_dist)

    # 6. Hot / Warm / Cold (최근 20회 기준)
    counts20 = Counter()
    for _, r in recent20.iterrows():
        counts20.update(row_numbers(r))

    hot_numbers = sorted([n for n in NUMBERS if counts20[n] >= 4])
    warm_numbers = sorted([n for n in NUMBERS if 2 <= counts20[n] <= 3])
    cold_numbers = sorted([n for n in NUMBERS if counts20[n] <= 1])

    # 7. Sum Balance (합계)
    latest_sum = int(sum(latest_numbers))
    sum_fc = forecast_next_sum(df, state_window=50, transition_lookback=min(100, len(df) - 1))

    # + 대각선 흐름(이웃·폭포·역폭포) 타깃 연동
    flow_targets = next_round_flow_targets(df)
    flow_history = analyze_flow_history(recent_df)
    flow_summary = summarize_flow_statistics(flow_history)

    return HowardSummaryReport(
        target_round=target_round,
        latest_round=latest_round,
        latest_numbers=latest_numbers,
        short_skips_count=len(short_skips),
        mid_skips_count=len(mid_skips),
        long_skips_count=len(long_skips),
        cold_numbers_15plus=cold_15plus,
        group_counts_latest=group_counts_latest,
        missing_groups_latest=missing_groups,
        group_frequency_recent20=group_frequency_recent20,
        last_digit_counts_latest=last_digit_counts_latest,
        same_digit_pairs_latest=same_digit_pairs_latest,
        digit_frequency_recent20=digit_frequency_recent20,
        latest_odd_even=latest_odd_even,
        odd_even_distribution_recent50=odd_even_distribution_recent50,
        latest_high_low=latest_high_low,
        high_low_distribution_recent50=high_low_distribution_recent50,
        hot_numbers=hot_numbers,
        warm_numbers=warm_numbers,
        cold_numbers=cold_numbers,
        latest_sum=latest_sum,
        sum_target_center=float(sum_fc.target_center),
        sum_target_low=float(sum_fc.target_low),
        sum_target_high=float(sum_fc.target_high),
        flow_targets=flow_targets,
        flow_summary=flow_summary,
    )
