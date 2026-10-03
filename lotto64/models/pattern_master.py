from __future__ import annotations

"""
[모듈 설명: Gail Howard 《Lottery Master Guide》 핵심 철학 + 대각선 흐름(이웃·폭포·역폭포) 통합 번호 평가 모델]
이 모듈은 게일 하워드의 핵심 분석 체계(Games Out, Skip Hazard, Skips Due, Hot/Cold, Number Groups, Last Digits)와
한국 로또에서 빈번하게 발생하는 대각선 흐름 패턴(이웃수, 우하향 폭포 +1, 좌하향 역폭포 -1, 3연속 계단식 폭포)을
현대적인 Python 통계 가중치 앙상블로 융합하여 1~45 각 번호의 최종 점수(master_score)를 산출합니다.
"""

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from lotto64.analysis.flow_pattern import next_round_flow_targets
from lotto64.analysis.gap import current_gap_table, row_numbers
from lotto64.config import MAIN_COLUMNS, NUMBERS
from lotto64.models.scoring import number_scores
from lotto64.utils.lotto_math import neighbor_set, zone_index


MASTER_WEIGHTS = {
    # 1. Gail Howard 계열 Games Out / Skips 가중치
    "skip_hit_recent50": 0.18,      # 최근 50회 기준 해당 건너띔 구간의 적중 위험율(Hazard)
    "skip_hit_recent100": 0.10,     # 최근 100회 기준 해당 건너띔 구간의 적중 위험율
    "drawings_since_hit": 0.10,     # 현재 미출현 회차(건너띔 백분위)
    "skips_due": 0.08,              # 개인 평균 건너띔 대비 현재 경과 비율 (출현 임박도)
    
    # 2. Python 시계열 롤링 빈도 / 추세 가중치
    "hot_20": 0.10,                 # 최근 20회 출현 빈도 (Hot 번호)
    "hot_50": 0.08,                 # 최근 50회 출현 빈도
    "frequency_trend": 0.08,        # 최근 50회 vs 이전 50회 출현 모멘텀 추세
    
    # 3. Gail Howard 번호대 및 끝수 복원 가중치
    "number_group_recovery": 0.06,  # 최근 침체된 번호대의 회복 탄력성 점수
    "last_digit_recovery": 0.05,    # 최근 침체된 끝수(0~9)의 회복 탄력성 점수
    
    # 4. 대각선 흐름(이웃·폭포·역폭포·이월수) 종합 가중치
    "multiple_hit_neighbor": 0.07,  # 이웃수(±1), 2단계/3단계 폭포(+1), 역폭포(-1) 모멘텀
    
    # 5. 기존 Python / DNA / GAP 통계 앙상블
    "python_base": 0.10,            # 기존 기본 스코어링 모델 결과
}


def gap_bucket(gap: int) -> str:
    """
    건너띔(GAP) 일수를 Gail Howard 스타일의 핵심 구간으로 분류합니다.
    """
    gap = int(gap)
    if gap == 0:
        return "0"
    if gap <= 2:
        return "1-2"
    if gap <= 5:
        return "3-5"
    if gap <= 10:
        return "6-10"
    if gap <= 16:
        return "11-16"
    return "17+"


def _minmax(values: Iterable[float]) -> np.ndarray:
    """
    값들의 리스트를 0.0 ~ 1.0 범위로 최소-최대 정규화합니다.
    """
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return arr
    low, high = float(arr.min()), float(arr.max())
    if np.isclose(low, high):
        return np.full(len(arr), 0.5)
    return (arr - low) / (high - low)


def _window_counts(df: pd.DataFrame) -> Counter:
    """
    주어진 회차 구간에서 각 번호의 출현 횟수를 카운트합니다.
    """
    counts: Counter = Counter()
    for _, row in df.iterrows():
        counts.update(row_numbers(row))
    return counts


def _games_out_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    과거 전 회차를 순회하며 각 번호가 출현했을 때의 직전 미출현 회차(Games Out / Skips)를 기록합니다.
    """
    last_seen = {n: None for n in NUMBERS}
    records = []

    for _, row in df.iterrows():
        round_no = int(row["round"])
        hits = set(row_numbers(row))

        for number in NUMBERS:
            if last_seen[number] is None:
                continue
            gap = round_no - int(last_seen[number]) - 1
            records.append({
                "round": round_no,
                "number": number,
                "gap": gap,
                "gap_bucket": gap_bucket(gap),
                "hit": int(number in hits),
            })

        for number in hits:
            last_seen[number] = round_no

    return pd.DataFrame(records)


def _hazard_by_bucket(
    df: pd.DataFrame,
    lookback: int,
) -> dict[str, float]:
    """
    특정 관찰 기간(lookback 회차) 동안 각 GAP 구간에서 실제 당첨이 발생한 경험적 위험율(Empirical Hazard)을 산출합니다.
    """
    records = _games_out_records(df)
    if records.empty:
        return {}

    latest = int(df["round"].max())
    records = records[records["round"] > latest - lookback]
    grouped = records.groupby("gap_bucket").agg(
        exposures=("hit", "size"),
        hits=("hit", "sum"),
    )
    grouped["rate"] = (grouped["hits"] + 1) / (grouped["exposures"] + 2)

    raw = grouped["rate"].to_dict()
    keys = list(raw)
    normalized = _minmax(raw.values())
    return dict(zip(keys, normalized))


def master_number_scores(
    df: pd.DataFrame,
    egr_threshold: int = 17,
    similarity_k: int = 15,
) -> pd.DataFrame:
    """
    1~45 각 번호의 최종 Master Score를 산출합니다.

    핵심 분석 결합 항목:
    - Drawings Since Hit / Games Out (미출현 간격)
    - Skip-and-Hit empirical hazard (경험적 출현 위험율)
    - Skips Due (현재 건너띔 / 개인 평균 건너띔 비율)
    - 최근 20/50회 Hot, 이전 50회 대비 모멘텀 Trend
    - Number Groups (번호대 복원력) / Last Digits (끝수 복원력)
    - 직전 회차 이월수 및 이웃수, 2단계/3단계 폭포(+1) 및 역폭포(-1) 대각선 흐름
    - 기존 Python 통계/DNA/GAP 앙상블 점수
    """
    if len(df) < 30:
        raise ValueError("Pattern Master에는 최소 30회 데이터가 필요합니다.")

    gaps = current_gap_table(df)
    gap_map = gaps.set_index("number")["current_gap"].to_dict()
    gap_pct = gaps.set_index("number")["gap_percentile"].to_dict()

    hazard50 = _hazard_by_bucket(df, 50)
    hazard100 = _hazard_by_bucket(df, 100)

    recent20 = _window_counts(df.tail(min(20, len(df))))
    recent50 = _window_counts(df.tail(min(50, len(df))))

    if len(df) >= 100:
        previous50 = _window_counts(df.iloc[-100:-50])
    elif len(df) > 50:
        previous50 = _window_counts(df.iloc[:-50])
    else:
        previous50 = Counter()

    hot20_norm = dict(zip(
        NUMBERS,
        _minmax(recent20[n] for n in NUMBERS),
    ))
    hot50_norm = dict(zip(
        NUMBERS,
        _minmax(recent50[n] for n in NUMBERS),
    ))
    trend_norm = dict(zip(
        NUMBERS,
        _minmax(recent50[n] - previous50[n] for n in NUMBERS),
    ))

    # 최근 20회 번호대 및 끝수 복원력 분석
    zone_counts_counter: Counter = Counter()
    digit_counts_counter: Counter = Counter()
    for _, row in df.tail(min(20, len(df))).iterrows():
        for number in row_numbers(row):
            zone_counts_counter[zone_index(number)] += 1
            digit_counts_counter[number % 10] += 1

    zone_values = [zone_counts_counter[z] for z in range(5)]
    zmin, zmax = min(zone_values), max(zone_values)
    zone_recovery = {
        z: 1 - (zone_counts_counter[z] - zmin) / (zmax - zmin or 1)
        for z in range(5)
    }

    digit_values = [digit_counts_counter[d] for d in range(10)]
    dmin, dmax = min(digit_values), max(digit_values)
    digit_recovery = {
        d: 1 - (digit_counts_counter[d] - dmin) / (dmax - dmin or 1)
        for d in range(10)
    }

    # 기본 베이스 스코어링
    base = number_scores(
        df,
        egr_threshold=egr_threshold,
        similarity_k=similarity_k,
    )
    base_map = dict(zip(base["number"], base["final_score"]))
    base_norm = dict(zip(
        NUMBERS,
        _minmax(base_map[n] for n in NUMBERS),
    ))

    # 대각선 흐름(이웃수, 폭포수, 역폭포수, 3연속 대각선) 타깃 및 점수 추출
    flow_targets = next_round_flow_targets(df)
    flow_score_map = flow_targets.number_flow_scores

    rows = []
    for number in NUMBERS:
        gap = int(gap_map[number])
        bucket = gap_bucket(gap)
        gap_row = gaps[gaps["number"] == number].iloc[0]
        average_gap = float(gap_row["average_gap"]) if pd.notna(
            gap_row["average_gap"]
        ) else np.nan

        # 출현 주기 도달 비율(Skips Due)
        if pd.notna(average_gap) and average_gap > 0:
            due_ratio = gap / average_gap
            skips_due = min(1.0, due_ratio / 1.2)
        else:
            skips_due = 0.5

        # 대각선 흐름 모멘텀 점수
        multiple_context = float(flow_score_map.get(number, 0.35))

        components = {
            "skip_hit_recent50": float(hazard50.get(bucket, 0.5)),
            "skip_hit_recent100": float(hazard100.get(bucket, 0.5)),
            "drawings_since_hit": float(gap_pct[number]),
            "skips_due": float(skips_due),
            "hot_20": float(hot20_norm[number]),
            "hot_50": float(hot50_norm[number]),
            "frequency_trend": float(trend_norm[number]),
            "number_group_recovery": float(zone_recovery[zone_index(number)]),
            "last_digit_recovery": float(digit_recovery[number % 10]),
            "multiple_hit_neighbor": multiple_context,
            "python_base": float(base_norm[number]),
        }

        master_score = sum(
            MASTER_WEIGHTS[key] * components[key]
            for key in MASTER_WEIGHTS
        )

        rows.append({
            "number": number,
            "master_score": master_score,
            "current_gap": gap,
            "gap_bucket": bucket,
            "average_gap": average_gap,
            "is_neighbor": int(number in flow_targets.neighbor_candidates),
            "is_cascade": int(number in flow_targets.cascade_step2_candidates),
            "is_reverse_cascade": int(number in flow_targets.reverse_step2_candidates),
            "is_cascade_3step": int(number in flow_targets.cascade_step3_candidates),
            "is_reverse_cascade_3step": int(number in flow_targets.reverse_step3_candidates),
            "is_carryover": int(number in flow_targets.carryover_candidates),
            **{f"score_{key}": value for key, value in components.items()},
        })

    out = pd.DataFrame(rows).sort_values(
        ["master_score", "number"],
        ascending=[False, True],
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def candidate_sets(scores: pd.DataFrame) -> dict[int, list[int]]:
    """
    점수 순으로 상위 11개, 13개, 15개 후보 번호 셋을 구성합니다.
    """
    return {
        size: sorted(scores.head(size)["number"].astype(int).tolist())
        for size in (11, 13, 15)
    }
