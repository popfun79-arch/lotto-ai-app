from __future__ import annotations

"""
[모듈 설명: Final Pattern 통합 추천 및 조합 생성 엔진]
이 모듈은 번호별 Master Score, 합계 시계열 예측 구간, 건너띔(GAP/Skip) 합계 및 전이 확률,
홀짝/고저/번호대/끝수 밸런스, 그리고 대각선 흐름(이웃수, 폭포수, 역폭포수, 이월수)을
종합적으로 평가하여 가장 수학적 균형성이 뛰어난 최적의 6개 번호 조합들을 도출합니다.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from lotto64.analysis.flow_pattern import next_round_flow_targets
from lotto64.analysis.gap import build_gap_tables, current_gap_table
from lotto64.analysis.gap_sum_series import (
    forecast_next_gap_sum,
    gap_sum_pattern_score,
)
from lotto64.analysis.sum_series import forecast_next_sum, sum_pattern_score
from lotto64.analysis.skip_pattern import (
    SKIP_BUCKETS,
    build_empirical_hazard,
    current_skip_profile,
    forecast_next_skip_pattern,
    skip_bucket,
    skip_sum_pattern_score,
)
from lotto64.models.pattern_master import (
    candidate_sets,
    gap_bucket,
    master_number_scores,
)
from lotto64.utils.lotto_math import (
    ac_value,
    low_count,
    max_consecutive_run,
    odd_count,
    zone_counts,
)


BUCKETS = ["0", "1-2", "3-5", "6-10", "11-16", "17+"]


def _feature_distributions(df: pd.DataFrame, window: int = 50) -> dict:
    """
    최근 관찰 회차(window) 동안의 홀수, 저번호, 번호대, 끝수, AC값 분포를 계산하여
    가장 자주 등장하는 최적 형태에 정규화된 가중치를 부여합니다.
    """
    recent = df.tail(min(window, len(df)))
    odd_values = Counter()
    low_values = Counter()
    zones_values = Counter()
    last_values = Counter()
    ac_values = Counter()

    for _, row in recent.iterrows():
        nums = [int(row[f"n{i}"]) for i in range(1, 7)]
        odd_values[odd_count(nums)] += 1
        low_values[low_count(nums)] += 1
        zones_values[sum(v > 0 for v in zone_counts(nums))] += 1
        last_values[len({n % 10 for n in nums})] += 1
        ac_values[ac_value(nums)] += 1

    def normalize(counter: Counter) -> dict:
        top = max(counter.values()) if counter else 1
        return {key: value / top for key, value in counter.items()}

    return {
        "odd": normalize(odd_values),
        "low": normalize(low_values),
        "zones": normalize(zones_values),
        "last": normalize(last_values),
        "ac": normalize(ac_values),
    }


def _gap_bucket_means(df: pd.DataFrame, window: int = 50) -> dict[str, float]:
    """
    최근 50회 동안 당첨된 6개 번호의 GAP 구간 구성 평균(타깃)을 계산합니다.
    """
    _, rounds = build_gap_tables(df)
    recent = rounds.tail(min(window, len(rounds)))
    rows = []

    for values in recent["gap_values"]:
        counts = Counter(
            gap_bucket(int(value))
            for value in values
            if pd.notna(value)
        )
        rows.append({bucket: counts[bucket] for bucket in BUCKETS})

    if not rows:
        return {bucket: 1.0 for bucket in BUCKETS}

    frame = pd.DataFrame(rows)
    return frame.mean().to_dict()


def _bucket_composition_score(
    combo: tuple[int, ...],
    gap_map: dict[int, int],
    target: dict[str, float],
) -> float:
    """
    조합 내 6개 번호의 GAP 구간 구성이 목표 구성과 얼마나 일치하는지 점수화합니다.
    """
    counts = Counter(gap_bucket(gap_map[n]) for n in combo)
    distance = sum(
        abs(float(counts[bucket]) - float(target[bucket]))
        for bucket in BUCKETS
    )
    return float(max(0.0, 1.0 - distance / 12.0))


def _candidate_pattern_ok(
    combo: tuple[int, ...],
    skip_map: dict[int, int],
    sum_low: float,
    sum_high: float,
    skip_low: float,
    skip_high: float,
) -> bool:
    """
    통계적 비정상 조합(예: 홀수 6개 몰림, 4연속 연속번호, 합계 이탈 등)을 사전에 걸러내는 엄격 필터입니다.
    """
    total = sum(combo)
    skip_total = sum(skip_map[n] for n in combo)
    buckets = Counter(skip_bucket(skip_map[n]) for n in combo)

    return (
        sum_low <= total <= sum_high
        and skip_low <= skip_total <= skip_high
        and max_consecutive_run(combo) < 4
        and odd_count(combo) not in (0, 6)
        and low_count(combo) not in (0, 6)
        and sum(v > 0 for v in zone_counts(combo)) >= 3
        and buckets["17+"] <= 1
        and buckets["0"] <= 1
        and buckets["6-10"] >= 1
        and buckets["3-5"] <= 2
        and buckets["11-16"] <= 2
    )


def rank_final_combinations(
    df: pd.DataFrame,
    pool_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    최상위 후보 번호 풀(pool_size=20)에서 가능한 모든 6개 조합(38,760개)을 생성하고,
    시계열 합계, GAP, Skip 전이, 밸런스, 대각선 흐름(이웃·폭포·역폭포)을 결합하여 순위를 매깁니다.
    """
    master = master_number_scores(df)
    pool = master.head(pool_size)["number"].astype(int).tolist()

    master_map = dict(zip(master["number"], master["master_score"]))
    pool_scores = np.asarray([master_map[n] for n in pool], dtype=float)
    low_score, high_score = float(pool_scores.min()), float(pool_scores.max())

    def normalized_number_score(number: int) -> float:
        if np.isclose(low_score, high_score):
            return 0.5
        return (master_map[number] - low_score) / (high_score - low_score)

    gaps = current_gap_table(df)
    gap_map = dict(zip(
        gaps["number"].astype(int),
        gaps["current_gap"].astype(int),
    ))

    sum_forecast = forecast_next_sum(
        df,
        state_window=50,
        transition_lookback=100,
    )
    gap_forecast = forecast_next_gap_sum(
        df,
        state_window=50,
        transition_lookback=100,
    )
    target_buckets = _gap_bucket_means(df, 50)
    feature_dist = _feature_distributions(df, 50)

    # skip=0 기준 합계와 전용 예측 구간
    skip_profile = current_skip_profile(df).set_index("number")
    skip_map = skip_profile["current_skip"].astype(int).to_dict()
    skip_forecast = forecast_next_skip_pattern(
        df,
        state_window=50,
        transition_lookback=min(120, max(30, len(df) - 1)),
    )
    skip_target = skip_forecast.bucket_target
    skip_hazard = build_empirical_hazard(df).set_index("number")
    skip_hazard["hazard_rank_score"] = skip_hazard["empirical_hazard"].rank(pct=True)

    # 직전 회차 당첨 번호 및 대각선 흐름 타깃
    flow_targets = next_round_flow_targets(df)
    latest = set(flow_targets.previous_numbers)
    neighbors_set = set(flow_targets.neighbor_candidates)
    cascade_set = set(flow_targets.cascade_step2_candidates)
    reverse_cascade_set = set(flow_targets.reverse_step2_candidates)
    cascade_3step_set = set(flow_targets.cascade_step3_candidates)
    reverse_3step_set = set(flow_targets.reverse_step3_candidates)

    rows = []
    for combo in combinations(pool, 6):
        combo = tuple(sorted(combo))
        total = sum(combo)
        skip_total = sum(skip_map[n] for n in combo)
        legacy_gap_total = skip_total + len(combo)

        if not _candidate_pattern_ok(
            combo,
            skip_map,
            sum_forecast.target_low,
            sum_forecast.target_high,
            max(0.0, skip_forecast.wide_low - 3.0),
            skip_forecast.wide_high + 3.0,
        ):
            continue

        number_score = float(np.mean([
            normalized_number_score(n)
            for n in combo
        ]))
        draw_sum_score = sum_pattern_score(total, sum_forecast)
        gap_total_score = gap_sum_pattern_score(legacy_gap_total, gap_forecast)
        skip_total_score = skip_sum_pattern_score(skip_total, skip_forecast)
        skip_counts = Counter(
            skip_bucket(skip_map[n])
            for n in combo
        )
        skip_distance = sum(
            abs(float(skip_counts[bucket]) - float(skip_target[bucket]))
            for bucket in SKIP_BUCKETS
        )
        skip_period_score = float(max(0.0, 1.0 - skip_distance / 12.0))
        skip_hazard_score_value = float(np.mean([
            float(skip_hazard.loc[n, "hazard_rank_score"])
            for n in combo
        ]))
        bucket_score = _bucket_composition_score(
            combo,
            gap_map,
            target_buckets,
        )

        odd = odd_count(combo)
        low = low_count(combo)
        used_zones = sum(v > 0 for v in zone_counts(combo))
        unique_last = len({n % 10 for n in combo})
        ac = ac_value(combo)
        carry = len(set(combo) & latest)

        # 대각선 흐름(이웃수, 폭포수, 역폭포수) 카운트
        combo_set = set(combo)
        neighbor_cnt = len(combo_set & neighbors_set)
        cascade_cnt = len(combo_set & cascade_set)
        rev_cascade_cnt = len(combo_set & reverse_cascade_set)
        cascade_3step_cnt = len(combo_set & cascade_3step_set)
        rev_3step_cnt = len(combo_set & reverse_3step_set)

        balance_score = (
            feature_dist["odd"].get(odd, 0.0)
            + feature_dist["low"].get(low, 0.0)
        ) / 2
        zone_score = feature_dist["zones"].get(used_zones, 0.0)
        last_score = feature_dist["last"].get(unique_last, 0.0)
        ac_score = feature_dist["ac"].get(ac, 0.0)

        # 이월수 점수
        if carry == 1:
            carry_score = 1.0
        elif carry in (0, 2):
            carry_score = 0.80
        else:
            carry_score = 0.35

        # 최종 스코어 앙상블 (기존 12개 가중치 인터페이스 100% 호환 보존)
        final_score = (
            0.25 * number_score
            + 0.17 * draw_sum_score
            + 0.08 * gap_total_score
            + 0.06 * bucket_score
            + 0.14 * skip_total_score
            + 0.10 * skip_period_score
            + 0.05 * skip_hazard_score_value
            + 0.05 * balance_score
            + 0.03 * zone_score
            + 0.02 * last_score
            + 0.02 * ac_score
            + 0.03 * carry_score
        )

        bucket_counts = Counter(gap_bucket(gap_map[n]) for n in combo)

        # 대각선 흐름 직관적 요약 문자열
        flow_parts = []
        if carry > 0:
            flow_parts.append(f"이월{carry}")
        if neighbor_cnt > 0:
            flow_parts.append(f"이웃{neighbor_cnt}")
        if cascade_cnt > 0:
            flow_parts.append(f"폭포{cascade_cnt}")
        if rev_cascade_cnt > 0:
            flow_parts.append(f"역폭포{rev_cascade_cnt}")
        if cascade_3step_cnt > 0:
            flow_parts.append(f"3연속폭포{cascade_3step_cnt}")
        if rev_3step_cnt > 0:
            flow_parts.append(f"3연속역폭포{rev_3step_cnt}")
        flow_summary_str = " · ".join(flow_parts) if flow_parts else "신규독립"

        rows.append({
            "combination": combo,
            "final_score": final_score,
            "number_score": number_score,
            "sum": total,
            "sum_pattern_score": draw_sum_score,
            "gap_sum": legacy_gap_total,
            "gap_sum_pattern_score": gap_total_score,
            "gap_bucket_score": bucket_score,
            "skip_sum": skip_total,
            "skip_sum_pattern_score": skip_total_score,
            "skip_period_pattern_score": skip_period_score,
            "skip_empirical_hazard_score": skip_hazard_score_value,
            "odd_count": odd,
            "low_count": low,
            "used_zones": used_zones,
            "unique_last_digits": unique_last,
            "ac": ac,
            "carryover_count": carry,
            "neighbor_count": neighbor_cnt,
            "cascade_count": cascade_cnt,
            "reverse_cascade_count": rev_cascade_cnt,
            "cascade_3step_count": cascade_3step_cnt,
            "reverse_cascade_3step_count": rev_3step_cnt,
            "flow_summary": flow_summary_str,
            "gap_pattern": "/".join(
                f"{bucket}:{bucket_counts[bucket]}"
                for bucket in BUCKETS
            ),
            "skip_pattern": "/".join(
                f"{bucket}:{skip_counts[bucket]}"
                for bucket in SKIP_BUCKETS
            ),
        })

    ranked = pd.DataFrame(rows)
    if not ranked.empty:
        ranked = ranked.sort_values(
            ["final_score", "combination"],
            ascending=[False, True],
        ).reset_index(drop=True)

    context = {
        "candidate_sets": candidate_sets(master),
        "sum_forecast": sum_forecast.to_dict(),
        "gap_sum_forecast": gap_forecast.to_dict(),
        "skip_pattern_forecast": skip_forecast.to_dict(),
        "flow_targets": flow_targets,
        "pool": sorted(pool),
        "gap_bucket_target_mean": target_buckets,
        "final_score_weights": {
            "number": 0.25,
            "number_sum": 0.17,
            "legacy_gap_sum": 0.08,
            "gap_bucket": 0.06,
            "skip_sum_transition": 0.14,
            "skip_bucket_transition": 0.10,
            "skip_empirical_hazard": 0.05,
            "balance": 0.05,
            "zone": 0.03,
            "last_digit": 0.02,
            "ac": 0.02,
            "carry": 0.03,
        },
        "skip_period_enabled": True,
        "skip_period_note": (
            "skip=0 기준 합계 상태·방향 전이 + 구간별 6개 구성 + "
            "empirical hazard + 대각선 흐름(이웃·폭포·역폭포)을 최종 점수에 반영"
        ),
    }
    return ranked, master, context


def build_final_portfolio(
    ranked: pd.DataFrame,
    size: int = 20,
    max_overlap: int = 4,
    max_exposure: int = 10,
    exposure_penalty: float = 0.005,
) -> pd.DataFrame:
    """
    최상위 랭킹 조합들 중 번호 쏠림을 방지하고 분산 투자를 위한 최종 포트폴리오를 구성합니다.
    """
    if ranked.empty:
        return ranked

    selected: list[dict] = []
    exposure: Counter = Counter()
    remaining = ranked.copy()

    while len(selected) < size and not remaining.empty:
        best = None
        best_adjusted = -1e9

        for _, row in remaining.iterrows():
            combo = tuple(row["combination"])

            if any(exposure[n] >= max_exposure for n in combo):
                continue
            if any(
                len(set(combo) & set(item["combination"])) > max_overlap
                for item in selected
            ):
                continue

            adjusted = float(row["final_score"]) - exposure_penalty * sum(
                exposure[n] for n in combo
            )
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best = row.to_dict()

        if best is None:
            max_exposure += 1
            if max_exposure > size:
                break
            continue

        selected.append(best)
        for number in best["combination"]:
            exposure[number] += 1

        chosen = tuple(best["combination"])
        remaining = remaining[
            remaining["combination"].apply(tuple) != chosen
        ]

    return pd.DataFrame(selected).reset_index(drop=True)


def final_recommendation_bundle(df: pd.DataFrame) -> dict:
    """
    Master Score, 후보 번호 풀, 전체 랭킹 조합 및 최종 분산 포트폴리오를 번들링하여 반환합니다.
    """
    ranked, master, context = rank_final_combinations(df, pool_size=20)
    portfolio = build_final_portfolio(ranked, size=20)

    return {
        "master_scores": master,
        "candidate_sets": context["candidate_sets"],
        "ranked": ranked,
        "portfolio": portfolio,
        "context": context,
    }
