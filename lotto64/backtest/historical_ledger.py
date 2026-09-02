from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from lotto64.analysis.gap import current_gap_table, row_numbers
from lotto64.analysis.skip_pattern import skip_bucket
from lotto64.models.pattern_master import gap_bucket
from lotto64.recommend.final_pattern import final_recommendation_bundle

try:
    from lotto64.utils.lotto_math import NUMBER_GROUP_LABELS, zone_counts
except ImportError:
    # Partial-deployment safety: keep the canonical number groups available
    # when lotto_math.py has not yet been updated on the target environment.
    NUMBER_GROUP_LABELS = (
        "1~9",
        "10~19",
        "20~29",
        "30~39",
        "40~45",
    )

    def zone_counts(numbers):
        counts = [0, 0, 0, 0, 0]
        for raw in numbers:
            number = int(raw)
            if not 1 <= number <= 45:
                raise ValueError(f"로또 번호 범위 오류: {number}")
            if number <= 9:
                index = 0
            elif number <= 19:
                index = 1
            elif number <= 29:
                index = 2
            elif number <= 39:
                index = 3
            else:
                index = 4
            counts[index] += 1
        return tuple(counts)


DEFAULT_LEDGER_PATH = Path("reports/historical_validation_ledger.csv")
DEFAULT_SUMMARY_PATH = Path("reports/historical_validation_summary.json")


def required_history_start(
    latest_round: int,
    validation_rounds: int = 100,
    train_window: int = 200,
) -> int:
    """
    최근 validation_rounds개를 엄격한 train_window로 검증할 때
    필요한 가장 이른 회차를 계산합니다.

    예: latest=1236, validation=100, train=200 -> 937
    """
    return int(latest_round) - int(validation_rounds) - int(train_window) + 1


def max_strict_validation_rounds(
    df: pd.DataFrame,
    train_window: int = 200,
) -> int:
    return max(0, len(df) - int(train_window))


def _assert_consecutive(df: pd.DataFrame) -> None:
    rounds = df["round"].astype(int).sort_values().tolist()
    if not rounds:
        raise ValueError("데이터가 비어 있습니다.")

    missing = [
        value
        for value in range(rounds[0], rounds[-1] + 1)
        if value not in set(rounds)
    ]
    if missing:
        sample = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"누락 회차가 있습니다: {sample}{suffix}")


def _combo_max_hit(
    portfolio: pd.DataFrame,
    actual: set[int],
    size: int,
) -> int:
    if portfolio is None or portfolio.empty:
        return 0
    hits = [
        len(actual & set(combo))
        for combo in portfolio.head(size)["combination"].tolist()
    ]
    return max(hits) if hits else 0


def _actual_group_pattern(numbers: list[int]) -> str:
    counts = zone_counts(numbers)
    return " / ".join(
        f"{label}:{count}"
        for label, count in zip(NUMBER_GROUP_LABELS, counts)
    )


def _actual_gap_pattern(
    numbers: list[int],
    skip_map: dict[int, int],
) -> str:
    counts = Counter(
        gap_bucket(skip_map[number] + 1)
        for number in numbers
    )
    keys = ["0", "1-2", "3-5", "6-10", "11-16", "17+"]
    return " / ".join(f"{key}:{counts[key]}" for key in keys)


def _actual_skip_pattern(
    numbers: list[int],
    skip_map: dict[int, int],
) -> str:
    counts = Counter(skip_bucket(skip_map[number]) for number in numbers)
    keys = ["0", "1-2", "3-5", "6-10", "11-16", "17+"]
    return " / ".join(f"{key}:{counts[key]}" for key in keys)


def classify_failure(record: dict) -> tuple[str, str]:
    """
    하나의 '주요 실패 원인'과 다중 원인 문자열을 반환합니다.

    순서는 예측 파이프라인 앞단에서 뒷단으로 갑니다.
    """
    reasons: list[str] = []

    if int(record["candidate_15_hits"]) < 3:
        reasons.append("후보번호 단계 미포착")

    if not int(record["sum_core_hit"]):
        reasons.append("번호합 핵심구간 이탈")

    if not int(record["gap_sum_core_hit"]):
        reasons.append("GAP합 핵심구간 이탈")

    if not int(record.get("skip_sum_core_hit", 1)):
        reasons.append("건너띔 합계 핵심구간 이탈")

    if (
        int(record["candidate_15_hits"]) >= 3
        and int(record["best20_max_hit"]) < 3
    ):
        reasons.append("조합 구성/포트폴리오 단계 손실")

    if int(record["master_top20_hits"]) < 3:
        reasons.append("Pattern Master 상위권 포착 부족")

    if not reasons:
        reasons.append("주요 구조 적합·무작위 변동")

    primary = reasons[0]
    return primary, " | ".join(reasons)


def evaluate_target_round(
    df: pd.DataFrame,
    target_index: int,
    train_window: int = 200,
) -> dict:
    """
    target_index 회차는 절대 학습에 포함하지 않고 직전 train_window만 사용합니다.
    """
    if target_index < train_window:
        raise ValueError(
            f"엄격 Walk-forward에는 target 이전 {train_window}회가 필요합니다."
        )
    if target_index >= len(df):
        raise IndexError("target_index 범위 오류")

    train = df.iloc[
        target_index - train_window : target_index
    ].reset_index(drop=True)
    actual_row = df.iloc[target_index]
    actual_numbers = row_numbers(actual_row)
    actual = set(actual_numbers)
    target_round = int(actual_row["round"])

    bundle = final_recommendation_bundle(train)
    candidates = bundle["candidate_sets"]
    portfolio = bundle["portfolio"]
    master = bundle["master_scores"]
    context = bundle["context"]

    gap_table = current_gap_table(train)
    gap_map = dict(
        zip(
            gap_table["number"].astype(int),
            gap_table["current_gap"].astype(int),
        )
    )
    actual_skip_values = [gap_map[number] for number in actual_numbers]
    actual_skip_sum = int(sum(actual_skip_values))
    actual_gap_sum = actual_skip_sum + len(actual_numbers)
    actual_sum = int(sum(actual_numbers))

    master_rank = dict(
        zip(
            master["number"].astype(int),
            master["rank"].astype(int),
        )
    )
    actual_master_ranks = [master_rank[number] for number in actual_numbers]
    master_top20_hits = sum(rank <= 20 for rank in actual_master_ranks)

    sum_fc = context["sum_forecast"]
    gap_fc = context["gap_sum_forecast"]
    skip_fc = context["skip_pattern_forecast"]

    record = {
        "round": target_round,
        "date": str(pd.to_datetime(actual_row["date"]).date()),
        "train_start_round": int(train.iloc[0]["round"]),
        "train_end_round": int(train.iloc[-1]["round"]),
        "train_window": int(train_window),
        "actual_numbers": " ".join(map(str, actual_numbers)),
        "actual_bonus": int(actual_row["bonus"]),
        "actual_sum": actual_sum,
        "actual_gap_sum": actual_gap_sum,
        "actual_skip_sum": actual_skip_sum,
        "actual_group_pattern": _actual_group_pattern(actual_numbers),
        "actual_gap_pattern": _actual_gap_pattern(actual_numbers, gap_map),
        "actual_skip_pattern": _actual_skip_pattern(actual_numbers, gap_map),
        "candidate_11_hits": len(
            actual & set(candidates[11])
        ),
        "candidate_13_hits": len(
            actual & set(candidates[13])
        ),
        "candidate_15_hits": len(
            actual & set(candidates[15])
        ),
        "best5_max_hit": _combo_max_hit(portfolio, actual, 5),
        "best10_max_hit": _combo_max_hit(portfolio, actual, 10),
        "best15_max_hit": _combo_max_hit(portfolio, actual, 15),
        "best20_max_hit": _combo_max_hit(portfolio, actual, 20),
        "predicted_sum_center": float(sum_fc["target_center"]),
        "predicted_sum_low": float(sum_fc["target_low"]),
        "predicted_sum_high": float(sum_fc["target_high"]),
        "predicted_sum_wide_low": float(sum_fc["wide_low"]),
        "predicted_sum_wide_high": float(sum_fc["wide_high"]),
        "sum_abs_error": float(
            abs(actual_sum - float(sum_fc["target_center"]))
        ),
        "sum_core_hit": int(
            float(sum_fc["target_low"])
            <= actual_sum
            <= float(sum_fc["target_high"])
        ),
        "sum_wide_hit": int(
            float(sum_fc["wide_low"])
            <= actual_sum
            <= float(sum_fc["wide_high"])
        ),
        "predicted_gap_sum_center": float(gap_fc["target_center"]),
        "predicted_gap_sum_low": float(gap_fc["target_low"]),
        "predicted_gap_sum_high": float(gap_fc["target_high"]),
        "predicted_gap_sum_wide_low": float(gap_fc["wide_low"]),
        "predicted_gap_sum_wide_high": float(gap_fc["wide_high"]),
        "gap_sum_abs_error": float(
            abs(actual_gap_sum - float(gap_fc["target_center"]))
        ),
        "gap_sum_core_hit": int(
            float(gap_fc["target_low"])
            <= actual_gap_sum
            <= float(gap_fc["target_high"])
        ),
        "gap_sum_wide_hit": int(
            float(gap_fc["wide_low"])
            <= actual_gap_sum
            <= float(gap_fc["wide_high"])
        ),
        "predicted_skip_sum_center": float(skip_fc["target_center"]),
        "predicted_skip_sum_low": float(skip_fc["target_low"]),
        "predicted_skip_sum_high": float(skip_fc["target_high"]),
        "predicted_skip_sum_wide_low": float(skip_fc["wide_low"]),
        "predicted_skip_sum_wide_high": float(skip_fc["wide_high"]),
        "skip_sum_abs_error": float(
            abs(actual_skip_sum - float(skip_fc["target_center"]))
        ),
        "skip_sum_core_hit": int(
            float(skip_fc["target_low"])
            <= actual_skip_sum
            <= float(skip_fc["target_high"])
        ),
        "skip_sum_wide_hit": int(
            float(skip_fc["wide_low"])
            <= actual_skip_sum
            <= float(skip_fc["wide_high"])
        ),
        "skip_transition_match_mode": str(skip_fc["match_mode"]),
        "skip_transition_matches": int(skip_fc["matched_transitions"]),
        "master_top20_hits": int(master_top20_hits),
        "master_actual_mean_rank": float(np.mean(actual_master_ranks)),
        "master_actual_best_rank": int(min(actual_master_ranks)),
        "master_actual_worst_rank": int(max(actual_master_ranks)),
        # Final Pattern / Top of the Best는 결정론적이므로 Seed 직접 영향 없음.
        "seed_sensitive": 0,
        "seed_note": "N/A - Final Pattern deterministic",
    }

    primary, reasons = classify_failure(record)
    record["primary_failure"] = primary
    record["failure_reasons"] = reasons
    return record


def build_historical_ledger(
    df: pd.DataFrame,
    validation_rounds: int = 100,
    train_window: int = 200,
    progress_callback: Optional[
        Callable[[float, str], None]
    ] = None,
) -> pd.DataFrame:
    if validation_rounds < 1:
        raise ValueError("검증 회차는 1 이상이어야 합니다.")
    if train_window < 60:
        raise ValueError("학습 창은 최소 60회 이상이어야 합니다.")

    ordered = df.sort_values("round").reset_index(drop=True)
    _assert_consecutive(ordered)

    available = max_strict_validation_rounds(ordered, train_window)
    if available < validation_rounds:
        latest = int(ordered["round"].max())
        required = required_history_start(
            latest,
            validation_rounds,
            train_window,
        )
        raise ValueError(
            f"최근 {validation_rounds}회 엄격 검증에는 "
            f"{train_window + validation_rounds}회 데이터가 필요합니다. "
            f"현재 가능 {available}회 / 필요한 시작 회차 {required}회."
        )

    start_index = len(ordered) - validation_rounds
    rows = []
    for position, idx in enumerate(
        range(start_index, len(ordered)),
        start=1,
    ):
        row = evaluate_target_round(
            ordered,
            target_index=idx,
            train_window=train_window,
        )
        rows.append(row)

        if progress_callback:
            progress_callback(
                position / validation_rounds,
                f"{row['round']}회 검증 ({position}/{validation_rounds})",
            )

    return pd.DataFrame(rows)


def summarize_ledger(ledger: pd.DataFrame) -> dict:
    if ledger is None or ledger.empty:
        return {}

    summary = {
        "rounds": int(len(ledger)),
        "start_round": int(ledger["round"].min()),
        "end_round": int(ledger["round"].max()),
        "candidate_11_mean": float(ledger["candidate_11_hits"].mean()),
        "candidate_13_mean": float(ledger["candidate_13_hits"].mean()),
        "candidate_15_mean": float(ledger["candidate_15_hits"].mean()),
        "best5_mean": float(ledger["best5_max_hit"].mean()),
        "best10_mean": float(ledger["best10_max_hit"].mean()),
        "best15_mean": float(ledger["best15_max_hit"].mean()),
        "best20_mean": float(ledger["best20_max_hit"].mean()),
        "best20_3plus_rate": float(
            (ledger["best20_max_hit"] >= 3).mean()
        ),
        "best20_4plus_rate": float(
            (ledger["best20_max_hit"] >= 4).mean()
        ),
        "sum_core_rate": float(ledger["sum_core_hit"].mean()),
        "sum_wide_rate": float(ledger["sum_wide_hit"].mean()),
        "sum_mae": float(ledger["sum_abs_error"].mean()),
        "gap_sum_core_rate": float(ledger["gap_sum_core_hit"].mean()),
        "gap_sum_wide_rate": float(ledger["gap_sum_wide_hit"].mean()),
        "gap_sum_mae": float(ledger["gap_sum_abs_error"].mean()),
        "master_top20_mean_hits": float(
            ledger["master_top20_hits"].mean()
        ),
    }
    if "skip_sum_core_hit" in ledger.columns:
        summary.update({
            "skip_sum_core_rate": float(ledger["skip_sum_core_hit"].mean()),
            "skip_sum_wide_rate": float(ledger["skip_sum_wide_hit"].mean()),
            "skip_sum_mae": float(ledger["skip_sum_abs_error"].mean()),
        })
    return summary


def compare_previous_recent(
    ledger: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:
    """
    100개 이상일 때 이전 50 vs 최근 50.
    데이터가 적으면 가능한 동일 크기 절반을 비교합니다.
    """
    if ledger is None or ledger.empty or len(ledger) < 2:
        return pd.DataFrame()

    size = min(
        int(window),
        len(ledger) // 2,
    )
    if size < 1:
        return pd.DataFrame()

    previous = ledger.iloc[-2 * size : -size]
    recent = ledger.iloc[-size:]

    metrics = [
        ("후보11 평균 적중", "candidate_11_hits", "mean"),
        ("후보13 평균 적중", "candidate_13_hits", "mean"),
        ("후보15 평균 적중", "candidate_15_hits", "mean"),
        ("BEST5 평균 최고 적중", "best5_max_hit", "mean"),
        ("BEST10 평균 최고 적중", "best10_max_hit", "mean"),
        ("BEST15 평균 최고 적중", "best15_max_hit", "mean"),
        ("BEST20 평균 최고 적중", "best20_max_hit", "mean"),
        ("BEST20 3+ 비율", "best20_max_hit", "3plus"),
        ("BEST20 4+ 비율", "best20_max_hit", "4plus"),
        ("번호합 핵심구간 적중률", "sum_core_hit", "mean"),
        ("번호합 MAE", "sum_abs_error", "mean"),
        ("GAP합 핵심구간 적중률", "gap_sum_core_hit", "mean"),
        ("GAP합 MAE", "gap_sum_abs_error", "mean"),
        ("Pattern Master TOP20 평균 포착", "master_top20_hits", "mean"),
    ]
    if "skip_sum_core_hit" in ledger.columns:
        metrics.extend([
            ("건너띔 합계 핵심구간 적중률", "skip_sum_core_hit", "mean"),
            ("건너띔 합계 MAE", "skip_sum_abs_error", "mean"),
        ])

    def calculate(frame: pd.DataFrame, column: str, mode: str) -> float:
        if mode == "3plus":
            return float((frame[column] >= 3).mean())
        if mode == "4plus":
            return float((frame[column] >= 4).mean())
        return float(frame[column].mean())

    rows = []
    for label, column, mode in metrics:
        prev = calculate(previous, column, mode)
        curr = calculate(recent, column, mode)
        rows.append({
            "metric": label,
            f"previous_{size}": prev,
            f"recent_{size}": curr,
            "change": curr - prev,
        })

    return pd.DataFrame(rows)


def failure_counts(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return pd.DataFrame(
            columns=["failure", "count", "rate"]
        )

    counts = (
        ledger["primary_failure"]
        .value_counts()
        .rename_axis("failure")
        .reset_index(name="count")
    )
    counts["rate"] = counts["count"] / len(ledger)
    return counts


def cumulative_metrics(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return pd.DataFrame()

    values = {
        "round": ledger["round"].astype(int),
        "candidate_15_mean": ledger["candidate_15_hits"].expanding().mean(),
        "best20_mean": ledger["best20_max_hit"].expanding().mean(),
        "sum_core_rate": ledger["sum_core_hit"].expanding().mean(),
        "gap_sum_core_rate": ledger["gap_sum_core_hit"].expanding().mean(),
    }
    if "skip_sum_core_hit" in ledger.columns:
        values["skip_sum_core_rate"] = (
            ledger["skip_sum_core_hit"].expanding().mean()
        )
    return pd.DataFrame(values)


def save_ledger(
    ledger: pd.DataFrame,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
) -> tuple[Path, Path]:
    ledger_path = Path(ledger_path)
    summary_path = Path(summary_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ledger.to_csv(
        ledger_path,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "summary": summarize_ledger(ledger),
        "failure_counts": failure_counts(ledger).to_dict(
            orient="records"
        ),
    }
    summary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ledger_path, summary_path


def load_saved_ledger(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
) -> pd.DataFrame:
    path = Path(ledger_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")
