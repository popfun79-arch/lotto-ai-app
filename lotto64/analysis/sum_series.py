from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lotto64.config import MAIN_COLUMNS


@dataclass(frozen=True)
class SumForecast:
    current_sum: int
    current_state: str
    target_center: float
    target_low: float
    target_high: float
    wide_low: float
    wide_high: float
    expected_delta: float
    matched_transitions: int
    state_window: int
    transition_lookback: int

    def to_dict(self) -> dict:
        return asdict(self)


def _draw_sum(df: pd.DataFrame) -> pd.Series:
    return df[MAIN_COLUMNS].astype(int).sum(axis=1)


def build_sum_series(
    df: pd.DataFrame,
    state_window: int = 50,
) -> pd.DataFrame:
    """
    회차별 당첨번호 6개의 합계를 시계열로 변환합니다.

    상태(LOW/MID/HIGH)는 각 회차 이전 데이터만 사용해 산출하므로
    Walk-forward 백테스트에서 미래 정보 누출을 피할 수 있습니다.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "round": df["round"].astype(int).to_numpy(),
        "sum": _draw_sum(df).to_numpy(),
    })

    out["delta1"] = out["sum"].diff()
    out["ma5"] = out["sum"].rolling(5, min_periods=3).mean()
    out["ma10"] = out["sum"].rolling(10, min_periods=5).mean()
    out["ma20"] = out["sum"].rolling(20, min_periods=10).mean()
    out["std10"] = out["sum"].rolling(10, min_periods=5).std(ddof=0)
    out["std20"] = out["sum"].rolling(20, min_periods=10).std(ddof=0)

    prior = out["sum"].shift(1)
    rolling = prior.rolling(state_window, min_periods=max(12, state_window // 3))
    out["prior_q25"] = rolling.quantile(0.25)
    out["prior_q50"] = rolling.quantile(0.50)
    out["prior_q75"] = rolling.quantile(0.75)

    denom = out["std20"].replace(0, np.nan)
    out["z20"] = (out["sum"] - out["ma20"]) / denom

    def state(row: pd.Series) -> str:
        if pd.isna(row["prior_q25"]) or pd.isna(row["prior_q75"]):
            return "UNKNOWN"
        if row["sum"] <= row["prior_q25"]:
            return "LOW"
        if row["sum"] >= row["prior_q75"]:
            return "HIGH"
        return "MID"

    out["sum_state"] = out.apply(state, axis=1)
    return out


def forecast_next_sum(
    df: pd.DataFrame,
    state_window: int = 50,
    transition_lookback: int = 100,
    min_matches: int = 8,
) -> SumForecast:
    """
    최근 합계 상태와 같은 과거 상태의 '다음 회차 합계' 분포를 이용해
    다음 회차 합계의 중심/구간을 추정합니다.

    예측값은 실제 확률이 아니라 조합 랭킹용 상태 prior입니다.
    """
    series = build_sum_series(df, state_window=state_window)
    if len(series) < 20:
        values = series["sum"].astype(float)
        return SumForecast(
            current_sum=int(values.iloc[-1]),
            current_state="UNKNOWN",
            target_center=float(values.median()),
            target_low=float(values.quantile(0.25)),
            target_high=float(values.quantile(0.75)),
            wide_low=float(values.quantile(0.10)),
            wide_high=float(values.quantile(0.90)),
            expected_delta=0.0,
            matched_transitions=max(0, len(values) - 1),
            state_window=state_window,
            transition_lookback=transition_lookback,
        )

    latest = series.iloc[-1]
    current_state = str(latest["sum_state"])
    start = max(0, len(series) - transition_lookback - 1)

    next_values: list[float] = []
    deltas: list[float] = []

    for i in range(start, len(series) - 1):
        row = series.iloc[i]
        if current_state != "UNKNOWN" and row["sum_state"] != current_state:
            continue
        current = float(row["sum"])
        nxt = float(series.iloc[i + 1]["sum"])
        next_values.append(nxt)
        deltas.append(nxt - current)

    if len(next_values) < min_matches:
        recent = series["sum"].tail(min(transition_lookback, len(series))).astype(float)
        next_values = recent.iloc[1:].tolist()
        deltas = recent.diff().dropna().tolist()

    values = pd.Series(next_values, dtype=float)
    delta_series = pd.Series(deltas, dtype=float)

    return SumForecast(
        current_sum=int(latest["sum"]),
        current_state=current_state,
        target_center=float(values.quantile(0.50)),
        target_low=float(values.quantile(0.25)),
        target_high=float(values.quantile(0.75)),
        wide_low=float(values.quantile(0.10)),
        wide_high=float(values.quantile(0.90)),
        expected_delta=float(delta_series.quantile(0.50)) if len(delta_series) else 0.0,
        matched_transitions=int(len(values)),
        state_window=state_window,
        transition_lookback=transition_lookback,
    )


def sum_pattern_score(total: int, forecast: SumForecast) -> float:
    """
    예측 합계 분포에 가까울수록 0~1 사이의 높은 점수를 반환합니다.
    핵심 IQR 구간은 최고점, 10~90% 구간은 완만한 감점을 적용합니다.
    """
    total = float(total)

    if forecast.target_low <= total <= forecast.target_high:
        center_span = max(8.0, (forecast.target_high - forecast.target_low) / 2)
        distance = abs(total - forecast.target_center)
        return float(max(0.85, 1.0 - 0.15 * distance / center_span))

    if forecast.wide_low <= total <= forecast.wide_high:
        if total < forecast.target_low:
            distance = forecast.target_low - total
        else:
            distance = total - forecast.target_high
        outer_span = max(
            8.0,
            max(
                forecast.target_low - forecast.wide_low,
                forecast.wide_high - forecast.target_high,
            ),
        )
        return float(max(0.45, 0.80 - 0.35 * distance / outer_span))

    if total < forecast.wide_low:
        distance = forecast.wide_low - total
    else:
        distance = total - forecast.wide_high

    return float(max(0.05, 0.40 * np.exp(-distance / 18.0)))


def compare_sum_windows(
    df: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:
    """
    최근 N회와 그 이전 N회의 합계 수준/변동성을 비교합니다.
    """
    sums = _draw_sum(df).astype(float)
    if len(sums) < window * 2:
        return pd.DataFrame()

    previous = sums.iloc[-window * 2:-window]
    recent = sums.iloc[-window:]

    rows = []
    for name, values in [("이전 50회", previous), ("최근 50회", recent)]:
        rows.append({
            "구간": name.replace("50", str(window)),
            "평균": float(values.mean()),
            "중앙값": float(values.median()),
            "표준편차": float(values.std(ddof=0)),
            "Q25": float(values.quantile(0.25)),
            "Q75": float(values.quantile(0.75)),
            "최소": int(values.min()),
            "최대": int(values.max()),
        })

    return pd.DataFrame(rows)
