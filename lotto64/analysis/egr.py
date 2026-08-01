from __future__ import annotations

import pandas as pd

from lotto64.analysis.gap import build_gap_tables, row_numbers
from lotto64.config import NUMBERS

def egr_backtest(df: pd.DataFrame, threshold: int = 17, horizon: int = 4) -> pd.DataFrame:
    matrix, _ = build_gap_tables(df)
    ordered_rounds = matrix["round"].astype(int).tolist()
    numbers_by_round = {int(r["round"]): set(row_numbers(r)) for _, r in df.iterrows()}
    active = {n: False for n in NUMBERS}
    events = []

    for idx, row in matrix.iterrows():
        current_round = int(row["round"])
        for number in NUMBERS:
            gap = row[f"gap_{number}"]
            if pd.isna(gap):
                continue

            if gap >= threshold and not active[number]:
                recovery_after = None
                for future_round in ordered_rounds[idx: idx + horizon + 1]:
                    if number in numbers_by_round.get(int(future_round), set()):
                        recovery_after = int(future_round) - current_round
                        break
                events.append({
                    "number": number,
                    "entry_round": current_round,
                    "entry_gap": int(gap),
                    "recovered_within_horizon": recovery_after is not None,
                    "recovery_after_rounds": recovery_after,
                })
                active[number] = True

            if number in numbers_by_round.get(current_round, set()):
                active[number] = False

    return pd.DataFrame(events)
