from __future__ import annotations

import numpy as np
import pandas as pd

from lotto64.analysis.dna import classify_states

def cec_drc_backtest(dna: pd.DataFrame) -> pd.DataFrame:
    states = classify_states(dna)
    rows = []

    for i in range(len(states) - 1):
        current, nxt = states.iloc[i], states.iloc[i + 1]
        cec_event = current["gap_state"] == "COMPRESSION"
        drc_event = current["gap_state"] == "EXTREME_EXPANSION"

        rows.append({
            "round": int(current["round"]),
            "next_round": int(nxt["round"]),
            "state": current["gap_state"],
            "cec_event": cec_event,
            "cec_success": (
                bool(nxt["gap_sum"] > current["gap_sum"] or nxt["gap_max"] > 10)
                if cec_event else np.nan
            ),
            "drc_event": drc_event,
            "drc_success": (
                bool(nxt["gap_sum"] < current["gap_sum"] or nxt["gap_le_10"] > current["gap_le_10"])
                if drc_event else np.nan
            ),
        })

    return pd.DataFrame(rows)
