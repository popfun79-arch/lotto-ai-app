from __future__ import annotations

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from lotto64.analysis.dna import DNA_FEATURES

def similar_rounds(dna: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    clean = dna.dropna(subset=DNA_FEATURES).reset_index(drop=True)
    if len(clean) < 8:
        return pd.DataFrame()

    history, target = clean.iloc[:-1], clean.iloc[[-1]]
    scaler = StandardScaler()
    x = scaler.fit_transform(history[DNA_FEATURES])
    y = scaler.transform(target[DNA_FEATURES])

    model = NearestNeighbors(
        n_neighbors=min(k, len(history)),
        metric="euclidean",
    ).fit(x)

    distances, indices = model.kneighbors(y)
    rows = []

    for distance, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx + 1 >= len(clean):
            continue
        source, nxt = history.iloc[idx], clean.iloc[idx + 1]
        rows.append({
            "similar_round": int(source["round"]),
            "distance": float(distance),
            "next_round": int(nxt["round"]),
            "next_sum": float(nxt["sum"]),
            "next_gap_sum": float(nxt["gap_sum"]),
            "next_carryover": int(nxt["carryover_count"]),
            "next_neighbor": int(nxt["neighbor_count"]),
            "next_missing_zones": int(nxt["missing_zones"]),
            "next_ac": int(nxt["ac"]),
        })

    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
