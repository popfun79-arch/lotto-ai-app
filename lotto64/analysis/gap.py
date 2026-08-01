from __future__ import annotations

import numpy as np
import pandas as pd

from lotto64.config import MAIN_COLUMNS, NUMBERS

def row_numbers(row: pd.Series) -> list[int]:
    return [int(row[c]) for c in MAIN_COLUMNS]

def build_gap_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_seen = {n: None for n in NUMBERS}
    matrix_rows, round_rows = [], []

    for _, row in df.iterrows():
        round_no = int(row["round"])
        gaps = {
            n: round_no - last_seen[n] - 1 if last_seen[n] is not None else np.nan
            for n in NUMBERS
        }
        matrix_rows.append({"round": round_no, **{f"gap_{n}": gaps[n] for n in NUMBERS}})

        selected = [gaps[n] for n in row_numbers(row)]
        valid = [g for g in selected if pd.notna(g)]
        round_rows.append({
            "round": round_no,
            "gap_values": selected,
            "gap_sum": float(np.sum(valid)) if valid else np.nan,
            "gap_mean": float(np.mean(valid)) if valid else np.nan,
            "gap_max": float(np.max(valid)) if valid else np.nan,
            "gap_std": float(np.std(valid)) if valid else np.nan,
            "gap_le_5": int(sum(g <= 5 for g in valid)),
            "gap_le_10": int(sum(g <= 10 for g in valid)),
            "gap_gt_10": int(sum(g > 10 for g in valid)),
            "gap_ge_17": int(sum(g >= 17 for g in valid)),
        })

        for n in row_numbers(row):
            last_seen[n] = round_no

    return pd.DataFrame(matrix_rows), pd.DataFrame(round_rows)

def current_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    latest = int(df["round"].max())
    last_seen = {n: None for n in NUMBERS}
    histories = {n: [] for n in NUMBERS}

    for _, row in df.iterrows():
        round_no = int(row["round"])
        for n in row_numbers(row):
            if last_seen[n] is not None:
                histories[n].append(round_no - int(last_seen[n]) - 1)
            last_seen[n] = round_no

    rows = []
    for n in NUMBERS:
        current_gap = latest - int(last_seen[n]) if last_seen[n] is not None else len(df)
        hist = histories[n]
        rows.append({
            "number": n,
            "current_gap": current_gap,
            "average_gap": float(np.mean(hist)) if hist else np.nan,
            "median_gap": float(np.median(hist)) if hist else np.nan,
            "gap_observations": len(hist),
        })

    out = pd.DataFrame(rows)
    out["gap_percentile"] = out["current_gap"].rank(pct=True)
    return out

def gap_distribution(df: pd.DataFrame) -> pd.DataFrame:
    _, rounds = build_gap_tables(df)
    values = [int(g) for arr in rounds["gap_values"] for g in arr if pd.notna(g)]
    counts = pd.Series(values).value_counts().sort_index()
    out = counts.rename_axis("gap").reset_index(name="count")
    out["ratio"] = out["count"] / out["count"].sum()
    return out
