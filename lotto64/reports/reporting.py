from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from lotto64.config import REPORT_DIR

def json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def build_backtest_report(result: pd.DataFrame, settings: dict) -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "summary": {
            "rounds": int(len(result)),
            "candidate_11_mean_hits": float(result["candidate_11_hits"].mean()),
            "candidate_13_mean_hits": float(result["candidate_13_hits"].mean()),
            "candidate_15_mean_hits": float(result["candidate_15_hits"].mean()),
            "top_combo_mean_max_hit": float(result["top_combo_max_hit"].mean()),
            "top_combo_3plus_rate": float(result["top_combo_3plus"].mean()),
            "top_combo_4plus_rate": float(result["top_combo_4plus"].mean()),
            "top_combo_5plus_rate": float(result["top_combo_5plus"].mean()),
            "top_combo_6_rate": float(result["top_combo_6"].mean()),
        },
    }

def save_report_files(result: pd.DataFrame, report: dict, prefix: str = "walk_forward") -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / f"{prefix}.csv"
    json_path = REPORT_DIR / f"{prefix}.json"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path
