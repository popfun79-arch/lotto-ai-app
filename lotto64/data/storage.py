from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from lotto64.config import CSV_PATH, DB_PATH
from lotto64.data.validation import validate_and_clean

def read_csv(path: Path = CSV_PATH) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size < 10:
        return None
    return pd.read_csv(path, encoding="utf-8-sig")

def write_csv(df: pd.DataFrame, path: Path = CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def sync_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("draws", conn, if_exists="replace", index=False)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_draws_round ON draws(round)")
        conn.commit()
    return db_path

def load_sqlite(db_path: Path = DB_PATH) -> Optional[pd.DataFrame]:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM draws ORDER BY round", conn)

def load_best_available(uploaded_file=None) -> Tuple[pd.DataFrame, str, list[str]]:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".csv":
            raw = pd.read_csv(uploaded_file)
        elif suffix == ".json":
            raw = pd.read_json(uploaded_file)
        else:
            raise ValueError("CSV 또는 JSON만 지원합니다.")
        source = f"업로드: {uploaded_file.name}"
    else:
        raw = read_csv()
        source = str(CSV_PATH)
        if raw is None:
            raw = load_sqlite()
            source = str(DB_PATH)

    if raw is None:
        raise FileNotFoundError("data/lotto_all.csv 또는 SQLite 데이터가 없습니다.")

    cleaned, notes = validate_and_clean(raw)
    return cleaned, source, notes
