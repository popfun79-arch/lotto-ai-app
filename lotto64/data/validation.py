from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from lotto64.config import MAIN_COLUMNS, REQUIRED_COLUMNS

COLUMN_ALIASES = {
    "회차": "round", "날짜": "date", "추첨일": "date",
    "번호1": "n1", "번호2": "n2", "번호3": "n3",
    "번호4": "n4", "번호5": "n5", "번호6": "n6",
    "보너스": "bonus", "보너스번호": "bonus",
    "draw_num": "round", "num1": "n1", "num2": "n2", "num3": "n3",
    "num4": "n4", "num5": "n5", "num6": "n6",
    "drwNo": "round", "drwNoDate": "date",
    "drwtNo1": "n1", "drwtNo2": "n2", "drwtNo3": "n3",
    "drwtNo4": "n4", "drwtNo5": "n5", "drwtNo6": "n6",
    "bnusNo": "bonus",
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns=COLUMN_ALIASES)

    if "bonus" not in out.columns:
        out["bonus"] = 0

    if "date" not in out.columns and "round" in out.columns:
        rounds = pd.to_numeric(out["round"], errors="coerce")
        out["date"] = pd.Timestamp("2002-12-07") + pd.to_timedelta((rounds - 1) * 7, unit="D")

    return out

def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty:
        raise ValueError("데이터가 비어 있습니다.")

    out = normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError("필수 컬럼 누락: " + ", ".join(missing))

    out = out[REQUIRED_COLUMNS].copy()
    notes: List[str] = []
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in ["round", *MAIN_COLUMNS, "bonus"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["round", *MAIN_COLUMNS])
    if len(out) < before:
        notes.append(f"필수 숫자 누락 행 {before-len(out)}개 제거")

    out["bonus"] = out["bonus"].fillna(0)
    out[["round", *MAIN_COLUMNS, "bonus"]] = out[["round", *MAIN_COLUMNS, "bonus"]].astype(int)

    def sort_main_numbers(row: pd.Series) -> pd.Series:
        numbers = sorted(int(row[c]) for c in MAIN_COLUMNS)
        for idx, col in enumerate(MAIN_COLUMNS):
            row[col] = numbers[idx]
        return row

    out = out.apply(sort_main_numbers, axis=1)

    mask = np.ones(len(out), dtype=bool)
    for col in MAIN_COLUMNS:
        mask &= out[col].between(1, 45).to_numpy()
    mask &= out["bonus"].between(0, 45).to_numpy()

    invalid_count = int((~mask).sum())
    if invalid_count:
        notes.append(f"번호 범위 오류 행 {invalid_count}개 제거")
        out = out.loc[mask].copy()

    if out.empty:
        raise ValueError("유효 데이터가 없습니다. 번호 범위와 컬럼 구성을 확인해 주세요.")

    unique_mask = out.apply(
        lambda row: len({int(row[c]) for c in MAIN_COLUMNS}) == 6,
        axis=1,
    ).astype(bool)

    duplicate_number_count = int((~unique_mask).sum())
    if duplicate_number_count:
        notes.append(f"본번호 중복 행 {duplicate_number_count}개 제거")
        out = out.loc[unique_mask].copy()

    before = len(out)
    out = out.drop_duplicates("round", keep="last").sort_values("round").reset_index(drop=True)
    if len(out) < before:
        notes.append(f"중복 회차 {before-len(out)}개 정리")

    if len(out) >= 2:
        expected = set(range(int(out["round"].min()), int(out["round"].max()) + 1))
        missing_rounds = sorted(expected - set(out["round"].astype(int)))
        if missing_rounds:
            notes.append(f"누락 회차 {len(missing_rounds)}개 존재")

    if int((out["bonus"] == 0).sum()) > 0:
        notes.append(f"보너스 미입력(0) 회차 {int((out['bonus'] == 0).sum())}개")

    if not notes:
        notes.append("기본 데이터 검증 통과")

    return out, notes
