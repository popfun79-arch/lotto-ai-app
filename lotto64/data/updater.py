from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from lotto64.data.storage import read_csv, sync_sqlite, write_csv
from lotto64.data.validation import validate_and_clean

OFFICIAL_LEGACY_URL = (
    "https://www.dhlottery.co.kr/common.do"
    "?method=getLottoNumber&drwNo={draw_no}"
)
FALLBACK_ALL_URL = "https://smok95.github.io/lotto/results/all.json"


def fetch_round_official(draw_no: int, timeout: int = 10) -> Optional[dict]:
    """
    동행복권의 기존 JSON 엔드포인트를 우선 사용합니다.
    사이트 개편으로 JSON이 아닌 HTML이 반환되면 None을 반환합니다.
    """
    try:
        response = requests.get(
            OFFICIAL_LEGACY_URL.format(draw_no=draw_no),
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 Lotto64-Ultimate/3.4"},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("returnValue") != "success":
            return None
        return {
            "round": int(data["drwNo"]),
            "date": data["drwNoDate"],
            **{f"n{i}": int(data[f"drwtNo{i}"]) for i in range(1, 7)},
            "bonus": int(data["bnusNo"]),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def fetch_all_fallback(timeout: int = 30) -> pd.DataFrame:
    """
    공식 엔드포인트가 작동하지 않을 때 공개 GitHub 데이터셋을
    보조 수단으로 사용합니다. 반환 데이터는 앱의 검증 모듈을 다시 거칩니다.
    """
    response = requests.get(
        FALLBACK_ALL_URL,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 Lotto64-Ultimate/3.4"},
    )
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError("보조 데이터 형식이 리스트가 아닙니다.")

    rows = []
    for item in payload:
        numbers = item.get("numbers", [])
        if len(numbers) != 6:
            continue
        rows.append({
            "round": int(item["draw_no"]),
            "date": str(item["date"])[:10],
            "n1": int(numbers[0]),
            "n2": int(numbers[1]),
            "n3": int(numbers[2]),
            "n4": int(numbers[3]),
            "n5": int(numbers[4]),
            "n6": int(numbers[5]),
            "bonus": int(item.get("bonus_no", 0)),
        })

    if not rows:
        raise ValueError("보조 데이터에서 유효한 회차를 찾지 못했습니다.")

    cleaned, _ = validate_and_clean(pd.DataFrame(rows))
    return cleaned


def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined, _ = validate_and_clean(combined)
    return combined


def update_latest(
    max_checks: int = 20,
    sleep_seconds: float = 0.15,
) -> tuple[pd.DataFrame, int]:
    existing = read_csv()
    if existing is None:
        existing = pd.DataFrame(
            columns=[
                "round", "date", "n1", "n2", "n3",
                "n4", "n5", "n6", "bonus",
            ]
        )
    elif len(existing):
        existing, _ = validate_and_clean(existing)

    before_round = int(existing["round"].max()) if len(existing) else 0
    rows = []
    failures = 0

    # 1차: 공식 기존 JSON 엔드포인트
    for draw_no in range(before_round + 1, before_round + max_checks + 1):
        item = fetch_round_official(draw_no)
        if item is None:
            failures += 1
            if failures >= 3:
                break
        else:
            failures = 0
            rows.append(item)
        time.sleep(sleep_seconds)

    result = existing
    if rows:
        result = _merge(existing, pd.DataFrame(rows))

    # 2차: 공식 API가 실패했거나 최신 데이터가 없으면 공개 데이터셋으로 보완
    if not rows:
        try:
            fallback = fetch_all_fallback()
            result = _merge(existing, fallback)
        except Exception as exc:
            if len(existing):
                return existing, 0
            raise RuntimeError(
                "공식 데이터와 보조 데이터 모두 불러오지 못했습니다."
            ) from exc

    write_csv(result)
    sync_sqlite(result)

    after_round = int(result["round"].max()) if len(result) else before_round
    new_count = max(0, after_round - before_round)
    return result, new_count
