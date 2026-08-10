from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from lotto64.data.storage import read_csv, sync_sqlite, write_csv
from lotto64.data.validation import validate_and_clean

def fetch_round(draw_no: int, timeout: int = 10) -> Optional[dict]:
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 Lotto64-Ultimate/3.0"},
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

def update_latest(max_checks: int = 20, sleep_seconds: float = 0.15) -> tuple[pd.DataFrame, int]:
    existing = read_csv()
    if existing is None:
        existing = pd.DataFrame(columns=["round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"])
    elif len(existing):
        existing, _ = validate_and_clean(existing)

    latest = int(existing["round"].max()) if len(existing) else 0
    new_rows = []
    failures = 0

    for draw_no in range(latest + 1, latest + max_checks + 1):
        item = fetch_round(draw_no)
        if item is None:
            failures += 1
            if failures >= 3:
                break
        else:
            failures = 0
            new_rows.append(item)
        time.sleep(sleep_seconds)

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    if len(combined):
        combined, _ = validate_and_clean(combined)
        write_csv(combined)
        sync_sqlite(combined)
    return combined, len(new_rows)
