from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from lotto64.data.storage import read_csv, sync_sqlite, write_csv
from lotto64.data.validation import validate_and_clean

SMOK95_ALL_URL = "https://smok95.github.io/lotto/results/all.json"
LEGACY_OFFICIAL_URL = (
    "https://www.dhlottery.co.kr/common.do"
    "?method=getLottoNumber&drwNo={draw_no}"
)


def _request_json(url: str, timeout: int = 20):
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 Lotto64-Ultimate/4.5"},
    )
    response.raise_for_status()
    return response.json()


def _normalize_smok95_item(item: dict) -> Optional[dict]:
    try:
        draw_no = int(item["draw_no"])
        numbers = [int(v) for v in item["numbers"]]
        bonus = int(item["bonus_no"])
        date = str(item["date"])[:10]
    except (KeyError, TypeError, ValueError):
        return None

    if len(numbers) != 6:
        return None
    if len(set(numbers)) != 6:
        return None
    if not all(1 <= n <= 45 for n in numbers):
        return None
    if not 1 <= bonus <= 45:
        return None
    if bonus in numbers:
        return None

    numbers = sorted(numbers)
    return {
        "round": draw_no,
        "date": date,
        **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
        "bonus": bonus,
    }


def fetch_remote_range(
    start_round: int,
    end_round: int,
    timeout: int = 20,
) -> list[dict]:
    """
    최신 공개 전체 데이터셋을 한 번만 내려받아 필요한 회차만 추출합니다.
    """
    payload = _request_json(SMOK95_ALL_URL, timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError("원격 로또 데이터 형식이 리스트가 아닙니다.")

    rows: list[dict] = []
    for item in payload:
        normalized = _normalize_smok95_item(item)
        if normalized is None:
            continue
        if start_round <= normalized["round"] <= end_round:
            rows.append(normalized)

    rows.sort(key=lambda row: row["round"])
    return rows


def fetch_round_legacy(
    draw_no: int,
    timeout: int = 10,
) -> Optional[dict]:
    """
    기존 동행복권 JSON endpoint fallback.
    현재 주 데이터 소스가 실패했을 때만 시도합니다.
    """
    try:
        data = _request_json(
            LEGACY_OFFICIAL_URL.format(draw_no=draw_no),
            timeout=timeout,
        )
        if data.get("returnValue") != "success":
            return None

        numbers = [int(data[f"drwtNo{i}"]) for i in range(1, 7)]
        bonus = int(data["bnusNo"])

        if len(set(numbers)) != 6:
            return None
        if bonus in numbers:
            return None

        numbers = sorted(numbers)
        return {
            "round": int(data["drwNo"]),
            "date": data["drwNoDate"],
            **{f"n{i}": numbers[i - 1] for i in range(1, 7)},
            "bonus": bonus,
        }
    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ):
        return None


def _validate_new_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    cleaned, _ = validate_and_clean(frame)

    valid_rows: list[dict] = []
    for _, row in cleaned.iterrows():
        main = [int(row[f"n{i}"]) for i in range(1, 7)]
        bonus = int(row["bonus"])

        if bonus == 0:
            continue
        if bonus in set(main):
            continue

        valid_rows.append({
            "round": int(row["round"]),
            "date": row["date"].strftime("%Y-%m-%d")
            if pd.notna(row["date"])
            else "",
            **{f"n{i}": int(row[f"n{i}"]) for i in range(1, 7)},
            "bonus": bonus,
        })

    return valid_rows


def update_latest(
    max_checks: int = 20,
    sleep_seconds: float = 0.15,
) -> tuple[pd.DataFrame, int]:
    existing = read_csv()

    if existing is None:
        existing = pd.DataFrame(
            columns=[
                "round", "date",
                "n1", "n2", "n3", "n4", "n5", "n6",
                "bonus",
            ]
        )
    elif len(existing):
        existing, _ = validate_and_clean(existing)

    latest = int(existing["round"].max()) if len(existing) else 0
    start_round = latest + 1
    end_round = latest + max_checks

    new_rows: list[dict] = []

    # 1순위: 최신 공개 전체 데이터셋
    try:
        new_rows = fetch_remote_range(
            start_round,
            end_round,
            timeout=20,
        )
    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        new_rows = []

    # 2순위: 기존 개별 회차 endpoint fallback
    if not new_rows:
        failures = 0
        for draw_no in range(start_round, end_round + 1):
            item = fetch_round_legacy(draw_no)
            if item is None:
                failures += 1
                if failures >= 3:
                    break
            else:
                failures = 0
                new_rows.append(item)

            time.sleep(sleep_seconds)

    new_rows = _validate_new_rows(new_rows)

    # 기존 최신 회차보다 큰 행만 허용
    new_rows = [
        row
        for row in new_rows
        if int(row["round"]) > latest
    ]

    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)],
        ignore_index=True,
    )

    if len(combined):
        combined, _ = validate_and_clean(combined)

        # 저장 직전 추가 안전검사
        if combined["round"].duplicated().any():
            raise ValueError("중복 회차가 있습니다.")

        for _, row in combined.iterrows():
            main = [int(row[f"n{i}"]) for i in range(1, 7)]
            bonus = int(row["bonus"])
            if bonus == 0:
                raise ValueError(
                    f"{int(row['round'])}회 보너스가 0입니다."
                )
            if bonus in set(main):
                raise ValueError(
                    f"{int(row['round'])}회 보너스/본번호 중복입니다."
                )

        write_csv(combined)
        sync_sqlite(combined)

    return combined, len(new_rows)


def ensure_history_start(
    start_round: int = 937,
) -> tuple[pd.DataFrame, int]:
    """
    기존 CSV의 최초 회차보다 앞선 과거 데이터를 공개 전체 데이터셋에서
    보충합니다. Historical Validation Ledger 100회 / 200회 학습창을
    위해 기본 시작점은 937회입니다.
    """
    existing = read_csv()

    if existing is None or existing.empty:
        raise ValueError(
            "기존 데이터가 없습니다. 먼저 기본 데이터를 준비해 주세요."
        )

    existing, _ = validate_and_clean(existing)
    earliest = int(existing["round"].min())

    if earliest <= int(start_round):
        return existing, 0

    historical_rows: list[dict] = []
    try:
        historical_rows = fetch_remote_range(
            int(start_round),
            earliest - 1,
            timeout=30,
        )
    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            f"과거 데이터 백필 실패: {exc}"
        ) from exc

    historical_rows = _validate_new_rows(historical_rows)

    expected_count = earliest - int(start_round)
    if len(historical_rows) != expected_count:
        got_rounds = {int(row["round"]) for row in historical_rows}
        expected_rounds = set(range(int(start_round), earliest))
        missing = sorted(expected_rounds - got_rounds)
        sample = ", ".join(map(str, missing[:10]))
        raise RuntimeError(
            f"과거 데이터 누락: 기대 {expected_count}회 / "
            f"수신 {len(historical_rows)}회 / 누락 {sample}"
        )

    combined = pd.concat(
        [pd.DataFrame(historical_rows), existing],
        ignore_index=True,
    )
    combined, _ = validate_and_clean(combined)

    if combined["round"].duplicated().any():
        raise ValueError("과거 데이터 백필 후 중복 회차가 있습니다.")

    write_csv(combined)
    sync_sqlite(combined)
    return combined, len(historical_rows)
