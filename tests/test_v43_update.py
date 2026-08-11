from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import lotto64.data.updater as updater
from update_lotto import expected_latest_round_kst

KST = ZoneInfo("Asia/Seoul")


def test_expected_round_after_1236_draw():
    now = datetime(2026, 8, 9, 4, 10, tzinfo=KST)
    assert expected_latest_round_kst(now) == 1236


def test_expected_round_next_week():
    now = datetime(2026, 8, 16, 4, 10, tzinfo=KST)
    assert expected_latest_round_kst(now) == 1237


def test_smok95_item_normalization():
    item = {
        "draw_no": 1236,
        "numbers": [12, 18, 21, 29, 34, 38],
        "bonus_no": 10,
        "date": "2026-08-08T00:00:00Z",
    }
    row = updater._normalize_smok95_item(item)

    assert row is not None
    assert row["round"] == 1236
    assert row["bonus"] == 10
    assert [row[f"n{i}"] for i in range(1, 7)] == [
        12, 18, 21, 29, 34, 38
    ]


def test_remote_range_parsing(monkeypatch):
    payload = [
        {
            "draw_no": 1235,
            "numbers": [6, 7, 11, 15, 39, 43],
            "bonus_no": 20,
            "date": "2026-08-01T00:00:00Z",
        },
        {
            "draw_no": 1236,
            "numbers": [12, 18, 21, 29, 34, 38],
            "bonus_no": 10,
            "date": "2026-08-08T00:00:00Z",
        },
    ]

    monkeypatch.setattr(
        updater,
        "_request_json",
        lambda *args, **kwargs: payload,
    )

    rows = updater.fetch_remote_range(1236, 1240)
    assert len(rows) == 1
    assert rows[0]["round"] == 1236
