from __future__ import annotations

import pandas as pd

import lotto64.data.updater as updater


def test_history_backfill_function_exists():
    assert callable(updater.ensure_history_start)


def test_backfill_merges_earlier_rounds(monkeypatch):
    existing = pd.DataFrame([
        {
            "round": 1000,
            "date": "2021-01-01",
            "n1": 1, "n2": 2, "n3": 3,
            "n4": 4, "n5": 5, "n6": 6,
            "bonus": 7,
        },
        {
            "round": 1001,
            "date": "2021-01-08",
            "n1": 8, "n2": 9, "n3": 10,
            "n4": 11, "n5": 12, "n6": 13,
            "bonus": 14,
        },
    ])

    backfill = [{
        "round": 999,
        "date": "2020-12-25",
        "n1": 15, "n2": 16, "n3": 17,
        "n4": 18, "n5": 19, "n6": 20,
        "bonus": 21,
    }]

    monkeypatch.setattr(updater, "read_csv", lambda: existing.copy())
    monkeypatch.setattr(
        updater,
        "fetch_remote_range",
        lambda start, end, timeout=30: backfill.copy(),
    )
    monkeypatch.setattr(updater, "write_csv", lambda df: None)
    monkeypatch.setattr(updater, "sync_sqlite", lambda df: None)

    combined, count = updater.ensure_history_start(999)

    assert count == 1
    assert int(combined["round"].min()) == 999
    assert int(combined["round"].max()) == 1001
