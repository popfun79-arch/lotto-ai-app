from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_defaults_to_latest_200():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"분석 회차 (최신)"' in text
    assert "value=200" in text
    assert "max_value=300" in text


def test_weekly_update_is_sunday_0400_kst():
    text = (
        ROOT / ".github" / "workflows" / "update_data.yml"
    ).read_text(encoding="utf-8")
    assert 'cron: "0 19 * * 6"' in text
    assert "일요일 04:00 KST" in text
