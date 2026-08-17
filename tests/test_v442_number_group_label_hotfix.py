from __future__ import annotations

from pathlib import Path

from lotto64.utils.lotto_math import NUMBER_GROUP_LABELS


ROOT = Path(__file__).resolve().parents[1]


def test_number_group_labels_are_available():
    assert NUMBER_GROUP_LABELS == (
        "1~9",
        "10~19",
        "20~29",
        "30~39",
        "40~45",
    )


def test_app_imports_and_uses_number_group_labels():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert (
        "from lotto64.utils.lotto_math import NUMBER_GROUP_LABELS"
        in text
    )
    assert "Number Groups 적용 구간:" in text
    assert "join(NUMBER_GROUP_LABELS)" in text
