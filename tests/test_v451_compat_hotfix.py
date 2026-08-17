from pathlib import Path

from lotto64.backtest.historical_ledger import (
    NUMBER_GROUP_LABELS,
    _actual_group_pattern,
)
from lotto64.utils.lotto_math import NUMBER_GROUPS, zone_counts


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_number_groups_are_present():
    assert NUMBER_GROUPS == (
        (1, 9),
        (10, 19),
        (20, 29),
        (30, 39),
        (40, 45),
    )
    assert NUMBER_GROUP_LABELS == (
        "1~9",
        "10~19",
        "20~29",
        "30~39",
        "40~45",
    )


def test_historical_ledger_uses_new_group_boundaries():
    assert zone_counts([9, 10, 19, 20, 29, 40]) == (1, 2, 2, 0, 1)
    assert _actual_group_pattern([9, 10, 19, 20, 29, 40]) == (
        "1~9:1 / 10~19:2 / 20~29:2 / 30~39:0 / 40~45:1"
    )


def test_historical_ledger_has_partial_deployment_fallback():
    text = (
        ROOT / "lotto64" / "backtest" / "historical_ledger.py"
    ).read_text(encoding="utf-8")
    assert "except ImportError:" in text
    assert '"1~9"' in text
    assert '"40~45"' in text
