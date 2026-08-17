from __future__ import annotations

import pandas as pd

from lotto64.backtest.seed_stability import build_seed_set
from lotto64.config import (
    DEFAULT_MULTI_SEED_COUNT,
    FIXED_SEED,
)


def test_fixed_seed_constant():
    assert FIXED_SEED == 20260720
    assert DEFAULT_MULTI_SEED_COUNT == 5


def test_seed_set_is_reproducible_and_unique():
    first = build_seed_set(FIXED_SEED, 5)
    second = build_seed_set(FIXED_SEED, 5)

    assert first == second
    assert first[0] == FIXED_SEED
    assert len(first) == 5
    assert len(set(first)) == 5


def test_seed_set_supports_3_5_7():
    for count in (3, 5, 7):
        seeds = build_seed_set(FIXED_SEED, count)
        assert len(seeds) == count
        assert seeds[0] == FIXED_SEED


def test_walk_forward_records_seed():
    path = "lotto64/backtest/walk_forward.py"
    text = open(path, encoding="utf-8").read()
    assert '"seed": int(seed)' in text


def test_app_has_fixed_and_multi_seed_controls():
    text = open("app.py", encoding="utf-8").read()

    assert '"고정 기준 Seed"' in text
    assert "disabled=True" in text
    assert '"다중 Seed 검증 수"' in text
    assert '"GA 다중 Seed 안정성 실행"' in text
    assert '"다중 Seed Walk-forward 실행"' in text
