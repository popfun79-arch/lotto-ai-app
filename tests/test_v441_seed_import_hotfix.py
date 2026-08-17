from lotto64.config import DEFAULT_MULTI_SEED_COUNT, FIXED_SEED
from lotto64.backtest.seed_stability import build_seed_set


def test_seed_config_exports_exist():
    assert FIXED_SEED == 20260720
    assert DEFAULT_MULTI_SEED_COUNT == 5


def test_seed_stability_imports_and_builds_default_set():
    seeds = build_seed_set()
    assert seeds[0] == FIXED_SEED
    assert len(seeds) == DEFAULT_MULTI_SEED_COUNT
