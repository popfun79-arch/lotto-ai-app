from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "lotto64.db"
CSV_PATH = DATA_DIR / "lotto_all.csv"

MAIN_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
REQUIRED_COLUMNS = ["round", "date", *MAIN_COLUMNS, "bonus"]
NUMBERS = list(range(1, 46))
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# 실전 추천/기본 백테스트 재현성을 위한 고정 기준 Seed
FIXED_SEED = 20260720
DEFAULT_MULTI_SEED_COUNT = 5

DEFAULT_WEIGHTS = {
    "long_frequency": 0.08,
    "short_frequency": 0.08,
    "gap": 0.12,
    "egr": 0.10,
    "carry": 0.07,
    "neighbor": 0.08,
    "bonus_window": 0.05,
    "pair": 0.05,
    "zone_recovery": 0.07,
    "dna_similarity": 0.15,
    "state": 0.10,
    "hot_cold": 0.05,
}

@dataclass(frozen=True)
class RunConfig:
    recent_window: int = 200
    backtest_rounds: int = 50
    candidate_count: int = 18
    similarity_k: int = 15
    egr_threshold: int = 17
    egr_horizon: int = 4
    seed: int = FIXED_SEED
    top_n: int = 20

    def to_dict(self) -> dict:
        return asdict(self)
