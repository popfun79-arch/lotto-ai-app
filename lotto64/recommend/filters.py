from __future__ import annotations

from typing import Sequence

from lotto64.utils.lotto_math import (
    ac_value, end_digit_sum, low_count, max_consecutive_run,
    odd_count, zone_counts,
)

def hard_filter(combo: Sequence[int]) -> bool:
    numbers = sorted(map(int, combo))
    zones = zone_counts(numbers)

    return (
        80 <= sum(numbers) <= 190
        and 10 <= end_digit_sum(numbers) <= 48
        and odd_count(numbers) not in (0, 1, 5, 6)
        and low_count(numbers) not in (0, 6)
        and 4 <= ac_value(numbers) <= 12
        and max_consecutive_run(numbers) < 4
        and max(zones) < 5
        and sum(v > 0 for v in zones) >= 3
    )
