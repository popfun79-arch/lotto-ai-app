from __future__ import annotations

from typing import Sequence

from lotto64.config import PRIMES

# Lotto64 v4.3.3 Number Groups
# 1~9 / 10~19 / 20~29 / 30~39 / 40~45
NUMBER_GROUPS: tuple[tuple[int, int], ...] = (
    (1, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 45),
)

NUMBER_GROUP_LABELS: tuple[str, ...] = tuple(
    f"{start}~{end}" for start, end in NUMBER_GROUPS
)

def zone_index(number: int) -> int:
    """Return the Number Group index for a Lotto 6/45 number."""
    value = int(number)
    if not 1 <= value <= 45:
        raise ValueError(f"로또 번호 범위 오류: {value}")
    for index, (start, end) in enumerate(NUMBER_GROUPS):
        if start <= value <= end:
            return index
    raise ValueError(f"Number Group을 찾지 못했습니다: {value}")

def zone_counts(numbers: Sequence[int]) -> tuple[int, int, int, int, int]:
    counts = [0] * len(NUMBER_GROUPS)
    for number in numbers:
        counts[zone_index(int(number))] += 1
    return tuple(counts)

def odd_count(numbers: Sequence[int]) -> int:
    return sum(int(n) % 2 == 1 for n in numbers)

def low_count(numbers: Sequence[int]) -> int:
    return sum(int(n) <= 22 for n in numbers)

def prime_count(numbers: Sequence[int]) -> int:
    return sum(int(n) in PRIMES for n in numbers)

def end_digit_sum(numbers: Sequence[int]) -> int:
    return sum(int(n) % 10 for n in numbers)

def ac_value(numbers: Sequence[int]) -> int:
    values = sorted(map(int, numbers))
    diffs = {
        values[j] - values[i]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    }
    return len(diffs) - (len(values) - 1)

def consecutive_pairs(numbers: Sequence[int]) -> int:
    values = sorted(map(int, numbers))
    return sum(values[i] == values[i - 1] + 1 for i in range(1, len(values)))

def max_consecutive_run(numbers: Sequence[int]) -> int:
    values = sorted(map(int, numbers))
    if not values:
        return 0
    best = run = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best

def neighbor_set(numbers: Sequence[int]) -> set[int]:
    result: set[int] = set()
    for number in numbers:
        if number > 1:
            result.add(number - 1)
        if number < 45:
            result.add(number + 1)
    return result

def format_combo(numbers: Sequence[int]) -> str:
    return " ".join(map(str, sorted(map(int, numbers))))
