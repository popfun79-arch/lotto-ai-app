from __future__ import annotations

from typing import Sequence

from lotto64.config import PRIMES

def zone_index(number: int) -> int:
    if number <= 10:
        return 0
    if number <= 20:
        return 1
    if number <= 30:
        return 2
    if number <= 40:
        return 3
    return 4

def zone_counts(numbers: Sequence[int]) -> tuple[int, int, int, int, int]:
    counts = [0, 0, 0, 0, 0]
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
