import pytest

from lotto64.utils.lotto_math import (
    ac_value, consecutive_pairs, end_digit_sum, low_count,
    max_consecutive_run, odd_count, prime_count, zone_counts,
)

@pytest.mark.parametrize("number", range(1, 46))
def test_single_number_zone_total(number):
    assert sum(zone_counts([number])) == 1

@pytest.mark.parametrize("number", range(1, 46))
def test_end_digit(number):
    assert end_digit_sum([number]) == number % 10

@pytest.mark.parametrize("number", range(1, 46))
def test_odd_count(number):
    assert odd_count([number]) in (0, 1)

@pytest.mark.parametrize("number", range(1, 46))
def test_low_count(number):
    assert low_count([number]) == int(number <= 22)

def test_ac_integer():
    assert isinstance(ac_value([1, 7, 13, 22, 34, 45]), int)

def test_consecutive_pairs():
    assert consecutive_pairs([1, 2, 3, 10, 20, 30]) == 2

def test_max_run():
    assert max_consecutive_run([1, 2, 3, 10, 20, 30]) == 3

def test_prime_count():
    assert prime_count([2, 3, 4, 5, 6, 7]) == 4
