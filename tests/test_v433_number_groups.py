from pathlib import Path
import pytest
from lotto64.utils.lotto_math import NUMBER_GROUPS, NUMBER_GROUP_LABELS, zone_counts, zone_index

def test_number_group_definition():
    assert NUMBER_GROUPS == ((1,9),(10,19),(20,29),(30,39),(40,45))
    assert NUMBER_GROUP_LABELS == ("1~9","10~19","20~29","30~39","40~45")

@pytest.mark.parametrize(("number","expected"),[(1,0),(9,0),(10,1),(19,1),(20,2),(29,2),(30,3),(39,3),(40,4),(45,4)])
def test_boundaries(number, expected):
    assert zone_index(number) == expected

def test_counts():
    assert zone_counts([9,10,19,20,29,40]) == (1,2,2,0,1)

@pytest.mark.parametrize("number",[0,46,-1,99])
def test_invalid(number):
    with pytest.raises(ValueError): zone_index(number)

def test_app_shows_groups():
    root=Path(__file__).resolve().parents[1]
    text=(root/'app.py').read_text(encoding='utf-8')
    assert 'NUMBER_GROUP_LABELS' in text
    assert 'Number Groups 적용 구간:' in text
