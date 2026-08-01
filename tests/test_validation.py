import pandas as pd

from lotto64.data.validation import validate_and_clean

def sample():
    return pd.DataFrame([
        {"round": 1, "date": "2002-12-07", "n1": 1, "n2": 2, "n3": 3, "n4": 10, "n5": 20, "n6": 30, "bonus": 40},
        {"round": 2, "date": "2002-12-14", "n1": 4, "n2": 11, "n3": 21, "n4": 31, "n5": 41, "n6": 45, "bonus": 0},
    ])

def test_validate_rows():
    cleaned, notes = validate_and_clean(sample())
    assert len(cleaned) == 2
    assert "보너스 미입력" in " ".join(notes)

def test_numbers_sorted():
    df = sample()
    df.loc[0, ["n1","n2","n3","n4","n5","n6"]] = [30, 20, 10, 3, 2, 1]
    cleaned, _ = validate_and_clean(df)
    assert cleaned.loc[0, "n1"] == 1
    assert cleaned.loc[0, "n6"] == 30
