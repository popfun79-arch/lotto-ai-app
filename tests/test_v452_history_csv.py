from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_history_csv_is_consecutive_from_937_through_latest():
    df = pd.read_csv(ROOT / "data" / "lotto_all.csv", encoding="utf-8-sig")

    latest_round = int(df["round"].max())
    expected_rounds = list(range(937, latest_round + 1))

    assert len(df) >= 300
    assert int(df["round"].min()) == 937
    assert df["round"].tolist() == expected_rounds
    assert not df["round"].duplicated().any()
    assert not (df["bonus"] == 0).any()
