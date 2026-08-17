from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_history_csv_is_exact_937_1236():
    df = pd.read_csv(ROOT / "data" / "lotto_all.csv", encoding="utf-8-sig")
    assert len(df) == 300
    assert int(df["round"].min()) == 937
    assert int(df["round"].max()) == 1236
    assert df["round"].tolist() == list(range(937, 1237))
    assert not df["round"].duplicated().any()
    assert not (df["bonus"] == 0).any()
