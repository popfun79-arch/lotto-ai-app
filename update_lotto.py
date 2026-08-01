from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "lotto_all.csv"
ROOT_COPY = ROOT / "lotto.csv"
COLUMNS = ["round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "draw_num": "round", "num1": "n1", "num2": "n2", "num3": "n3",
        "num4": "n4", "num5": "n5", "num6": "n6",
        "회차": "round", "날짜": "date", "보너스": "bonus",
    }
    out = df.rename(columns=aliases).copy()

    if "date" not in out and "round" in out:
        rounds = pd.to_numeric(out["round"], errors="coerce")
        out["date"] = pd.Timestamp("2002-12-07") + pd.to_timedelta((rounds - 1) * 7, unit="D")
    if "bonus" not in out:
        out["bonus"] = 0

    missing = [c for c in COLUMNS if c not in out]
    if missing:
        raise ValueError("필수 컬럼 누락: " + ", ".join(missing))

    out = out[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["round", "n1", "n2", "n3", "n4", "n5", "n6"])
    out["bonus"] = out["bonus"].fillna(0)
    out[["round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]] = (
        out[["round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"]].astype(int)
    )

    def sort_numbers(row):
        nums = sorted(row[f"n{i}"] for i in range(1, 7))
        for i, n in enumerate(nums, 1):
            row[f"n{i}"] = n
        return row

    out = out.apply(sort_numbers, axis=1)
    valid = out.apply(
        lambda r: len({r[f"n{i}"] for i in range(1, 7)}) == 6
        and all(1 <= r[f"n{i}"] <= 45 for i in range(1, 7))
        and 0 <= r["bonus"] <= 45,
        axis=1,
    )
    return (
        out.loc[valid]
        .drop_duplicates("round", keep="last")
        .sort_values("round")
        .reset_index(drop=True)
    )


def fetch_round(draw_no: int, timeout: int = 10) -> Optional[dict]:
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 Lotto64-Updater/2.0"},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("returnValue") != "success":
            return None
        return {
            "round": int(data["drwNo"]),
            "date": data["drwNoDate"],
            **{f"n{i}": int(data[f"drwtNo{i}"]) for i in range(1, 7)},
            "bonus": int(data["bnusNo"]),
        }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def save(df: pd.DataFrame) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    df.to_csv(ROOT_COPY, index=False, encoding="utf-8-sig")


def update(max_checks: int = 20) -> pd.DataFrame:
    current = validate(pd.read_csv(OUTPUT, encoding="utf-8-sig")) if OUTPUT.exists() else pd.DataFrame(columns=COLUMNS)
    latest = int(current["round"].max()) if len(current) else 0
    new_rows = []
    failures = 0

    for draw_no in range(latest + 1, latest + max_checks + 1):
        item = fetch_round(draw_no)
        if item is None:
            failures += 1
            if failures >= 3:
                break
        else:
            failures = 0
            new_rows.append(item)
        time.sleep(0.15)

    if new_rows:
        result = validate(pd.concat([current, pd.DataFrame(new_rows)], ignore_index=True))
        save(result)
        print(f"신규 {len(new_rows)}회 저장, 최신 {int(result['round'].max())}회")
        return result

    print("자동 업데이트 자료를 가져오지 못했습니다.")
    print("사이트 응답 구조가 변경된 경우 --import-csv를 사용하세요.")
    return current


def import_csv(path: str) -> pd.DataFrame:
    result = validate(pd.read_csv(path, encoding="utf-8-sig"))
    save(result)
    print(f"{len(result)}회 저장, 최신 {int(result['round'].max())}회")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-csv", help="외부 CSV로 안전하게 교체")
    parser.add_argument("--max-checks", type=int, default=20)
    args = parser.parse_args()
    import_csv(args.import_csv) if args.import_csv else update(args.max_checks)


if __name__ == "__main__":
    main()
