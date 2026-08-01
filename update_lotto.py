from __future__ import annotations

import argparse

from lotto64.data.updater import update_latest

def main():
    parser = argparse.ArgumentParser(description="Lotto64 최신 데이터 업데이트")
    parser.add_argument("--max-checks", type=int, default=20)
    args = parser.parse_args()

    df, new_count = update_latest(max_checks=args.max_checks)
    latest = int(df["round"].max()) if len(df) else 0
    print(f"신규 {new_count}회 / 최신 {latest}회 / 전체 {len(df)}회")

if __name__ == "__main__":
    main()
