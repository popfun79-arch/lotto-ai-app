from __future__ import annotations

import argparse

import pandas as pd

from lotto64.backtest.historical_ledger import (
    build_historical_ledger,
    evaluate_target_round,
    load_saved_ledger,
    save_ledger,
)
from lotto64.data.storage import read_csv
from lotto64.data.validation import validate_and_clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lotto64 Historical Validation Ledger"
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--train-window", type=int, default=200)
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "원장이 최신이면 종료, 1회만 뒤처지면 최신 회차만 추가, "
            "그 외에는 최근 N회를 재생성"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 원장과 관계없이 최근 N회를 재생성",
    )
    args = parser.parse_args()

    df = read_csv()
    if df is None or df.empty:
        raise RuntimeError("data/lotto_all.csv를 찾지 못했습니다.")
    df, _ = validate_and_clean(df)
    df = df.sort_values("round").reset_index(drop=True)

    latest = int(df["round"].max())
    saved = load_saved_ledger()

    if args.auto and not args.force and not saved.empty:
        saved = saved.sort_values("round").drop_duplicates(
            "round", keep="last"
        ).reset_index(drop=True)
        saved_latest = int(saved["round"].max())

        if saved_latest == latest:
            print(
                f"Historical Ledger 최신 상태 / "
                f"{int(saved['round'].min())}~{saved_latest} / "
                f"{len(saved)}회"
            )
            return

        if saved_latest == latest - 1:
            target_index = int(df.index[df["round"] == latest][0])
            if target_index >= args.train_window:
                print(f"Historical Ledger 최신 {latest}회 1행 추가")
                row = evaluate_target_round(
                    df,
                    target_index=target_index,
                    train_window=args.train_window,
                )
                combined = pd.concat(
                    [saved, pd.DataFrame([row])],
                    ignore_index=True,
                )
                combined = (
                    combined.sort_values("round")
                    .drop_duplicates("round", keep="last")
                    .tail(args.rounds)
                    .reset_index(drop=True)
                )
                ledger_path, summary_path = save_ledger(combined)
                print(
                    f"증분 저장 완료: {ledger_path} / {summary_path} / "
                    f"{int(combined['round'].min())}~"
                    f"{int(combined['round'].max())}"
                )
                return

    print(
        f"Historical Ledger 전체 생성 / 최근 {args.rounds}회 / "
        f"학습창 {args.train_window}회"
    )
    ledger = build_historical_ledger(
        df,
        validation_rounds=args.rounds,
        train_window=args.train_window,
        progress_callback=lambda value, text: print(
            f"[{value:6.1%}] {text}"
        ),
    )
    ledger_path, summary_path = save_ledger(ledger)
    print(
        f"저장 완료: {ledger_path} / {summary_path} / "
        f"{int(ledger['round'].min())}~{int(ledger['round'].max())}"
    )


if __name__ == "__main__":
    main()
