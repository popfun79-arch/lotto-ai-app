from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from lotto64.data.updater import update_latest

KST = ZoneInfo("Asia/Seoul")

# 검증 기준점:
# 1236회 = 2026-08-08 토요일 추첨
ANCHOR_ROUND = 1236
ANCHOR_DRAW_DATE = datetime(2026, 8, 8, tzinfo=KST).date()


def expected_latest_round_kst(
    now: datetime | None = None,
) -> int:
    """
    현재 한국 시각 기준 이미 끝났어야 하는 최신 토요일 추첨 회차를 계산합니다.
    토요일 20:35경 추첨을 고려해 당일 21:30 이후부터 새 회차로 간주합니다.
    자동 workflow는 일요일 04:10이므로 정상적으로 직전 토요일 회차를 기대합니다.
    """
    now = now or datetime.now(KST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=KST)
    else:
        now = now.astimezone(KST)

    effective_date = now.date()

    # 토요일 추첨 당일에는 결과 반영 여유시간을 둡니다.
    if now.weekday() == 5 and (now.hour, now.minute) < (21, 30):
        effective_date = effective_date - timedelta(days=1)

    days = (effective_date - ANCHOR_DRAW_DATE).days

    if days < 0:
        return ANCHOR_ROUND - 1

    return ANCHOR_ROUND + days // 7


def main():
    parser = argparse.ArgumentParser(
        description="Lotto64 최신 데이터 업데이트"
    )
    parser.add_argument("--max-checks", type=int, default=20)
    parser.add_argument(
        "--require-expected",
        action="store_true",
        help="한국 시각 기준 예상 최신 회차보다 데이터가 뒤처지면 실패",
    )
    args = parser.parse_args()

    df, new_count = update_latest(max_checks=args.max_checks)

    latest = int(df["round"].max()) if len(df) else 0
    expected = expected_latest_round_kst()

    print(
        f"신규 {new_count}회 / 최신 {latest}회 / "
        f"예상 최신 {expected}회 / 전체 {len(df)}회"
    )

    if args.require_expected and latest < expected:
        raise RuntimeError(
            f"최신 데이터가 뒤처져 있습니다: "
            f"현재 {latest}회 / 예상 {expected}회"
        )


if __name__ == "__main__":
    main()
