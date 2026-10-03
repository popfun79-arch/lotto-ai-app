from __future__ import annotations

import pandas as pd
import pytest

from lotto64.analysis.howard_guide import build_howard_summary


def test_build_howard_summary():
    # 35회차 합성 데이터 생성
    rows = []
    for r in range(1, 40):
        # 유효한 6개 번호 (정렬됨, 중복 없음)
        base = (r % 7) * 5 + 1
        nums = sorted([
            base,
            (base + 3) % 45 + 1,
            (base + 8) % 45 + 1,
            (base + 15) % 45 + 1,
            (base + 22) % 45 + 1,
            (base + 31) % 45 + 1,
        ])
        # 중복 방지
        unique_nums = []
        curr = 1
        for n in nums:
            val = max(curr, n)
            if val > 45:
                val = 45
            unique_nums.append(val)
            curr = val + 1
        unique_nums = sorted(list(set(unique_nums)))
        while len(unique_nums) < 6:
            unique_nums.append(unique_nums[-1] + 1)

        rows.append({
            "round": r,
            "n1": unique_nums[0],
            "n2": unique_nums[1],
            "n3": unique_nums[2],
            "n4": unique_nums[3],
            "n5": unique_nums[4],
            "n6": unique_nums[5],
            "bonus": (unique_nums[5] % 45) + 1,
        })

    df = pd.DataFrame(rows)
    report = build_howard_summary(df, window=30)

    assert report.latest_round == 39
    assert report.target_round == 40
    assert len(report.latest_numbers) == 6
    assert isinstance(report.group_counts_latest, dict)
    assert isinstance(report.last_digit_counts_latest, dict)
    assert ":" in report.latest_odd_even
    assert ":" in report.latest_high_low
    assert report.sum_target_low <= report.sum_target_high
    assert report.flow_targets is not None
    assert report.flow_targets.target_round == 40
