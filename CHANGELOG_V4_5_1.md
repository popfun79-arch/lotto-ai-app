# Lotto64 v4.5.1 — Historical Ledger Compatibility Hotfix

## 원인
v4.5 Historical Ledger는 `NUMBER_GROUP_LABELS`를 직접 import합니다.
GitHub에 patch 파일을 여러 번 나누어 업로드하는 과정에서
`historical_ledger.py`는 v4.5인데 `lotto_math.py`가 이전 버전으로 남으면
CI Import Check 단계에서 실패할 수 있습니다.

## 수정
- v4.3.3+ 기준 `lotto_math.py`를 이번 patch에 반드시 포함
- Historical Ledger에 partial-deployment fallback 추가
- Number Groups를 항상
  `1~9 / 10~19 / 20~29 / 30~39 / 40~45`
  로 유지
- CI에서 Historical Ledger의 Number Group import를 직접 검증
- 전용 회귀 테스트 추가
