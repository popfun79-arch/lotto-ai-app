# Lotto64 v4.5 — Historical Validation Ledger

## 핵심
- 엄격한 200회 Rolling Walk-forward 기반 회차별 검증 원장
- 최근 50회 / 최근 100회 선택 검증
- 후보 11·13·15수 적중 기록
- BEST 5·10·15·20 최고 적중 기록
- 번호합 실제값 / 예상 중심 / 핵심·확장구간 / 오차
- GAP합 실제값 / 예상 중심 / 핵심·확장구간 / 오차
- Number Groups 실제 구성
- 실제 당첨번호의 GAP 구간 구성
- Pattern Master 실제 당첨번호 순위
- 회차별 주요 실패 원인 자동 분류
- 이전 50회 vs 최근 50회 비교
- 실패 원인 누적 분포
- 누적 검증 추세
- CSV/JSON 저장 및 다운로드

## Historical Backfill
- 100회 검증 + 200회 학습창에 필요한 과거 데이터를 자동 보충
- 기본 백필 시작: 937회
- Weekly Lotto Data Update가 `--history-start 937` 실행
- 원격 전체 데이터셋에서 누락 과거 회차를 보충
- 백필 후 누락/중복을 검증

## Weekly Ledger
- 주간 데이터 업데이트 후 Historical Ledger 100회를 자동 생성/갱신
- reports/historical_validation_ledger.csv
- reports/historical_validation_summary.json
- GitHub에 데이터와 함께 자동 커밋

## 중요한 해석
Historical Ledger는 과거 100회에 맞춰 규칙을 계속 튜닝하는 용도가 아닙니다.
v4.5의 기준선을 구축하고 실패 위치를 진단한 뒤,
향후 실제 회차에서 같은 규칙을 고정해 검증하기 위한 장부입니다.
