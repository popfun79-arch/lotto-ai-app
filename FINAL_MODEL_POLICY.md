# Lotto64 Final Pattern Model Policy

## 기본 베이스

### Python 분석
- 최근 20/50/100회 rolling frequency
- 최근 50회 vs 이전 50회 regime 비교
- GAP empirical hazard
- 현재 GAP percentile
- 개인 평균 GAP 대비 Skips Due
- 회차 DNA / 유사 회차
- 회차별 번호합 시계열
- 회차별 GAP합 시계열
- Walk-forward 검증
- GA/포트폴리오 분산

### Gail Howard에서 참고한 공개적 분석 주제
- Drawings Since Hit / Games Out
- Skip and Hit
- Skips Due
- Number Groups
- Last Digits
- Odd-Even
- High-Low
- Hot / Cold
- Sum balance / wheeling의 분산 개념

상용 프로그램의 비공개 계산식을 복제하지 않고,
위 공개 분석 주제를 한국 로또 6/45 데이터에 맞춰 재구성합니다.

## 최종 우선순위

1. GAP합 시계열 + GAP구간별 당첨 분포
2. 당첨번호 6개 합계 시계열
3. 최근 50/100회의 regime 변화
4. Drawings Since Hit / Skip-Hit / Skips Due
5. 후보번호 11/13/15
6. Odd-Even / High-Low / Number Groups / Last Digits
7. DNA / 유사 회차 / 빈도
8. GA 및 TOP20 포트폴리오 다양성

## 백테스트 원인 분석

단순 적중 개수 외에 아래를 기록합니다.

- 후보 11/13/15 적중수
- 실제 번호합 vs 예측 번호합 구간
- 실제 GAP합 vs 예측 GAP합 구간
- TOP 포트폴리오 최고 적중
- 후보번호 단계 실패
- 합계 상태 예측 실패
- GAP합 상태 예측 실패
- 조합구성 단계 손실

로또는 독립적인 무작위 추첨이며, 과거 패턴은 미래 당첨을 보장하지 않습니다.
