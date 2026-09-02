# Lotto64 Ultimate AI v4.5 — Historical Validation Ledger

Python 통계·시계열 분석을 기본 베이스로 하고, Gail Howard의 Lottery Master Guide 계열에서 공개적으로 강조되는 Games Out/Skips, Number Groups, Last Digits, Odd-Even, High-Low, Hot/Cold, Sum Balance 성격의 분석 주제를 결합한 연구용 Lotto64 프로젝트입니다.

## 최종 핵심

1. 회차별 **건너띔 합계 시계열·구간 분포** (`skip=0` 기준)
2. **현재 상태+방향 기반 다음 회차 전이 예측**
3. **건너띔 기간 구간별 6개 구성 / empirical hazard**
4. 회차별 **GAP 합계·당첨번호 6개 합계 시계열**
5. 최근 **50회 vs 이전 50회 / 최근 100회 regime**
6. Drawings Since Hit / Skip-Hit / Skips Due
7. 후보 **11·13·15수**
8. **Top of the Best 5·10·15·20조합**
9. Walk-forward + 실패 원인 분석

## 데이터

포함 데이터:

- **937~1239회**
- **303회**
- 본번호 6개 + 실제 보너스 번호
- 최신 포함: **1239회**

전체 이력은 보존하고, Streamlit 분석 기본 범위는 **최신 200회**를 사용합니다.

## Top of the Best

기존 안정형·균형형·공격형 3세트 TOP20은 사용하지 않습니다.

하나의 Final Pattern 통합 포트폴리오를 사용합니다.

- BEST 5 = 최종 1~5위
- BEST 10 = 최종 1~10위
- BEST 15 = 최종 1~15위
- BEST 20 = 최종 1~20위

`BEST 5 ⊂ BEST 10 ⊂ BEST 15 ⊂ BEST 20` 구조이며, 실제 고유 추천 조합은 최대 20개입니다.

## 자동 데이터 업데이트

- 매주 **일요일 오전 04:10 KST**
- GitHub Actions timezone: `Asia/Seoul`
- 정각 혼잡을 피하기 위해 04:00 대신 04:10 사용
- 최신 공개 전체 데이터셋을 우선 확인
- 기존 개별 회차 endpoint는 fallback
- 예상 최신 회차보다 데이터가 뒤처지면 workflow를 실패 처리하여 누락을 숨기지 않음

## Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud:

- Repository: `popfun79-arch/lotto-ai-app`
- Branch: `main`
- Main file: `app.py`

## 주의

로또는 무작위 추첨입니다. 이 프로젝트의 점수와 패턴은 과거 데이터에 대한 상대적 순위·검증 지표이며 실제 당첨을 보장하지 않습니다.


## Number Groups v4.3.3

`1~9 / 10~19 / 20~29 / 30~39 / 40~45` 구간을 사용합니다.
이 기준은 후보번호 점수, Number Group Recovery, 회차 DNA, 조합 구간 분산/필터, Top of the Best 최종 랭킹에 공통 적용됩니다.


## Seed Stability

v4.4부터 Lotto64의 랜덤 탐색은 다음 원칙으로 운영합니다.

- 고정 기준 Seed: `20260720`
- 기본 다중 Seed 검증: 5개
- 선택 가능한 검증 수: 3 / 5 / 7
- 좋은 Seed를 사후 선택하지 않음
- 동일 설정에서 결과가 여러 Seed에서도 반복되는지를 검증

### GA 안정성
정확히 같은 조합이 여러 Seed에서 반복되는 비율과 번호별 노출 패턴을
분석합니다.

### Walk-forward 안정성
같은 과거 회차를 여러 Seed로 다시 예측해 TOP 조합 최고 적중수의
표준편차와 Seed 일치율을 측정합니다.

Final Pattern 기반 Top of the Best는 완전탐색/결정론적 랭킹이므로
Seed가 바뀌어도 자체 결과는 바뀌지 않습니다.


## v4.5 Historical Validation Ledger

과거 회차를 단순 재분석하지 않고, 각 목표 회차 직전 200회만 사용하여
한 회차씩 다시 예측하는 엄격한 Rolling Walk-forward 검증 장부입니다.

기록 항목:
- 후보 11/13/15수 적중
- BEST 5/10/15/20 최고 적중
- 번호합 예측
- GAP합 예측
- Number Groups
- GAP 구간 구성
- Pattern Master 순위
- 주요 실패 원인

최근 100회를 엄격하게 검증하려면 300개 연속 회차가 필요합니다.
Weekly Lotto Data Update는 v4.5부터 937회까지 과거 데이터를 자동 백필하고,
검증 원장을 생성하여 `reports/`에 저장합니다.

Historical Ledger의 목적은 과거 적중률을 최대화하는 것이 아니라
`후보 → 번호합/GAP합 → 조합 → 포트폴리오` 중 어느 단계에서
손실이 반복되는지 확인하는 것입니다.

## v4.5.4 Skip Sum Transition

첨부 분석표의 건너띔 기준을 코드에 직접 반영했습니다.

- `skip=0`: 직전 회차 출현 번호가 연속 재출현
- 당첨번호 6개의 회차별 skip 값·합계·평균·최대·구간 구성 기록
- 개별 skip 기간 분포와 회차별 skip 합계 구간 분포 제공
- 최근 합계 상태(LOW/MID/HIGH/EXTREME)와 방향(UP/DOWN/FLAT)이
  같았던 과거 회차의 다음 회차를 우선 사용
- 일치 표본이 부족하면 동일 상태, 최근 전이 순으로 자동 후퇴
- 전용 skip 합계·전이 구성·empirical hazard를 최종 점수의 29%로 반영
- 기존 GAP은 연속 재출현을 1로 계산하므로 skip 합계와 비교할 때
  6개 번호 기준 차이를 보정

이 지표는 과거 조건부 분포를 정량화한 보조 신호이며 당첨을 보장하지 않습니다.
