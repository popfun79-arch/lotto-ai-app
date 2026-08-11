# Lotto64 Ultimate AI v4.3 — Top of the Best Edition

Python 통계·시계열 분석을 기본 베이스로 하고, Gail Howard의 Lottery Master Guide 계열에서 공개적으로 강조되는 Games Out/Skips, Number Groups, Last Digits, Odd-Even, High-Low, Hot/Cold, Sum Balance 성격의 분석 주제를 결합한 연구용 Lotto64 프로젝트입니다.

## 최종 핵심

1. 회차별 **GAP 합계 시계열**
2. **GAP 구간별 당첨 분포 / empirical hazard**
3. 회차별 **당첨번호 6개 합계 시계열**
4. 최근 **50회 vs 이전 50회 / 최근 100회 regime**
5. Drawings Since Hit / Skip-Hit / Skips Due
6. 후보 **11·13·15수**
7. **Top of the Best 5·10·15·20조합**
8. Walk-forward + 실패 원인 분석

## 데이터

포함 데이터:

- **1024~1236회**
- **213회**
- 본번호 6개 + 실제 보너스 번호
- 최신 포함: **1236회 (2026-08-08)**
- 1236회: **12, 18, 21, 29, 34, 38 + 보너스 10**

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
