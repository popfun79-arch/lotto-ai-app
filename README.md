# Lotto64 Ultimate AI v4.2 — Top of the Best Edition

Python 통계/시계열 분석을 기본 베이스로 하고, Gail Howard의 Lottery Master Guide 계열에서 공개적으로 강조되는 Games Out/Skips, Number Groups, Last Digits, Odd-Even, High-Low, Hot/Cold, Sum Balance 성격의 분석 주제를 결합한 연구용 Lotto64 프로젝트입니다.

## 최종 핵심

1. 회차별 **GAP 합계 시계열**
2. **GAP 구간별 당첨 분포 / empirical hazard**
3. 회차별 **당첨번호 6개 합계 시계열**
4. 최근 **50회 vs 이전 50회 / 최근 100회 regime**
5. Drawings Since Hit / Skip-Hit / Skips Due
6. 후보 11·13·15수
7. 베스트 5·10·15·20조합
8. Walk-forward + 실패 원인 분석

## 데이터

포함 데이터:
- 1024~1235회
- 212회
- 본번호 6개 + 실제 보너스 번호
- 최신 포함: 1235회 (2026-08-01)

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

로또는 무작위 추첨입니다. 이 프로젝트의 점수와 패턴은 과거 데이터에 대한 상대적 순위/검증 지표이지 실제 당첨 확률을 높인다는 보장이 아닙니다.


## v4.1 Top of the Best

기존 `TOP20` 탭의 안정형·균형형·공격형 3세트 추천을 제거했습니다.

이제 하나의 Final Pattern 통합 포트폴리오만 사용합니다.

- BEST 5 = 최종 1~5위
- BEST 10 = 최종 1~10위
- BEST 15 = 최종 1~15위
- BEST 20 = 최종 1~20위

따라서 `BEST 5 ⊂ BEST 10 ⊂ BEST 15 ⊂ BEST 20`입니다.
네 화면을 합쳐 50개를 새로 만드는 것이 아니라,
실제 고유 추천 조합은 최대 20개입니다.


## v4.2 운영 설정

- 자동 데이터 업데이트: 매주 **일요일 오전 04:00 KST**
- GitHub Actions cron: `0 19 * * 6` (UTC 기준)
- 기본 분석 범위: **최신 200회**
- 원본 데이터는 전체 이력을 보관하고, 분석만 최근 200회를 기본 사용
- 사이드바에서 필요할 경우 분석 범위를 100~300회로 조정 가능

이 설정은 토요일 추첨 직후가 아니라 충분한 시간 간격을 둔 뒤 데이터를
업데이트하도록 하여 외부 데이터 소스 반영 지연 가능성을 줄이기 위한 운영 설정입니다.
