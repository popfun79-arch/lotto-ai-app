# Lotto64 Ultimate AI v4.0 — Final Pattern Edition

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
