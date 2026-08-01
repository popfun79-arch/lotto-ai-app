# Lotto64 Ultimate AI v3.0

Streamlit, GitHub Actions, SQLite, GAP/EGR/CEC/DRC, 회차 DNA, 유사 회차, GA 조합 최적화, Walk-forward 백테스트와 가중치 탐색을 통합한 연구용 프로젝트입니다.

## 주요 기능

- CSV/JSON 업로드와 데이터 품질 검증
- 자동 최신 회차 업데이트 시도
- CSV와 SQLite 동기화
- GAP 분포와 번호별 현재 GAP
- EGR 회복률 검증
- CEC·DRC 상태전이 검증
- 회차 DNA와 유사 회차 검색
- 후보 15수·13수·11수
- 번호별 상대확률 지표와 설명
- 안정형·균형형·공격형 TOP20
- 유전자 알고리즘 기반 조합 최적화
- Walk-forward 백테스트와 성능 그래프
- 확률적 가중치 탐색
- GitHub Actions CI
- 주간 데이터 업데이트 Workflow

> 로또는 무작위 추첨입니다. 이 프로젝트는 당첨을 보장하지 않으며 과거 데이터 연구와 가설 검증을 위한 도구입니다.

## GitHub에 교체 업로드

압축을 푼 뒤 저장소의 기존 내용을 백업하고, 이 프로젝트의 모든 파일과 폴더를 저장소 루트에 업로드합니다.

최종 구조:

```text
lotto-ai-app/
├─ app.py
├─ update_lotto.py
├─ requirements.txt
├─ README.md
├─ lotto64/
├─ data/
├─ reports/
├─ tests/
├─ .streamlit/
└─ .github/workflows/
```

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

- Repository: `popfun79-arch/lotto-ai-app`
- Branch: `main`
- Main file path: `app.py`

## 주간 데이터 자동 업데이트

`.github/workflows/update_data.yml`은 매주 토요일 UTC 13:30에 실행되도록 설정되어 있습니다. 공식 사이트 응답 형식이 바뀌면 자동 수집이 실패할 수 있으며, 그 경우 CSV 업로드로 갱신하세요.

## 테스트 수

`tests/test_math.py`는 번호 1~45를 매개변수화해 180개 이상의 개별 검증 사례를 수행합니다.
