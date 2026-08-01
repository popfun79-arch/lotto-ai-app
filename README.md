# Lotto64 Ultimate AI

`popfun79-arch/lotto-ai-app` 저장소의 기존 파일을 교체하기 위한 실행형 프로젝트입니다.

## 교체되는 기존 파일
- `app.py`
- `lotto.csv`
- `lotto_200.csv`
- `requirements.txt`
- `update_lotto.py`

## 새로 추가되는 파일
- `README.md`
- `data/lotto_all.csv`
- `tests/test_smoke.py`
- `.github/workflows/ci.yml`
- `.streamlit/config.toml`

## 포함 기능
- 데이터 검증 및 컬럼 자동 정리
- GAP 기간별 당첨 분포
- 번호별 현재 GAP
- EGR 실제 검증
- CEC·DRC 상태전이 검증
- 회차 DNA와 유사 회차 검색
- 후보 번호 15수·13수·11수
- 안정형·균형형·공격형 TOP20
- 구매용 5·10·20게임
- 동일 난수 시드 기반 결과 재현

## 보너스 번호 안내
기존 `lotto_200.csv`에는 보너스 번호가 없으므로 마이그레이션 자료에서는 `bonus=0`입니다.
보너스 전이 분석을 정확히 사용하려면 보너스 번호가 포함된 전체 CSV를 업로드하거나 다음 명령으로 교체하세요.

```bash
python update_lotto.py --import-csv 최신_전체데이터.csv
```

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- Repository: `popfun79-arch/lotto-ai-app`
- Branch: `main`
- Main file path: `app.py`

로또는 무작위 추첨입니다. 본 앱은 당첨을 보장하지 않는 연구·검증 도구입니다.
