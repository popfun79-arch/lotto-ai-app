# Lotto64 v4.4.2 — Number Group Label Hotfix

## 원인
`app.py`가 `NUMBER_GROUP_LABELS`를 화면에 표시하면서
해당 이름을 import하지 않아 `NameError`가 발생했습니다.

Streamlit의 tabs는 선택된 탭만 실행하는 구조가 아니라
전체 Python 스크립트를 위에서 아래로 실행하므로,
한 탭의 NameError가 사실상 모든 탭을 막았습니다.

## 수정
- `app.py`에 `NUMBER_GROUP_LABELS` 안전 import 추가
- 부분 업로드 상황을 위한 fallback 라벨 추가
- CI에 Number Group labels 실제 import 검사 추가
- 전용 회귀 테스트 추가
- v4.4.2로 버전 갱신
