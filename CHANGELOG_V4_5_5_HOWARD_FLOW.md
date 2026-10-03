# Lotto64 v4.5.5 변경 이력 (Changelog)

## 📌 버전 요약: Gail Howard 7대 분석 및 대각선 흐름(이웃·폭포·역폭포) 패턴 종합 결합

이번 버전은 세계적인 복권 분석가 게일 하워드(Gail Howard)의 핵심 분석 체계와 현대 Python 시계열·통계 분석 기법을 융합하고, 한국 로또에서 매우 빈번하게 발생하는 **대각선 흐름(이웃수 ±1, 우하향 폭포 +1, 좌하향 역폭포 -1, 3연속 계단식 폭포/역폭포)** 패턴을 정식 모델과 추천 엔진, 그리고 Streamlit 대시보드에 전면 결합한 업데이트입니다.

---

### 1. 신규 분석 모듈 추가
- **`lotto64/analysis/flow_pattern.py`**:
  - **이웃수(Neighbor)**: 직전 회차 당첨번호 $n$의 인접 번호($n-1, n+1$) 자동 추출 및 통계 분석 (평균 회차당 1~2개 출현)
  - **폭포 패턴(Cascade / Waterfall: +1)**: $R-1$회차 $n \to R$회차 $n+1$ 우하향 대각선 흐름 및 $n \to n+1 \to \mathbf{n+2}$ 3연속 계단식 폭포 추적
  - **역폭포 패턴(Reverse Cascade: -1)**: $R-1$회차 $n \to R$회차 $n-1$ 좌하향 대각선 흐름 및 $n \to n-1 \to \mathbf{n-2}$ 3연속 계단식 역폭포 추적
  - **이월수(Repeat / Carryover)**: 직전 회차 번호 연속 출현 추적
  - 회차별 대각선 흐름 발생 이력 데이터프레임 생성(`analyze_flow_history`) 및 다음 회차 대각선 타깃 산출(`next_round_flow_targets`)
- **`lotto64/analysis/howard_guide.py`**:
  - Gail Howard의 7대 핵심 분석을 일원화하여 종합 보고서(`HowardSummaryReport`)로 도출
  1. Games Out / Skips: 단기(0~5), 중기(6~10), 장기(11+), 15회 이상 냉각 번호 분석
  2. Number Groups: 1~9, 10~19, 20~29, 30~39, 40~45 구간 밸런스 및 멸구간(Missing Group) 진단
  3. Last Digits: 끝수(0~9) 분포 및 동끝수(Pair) 출현 통계
  4. Odd-Even: 홀짝 3:3, 4:2, 2:4 정상 분포 진단
  5. High-Low: 저번호(1~22) vs 고번호(23~45) 황금 비율 분석
  6. Hot / Cold: 최근 20회/50회 기준 Hot(과열), Warm(보온), Cold(냉각) 번호 분류
  7. Sum Balance: 6개 번호 총합 시계열 이동평균 및 중심/안정 예측 범위(100~175) 진단

---

### 2. 추천 엔진 및 스코어링 고도화
- **`lotto64/models/pattern_master.py`**:
  - 1~45 각 번호의 `master_score` 산출 시 대각선 흐름 모멘텀(이웃수, 폭포수 +1, 역폭포수 -1, 3연속 완성 타깃)을 정밀 결합
  - 번호별 대각선 분류 플래그(`is_neighbor`, `is_cascade`, `is_reverse_cascade`, `is_cascade_3step`, `is_reverse_cascade_3step`, `is_carryover`) 추가
- **`lotto64/recommend/final_pattern.py`**:
  - 최종 6개 번호 조합 평가 시 대각선 흐름 지표(`neighbor_count`, `cascade_count`, `reverse_cascade_count`, `cascade_3step_count`, `reverse_cascade_3step_count`)와 직관적 요약 문자열(`flow_summary`, 예: `이웃2 · 폭포1 · 이월1`)을 메타데이터로 함께 산출
  - 기존 12개 가중치 인터페이스 및 테스트 100% 하위 호환 보존

---

### 3. Streamlit 대시보드 (`app.py`) UI 확장
- 신규 탭: **"Gail Howard·흐름 패턴"** (5번째 탭)
  - Gail Howard 7대 분석 종합 요약 브리핑 카드
  - 대각선 흐름(이웃/폭포/역폭포/이월) 실시간 후보군 및 평균 출현율 표시
  - **3연속 폭포(+1) 및 역폭포(-1) 진행 중인 강력 추천 타깃 특별 알림**
  - 최근 20회차 대각선 흐름 발생 이력 테이블
  - 번호대별 멸구간 진단 및 끝수 동끝수 현황
  - Hot / Warm / Cold 번호 그룹 리스트
- **Final Pattern** 및 **Top of the Best** 탭:
  - 추천 조합 테이블에 이웃/폭포/역폭포 흐름 요약(`flow_summary`) 컬럼 제공

---

### 4. 무결성 검증 (Verification)
- 신규 단위 테스트 추가: `tests/test_flow_pattern.py` (3개 테스트), `tests/test_howard_guide.py` (1개 테스트)
- 전체 253개 테스트 100% 통과 완료 (`253 passed in 5.87s`)
