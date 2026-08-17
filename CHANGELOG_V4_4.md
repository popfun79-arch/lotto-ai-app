# Lotto64 v4.4 — Fixed Seed + Multi-Seed Stability

## Fixed Seed
- 기준 Seed를 `20260720`으로 고정
- 실전/기본 GA/기본 Walk-forward 결과 재현성 유지
- Streamlit에서 Seed 직접 변경을 잠금
- Seed 값 자체에는 예측력이 없음을 UI에 명시

## Multi-Seed GA Stability
- 기본 5개 Seed, 선택 3/5/7개
- 기준 Seed에서 결정적으로 파생된 검증 Seed 사용
- exact combination Seed coverage
- 평균/최고 순위
- robustness score / S~D 안정성 등급
- 번호별 exposure 평균/표준편차/Seed presence
- Seed 간 조합 Jaccard
- 번호 노출 상관계수

## Multi-Seed Walk-forward
- 최근 10~30회에서 3/5/7 Seed 반복
- Seed별 TOP 조합 최고 적중 평균
- 회차별 Seed 결과 표준편차
- Seed 완전 일치율
- 3+/4+ 적중률 비교

## 원칙
다중 Seed는 '좋은 Seed를 고르는 기능'이 아닙니다.
같은 모델이 Seed가 달라도 비슷한 결론을 내는지 검증하는 용도입니다.
Top of the Best Final Pattern은 결정론적이므로 Seed에 직접 의존하지 않습니다.
