# 필수 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analysis.data_loader import LottoDataLoader
from analysis.statistics import LottoStatistics
from analysis.patterns import LottoPatternAnalysis
from analysis.gail_howard import GailHowardStrategy
from visualization.charts import LottoVisualizer
from prediction.predictor import LottoPredictor

# 프로젝트 모듈 임포트
sys.path.insert(0, str(Path.cwd()))
from analysis.data_loader import LottoDataLoader
from analysis.statistics import LottoStatistics
from analysis.patterns import LottoPatternAnalysis
from analysis.gail_howard import GailHowardStrategy
from visualization.charts import LottoVisualizer
from prediction.predictor import LottoPredictor

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')
sns.set_palette('husl')

print('✓ 모든 라이브러리 임포트 완료')



# 데이터 로드
loader = LottoDataLoader('data/lotto.csv')
data = loader.load()
data = loader.preprocess()

# 데이터 요약
summary = loader.get_summary()
print('\n📊 데이터 요약:')
for key, value in summary.items():
    print(f'  {key}: {value}')

print('\n첫 10행 미리보기:')
data.head(10)



# 통계 분석 객체 생성
stats = LottoStatistics(data)

# 기본 통계량
print('📈 기본 통계량:')
stats_dict = stats.get_statistics()
for key, value in stats_dict.items():
    print(f'  {key}: {value:.2f}' if isinstance(value, float) else f'  {key}: {value}')


# 핫번호 (자주 나오는 번호)
print('\n🔥 핫번호 TOP 15:')
hot = stats.get_hot_numbers(15)
print(hot.to_string())


# 콜드번호 (잘 안 나오는 번호)
print('\n❄️ 콜드번호 TOP 15:')
cold = stats.get_cold_numbers(15)
print(cold.to_string())


# 홀짝 비율
print('\n🔢 홀짝 비율:')
odd_even = stats.get_odd_even_ratio()
for key, value in odd_even.items():
    print(f'  {key}: {value}')


# 패턴 분석 객체 생성
pattern = LottoPatternAnalysis(data)

# 연번 패턴
print('📊 연번 패턴 분석:')
consecutive = pattern.get_consecutive_numbers()
print(consecutive.to_string())


# 번호 간 간격 분석
print('\n📏 번호 간 간격 분석 (TOP 10):')
gap = pattern.get_number_gap_analysis()
print(gap.head(10).to_string())


# 구간별 분포
print('\n📍 구간별 분포:')
section = pattern.get_section_distribution()
print(section.to_string())


# 시각화 객체 생성
viz = LottoVisualizer(figsize=(14, 8))

# 번호별 빈도 막대 그래프
freq_df = stats.get_frequency()
freq_df.columns = ['num', 'frequency']

fig = viz.plot_frequency_bar(freq_df, 'Lotto Number Frequency (1-45)')
plt.show()
viz.save_figure(fig, 'results/01_frequency_bar.png')


# 홀짝 비율 파이 차트
odd_even = stats.get_odd_even_ratio()
fig = viz.plot_odd_even_pie(odd_even['홀수'], odd_even['짝수'])
plt.show()
viz.save_figure(fig, 'results/02_odd_even_pie.png')


# 구간별 분포 막대 그래프
section_df = pattern.get_section_distribution()
section_df.columns = ['section', 'frequency']

fig = viz.plot_section_bar(section_df)
plt.show()
viz.save_figure(fig, 'results/03_section_distribution.png')


# 예측 객체 생성
predictor = LottoPredictor(data, stats)

# 다양한 전략으로 추천
predictor.generate_recommendations(count=5)


# 번호별 확률 예측
print('\n📊 번호별 예상 확률 (TOP 15):')
probability_df = predictor.predict_probability()
print(probability_df.to_string())


print('\n' + '='*60)
print('🎰 로또645 분석 최종 요약')
print('='*60)

print('\n1️⃣ 추천 번호 (최적 조합):')
recommended = predictor.recommend_balanced()
print(f'   {" ".join(map(str, recommended))}')

print('\n2️⃣ 최고 빈도 번호 TOP 5:')
hot_5 = stats.get_hot_numbers(5)
for _, row in hot_5.iterrows():
    print(f'   {int(row["num"])}: {int(row["frequency"])}회')

print('\n3️⃣ 최저 빈도 번호 TOP 5:')
cold_5 = stats.get_cold_numbers(5)
for _, row in cold_5.iterrows():
    print(f'   {int(row["num"])}: {int(row["frequency"])}회')

print('\n4️⃣ 홀짝 비율:')
odd_even = stats.get_odd_even_ratio()
print(f'   홀수: {odd_even["홀수 비율(%)"]}% | 짝수: {odd_even["짝수 비율(%)"]}%')

print('\n5️⃣ 주요 통계:')
stats_info = stats.get_statistics()
print(f'   평균: {stats_info["평균"]:.1f} | 중앙값: {stats_info["중앙값"]:.1f} | 표준편차: {stats_info["표준편차"]:.2f}')
print(f'   범위: {stats_info["최소값"]:.0f}-{stats_info["최대값"]:.0f}')

print('\n' + '='*60)


# 최종 분석 결론
print("\n" + "="*70)
print("🎯 최종 분석 결론")
print("="*70)

print("""
✅ 이 분석의 목적
   - 로또 데이터의 통계적 특성 파악
   - 다양한 분석 관점 제시
   - 과학적 사고방식 제공

⚠️ 중요한 주의사항
   - 로또는 완벽한 무작위 추첨입니다
   - 과거 패턴이 미래를 보장하지 않습니다
   - 이 분석으로 수익을 보장할 수 없습니다
   - 책임감 있는 도박 문화를 권장합니다

📈 게일하워드 전략의 의미
   - 무작위보다 더 균형잡힌 조합을 만드는 방법
   - 확률을 수학적으로 생각하는 방식 제시
   - 엔터테인먼트 목적의 과학적 접근

💡 현명한 로또 구매 방법
   1. 절대 수익 창출 수단으로 보지 않기
   2. 여유 자금으로만 참여하기
   3. 다양한 조합 체험해보기
   4. 즐거움을 우선으로 생각하기
""")

print("="*70)
print("\n✨ 분석 완료! 행운을 빕니다! 🍀")
print("="*70)


# 게일하워드 전략과 다른 전략의 최종 비교
print("\n" + "="*70)
print("📊 전략 비교 (최적 조합을 기준으로)")
print("="*70)

strategies_comparison = {
    '게일하워드 전략': {
        'description': '핫/콜드 균형 + 필터링',
        'recommended': combinations[0]['numbers'] if combinations else [],
        'strength': '균형잡힌 조합, 과학적 기준',
        'weakness': 'CPU 비용 높음, 완벽한 예측은 불가능'
    },
    '빈도 기반': {
        'description': '자주 나오는 번호 중심',
        'recommended': predictor.recommend_by_frequency(),
        'strength': '구현 간단, 빠름',
        'weakness': '편향될 수 있음, 다양성 낮음'
    },
    '균형 중심': {
        'description': '구간별 균형 고려',
        'recommended': predictor.recommend_balanced(),
        'strength': '다양성 좋음, 균형잡힘',
        'weakness': '고빈도 번호 놓칠 수 있음'
    },
    '대조 전략': {
        'description': '핫+콜드 혼합',
        'recommended': predictor.recommend_contrast(),
        'strength': '다양한 시각, 위험도 분산',
        'weakness': '불안정성, 이론적 근거 약함'
    }
}

for strategy_name, details in strategies_comparison.items():
    print(f"\n💡 {strategy_name}")
    print(f"   설명: {details['description']}")
    print(f"   추천: {details['recommended']}")
    print(f"   장점: {details['strength']}")
    print(f"   약점: {details['weakness']}")

print("\n" + "="*70)


# 각 필터링 규칙 개별 검증
test_combinations = [
    ([1, 2, 3, 4, 5, 6], "3개 이상 연번 - 제거"),
    ([1, 3, 11, 21, 31, 41], "구간 좋음 - 통과"),
    ([1, 2, 3, 24, 25, 26], "과도한 연번 - 제거"),
    ([2, 4, 6, 8, 10, 12], "모두 짝수 - 제거"),
    ([1, 3, 5, 7, 9, 11], "모두 홀수 - 제거"),
    ([5, 10, 15, 20, 25, 30], "저/고 불균형 - 제거"),
]

print("\n필터링 규칙 검증 테스트:")
print("-" * 70)

for nums, description in test_combinations:
    result = gail.apply_all_filters(nums)
    status = "✓ 통과" if result['passed'] else "✗ 제거"
    print(f"{status} | {nums}")
    print(f"       {description}")
    print(f"       필터 결과: {result['filters']}")
    print()


# 게일하워드 원리 상세 설명
print("\n" + "="*70)
print("📚 게일하워드 로또마스터 전략의 핵심 원리")
print("="*70)

print("\n🎯 핵심 전략:")
print("""
1. 핫/콜드 번호 균형 조합
   - 자주 나오는 번호(핫)와 잘 안 나오는 번호(콜드) 혼합
   - 과거 빈도 데이터를 기반으로 한 과학적 접근
   
2. 필터링 규칙 (불균형 조합 제거)
   - 연번 제거: 3개 이상 연속된 번호 제외
   - 구간 제한: 같은 10단위 구간에 3개 이상 번호 제외
   - 홀짝 균형: 3:3 비율 유지 (2 허용)
   - 고저 균형: 3:3 비율 유지 (2 허용)
   
3. 통계적 균형
   - 이상적인 조합의 특성을 정의하고 유지
   - 확률을 높이기 위한 과학적 필터링
   
4. 장점
   - 단순 랜덤보다 더 균형잡힌 조합 생성
   - 과도한 패턴 제거로 확률 향상
   - 재현 가능한 과학적 방법론
""")

print("="*70)


# 게일하워드 전략 적용
gail = GailHowardStrategy(data, stats)

# 게일하워드 전략 보고서 출력
combinations = gail.get_strategy_report()


import matplotlib.pyplot as plt
import numpy as np

# 시각화 1: 당첨번호 합계 분포
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1-1: 합계 분포 그래프
ax = axes[0, 0]
sums = np.array(sum_result['all_sums'])
ax.hist(sums, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(sum_result['stats']['평균 합계'], color='red', linestyle='--', linewidth=2, label=f"평균: {sum_result['stats']['평균 합계']:.0f}")
ax.axvline(sum_result['stats']['중앙값'], color='green', linestyle='--', linewidth=2, label=f"중앙값: {sum_result['stats']['중앙값']:.0f}")
ax.set_xlabel('당첨번호 합계')
ax.set_ylabel('빈도')
ax.set_title('당첨번호 합계 분포', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 1-2: 폭포 패턴 분포
ax = axes[0, 1]
waterfall_lens = waterfall_result['waterfall_df']['max_waterfall_length'].value_counts().sort_index()
ax.bar(waterfall_lens.index, waterfall_lens.values, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('폭포 길이 (연속 번호 개수)')
ax.set_ylabel('빈도')
ax.set_title('폭포 패턴 분포', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 1-3: 건너띔 기간 합계
ax = axes[1, 0]
skip_sums = np.array(skip_sum_result['skip_sums'])
ax.plot(skip_sums, marker='o', markersize=3, linewidth=1, color='purple', alpha=0.6)
ax.axhline(skip_sum_result['stats']['평균'], color='red', linestyle='--', linewidth=2, label=f"평균: {skip_sum_result['stats']['평균']:.0f}")
ax.set_xlabel('회차')
ax.set_ylabel('건너띔 기간 합계')
ax.set_title('회차별 건너띔 기간 합계 추이', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 1-4: 번호별 평균 건너띔 기간
ax = axes[1, 1]
nums_for_plot = []
avg_skips_for_plot = []

for num in range(1, 46):
    if num in individual_result['all_patterns']:
        analysis = individual_result['all_patterns'][num]
        if '평균 건너띔' in analysis:
            nums_for_plot.append(num)
            avg_skips_for_plot.append(analysis['평균 건너띔'])

# 색상 분류 (낮음/중간/높음)
colors = ['green' if x <= 10 else 'orange' if x <= 20 else 'red' for x in avg_skips_for_plot]

ax.bar(nums_for_plot, avg_skips_for_plot, color=colors, edgecolor='black', alpha=0.7)
ax.set_xlabel('번호')
ax.set_ylabel('평균 건너띔 기간 (회차)')
ax.set_title('번호별 평균 건너띔 기간 (초록=자주나옴, 빨강=잘안나옴)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(avg_skips_for_plot) * 1.1)

plt.tight_layout()
plt.show()

print("\n✓ 고급 분석 시각화 완료")


# 패턴 6: 각 번호의 건너띔 기간 패턴
print("\n📊 [패턴 6] 각 번호의 건너띔 기간 패턴\n")
individual_result = advanced.get_individual_skip_pattern()

print("건너띔 기간이 짧은 번호 (자주 나오는 번호) TOP 15:")
print("-" * 85)
print(f"{'번호':<6} {'출현횟':<8} {'평균건너':<10} {'최대건너':<10} {'최소건너':<10} {'표준편':<10}")
print("-" * 85)

frequent = list(individual_result['frequent_numbers'].items())[:15]
for num, analysis in frequent:
    print(f"{num:<6} {analysis['출현 횟수']:<8} "
          f"{analysis['평균 건너띔']:<10.1f} {analysis['최대 건너띔']:<10} "
          f"{analysis['최소 건너띔']:<10} {analysis['표준편차']:<10.2f}")

print("\n\n건너띔 기간이 긴 번호 (잘 안 나오는 번호) TOP 15:")
print("-" * 85)
print(f"{'번호':<6} {'출현횟':<8} {'평균건너':<10} {'최대건너':<10} {'최소건너':<10} {'표준편':<10}")
print("-" * 85)

rare = list(individual_result['rare_numbers'].items())[:15]
for num, analysis in rare:
    print(f"{num:<6} {analysis['출현 횟수']:<8} "
          f"{analysis['평균 건너띔']:<10.1f} {analysis['최대 건너띔']:<10} "
          f"{analysis['최소 건너띔']:<10} {analysis['표준편차']:<10.2f}")


# 패턴 5: 역폭포 패턴 - 간격이 감소하는 패턴
print("\n📊 [패턴 5] 역폭포 패턴 (간격이 감소하는 패턴)\n")
reverse_result = advanced.get_reverse_waterfall_pattern()

print("역폭포 패턴 통계:")
print("-" * 50)
for key, value in reverse_result['stats'].items():
    print(f"  {key:20s}: {value:8.2f}")

print("\n\n간격 감소 상위 10회 (역폭포가 강한 회차):")
print("-" * 80)
print("회차    당첨번호           간격           간격감소횟수")
print("-" * 80)
for idx, row in reverse_result['high_gap_decrease'].iterrows():
    nums_str = ",".join(map(str, row['numbers']))
    gaps_str = ",".join(map(str, row['gaps']))
    print(f"{row['draw']:<6} {nums_str:<25} {gaps_str:<15} {row['gap_decrease_count']}")

print("\n예시: [1,3,6,8,10,11] → 간격 [2,3,2,2,1] → 감소 횟수 3회")


# 패턴 4: 폭포 패턴 - 연속으로 증가하는 번호
print("\n📊 [패턴 4] 폭포 패턴 (연속으로 증가하는 번호)\n")
waterfall_result = advanced.get_waterfall_pattern()

print("폭포 패턴 통계:")
print("-" * 50)
for key, value in waterfall_result['stats'].items():
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            print(f"  {key:20s}: {value:8.2f}")
        else:
            print(f"  {key:20s}: {value:8d}")
    else:
        print(f"  {key:20s}: {value}")

print("\n\n3개 이상 폭포 사례 (상위 15개):")
print("-" * 80)
print("회차    당첨번호           폭포길이")
print("-" * 80)
for idx, row in waterfall_result['waterfall_3_plus'].iterrows():
    nums_str = ",".join(map(str, row['numbers']))
    print(f"{row['draw']:<6} {nums_str:<25} {row['max_waterfall_length']}")

print("\n예시: 폭포 길이 4 = [1,2,3,4] 같은 4개 연속 번호")


# 패턴 3: 각 회차별 당첨번호 건너띔 기간 합계
print("\n📊 [패턴 3] 각 회차별 건너띔 기간 합계의 패턴\n")
skip_sum_result = advanced.get_skip_period_sum()

print("건너띔 기간 합계 통계:")
print("-" * 50)
for key, value in skip_sum_result['stats'].items():
    print(f"  {key:15s}: {value:8.2f}")

print("\n\n건너띔 합계 상위 10회:")
print("-" * 50)
print(skip_sum_result['top_skip'].to_string(index=False))

print("\n\n건너띔 합계 하위 10회 (안정적인 조합):")
print("-" * 50)
print(skip_sum_result['bottom_skip'].to_string(index=False))


# 패턴 2: 낙첨 후 10회 기준으로 당첨 패턴
print("\n📊 [패턴 2] 낙첨 후 10회 기준 당첨 패턴 분석\n")
gap_result = advanced.get_gap_pattern()

print(f"10회 이내 재당첨된 번호: {len(gap_result['short_gap_summary'])}개")
print("-" * 70)

short_gap_nums = sorted(gap_result['short_gap_summary'].items(), 
                        key=lambda x: x[1]['count'], reverse=True)[:15]

print(f"{'번호':<6} {'재당첨회':<8} {'평균건너띔':<10} {'건너띔패턴':<30}")
print("-" * 70)

for num, info in short_gap_nums:
    gap_pattern = str(sorted(info['gaps']))[:25]
    print(f"{num:<6} {info['count']:<8} {info['avg']:<10.1f} {gap_pattern}")

print(f"\n\n10회 초과 건너띔 번호: {len(gap_result['long_gap_summary'])}개")
print("-" * 70)

long_gap_nums = sorted(gap_result['long_gap_summary'].items(), 
                       key=lambda x: x[1]['count'], reverse=True)[:15]

print(f"{'번호':<6} {'장기건너띔':<8} {'평균건너띔':<10} {'최대건너띔':<10}")
print("-" * 70)

for num, info in long_gap_nums:
    max_gap = max(info['gaps']) if info['gaps'] else 0
    print(f"{num:<6} {info['count']:<8} {info['avg']:<10.1f} {max_gap:<10}")


# 패턴 1: 당첨번호 합계의 패턴
print("\n📊 [패턴 1] 당첨번호 합계 상세 분석\n")
sum_result = advanced.get_sum_pattern()

print("당첨번호 합계 통계량:")
print("-" * 50)
for key, value in sum_result['stats'].items():
    if isinstance(value, float):
        print(f"  {key:15s}: {value:8.2f}")
    else:
        print(f"  {key:15s}: {value}")

print("\n상위 15개 합계별 빈도:")
print("-" * 50)
print(sum_result['distribution'].to_string(index=False))


from analysis.advanced_analysis import AdvancedLottoAnalysis

# 고급 분석 객체 생성
advanced = AdvancedLottoAnalysis(data)

# 종합 분석 보고서 생성
advanced.generate_comprehensive_report()


import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. 건너띔 합계 시계열
ax = axes[0, 0]
ax.plot(range(1, len(skip_analyzer.skip_sums) + 1), skip_analyzer.skip_sums, 
        marker='o', markersize=3, linewidth=1.5, label='건너띔 합계', color='steelblue', alpha=0.7)
ax.axhline(np.mean(skip_analyzer.skip_sums), color='red', linestyle='--', linewidth=2, 
           label=f'평균: {np.mean(skip_analyzer.skip_sums):.0f}')

# 트렌드선 추가
from scipy.stats import linregress
x = np.arange(len(skip_analyzer.skip_sums))
slope, intercept, _, _, _ = linregress(x, skip_analyzer.skip_sums)
trendline = slope * x + intercept
ax.plot(range(1, len(skip_analyzer.skip_sums) + 1), trendline, 
        linestyle='-', linewidth=2, label=f'추세선 (기울기: {slope:.4f})', color='green', alpha=0.8)

ax.set_xlabel('회차')
ax.set_ylabel('건너띔 기간 합계')
ax.set_title('건너띔 합계 시계열 추이', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. 이동평균
ax = axes[0, 1]
trend = skip_analyzer.get_trend_analysis(window=10)
moving_avg = trend['moving_average']
ax.plot(range(1, len(skip_analyzer.skip_sums) + 1), skip_analyzer.skip_sums, 
        alpha=0.3, label='원본', color='gray')
ax.plot(range(1, len(skip_analyzer.skip_sums) + 1), moving_avg, 
        linewidth=2, label='10회 이동평균', color='darkblue')
ax.set_xlabel('회차')
ax.set_ylabel('건너띔 합계')
ax.set_title('이동평균으로 보는 평활화된 추세', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 3. 범위 분포
ax = axes[1, 0]
classification = skip_analyzer.get_range_classification()
categories = list(classification.keys())
counts = [classification[c]['count'] for c in categories]
colors_dist = ['green', 'lightgreen', 'orange', 'red']
bars = ax.bar(range(len(categories)), counts, color=colors_dist, edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=15, ha='right')
ax.set_ylabel('회차 수')
ax.set_title('범위별 회차 분포', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 값 레이블 추가
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. 분포 히스토그램
ax = axes[1, 1]
ax.hist(skip_analyzer.skip_sums, bins=20, color='purple', edgecolor='black', alpha=0.7)
ax.axvline(np.mean(skip_analyzer.skip_sums), color='red', linestyle='--', linewidth=2, 
           label=f'평균: {np.mean(skip_analyzer.skip_sums):.0f}')
ax.axvline(np.median(skip_analyzer.skip_sums), color='green', linestyle='--', linewidth=2, 
           label=f'중앙값: {np.median(skip_analyzer.skip_sums):.0f}')
ax.set_xlabel('건너띔 기간 합계')
ax.set_ylabel('빈도')
ax.set_title('건너띔 합계 분포 (히스토그램)', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ 건너띔 합계 심화 분석 시각화 완료")


# 예측 인사이트
print("\n🔮 예측 인사이트\n")
predict = skip_analyzer.get_predictive_insights()

print("최근 평균값:")
print(f"  최근 5회: {predict['recent_5_avg']:.2f}")
print(f"  최근 10회: {predict['recent_10_avg']:.2f}")
print(f"  최근 20회: {predict['recent_20_avg']:.2f}")
print(f"  전체 평균: {predict['overall_avg']:.2f}")

print(f"\n현재 추세: {predict['trend']}")
print(f"신호: {predict['signal']}")

print(f"\n📍 예상 범위:")
print(f"  하한: {predict['lower_bound']:.0f}")
print(f"  상한: {predict['upper_bound']:.0f}")
print(f"  예상 범위: {predict['predicted_range']}")

print("\n💡 해석:")
if predict['recent_5_avg'] > predict['overall_avg']:
    print("  최근 건너띔이 증가 추세 → 좀 더 많은 번호를 피할 준비 필요")
else:
    print("  최근 건너띔이 감소 추세 → 좀 더 많은 번호를 도전할 수 있음")


# 극값 분석
print("\n📊 극값 (Extreme Values) 분석\n")
extreme = skip_analyzer.get_extreme_values(10)

print("🔥 상위 10개 (건너띔 합계 높음 - 어려운 회차):")
print("-" * 50)
print(extreme['top_n'].to_string(index=False))

print("\n\n❄️ 하위 10개 (건너띔 합계 낮음 - 쉬운 회차):")
print("-" * 50)
print(extreme['bottom_n'].to_string(index=False))


# 주기성 분석
print("\n📊 주기성 (Cyclicity) 분석\n")
cycle = skip_analyzer.get_cycle_analysis()

if cycle['has_cyclicity']:
    print("✓ 주기성 감지됨")
    print(f"  주요 주기: {cycle['main_cycle']}회차")
    print(f"  유의한 모든 주기: {cycle['significant_lags']}")
    print(f"\n💡 해석:")
    print(f"  약 {cycle['main_cycle']}회마다 패턴이 반복되는 경향이 있습니다")
else:
    print("✗ 주기성 감지 안 됨")
    print("  → 건너띔 합계가 무작위성이 강함")
    print("  → 특정 주기 패턴을 찾기 어려움")


# 변동성 분석
print("\n📊 변동성 (Volatility) 분석\n")
volatility = skip_analyzer.get_volatility_analysis()

print(f"표준편차: {volatility['standard_deviation']:.2f}")
print(f"변동성 계수 (CV): {volatility['coefficient_of_variation']:.2f}%")
print(f"  → 평균 대비 {volatility['coefficient_of_variation']:.1f}% 정도 변함")

print(f"\n구간별 변동성:")
print(f"  전반부: {volatility['first_half_std']:.2f}")
print(f"  후반부: {volatility['second_half_std']:.2f}")
print(f"  추세: {volatility['volatility_trend']}")

print(f"\n회차간 변동:")
print(f"  최대 변동: {volatility['max_change']:.0f}")
print(f"  최소 변동: {volatility['min_change']:.0f}")
print(f"  평균 변동: {volatility['avg_change']:.2f}")


# 연속 상승/하강 분석
print("\n📊 연속 상승/하강 구간 분석\n")
consecutive = skip_analyzer.get_consecutive_analysis()

print("📈 상승 패턴:")
print(f"  최대 상승 연속: {consecutive['max_up_streak']}회차")
print(f"  평균 상승 연속: {consecutive['avg_up_streak']:.1f}회차")

print("\n📉 하강 패턴:")
print(f"  최대 하강 연속: {consecutive['max_down_streak']}회차")
print(f"  평균 하강 연속: {consecutive['avg_down_streak']:.1f}회차")

print("\n💡 해석:")
print(f"  상승이 {consecutive['max_up_streak']}회 연속으로 나타나는 경우가 있음")
print(f"  → 건너띔이 계속 커지는 구간이 존재")


# 트렌드 분석
print("\n📈 시계열 트렌드 분석\n")
trend = skip_analyzer.get_trend_analysis()

print(f"전체 추세: {trend['trend_direction']}")
print(f"기울기: {trend['slope']:.6f}")
print(f"  → 회차당 평균 {trend['slope']:.2f} {'증가' if trend['slope'] > 0 else '감소'}")
print(f"\n상승 회차: {trend['ups']}회 ({trend['up_percentage']:.1f}%)")
print(f"하강 회차: {trend['downs']}회 ({trend['down_percentage']:.1f}%)")
print(f"\n추세 강도: {trend['trend_strength']}")
print(f"R² (설명력): {trend['r_squared']:.3f}")
print(f"  → {trend['r_squared']*100:.1f}%의 변동을 추세로 설명 가능")


# 범위 분류 상세 분석
print("\n📊 건너띔 합계 범위별 상세 분류\n")
classification = skip_analyzer.get_range_classification()

for category, info in classification.items():
    print(f"🔹 {category}")
    print(f"   범위: {info['range']}")
    print(f"   회차 수: {info['count']}회 ({info['percentage']:.1f}%)")
    print(f"   회차: {info['indices'][:5]}" + ("..." if len(info['indices']) > 5 else ""))
    print()


from analysis.skip_sum_detailed import SkipSumPatternAnalysis

# 건너띔 기간 합계 심화 분석
skip_analyzer = SkipSumPatternAnalysis(data)

# 상세 보고서 생성
skip_analyzer.generate_detailed_report()


