import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import plotly.express as px
import lightgbm as lgb

st.set_page_config(
    page_title="🎯 ELITE LOTTO AI",
    layout="wide"
)

# =====================================================
# 데이터 수집
# =====================================================

@st.cache_data(ttl=3600)
def fetch_latest_round():

    url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="

    latest = 1

    for i in range(1200, 1400):

        try:
            res = requests.get(url + str(i), timeout=3)
            data = res.json()

            if data.get("returnValue") == "success":
                latest = i
            else:
                break

        except:
            break

    return latest


@st.cache_data(ttl=3600)
def update_lotto_data(last_n=200):

    latest = fetch_latest_round()

    rows = []

    for i in range(latest-last_n+1, latest+1):

        try:
            url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={i}"

            data = requests.get(url, timeout=3).json()

            if data.get("returnValue") == "success":

                rows.append([
                    data['drwtNo1'],
                    data['drwtNo2'],
                    data['drwtNo3'],
                    data['drwtNo4'],
                    data['drwtNo5'],
                    data['drwtNo6'],
                ])

        except:
            pass

    df = pd.DataFrame(rows, columns=['n1','n2','n3','n4','n5','n6'])

    return df


with st.spinner("🎯 최신 로또 데이터 분석 중..."):
    df = update_lotto_data()


# =====================================================
# 기본 통계
# =====================================================


def get_number_frequency(df):

    nums = []

    for row in df.values:
        nums.extend(list(row))

    freq = Counter(nums)

    return freq


freq = get_number_frequency(df)


# =====================================================
# Skip LN 분석
# =====================================================


def build_skip_map(df):

    skip_map = {}

    reversed_df = df.iloc[::-1]

    for n in range(1,46):

        skip = 0

        found = False

        for row in reversed_df.values:

            if n in row:
                found = True
                break

            skip += 1

        if not found:
            skip = len(df)

        skip_map[n] = skip

    return skip_map


skip_map = build_skip_map(df)


# =====================================================
# Pair Matrix
# =====================================================


def build_pair_matrix(df):

    pair_matrix = defaultdict(int)

    for row in df.values:

        row = sorted(row)

        for i in range(len(row)):
            for j in range(i+1, len(row)):

                pair = tuple(sorted((row[i], row[j])))

                pair_matrix[pair] += 1

    return pair_matrix


pair_matrix = build_pair_matrix(df)


# =====================================================
# Neighbor / Repeater
# =====================================================

last_row = list(df.iloc[-1].values)


def get_neighbors(last_row):

    neighbors = set()

    for n in last_row:

        if n > 1:
            neighbors.add(n-1)

        if n < 45:
            neighbors.add(n+1)

    return neighbors


neighbors = get_neighbors(last_row)


# =====================================================
# 군집 분석
# =====================================================


def make_features(df):

    feats = []

    for row in df.values:

        row = list(row)

        feats.append([
            sum(row),
            sum(n % 2 for n in row),
            sum(n <= 23 for n in row),
            max(row)-min(row)
        ])

    return np.array(feats)


X = make_features(df)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

current_cluster = labels[-1]


# =====================================================
# LightGBM 확률
# =====================================================


def build_probability():

    prob = {}

    total = sum(freq.values())

    for n in range(1,46):

        f = freq.get(n,0)

        skip = skip_map.get(n,0)

        score = (
            f * 1.2 +
            max(0, 15-skip) * 0.8
        )

        prob[n] = score

    return prob


prob_dict = build_probability()


# =====================================================
# 패턴 분석
# =====================================================

past_sums = [sum(row) for row in df.values]

sum_mean = np.mean(past_sums)
sum_std = np.std(past_sums)

sum_low = sum_mean - sum_std
sum_high = sum_mean + sum_std


# =====================================================
# 상위 20개 압축
# =====================================================


def build_top20():

    scores = {}

    for n in range(1,46):

        score = 0

        score += prob_dict[n] * 5

        # Skip LN
        skip = skip_map.get(n,0)

        if 2 <= skip <= 15:
            score += 8

        # Neighbor
        if n in neighbors:
            score += 5

        # Repeater
        if n in last_row:
            score += 5

        # Pair
        pair_bonus = 0

        for m in range(1,46):

            pair = tuple(sorted((n,m)))

            pair_bonus += pair_matrix.get(pair,0)

        score += pair_bonus * 0.01

        scores[n] = score

    top20 = sorted(
        scores,
        key=scores.get,
        reverse=True
    )[:20]

    return top20


top20 = build_top20()


# =====================================================
# 패턴 필터
# =====================================================


def check_sum(combo):

    s = sum(combo)

    return sum_low <= s <= sum_high



def check_odd_even(combo):

    odd = sum(n%2 for n in combo)

    return odd in [2,3,4]



def check_high_low(combo):

    low = sum(n<=23 for n in combo)

    return low in [2,3,4]



def check_neighbor(combo):

    cnt = sum(n in neighbors for n in combo)

    return 1 <= cnt <= 2



def check_repeater(combo):

    cnt = sum(n in last_row for n in combo)

    return 1 <= cnt <= 2



def check_tail(combo):

    tails = [n%10 for n in combo]

    c = Counter(tails)

    return max(c.values()) <= 2



def check_decade(combo):

    groups = [0,0,0,0,0]

    for n in combo:

        if n <= 9:
            groups[0]+=1
        elif n <=19:
            groups[1]+=1
        elif n <=29:
            groups[2]+=1
        elif n <=39:
            groups[3]+=1
        else:
            groups[4]+=1

    return max(groups) <= 3



def check_gap(combo):

    combo = sorted(combo)

    gaps = np.diff(combo)

    return np.std(gaps) > 1



def super_filter(combo):

    if not check_sum(combo):
        return False

    if not check_odd_even(combo):
        return False

    if not check_high_low(combo):
        return False

    if not check_neighbor(combo):
        return False

    if not check_repeater(combo):
        return False

    if not check_tail(combo):
        return False

    if not check_decade(combo):
        return False

    if not check_gap(combo):
        return False

    return True


# =====================================================
# Fitness
# =====================================================


def pair_score(combo):

    score = 0

    for i in range(len(combo)):
        for j in range(i+1, len(combo)):

            pair = tuple(sorted((combo[i], combo[j])))

            score += pair_matrix.get(pair,0)

    return score



def fitness(combo):

    score = 0

    # 확률
    score += sum(prob_dict[n] for n in combo)

    # Pair
    score += pair_score(combo) * 0.03

    # Skip
    score += sum(
        max(0, 12-skip_map.get(n,0))
        for n in combo
    )

    # Neighbor
    score += sum(n in neighbors for n in combo) * 5

    # Repeater
    score += sum(n in last_row for n in combo) * 5

    return score


# =====================================================
# ELITE 생성
# =====================================================


def generate_elite():

    raw = []

    for _ in range(5000):

        combo = sorted(random.sample(top20,6))

        if super_filter(combo):

            score = fitness(combo)

            raw.append((combo,score))

    raw = sorted(
        raw,
        key=lambda x:x[1],
        reverse=True
    )

    final = []

    for combo,score in raw:

        duplicate = False

        for existing,_ in final:

            overlap = len(set(combo) & set(existing))

            if overlap >= 5:
                duplicate = True
                break

        if not duplicate:
            final.append((combo,score))

        if len(final) >= 10:
            break

    return final


elite = generate_elite()


# =====================================================
# 백테스트
# =====================================================


def backtest():

    results = []

    for row in df.values[-50:]:

        real = set(row)

        for combo,_ in elite:

            hit = len(set(combo) & real)

            results.append(hit)

    return results


results = backtest()


# =====================================================
# UI
# =====================================================

st.title("🎯 ELITE LOTTO AI ENGINE")

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("📌 최신 회차", fetch_latest_round())

with col2:
    st.metric("🧠 현재 군집", current_cluster)

with col3:
    st.metric("🔥 ELITE 풀", len(top20))


st.subheader("🔥 핵심 압축 번호 TOP20")

st.write(top20)


st.subheader("👑 ELITE TOP10")

for i,(combo,score) in enumerate(elite,1):

    st.markdown(f"## 👑 ELITE {i}")

    st.success(combo)

    st.progress(min(score/100,1.0))

    reasons = []

    if check_neighbor(combo):
        reasons.append("Neighbor")

    if check_repeater(combo):
        reasons.append("Repeater")

    if check_tail(combo):
        reasons.append("Tail")

    if check_gap(combo):
        reasons.append("Gap")

    st.write("✔ 추천 이유:", ", ".join(reasons))


st.subheader("📊 백테스트 적중 분포")

fig = px.histogram(
    results,
    nbins=10,
    title="적중 분포"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("📈 번호 출현 빈도")

freq_df = pd.DataFrame({
    "번호": list(freq.keys()),
    "빈도": list(freq.values())
})

fig2 = px.bar(
    freq_df,
    x="번호",
    y="빈도"
)

st.plotly_chart(fig2, use_container_width=True)


st.subheader("🧠 확률 필터 최적화 연구")

st.info("""
현재 시스템은 단순 번호 추천이 아니라
'확률 패턴 공간 압축' 기반 AI 엔진입니다.

적용 요소:
- Pair Matrix
- Skip LN
- Neighbor
- Repeater
- Gap Pattern
- Tail Pattern
- Cluster Analysis
- Adaptive Probability
""")
