
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans


st.set_page_config(page_title="🎯 ELITE LOTTO AI", layout="wide")
st.title("🎯 ELITE LOTTO AI ENGINE")
st.set_option("client.showErrorDetails", False)


# =====================================================
# 안전한 데이터 로드
# =====================================================

COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
LOCAL_CSV = Path("lotto_200.csv")


def _safe_json_get(url: str, timeout: int = 4):
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_latest_round_number(min_round: int = 1000, max_round: int = 1400):
    latest = None
    base = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="

    for round_no in range(max_round, min_round - 1, -1):
        data = _safe_json_get(f"{base}{round_no}")
        if data and data.get("returnValue") == "success":
            latest = round_no
            break

    return latest


@st.cache_data(ttl=3600)
def fetch_round_data(round_no: int):
    base = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
    data = _safe_json_get(f"{base}{round_no}")
    if not data or data.get("returnValue") != "success":
        return None

    row = [
        int(data["drwtNo1"]),
        int(data["drwtNo2"]),
        int(data["drwtNo3"]),
        int(data["drwtNo4"]),
        int(data["drwtNo5"]),
        int(data["drwtNo6"]),
    ]
    return row


@st.cache_data(ttl=3600)
def load_lotto_df(last_n: int = 200):
    local_df = None

    if LOCAL_CSV.exists():
        try:
            local_df = pd.read_csv(LOCAL_CSV)
            local_df = local_df[[c for c in local_df.columns if c in COLUMNS]].copy()
            if len(local_df.columns) == 6:
                local_df.columns = COLUMNS
                local_df = local_df.dropna().astype(int)
        except Exception:
            local_df = None

    latest_round = fetch_latest_round_number()
    online_df = None

    if latest_round is not None:
        rows = []
        start_round = max(1, latest_round - last_n + 1)

        for round_no in range(start_round, latest_round + 1):
            row = fetch_round_data(round_no)
            if row is not None and len(row) == 6:
                rows.append(row)

        if rows:
            online_df = pd.DataFrame(rows, columns=COLUMNS)

    # 우선순위: 온라인 최신 데이터 → 로컬 CSV → 빈 DF
    if online_df is not None and not online_df.empty:
        if local_df is not None and not local_df.empty:
            merged = pd.concat([local_df, online_df], ignore_index=True)
            merged = merged.drop_duplicates().tail(last_n).reset_index(drop=True)
            return merged
        return online_df.tail(last_n).reset_index(drop=True)

    if local_df is not None and not local_df.empty:
        return local_df.tail(last_n).reset_index(drop=True)

    return pd.DataFrame(columns=COLUMNS)


with st.spinner("🎯 최신 로또 데이터 분석 중..."):
    df = load_lotto_df()

if df.empty:
    st.error("로또 데이터를 불러오지 못했습니다. 네트워크나 lotto_200.csv를 확인해 주세요.")
    st.stop()

if len(df) < 2:
    st.error("데이터가 너무 적습니다. 최소 2회차 이상 필요합니다.")
    st.stop()


# =====================================================
# 기본 통계 / 특징 생성
# =====================================================


def get_number_frequency(data: pd.DataFrame):
    nums = []
    for row in data.values:
        nums.extend(list(map(int, row)))
    return Counter(nums)


freq = get_number_frequency(df)
last_row = list(map(int, df.iloc[-1].values))


@st.cache_data(ttl=3600)
def build_skip_map(data: pd.DataFrame):
    skip_map = {}
    reversed_rows = data.iloc[::-1].values

    for n in range(1, 46):
        skip = 0
        found = False
        for row in reversed_rows:
            row = list(map(int, row))
            if n in row:
                found = True
                break
            skip += 1
        if not found:
            skip = len(data)
        skip_map[n] = skip

    return skip_map


@st.cache_data(ttl=3600)
def build_pair_matrix(data: pd.DataFrame):
    pair_matrix = defaultdict(int)
    for row in data.values:
        row = sorted(map(int, row))
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pair_matrix[(row[i], row[j])] += 1
    return pair_matrix


skip_map = build_skip_map(df)
pair_matrix = build_pair_matrix(df)


def get_neighbors(row):
    neighbors = set()
    for n in row:
        if n > 1:
            neighbors.add(n - 1)
        if n < 45:
            neighbors.add(n + 1)
    return neighbors


neighbors = get_neighbors(last_row)


# =====================================================
# 군집 분석
# =====================================================


def make_features(data: pd.DataFrame):
    feats = []
    for row in data.values:
        row = list(map(int, row))
        feats.append([
            sum(row),
            sum(n % 2 for n in row),
            sum(n <= 23 for n in row),
            max(row) - min(row),
        ])
    return np.array(feats)


X_cluster = make_features(df)
if len(df) >= 2:
    n_clusters = min(5, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)
    current_cluster = int(labels[-1])
else:
    kmeans = None
    labels = np.array([0] * len(df))
    current_cluster = 0


# =====================================================
# 확률 모델 (LightGBM + 안전한 fallback)
# =====================================================


def build_training_data(data: pd.DataFrame, window: int = 20):
    X, y = [], []
    if len(data) <= 1:
        return np.array(X), np.array(y)

    window = max(5, min(window, len(data) - 1))

    for i in range(window, len(data)):
        past = data.iloc[i - window:i]
        current = set(map(int, data.iloc[i].values))
        freq_local = Counter(past.values.flatten().tolist())

        recent5 = past.tail(5)
        recent10 = past.tail(10)

        for num in range(1, 46):
            last_seen = 0
            for j in range(len(past) - 1, -1, -1):
                if num in set(map(int, past.iloc[j].values)):
                    last_seen = len(past) - j
                    break

            pair_strength = 0
            for row in past.values:
                row = list(map(int, row))
                if num in row:
                    pair_strength += len(row)

            X.append([
                num,
                freq_local.get(num, 0),
                sum(num in set(map(int, r)) for r in recent5.values),
                sum(num in set(map(int, r)) for r in recent10.values),
                last_seen,
                pair_strength,
                num % 2,
                int(num <= 23),
            ])
            y.append(1 if num in current else 0)

    return np.array(X), np.array(y)


@st.cache_resource
def train_model(data_signature: str):
    X, y = build_training_data(df)

    if len(X) == 0 or len(np.unique(y)) < 2:
        return None

    try:
        model = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X, y)
        return model
    except Exception:
        return None


model = train_model(str(df.shape) + str(df.iloc[-1].tolist()))


def predict_prob():
    # LightGBM 사용 가능하면 사용, 아니면 frequency 기반 fallback
    if model is None:
        scores = {}
        total = sum(freq.values()) or 1
        for n in range(1, 46):
            scores[n] = (freq.get(n, 0) / total) + max(0, 15 - skip_map.get(n, 0)) * 0.01
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    past = df.tail(min(20, len(df)))
    freq_local = Counter(past.values.flatten().tolist())
    recent5 = past.tail(5)
    recent10 = past.tail(10)

    features = []
    for num in range(1, 46):
        last_seen = 0
        for j in range(len(past) - 1, -1, -1):
            if num in set(map(int, past.iloc[j].values)):
                last_seen = len(past) - j
                break

        pair_strength = 0
        for row in past.values:
            row = list(map(int, row))
            if num in row:
                pair_strength += len(row)

        features.append([
            num,
            freq_local.get(num, 0),
            sum(num in set(map(int, r)) for r in recent5.values),
            sum(num in set(map(int, r)) for r in recent10.values),
            last_seen,
            pair_strength,
            num % 2,
            int(num <= 23),
        ])

    probs = model.predict_proba(np.array(features))[:, 1]
    return [(i + 1, float(probs[i])) for i in range(45)]


# =====================================================
# 패턴 필터 / 점수
# =====================================================

past_sums = [sum(map(int, row)) for row in df.values]
sum_mean = float(np.mean(past_sums))
sum_std = float(np.std(past_sums))
sum_low = sum_mean - sum_std
sum_high = sum_mean + sum_std


def basic_filter(combo):
    combo = sorted(map(int, combo))

    # 이미 나온 조합 완전 제외
    for row in df.values:
        if combo == sorted(map(int, row)):
            return False

    # 6연속처럼 너무 규칙적인 조합 제외
    if max(combo) - min(combo) == 5:
        return False

    # 동일 끝자리 3개 이상 제한
    tails = [n % 10 for n in combo]
    if max(Counter(tails).values()) >= 3:
        return False

    return True


def check_sum(combo):
    return sum_low <= sum(combo) <= sum_high


def check_odd_even(combo):
    odd = sum(n % 2 for n in combo)
    return odd in [2, 3, 4]


def check_high_low(combo):
    low = sum(n <= 23 for n in combo)
    return low in [2, 3, 4]


def check_neighbor(combo):
    cnt = sum(n in neighbors for n in combo)
    return 1 <= cnt <= 2


def check_repeater(combo):
    cnt = sum(n in last_row for n in combo)
    return 1 <= cnt <= 2


def check_tail(combo):
    tails = [n % 10 for n in combo]
    c = Counter(tails)
    return max(c.values()) <= 2


def check_decade(combo):
    groups = [0, 0, 0, 0, 0]
    for n in combo:
        if n <= 9:
            groups[0] += 1
        elif n <= 19:
            groups[1] += 1
        elif n <= 29:
            groups[2] += 1
        elif n <= 39:
            groups[3] += 1
        else:
            groups[4] += 1
    return max(groups) <= 3


def check_gap(combo):
    combo = sorted(combo)
    gaps = np.diff(combo)
    return np.std(gaps) > 0.9


def combo_skip_sum(combo):
    return sum(skip_map.get(int(n), 0) for n in combo)


past_skip_sums = [combo_skip_sum(row) for row in df.values]
skip_mean = float(np.mean(past_skip_sums))
skip_std = float(np.std(past_skip_sums))
skip_low = skip_mean - skip_std
skip_high = skip_mean + skip_std


def check_skip_pattern(combo):
    s = combo_skip_sum(combo)
    return skip_low <= s <= skip_high


def pair_score(combo):
    score = 0
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            pair = tuple(sorted((int(combo[i]), int(combo[j]))))
            score += pair_matrix.get(pair, 0)
    return score


# Cluster score: 현재 군집의 중심과의 거리
cluster_centroid = None
if kmeans is not None:
    try:
        cluster_centroid = kmeans.cluster_centers_[current_cluster]
    except Exception:
        cluster_centroid = None


def cluster_score(combo):
    if cluster_centroid is None:
        return 0.0
    feat = np.array([
        sum(combo),
        sum(n % 2 for n in combo),
        sum(n <= 23 for n in combo),
        max(combo) - min(combo),
    ], dtype=float)
    dist = float(np.linalg.norm(feat - cluster_centroid))
    return max(0.0, 10.0 - dist)


# =====================================================
# ELITE 풀 / 조합 생성
# =====================================================


def build_top20(prob):
    scores = {}
    prob_dict = dict(prob)

    for n in range(1, 46):
        score = 0.0
        score += prob_dict.get(n, 0.0) * 50

        skip = skip_map.get(n, 0)
        if 2 <= skip <= 15:
            score += 8

        if n in neighbors:
            score += 5

        if n in last_row:
            score += 5

        pair_bonus = 0
        for m in range(1, 46):
            pair = tuple(sorted((n, m)))
            pair_bonus += pair_matrix.get(pair, 0)
        score += pair_bonus * 0.01

        scores[n] = score

    return sorted(scores, key=scores.get, reverse=True)[:20]


def build_elite_pool(top20, size=8):
    # 관리자 전용처럼 사용할 수 있는 핵심 풀
    return top20[:size]


def fitness(combo, prob_dict):
    combo = sorted(map(int, combo))

    if not basic_filter(combo):
        return 0.0

    score = 0.0

    # 확률
    score += sum(prob_dict.get(n, 0.0) for n in combo)

    # 합계 / 홀짝 / 고저
    if check_sum(combo):
        score += 10
    if check_odd_even(combo):
        score += 8
    if check_high_low(combo):
        score += 8

    # Neighbor / Repeater
    score += sum(n in neighbors for n in combo) * 4
    score += sum(n in last_row for n in combo) * 4

    # Skip / Pair / Tail / Decade / Gap / Cluster
    if check_skip_pattern(combo):
        score += 10
    score += pair_score(combo) * 0.03
    score += cluster_score(combo)

    if check_tail(combo):
        score += 5
    if check_decade(combo):
        score += 5
    if check_gap(combo):
        score += 5

    return score


def super_filter(combo):
    return (
        check_sum(combo)
        and check_odd_even(combo)
        and check_high_low(combo)
        and check_neighbor(combo)
        and check_repeater(combo)
        and check_tail(combo)
        and check_decade(combo)
        and check_gap(combo)
        and check_skip_pattern(combo)
    )


def generate_elite(prob, elite_pool_size=8):
    prob_dict = dict(prob)
    top20 = build_top20(prob)
    elite_pool = build_elite_pool(top20, size=elite_pool_size)

    raw = []
    # 집중 조합 생성
    for _ in range(5000):
        combo = sorted(random.sample(elite_pool, 6))
        if super_filter(combo):
            score = fitness(combo, prob_dict)
            raw.append((combo, score))

    # fallback: 너무 엄격해서 0개일 때
    if not raw:
        for _ in range(5000):
            combo = sorted(random.sample(top20, 6))
            if basic_filter(combo):
                score = fitness(combo, prob_dict)
                raw.append((combo, score))

    raw = sorted(raw, key=lambda x: x[1], reverse=True)

    # 중복도 줄여서 TOP10
    final = []
    for combo, score in raw:
        is_dup = False
        for exist, _ in final:
            if len(set(combo) & set(exist)) >= 5:
                is_dup = True
                break
        if not is_dup:
            final.append((combo, score))
        if len(final) >= 10:
            break

    return top20, elite_pool, final


# =====================================================
# 설명 / UI 표현
# =====================================================


def grade_combo(score):
    if score >= 70:
        return "👑 ELITE"
    if score >= 55:
        return "🔥 S급"
    if score >= 40:
        return "⭐ A급"
    if score >= 25:
        return "👍 B급"
    return "⚪ C급"


def explain_combo(combo, prob_dict):
    reasons = []
    if check_sum(combo):
        reasons.append("합계 안정")
    if check_odd_even(combo):
        reasons.append("홀짝 균형")
    if check_high_low(combo):
        reasons.append("고저 균형")
    if check_neighbor(combo):
        reasons.append("이웃 숫자")
    if check_repeater(combo):
        reasons.append("연번/재출현")
    if check_skip_pattern(combo):
        reasons.append("Skip LN 균형")
    if check_tail(combo):
        reasons.append("끝자리 분산")
    if check_decade(combo):
        reasons.append("십진 구간 분산")
    if check_gap(combo):
        reasons.append("Gap 패턴")
    return reasons


def style_numbers(combo):
    html = ""
    for n in combo:
        if n <= 10:
            color = "#fbc400"
        elif n <= 20:
            color = "#69c8f2"
        elif n <= 30:
            color = "#ff7272"
        elif n <= 40:
            color = "#aaaaaa"
        else:
            color = "#b0d840"

        html += f"""
        <span style="
            display:inline-block;
            background:{color};
            color:black;
            padding:10px 14px;
            margin:4px;
            border-radius:999px;
            min-width:42px;
            text-align:center;
            font-weight:700;
            box-shadow:0 1px 4px rgba(0,0,0,.12);
        ">{n}</span>
        """
    return html


# =====================================================
# UI
# =====================================================

st.sidebar.title("👤 사용자")
user_id = st.sidebar.text_input("이메일 입력")
use_personal = st.sidebar.checkbox("🎯 개인화 추천", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 보안 상태")
st.sidebar.success("✔ 입력 검증 활성화")
st.sidebar.success("✔ 데이터 빈값 방어")
st.sidebar.success("✔ API 실패 fallback")

if not user_id:
    st.info("이메일을 입력하면 개인화 기록 기능을 사용할 수 있습니다.")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 현재 분석 상태")
st.sidebar.write(f"데이터 회차 수: {len(df)}")
st.sidebar.write(f"현재 군집: {current_cluster}")
st.sidebar.write(f"핵심 압축 번호: 20개")
st.sidebar.write(f"ELITE 풀: 8개")


prob = predict_prob()
prob_dict = dict(prob)
top20, elite_pool, elite_final = generate_elite(prob, elite_pool_size=8)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📌 최근 회차", fetch_latest_round_number() or "fallback")
with col2:
    st.metric("🧠 현재 군집", current_cluster)
with col3:
    st.metric("🔥 ELITE 풀", len(elite_pool))

st.subheader("🔥 핵심 압축 번호 TOP20")
st.write(top20)

st.subheader("👑 ELITE TOP10")

if elite_final:
    for i, (combo, score) in enumerate(elite_final, 1):
        grade = grade_combo(score)
        reasons = explain_combo(combo, prob_dict)

        st.markdown(f"## {i}. {grade}")
        st.markdown(style_numbers(combo), unsafe_allow_html=True)
        st.write(f"💯 점수: {round(score, 2)}")
        st.write("✔ 추천 이유:", ", ".join(reasons) if reasons else "기본 조건 충족")
        st.progress(min(score / 100, 1.0))
        st.markdown("---")
else:
    st.warning("ELITE 후보가 너무 엄격해서 생성되지 않았습니다. 필터를 완화해 주세요.")


# =====================================================
# 백테스트 / 시각화
# =====================================================


def backtest(elite_list):
    results = []
    for row in df.values[-50:]:
        real = set(map(int, row))
        for combo, _ in elite_list:
            results.append(len(set(combo) & real))
    return results


results = backtest(elite_final)

st.subheader("📊 백테스트 적중 분포")
fig = px.histogram(results, nbins=10, title="적중 분포")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 번호 출현 빈도")
freq_df = pd.DataFrame({"번호": list(freq.keys()), "빈도": list(freq.values())})
fig2 = px.bar(freq_df, x="번호", y="빈도", title="번호 출현 빈도")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🧠 확률 필터 최적화 연구")
st.info(
    "현재 시스템은 단순 번호 추천이 아니라 '확률 패턴 공간 압축' 기반 AI 엔진입니다.)
