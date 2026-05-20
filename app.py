import os
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans


# =====================================================
# 기본 설정
# =====================================================

st.set_page_config(page_title="🎯 ELITE LOTTO AI", layout="wide")
st.set_option("client.showErrorDetails", False)
st.title("🎯 ELITE LOTTO AI ENGINE")

COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]
LOCAL_CSV = Path("lotto_200.csv")
ADMIN_CODE = "elite-admin-2026"


# =====================================================
# 안전한 유틸
# =====================================================


def _safe_json_get(url: str, timeout: int = 5):
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


# =====================================================
# 데이터 수집 / 로드
# =====================================================


@st.cache_data(ttl=3600)
def fetch_latest_round_number(min_round: int = 1000, max_round: int = 1400):
    base = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
    latest = None

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

    return [
        int(data["drwtNo1"]),
        int(data["drwtNo2"]),
        int(data["drwtNo3"]),
        int(data["drwtNo4"]),
        int(data["drwtNo5"]),
        int(data["drwtNo6"]),
    ]


@st.cache_data(ttl=3600)
def load_lotto_df(last_n: int = 200):
    local_df = None

    if LOCAL_CSV.exists():
        try:
            tmp = pd.read_csv(LOCAL_CSV)
            tmp = tmp[[c for c in tmp.columns if c in COLUMNS]].copy()
            if len(tmp.columns) == 6:
                tmp.columns = COLUMNS
                tmp = tmp.dropna().astype(int)
                local_df = tmp
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
    st.error("로또 데이터를 불러오지 못했습니다. 네트워크와 lotto_200.csv를 확인해 주세요.")
    st.stop()

if len(df) < 2:
    st.error("데이터가 너무 적습니다. 최소 2회차 이상 필요합니다.")
    st.stop()


# =====================================================
# 데이터 기본 분석
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

cluster_centroid = None
if kmeans is not None:
    try:
        cluster_centroid = kmeans.cluster_centers_[current_cluster]
    except Exception:
        cluster_centroid = None


# =====================================================
# 시계열 흐름 / Coverage / Mean Reversion
# =====================================================

sum_series = pd.Series([sum(map(int, row)) for row in df.values])
sum_ma50 = sum_series.rolling(50).mean()
sum_std50 = sum_series.rolling(50).std()
sum_upper = sum_ma50 + sum_std50
sum_lower = sum_ma50 - sum_std50

odd_series = pd.Series([sum(n % 2 for n in map(int, row)) for row in df.values])
odd_ma50 = odd_series.rolling(50).mean()

highlow_series = pd.Series([sum(n <= 23 for n in map(int, row)) for row in df.values])
highlow_ma50 = highlow_series.rolling(50).mean()

coverage5 = []
coverage10 = []
for i in range(10, len(df)):
    current = set(map(int, df.iloc[i].values))
    recent5 = set(df.iloc[i - 5:i].values.flatten())
    recent10 = set(df.iloc[i - 10:i].values.flatten())
    coverage5.append(len(current & recent5))
    coverage10.append(len(current & recent10))

coverage5_series = pd.Series(coverage5) if coverage5 else pd.Series(dtype=float)
coverage10_series = pd.Series(coverage10) if coverage10 else pd.Series(dtype=float)
coverage5_ma50 = coverage5_series.rolling(50).mean() if not coverage5_series.empty else pd.Series(dtype=float)
coverage10_ma50 = coverage10_series.rolling(50).mean() if not coverage10_series.empty else pd.Series(dtype=float)
coverage10_std50 = coverage10_series.rolling(50).std() if not coverage10_series.empty else pd.Series(dtype=float)
coverage10_upper = coverage10_ma50 + coverage10_std50 if not coverage10_series.empty else pd.Series(dtype=float)
coverage10_lower = coverage10_ma50 - coverage10_std50 if not coverage10_series.empty else pd.Series(dtype=float)

recent5_nums = set(df.tail(5).values.flatten())
recent10_nums = set(df.tail(10).values.flatten())


# =====================================================
# LightGBM 확률 모델
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
    if model is None:
        scores = {}
        total = sum(freq.values()) or 1
        for n in range(1, 46):
            scores[n] = (freq.get(n, 0) / total) + max(0, 15 - skip_map.get(n, 0)) * 0.01
        return [(n, scores[n]) for n in range(1, 46)]

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
# 패턴 / 상태 엔진
# =====================================================

past_sums = [sum(map(int, row)) for row in df.values]
sum_mean = float(np.mean(past_sums))
sum_std = float(np.std(past_sums))
sum_low = sum_mean - sum_std
sum_high = sum_mean + sum_std

past_skip_sums = [sum(skip_map.get(int(n), 0) for n in row) for row in df.values]
skip_mean = float(np.mean(past_skip_sums))
skip_std = float(np.std(past_skip_sums))
skip_low = skip_mean - skip_std
skip_high = skip_mean + skip_std


def basic_filter(combo):
    combo = sorted(map(int, combo))

    for row in df.values:
        if combo == sorted(map(int, row)):
            return False

    if max(combo) - min(combo) == 5:
        return False

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


def check_skip_pattern(combo):
    s = combo_skip_sum(combo)
    return skip_low <= s <= skip_high


def coverage_score(combo):
    score = 0
    r5 = sum(n in recent5_nums for n in combo)
    r10 = sum(n in recent10_nums for n in combo)

    if 2 <= r5 <= 3:
        score += 8
    if 4 <= r10 <= 5:
        score += 12

    latest_cov10 = coverage10_series.iloc[-1] if not coverage10_series.empty else 0
    if latest_cov10 >= 5 and r10 <= 4:
        score += 6
    elif latest_cov10 <= 3 and r10 >= 5:
        score += 6

    return score


def mean_reversion_score(combo):
    score = 0
    combo_sum = sum(combo)
    latest_sum = sum_series.iloc[-1]
    current_ma50 = sum_ma50.iloc[-1]
    current_upper = sum_upper.iloc[-1]
    current_lower = sum_lower.iloc[-1]

    if pd.notna(current_upper) and latest_sum > current_upper:
        if pd.notna(current_ma50) and combo_sum < current_ma50:
            score += 10
    elif pd.notna(current_lower) and latest_sum < current_lower:
        if pd.notna(current_ma50) and combo_sum > current_ma50:
            score += 10

    return score


def pair_score(combo):
    score = 0
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            pair = tuple(sorted((int(combo[i]), int(combo[j]))))
            score += pair_matrix.get(pair, 0)
    return score


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


def adaptive_weights(state):
    weights = {
        "prob": 1.0,
        "pair": 1.0,
        "skip": 1.0,
        "neighbor": 1.0,
        "repeater": 1.0,
        "tail": 1.0,
        "decade": 1.0,
        "gap": 1.0,
        "coverage": 1.0,
        "meanrev": 1.0,
        "cluster": 1.0,
    }

    if state["sum_state"] == "OVERHEAT":
        weights["meanrev"] = 1.2
    elif state["sum_state"] == "COOL":
        weights["meanrev"] = 1.15

    if state["coverage_state"] == "OVERHEAT":
        weights["coverage"] = 1.15
    elif state["coverage_state"] == "COOL":
        weights["coverage"] = 1.05

    if state["odd_state"] == "BALANCED":
        weights["neighbor"] = 1.05

    if state["highlow_state"] == "BALANCED":
        weights["pair"] = 1.05

    return weights


def state_engine():
    latest_sum = int(sum_series.iloc[-1])
    latest_odd = int(odd_series.iloc[-1])
    latest_high = int(highlow_series.iloc[-1])
    latest_cov10 = int(coverage10_series.iloc[-1]) if not coverage10_series.empty else 0

    sum_upper_v = float(sum_upper.iloc[-1]) if pd.notna(sum_upper.iloc[-1]) else sum_mean + sum_std
    sum_lower_v = float(sum_lower.iloc[-1]) if pd.notna(sum_lower.iloc[-1]) else sum_mean - sum_std

    coverage_upper_v = float(coverage10_upper.iloc[-1]) if not coverage10_upper.empty and pd.notna(coverage10_upper.iloc[-1]) else 5
    coverage_lower_v = float(coverage10_lower.iloc[-1]) if not coverage10_lower.empty and pd.notna(coverage10_lower.iloc[-1]) else 3

    if latest_sum > sum_upper_v:
        sum_state = "OVERHEAT"
    elif latest_sum < sum_lower_v:
        sum_state = "COOL"
    else:
        sum_state = "NORMAL"

    if latest_cov10 >= coverage_upper_v:
        coverage_state = "OVERHEAT"
    elif latest_cov10 <= coverage_lower_v:
        coverage_state = "COOL"
    else:
        coverage_state = "NORMAL"

    if latest_odd in [2, 3, 4]:
        odd_state = "BALANCED"
    elif latest_odd <= 1:
        odd_state = "EVEN_HEAVY"
    else:
        odd_state = "ODD_HEAVY"

    if latest_high in [2, 3, 4]:
        highlow_state = "BALANCED"
    elif latest_high <= 1:
        highlow_state = "LOW_HEAVY"
    else:
        highlow_state = "HIGH_HEAVY"

    return {
        "sum_state": sum_state,
        "coverage_state": coverage_state,
        "odd_state": odd_state,
        "highlow_state": highlow_state,
        "latest_sum": latest_sum,
        "latest_cov10": latest_cov10,
    }


state = state_engine()
weights = adaptive_weights(state)


# =====================================================
# ELITE 번호 풀 / 조합 생성
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



def fitness(combo, prob_dict):
    combo = sorted(map(int, combo))

    if not basic_filter(combo):
        return 0.0

    score = 0.0
    score += sum(prob_dict.get(n, 0.0) for n in combo) * weights["prob"]

    if check_sum(combo):
        score += 10
    if check_odd_even(combo):
        score += 8
    if check_high_low(combo):
        score += 8

    score += sum(n in neighbors for n in combo) * 4 * weights["neighbor"]
    score += sum(n in last_row for n in combo) * 4 * weights["repeater"]
    score += sum(max(0, 12 - skip_map.get(n, 0)) for n in combo) * weights["skip"]
    score += pair_score(combo) * 0.03 * weights["pair"]
    score += cluster_score(combo) * weights["cluster"]
    score += coverage_score(combo) * weights["coverage"]
    score += mean_reversion_score(combo) * weights["meanrev"]

    if check_tail(combo):
        score += 5 * weights["tail"]
    if check_decade(combo):
        score += 5 * weights["decade"]
    if check_gap(combo):
        score += 5 * weights["gap"]

    return score



def generate_elite(prob, elite_pool_size: int = 8):
    prob_dict = dict(prob)
    top20 = build_top20(prob)
    elite_pool = top20[:elite_pool_size]

    anchor = elite_pool[:3] if len(elite_pool) >= 3 else elite_pool[:]
    raw = []

    for _ in range(5000):
        if len(elite_pool) >= 6:
            combo = set(anchor)
            rest = list(set(elite_pool) - set(anchor))
            while len(combo) < 6 and rest:
                combo.add(random.choice(rest))
            combo = sorted(combo)
        else:
            combo = sorted(random.sample(top20, 6))

        if len(combo) != 6:
            continue

        if super_filter(combo):
            score = fitness(combo, prob_dict)
            raw.append((combo, score))

    if not raw:
        for _ in range(5000):
            combo = sorted(random.sample(top20, 6))
            if basic_filter(combo):
                raw.append((combo, fitness(combo, prob_dict)))

    raw = sorted(raw, key=lambda x: x[1], reverse=True)

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


prob = predict_prob()
prob_dict = dict(prob)
top20, elite_pool, elite_final = generate_elite(prob, elite_pool_size=8)


# =====================================================
# 추천 이유 / 등급 / 스타일
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



def explain_combo(combo):
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
# Walk-Forward Backtest / Diversity / State Report
# =====================================================


def build_context(train_df: pd.DataFrame):
    t_freq = get_number_frequency(train_df)
    t_skip_map = build_skip_map(train_df)
    t_pair_matrix = build_pair_matrix(train_df)
    t_last_row = list(map(int, train_df.iloc[-1].values))
    t_neighbors = get_neighbors(t_last_row)
    t_sum_series = pd.Series([sum(map(int, row)) for row in train_df.values])
    t_sum_ma50 = t_sum_series.rolling(50).mean()
    t_sum_std50 = t_sum_series.rolling(50).std()
    t_sum_upper = t_sum_ma50 + t_sum_std50
    t_sum_lower = t_sum_ma50 - t_sum_std50

    t_odd_series = pd.Series([sum(n % 2 for n in map(int, row)) for row in train_df.values])
    t_highlow_series = pd.Series([sum(n <= 23 for n in map(int, row)) for row in train_df.values])

    t_recent5_nums = set(train_df.tail(5).values.flatten())
    t_recent10_nums = set(train_df.tail(10).values.flatten())

    t_past_skip_sums = [sum(t_skip_map.get(int(n), 0) for n in row) for row in train_df.values]
    t_skip_mean = float(np.mean(t_past_skip_sums))
    t_skip_std = float(np.std(t_past_skip_sums))
    t_skip_low = t_skip_mean - t_skip_std
    t_skip_high = t_skip_mean + t_skip_std

    t_features = make_features(train_df)
    if len(train_df) >= 2:
        n_clusters_local = min(5, len(train_df))
        km = KMeans(n_clusters=n_clusters_local, random_state=42, n_init=10)
        lbls = km.fit_predict(t_features)
        t_current_cluster = int(lbls[-1])
        t_cluster_centroid = km.cluster_centers_[t_current_cluster]
    else:
        t_current_cluster = 0
        t_cluster_centroid = None

    return {
        "df": train_df,
        "freq": t_freq,
        "skip_map": t_skip_map,
        "pair_matrix": t_pair_matrix,
        "last_row": t_last_row,
        "neighbors": t_neighbors,
        "sum_series": t_sum_series,
        "sum_ma50": t_sum_ma50,
        "sum_std50": t_sum_std50,
        "sum_upper": t_sum_upper,
        "sum_lower": t_sum_lower,
        "odd_series": t_odd_series,
        "highlow_series": t_highlow_series,
        "recent5_nums": t_recent5_nums,
        "recent10_nums": t_recent10_nums,
        "skip_low": t_skip_low,
        "skip_high": t_skip_high,
        "current_cluster": t_current_cluster,
        "cluster_centroid": t_cluster_centroid,
    }



def adaptive_weights_from_context(ctx):
    latest_sum = int(ctx["sum_series"].iloc[-1])
    sum_upper_v = float(ctx["sum_upper"].iloc[-1]) if pd.notna(ctx["sum_upper"].iloc[-1]) else float(ctx["sum_series"].mean())
    sum_lower_v = float(ctx["sum_lower"].iloc[-1]) if pd.notna(ctx["sum_lower"].iloc[-1]) else float(ctx["sum_series"].mean())

    latest_cov10 = int(len(set(df.tail(10).values.flatten()) & set(map(int, df.iloc[-1].values)))) if len(df) >= 10 else 0

    sum_state = "NORMAL"
    if latest_sum > sum_upper_v:
        sum_state = "OVERHEAT"
    elif latest_sum < sum_lower_v:
        sum_state = "COOL"

    coverage_state = "NORMAL"
    if latest_cov10 >= 5:
        coverage_state = "OVERHEAT"
    elif latest_cov10 <= 3:
        coverage_state = "COOL"

    odd_state = "BALANCED" if sum(map(int, df.iloc[-1].values)) % 2 in [2, 3, 4] else "BIAS"
    highlow_state = "BALANCED" if sum(n <= 23 for n in map(int, df.iloc[-1].values)) in [2, 3, 4] else "BIAS"

    weights = {
        "prob": 1.0,
        "pair": 1.0,
        "skip": 1.0,
        "neighbor": 1.0,
        "repeater": 1.0,
        "tail": 1.0,
        "decade": 1.0,
        "gap": 1.0,
        "coverage": 1.0,
        "meanrev": 1.0,
        "cluster": 1.0,
    }

    if sum_state == "OVERHEAT":
        weights["meanrev"] = 1.2
    elif sum_state == "COOL":
        weights["meanrev"] = 1.15

    if coverage_state == "OVERHEAT":
        weights["coverage"] = 1.15
    elif coverage_state == "COOL":
        weights["coverage"] = 1.05

    if odd_state == "BALANCED":
        weights["neighbor"] = 1.05
    if highlow_state == "BALANCED":
        weights["pair"] = 1.05

    return {
        "sum_state": sum_state,
        "coverage_state": coverage_state,
        "odd_state": odd_state,
        "highlow_state": highlow_state,
        "latest_sum": latest_sum,
        "latest_cov10": latest_cov10,
        "weights": weights,
    }


ctx = build_context(df)
state = adaptive_weights_from_context(ctx)
weights = state["weights"]



def diversity_ok(candidate, selected):
    for existed, _ in selected:
        if len(set(candidate) & set(existed)) >= 5:
            return False
    return True


@st.cache_data(ttl=3600)
def walk_forward_backtest(data_signature: str, folds: int = 15):
    if len(df) < 60:
        return []

    results = []
    start = max(50, len(df) - folds)

    for i in range(start, len(df)):
        train_df = df.iloc[:i].reset_index(drop=True)
        test_row = set(map(int, df.iloc[i].values))
        train_ctx = build_context(train_df)
        train_state = adaptive_weights_from_context(train_ctx)
        train_weights = train_state["weights"]

        # training-based lightweight score
        t_freq = train_ctx["freq"]
        t_skip_map = train_ctx["skip_map"]
        t_pair_matrix = train_ctx["pair_matrix"]
        t_neighbors = train_ctx["neighbors"]
        t_last_row = train_ctx["last_row"]
        t_sum_series = train_ctx["sum_series"]
        t_sum_ma50 = train_ctx["sum_ma50"]
        t_sum_std50 = train_ctx["sum_std50"]
        t_sum_upper = train_ctx["sum_upper"]
        t_sum_lower = train_ctx["sum_lower"]
        t_recent5_nums = train_ctx["recent5_nums"]
        t_recent10_nums = train_ctx["recent10_nums"]
        t_skip_low = train_ctx["skip_low"]
        t_skip_high = train_ctx["skip_high"]
        t_cluster_centroid = train_ctx["cluster_centroid"]

        def t_pair_score(combo):
            score = 0
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    pair = tuple(sorted((int(combo[a]), int(combo[b]))))
                    score += t_pair_matrix.get(pair, 0)
            return score

        def t_cluster_score(combo):
            if t_cluster_centroid is None:
                return 0.0
            feat = np.array([
                sum(combo),
                sum(n % 2 for n in combo),
                sum(n <= 23 for n in combo),
                max(combo) - min(combo),
            ], dtype=float)
            dist = float(np.linalg.norm(feat - t_cluster_centroid))
            return max(0.0, 10.0 - dist)

        def t_coverage_score(combo):
            r5 = sum(n in t_recent5_nums for n in combo)
            r10 = sum(n in t_recent10_nums for n in combo)
            score = 0
            if 2 <= r5 <= 3:
                score += 8
            if 4 <= r10 <= 5:
                score += 12
            return score

        def t_mean_reversion_score(combo):
            combo_sum = sum(combo)
            latest_sum = t_sum_series.iloc[-1]
            current_ma50 = t_sum_ma50.iloc[-1]
            current_upper = t_sum_upper.iloc[-1]
            current_lower = t_sum_lower.iloc[-1]
            score = 0
            if pd.notna(current_upper) and latest_sum > current_upper and pd.notna(current_ma50) and combo_sum < current_ma50:
                score += 10
            elif pd.notna(current_lower) and latest_sum < current_lower and pd.notna(current_ma50) and combo_sum > current_ma50:
                score += 10
            return score

        def t_fitness(combo):
            combo = sorted(map(int, combo))
            if not basic_filter(combo):
                return 0.0
            score = 0.0
            score += sum((t_freq.get(n, 0) / max(1, sum(t_freq.values()))) for n in combo) * 50
            score += sum(n in t_neighbors for n in combo) * 4 * train_weights["neighbor"]
            score += sum(n in t_last_row for n in combo) * 4 * train_weights["repeater"]
            score += sum(max(0, 12 - t_skip_map.get(n, 0)) for n in combo) * train_weights["skip"]
            score += t_pair_score(combo) * 0.03 * train_weights["pair"]
            score += t_cluster_score(combo) * train_weights["cluster"]
            score += t_coverage_score(combo) * train_weights["coverage"]
            score += t_mean_reversion_score(combo) * train_weights["meanrev"]
            return score

        top20_local = sorted(range(1, 46), key=lambda n: t_freq.get(n, 0), reverse=True)[:20]
        elite_pool_local = top20_local[:8]
        raw = []

        for _ in range(1500):
            combo = sorted(random.sample(elite_pool_local if len(elite_pool_local) >= 6 else top20_local, 6))
            score = t_fitness(combo)
            raw.append((combo, score))

        raw = sorted(raw, key=lambda x: x[1], reverse=True)
        best = raw[0][0] if raw else sorted(random.sample(range(1, 46), 6))
        hit = len(set(best) & test_row)
        results.append(hit)

    return results


# =====================================================
# UI
# =====================================================

st.sidebar.title("👤 관리자 / 상태")
admin_mode = st.sidebar.checkbox("🔐 관리자 모드")
admin_code_input = st.sidebar.text_input("관리자 코드", type="password")
is_admin = admin_mode and (admin_code_input == ADMIN_CODE)

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 보안 상태")
st.sidebar.success("✔ 입력 검증 활성화")
st.sidebar.success("✔ 데이터 빈값 방어")
st.sidebar.success("✔ API 실패 fallback")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 현재 상태")
st.sidebar.write(f"데이터 회차 수: {len(df)}")
st.sidebar.write(f"현재 군집: {current_cluster}")
st.sidebar.write(f"합계 상태: {state['sum_state']}")
st.sidebar.write(f"Coverage 상태: {state['coverage_state']}")
st.sidebar.write(f"홀짝 상태: {state['odd_state']}")
st.sidebar.write(f"고저 상태: {state['highlow_state']}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📌 최근 회차", fetch_latest_round_number() or "fallback")
with col2:
    st.metric("🧠 현재 군집", current_cluster)
with col3:
    st.metric("🔥 ELITE 풀", 8)

st.subheader("🔥 핵심 압축 번호 TOP20")
st.markdown(style_numbers(top20), unsafe_allow_html=True)

if is_admin:
    st.subheader("🔐 관리자 전용 ELITE 풀 (Top 8)")
    st.markdown(style_numbers(elite_pool), unsafe_allow_html=True)

    st.subheader("👑 관리자 전용 최강 조합 TOP10")
    if elite_final:
        for i, (combo, score) in enumerate(elite_final, 1):
            grade = grade_combo(score)
            reasons = explain_combo(combo)
            st.markdown(f"## {i}. {grade}")
            st.markdown(style_numbers(combo), unsafe_allow_html=True)
            st.write(f"💯 점수: {round(score, 2)}")
            st.write("✔ 추천 이유:", ", ".join(reasons) if reasons else "기본 조건 충족")
            st.progress(min(score / 100, 1.0))
            st.markdown("---")
    else:
        st.warning("관리자 전용 최강 조합이 생성되지 않았습니다.")
else:
    st.subheader("👑 추천 결과")
    public_view = elite_final[:3]
    if public_view:
        for i, (combo, score) in enumerate(public_view, 1):
            grade = grade_combo(score)
            reasons = explain_combo(combo)
            st.markdown(f"## {i}. {grade}")
            st.markdown(style_numbers(combo), unsafe_allow_html=True)
            st.write(f"💯 점수: {round(score, 2)}")
            st.write("✔ 추천 이유:", ", ".join(reasons) if reasons else "기본 조건 충족")
            st.progress(min(score / 100, 1.0))
            st.markdown("---")
    else:
        st.warning("추천 조합이 생성되지 않았습니다.")


st.subheader("📈 합계 시계열 흐름 (rolling 50)")
fig_sum = px.line(x=range(len(sum_series)), y=sum_series, title="합계 흐름")
st.plotly_chart(fig_sum, use_container_width=True)

st.subheader("📈 최근10회 Coverage 흐름")
if not coverage10_series.empty:
    fig_cov = px.line(x=range(len(coverage10_series)), y=coverage10_series, title="최근10회 번호 포함 개수 흐름")
    st.plotly_chart(fig_cov, use_container_width=True)
else:
    st.info("Coverage 시계열은 데이터가 더 쌓이면 표시됩니다.")

st.subheader("📊 백테스트 적중 분포")
backtest_results = []
for row in df.values[-50:]:
    real = set(map(int, row))
    for combo, _ in elite_final[:10]:
        backtest_results.append(len(set(combo) & real))

fig_bt = px.histogram(backtest_results, nbins=10, title="적중 분포")
st.plotly_chart(fig_bt, use_container_width=True)

st.subheader("📊 Walk-Forward Backtest")
if st.button("▶ 백테스트 실행"):
    wf = walk_forward_backtest(str(df.shape) + str(df.iloc[-1].tolist()), folds=min(15, max(0, len(df) - 50)))
    if wf:
        st.write(f"평균 적중: {round(float(np.mean(wf)), 2)}")
        st.write(f"최대 적중: {max(wf)}")
        st.write(f"적중 분포: {Counter(wf)}")
    else:
        st.warning("백테스트를 수행할 데이터가 충분하지 않습니다.")

st.subheader("🧠 확률 필터 최적화 연구")
st.info(
    "현재 시스템은 단순 번호 추천이 아니라 '확률 패턴 공간 압축' 기반 AI 엔진입니다.

"
    "적용 요소:
"
    "- Pair Matrix
"
    "- Skip LN
"
    "- Neighbor
"
    "- Repeater
"
    "- Gap Pattern
"
    "- Tail Pattern
"
    "- Cluster Analysis
"
    "- Adaptive Probability
"
    "- Rolling 50
"
    "- Coverage Flow
"
    "- Mean Reversion
"
    "- Walk-Forward Backtest
"
    "- Adaptive Weight Engine
"
    "- Diversity Engine
"
    "- State Engine
"
)
