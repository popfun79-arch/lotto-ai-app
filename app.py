import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
import os
import json
from lightgbm import LGBMClassifier
import plotly.express as px
from collections import Counter
from itertools import combinations

# 🔐 보안 설정
st.set_option('client.showErrorDetails', False)

st.title("🎯 로또 AI 초강력 추천 시스템")

# ------------------------
# 👤 사용자 로그인
# ------------------------
st.sidebar.title("👤 사용자")

user_id = st.sidebar.text_input("이메일 입력")

if not user_id:
    st.warning("이메일 입력 후 사용해주세요")
    st.stop()

use_personal = st.sidebar.checkbox("🎯 개인화 추천", True)

# ------------------------
# 📁 사용자 데이터
# ------------------------
def get_user_file(user_id):
    return f"user_{user_id.replace('@','_').replace('.','_')}.json"

def load_user_data(user_id):
    file = get_user_file(user_id)
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {"history":[]}

def save_user_data(user_id, history):
    file = get_user_file(user_id)
    with open(file, "w") as f:
        json.dump({"history": history}, f)

# ------------------------
# 🌐 최신 회차 (안정화)
# ------------------------
def fetch_latest_round():
    url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(1300, 1000, -1):
        try:
            res = requests.get(url + str(i), headers=headers, timeout=5)
            if res.status_code != 200:
                continue
            data = res.json()
            if data.get("returnValue") == "success":
                return [
                    data['drwtNo1'], data['drwtNo2'], data['drwtNo3'],
                    data['drwtNo4'], data['drwtNo5'], data['drwtNo6']
                ]
        except:
            continue
    return None

# ------------------------
# 📊 데이터 업데이트
# ------------------------
def update_lotto_data():
    df = pd.read_csv("lotto_200.csv")

    latest = fetch_latest_round()

    if latest and latest not in df.values.tolist():
        df.loc[len(df)] = latest
        df = df.tail(200)
        df.to_csv("lotto_200.csv", index=False)

    return df

@st.cache_data(ttl=3600)
def load_data():
    return update_lotto_data()

df = load_data()

# ------------------------
# 🤖 모델
# ------------------------
def train_model(df):
    X, y = [], []
    for row in df.values:
        for n in range(1,46):
            X.append([n])
            y.append(1 if n in row else 0)
    model = LGBMClassifier()
    model.fit(X,y)
    return model

model = train_model(df)

def predict_prob():
    return [(n, model.predict_proba([[n]])[0][1]) for n in range(1,46)]

# ------------------------
# 🧠 개인화
# ------------------------
def personal_preference(user_data):
    freq = Counter()
    for hist in user_data["history"]:
        for combo in hist:
            for n in combo:
                freq[n]+=1
    return freq

def personal_score(combo, pref):
    return sum(pref.get(n,0) for n in combo) * 0.2

# ------------------------
# ⚙️ 점수
# ------------------------
def basic_filter(c):
    if max(c)-min(c)==5: return False
    if max([list(map(lambda x:x%10,c)).count(x) for x in set(c)])>=3: return False
    return True

def balance_score(c):
    score=0
    if sum(n%2 for n in c) in [2,3,4]: score+=10
    if sum(n<=23 for n in c) in [2,3,4]: score+=10
    if 100<=sum(c)<=180: score+=10
    return score

def neighbor_score(c):
    last = df.iloc[-1].values
    neighbors = [x+i for x in last for i in [-1,1]]
    return sum(n in neighbors for n in c)*5

def hot_cold_score(c):
    freq=Counter([n for row in df.values for n in row])
    hot=sorted(freq,key=freq.get,reverse=True)[:10]
    cold=sorted(freq,key=freq.get)[:10]
    return sum(n in hot for n in c)*3 + sum(n in cold for n in c)*2

def build_pair_matrix():
    pair={}
    for row in df.values:
        for a,b in combinations(row,2):
            pair[(a,b)] = pair.get((a,b),0)+1
    return pair

pair_matrix = build_pair_matrix()

def pair_score(c):
    return sum(pair_matrix.get(tuple(sorted((a,b))),0) for a,b in combinations(c,2))/5

def compute_skip_map():
    last={}
    for i,row in enumerate(df.values[::-1]):
        for n in row:
            if n not in last:
                last[n]=i
    return {n:last.get(n,len(df)) for n in range(1,46)}

skip_map = compute_skip_map()

def skip_pattern_score(c):
    s = sum(skip_map[n] for n in c)
    avg = np.mean([sum(skip_map[n] for n in row) for row in df.values])
    return 15 if avg*0.7 <= s <= avg*1.3 else 0

# ------------------------
# 🎯 fitness
# ------------------------
def fitness(c, prob_dict, pref):

    if not basic_filter(c):
        return 0

    score = sum(prob_dict[n] for n in c)
    score += balance_score(c)
    score += neighbor_score(c)
    score += hot_cold_score(c)
    score += pair_score(c)
    score += skip_pattern_score(c)

    if use_personal:
        score += personal_score(c, pref)

    return score

# ------------------------
# 🧬 GA
# ------------------------
def generate_best(prob):

    prob_dict = dict(prob)
    user_data = load_user_data(user_id)
    pref = personal_preference(user_data)

    pop=[sorted(random.sample(range(1,46),6)) for _ in range(50)]

    for _ in range(10):
        pop=sorted(pop,key=lambda x: fitness(x, prob_dict, pref),reverse=True)[:20]
        new=pop.copy()
        while len(new)<50:
            a,b=random.sample(pop,2)
            child=sorted(list(set(a[:3]+b[3:])))
            while len(child)<6:
                child.append(random.randint(1,45))
            new.append(sorted(child[:6]))
        pop=new

    return pop[:5]

# ------------------------
# 🎨 UI
# ------------------------
def style_numbers(c):
    html=""
    for n in c:
        color = "#fbc400" if n<=10 else "#69c8f2" if n<=20 else "#ff7272" if n<=30 else "#aaa" if n<=40 else "#b0d840"
        html += f"<span style='background:{color};padding:10px;margin:5px;border-radius:50%;'>{n}</span>"
    return html

def grade(score):
    return "🔥S" if score>80 else "⭐A" if score>60 else "👍B" if score>40 else "C"

# ------------------------
# 🚀 실행
# ------------------------
if st.button("🚀 AI 추천 실행"):

    prob = predict_prob()
    best = generate_best(prob)
    prob_dict = dict(prob)

    user_data = load_user_data(user_id)
    user_data["history"].append(best)
    save_user_data(user_id, user_data["history"])

    st.subheader("🎯 추천 결과")

    for i,c in enumerate(best,1):
        score = fitness(c, prob_dict, personal_preference(user_data))
        st.markdown(f"## {i}번 {grade(score)}")
        st.markdown(style_numbers(c), unsafe_allow_html=True)
        st.write(f"점수: {round(score,1)}")
        st.markdown("---")

    st.subheader("📂 나의 기록")
    for hist in user_data["history"][-3:]:
        st.write(hist)

    results = [len(set(b)&set(row)) for b in best for row in df.values]

    fig = px.histogram(results, nbins=10, title="적중 분포")
    st.plotly_chart(fig, use_container_width=True)
