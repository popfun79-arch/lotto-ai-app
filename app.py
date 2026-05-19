import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
from lightgbm import LGBMClassifier
import plotly.express as px
from collections import Counter
from itertools import combinations

st.title("🎯 로또 AI 초강력 추천 시스템")

# ------------------------
# 최신 회차 가져오기
# ------------------------
def fetch_latest_round():
    url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="
    for i in range(1300, 1000, -1):
        res = requests.get(url + str(i)).json()
        if res['returnValue'] == 'success':
            return [
                res['drwtNo1'], res['drwtNo2'], res['drwtNo3'],
                res['drwtNo4'], res['drwtNo5'], res['drwtNo6']
            ]

# ------------------------
# 데이터 업데이트
# ------------------------
def update_lotto_data():
    df = pd.read_csv("lotto_200.csv")

    latest_nums = fetch_latest_round()

    if latest_nums not in df.values.tolist():
        df.loc[len(df)] = latest_nums
        df = df.tail(200)
        df.to_csv("lotto_200.csv", index=False)

    return df

df = update_lotto_data()

# ------------------------
# 모델
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
    probs = []
    for n in range(1,46):
        p = model.predict_proba([[n]])[0][1]
        probs.append((n,p))
    return sorted(probs, key=lambda x: x[1], reverse=True)

# ------------------------
# 기본 필터
# ------------------------
def basic_filter(combo):
    if max(combo)-min(combo) == 5:
        return False
    last_digits = [n%10 for n in combo]
    if max(last_digits.count(x) for x in last_digits) >= 3:
        return False
    return True

# ------------------------
# 점수 요소들
# ------------------------
def balance_score(combo):
    score = 0
    odd = sum(n%2 for n in combo)
    if odd in [2,3,4]: score += 10
    low = sum(n<=23 for n in combo)
    if low in [2,3,4]: score += 10
    s = sum(combo)
    if 100<=s<=180: score += 10
    return score

def neighbor_score(combo, last):
    neighbors = []
    for n in last:
        neighbors += [n-1,n+1]
    return sum(n in neighbors for n in combo)*5

def hot_cold_score(combo, freq):
    hot = sorted(freq, key=freq.get, reverse=True)[:10]
    cold = sorted(freq, key=freq.get)[:10]
    score = 0
    score += sum(n in hot for n in combo)*3
    score += sum(n in cold for n in combo)*2
    return score

def skip_score(combo, df):
    last_seen = {}
    for i,row in enumerate(df.values[::-1]):
        for n in row:
            if n not in last_seen:
                last_seen[n]=i
    score=0
    for n in combo:
        skip = last_seen.get(n,len(df))
        if skip>10: score+=3
        if skip>20: score+=5
    return score

def build_pair_matrix(df):
    pair={}
    for row in df.values:
        for a,b in combinations(row,2):
            pair[(a,b)] = pair.get((a,b),0)+1
    return pair

def pair_score(combo, pair_matrix):
    return sum(pair_matrix.get(tuple(sorted((a,b))),0)
               for a,b in combinations(combo,2)) / 5

def last_digit_score(combo, df):
    counter={}
    for row in df.tail(50).values:
        for n in row:
            d=n%10
            counter[d]=counter.get(d,0)+1
    top=sorted(counter,key=counter.get,reverse=True)[:3]
    return sum(n%10 in top for n in combo)*3

# ------------------------
# 🔥 Skip 패턴 (핵심)
# ------------------------
def compute_skip_map(df):
    last_seen={}
    for i,row in enumerate(df.values[::-1]):
        for n in row:
            if n not in last_seen:
                last_seen[n]=i
    return {n:last_seen.get(n,len(df)) for n in range(1,46)}

def skip_sum(combo, skip_map):
    return sum(skip_map[n] for n in combo)

def compute_skip_sum_range(df):
    sums=[]
    for i in range(len(df)):
        sub=df.iloc[:i+1]
        skip_map=compute_skip_map(sub)
        s=sum(skip_map[n] for n in df.iloc[i])
        sums.append(s)
    avg=np.mean(sums)
    return avg*0.7, avg*1.3

def skip_pattern_score(combo, skip_map, low, high):
    s = skip_sum(combo, skip_map)
    if low<=s<=high:
        return 15
    return 0

# ------------------------
# 최종 fitness
# ------------------------
pair_matrix = build_pair_matrix(df)
skip_map = compute_skip_map(df)
low, high = compute_skip_sum_range(df)

def fitness(combo, prob_dict):
    if not basic_filter(combo):
        return 0

    freq=Counter()
    for row in df.values:
        for n in row:
            freq[n]+=1

    score = sum(prob_dict[n] for n in combo)
    score += balance_score(combo)
    score += neighbor_score(combo, df.iloc[-1].values)
    score += hot_cold_score(combo, freq)
    score += skip_score(combo, df)
    score += pair_score(combo, pair_matrix)
    score += last_digit_score(combo, df)
    score += skip_pattern_score(combo, skip_map, low, high)

    return score

# ------------------------
# GA
# ------------------------
def generate_best(prob):
    prob_dict=dict(prob)
    pop=[sorted(random.sample(range(1,46),6)) for _ in range(50)]

    for _ in range(10):
        pop=sorted(pop,key=lambda x: fitness(x, prob_dict),reverse=True)[:20]
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
# UI
# ------------------------
if st.button("🚀 AI 초강력 추천"):
    prob = predict_prob()
    best = generate_best(prob)

    st.subheader("🔥 추천 번호")
    for i,b in enumerate(best,1):
        st.write(f"{i}. {b}")

    results = [len(set(b)&set(row)) for b in best for row in df.values]

    st.subheader("📊 정확도")
    st.write(f"평균 적중: {round(np.mean(results),2)}")

    fig = px.histogram(results, nbins=10, title="적중 분포")
    st.plotly_chart(fig, use_container_width=True)
