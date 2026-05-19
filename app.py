import streamlit as st
import pandas as pd
import numpy as np
import random
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from collections import Counter

st.title("🎯 로또 AI 추천 시스템")

# ------------------------
# 데이터
# ------------------------
def load_data():
    data = [
        [9,18,21,27,44,45],
        [3,7,11,12,20,27],
        [7,8,10,12,34,44],
        [13,14,15,24,31,34],
        [12,16,20,23,38,40]
    ]
    return pd.DataFrame(data, columns=['n1','n2','n3','n4','n5','n6'])

df = load_data()

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
# GA
# ------------------------
def fitness(combo, prob_dict):
    return sum(prob_dict[n] for n in combo)

def generate_best(prob):
    prob_dict = dict(prob)
    population = [sorted(random.sample(range(1,46),6)) for _ in range(50)]

    for _ in range(10):
        population = sorted(population, key=lambda x: fitness(x, prob_dict), reverse=True)
        population = population[:20]

        new_pop = population.copy()
        while len(new_pop) < 50:
            a, b = random.sample(population, 2)
            child = sorted(list(set(a[:3] + b[3:])))
            while len(child) < 6:
                child.append(random.randint(1,45))
            child = sorted(child[:6])
            new_pop.append(child)

        population = new_pop

    return population[:5]

# ------------------------
# 백테스트
# ------------------------
def backtest_model(df, trials=30):
    results = []

    for _ in range(trials):
        prob = predict_prob()
        combos = generate_best(prob)

        for combo in combos:
            score = 0
            for row in df.values:
                score += len(set(combo) & set(row))

            avg_score = score / len(df)
            results.append(avg_score)

    return results

# ------------------------
# UI
# ------------------------
if st.button("🚀 AI 초강력 추천"):
    
    prob = predict_prob()
    best = generate_best(prob)

    st.subheader("🔥 추천 번호")
    for i, b in enumerate(best,1):
        st.write(f"{i}. {b}")

    # ------------------------
    # 정확도 분석
    # ------------------------
    st.subheader("📊 정확도 분석")

    results = backtest_model(df)

    avg = np.mean(results)
    high_hit = sum(r >= 3 for r in results) / len(results)

    st.write(f"평균 적중 개수: {round(avg,2)}")
    st.write(f"3개 이상 적중 확률: {round(high_hit*100,1)}%")

    # ------------------------
    # 그래프
    # ------------------------
    st.subheader("📈 적중 분포")

    fig, ax = plt.subplots()
    ax.hist(results, bins=10)
    st.pyplot(fig)

    # ------------------------
    # 히트맵
    # ------------------------
    st.subheader("🔥 번호 히트맵")

    counter = Counter()
    for row in df.values:
        for n in row:
            counter[n] += 1

    fig2, ax2 = plt.subplots()
    ax2.bar(counter.keys(), counter.values())
    st.pyplot(fig2)
