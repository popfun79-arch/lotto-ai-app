# 1️⃣ 라이브러리
import streamlit as st
import pandas as pd
import numpy as np
import random
from lightgbm import LGBMClassifier

# 2️⃣ UI 제목
st.title("🎯 로또 AI 추천 시스템")

# 3️⃣ 데이터 로드
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

# 4️⃣ 모델 학습
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

# 5️⃣ 확률 예측
def predict_prob():
    probs = []
    for n in range(1,46):
        p = model.predict_proba([[n]])[0][1]
        probs.append((n,p))
    return sorted(probs, key=lambda x: x[1], reverse=True)

# 6️⃣ GA 알고리즘
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

# 7️⃣ 평가 (백테스트)
def evaluate(combo, history):
    score = 0
    for h in history.values:
        score += len(set(combo) & set(h))
    return score

# 8️⃣ UI 설정
st.sidebar.title("⚙ 설정")
num_sets = st.sidebar.slider("추천 개수", 1, 10, 5)

# 9️⃣ 실행 버튼 (핵심)
if st.button("🚀 AI 초강력 추천"):
    prob = predict_prob()
    best = generate_best(prob)

    st.subheader("🔥 추천 번호")
    for i, b in enumerate(best[:num_sets],1):
        st.write(f"{i}. {b}")

    st.subheader("📊 백테스트")
    for b in best[:num_sets]:
        st.write(f"{b} → 점수: {evaluate(b, df)}")


# 📊 1. 백테스트 함수 (핵심)
def backtest_model(df, trials=50):
    results = []

    for _ in range(trials):
        prob = predict_prob()
        combos = generate_best(prob)

        for combo in combos:
            score = 0
            for row in df.values:
                match = len(set(combo) & set(row))
                score += match

            avg_score = score / len(df)
            results.append(avg_score)

    return results


# 📊 2. 통계 계산
def analyze_results(results):
    avg = np.mean(results)
    high_hit = sum(r >= 3 for r in results) / len(results)

    return avg, high_hit


# 📊 3. UI 출력 (핵심)  👉 버튼 아래쪽에 추가
st.subheader("📊 AI 정확도 분석")

results = backtest_model(df)

avg, high_hit = analyze_results(results)

st.write(f"평균 적중 개수: {round(avg,2)}")
st.write(f"3개 이상 적중 확률: {round(high_hit*100,1)}%")


# 🚀 ✅ 2️⃣ 그래프 추가 (중요)  👉 시각화 = 신뢰도
import matplotlib.pyplot as plt

st.subheader("📈 적중 분포")

fig, ax = plt.subplots()
ax.hist(results, bins=10)
st.pyplot(fig)


# 🚀 ✅ 3️⃣ 히트맵 (핵심)  👉 어떤 숫자가 많이 맞는지 보여줌
from collections import Counter

st.subheader("🔥 번호 히트맵")

counter = Counter()

for row in df.values:
    for n in row:
        counter[n] += 1

numbers = list(counter.keys())
counts = list(counter.values())

fig, ax = plt.subplots()
ax.bar(numbers, counts)
st.pyplot(fig)
