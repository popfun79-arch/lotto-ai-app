import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
import lightgbm as lgb

st.title("🎯 로또 AI 추천")

uploaded_file = st.file_uploader("CSV 업로드", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    return df[['n1','n2','n3','n4','n5','n6']].values.tolist()

def build_features(history, window=20):
    X, y = [], []

    for i in range(window, len(history)):
        past = history[i-window:i]
        current = set(history[i])
        freq = Counter(sum(past, []))

        for num in range(1,46):
            f = freq[num]
            r5 = sum(num in h for h in past[-5:])
            r10 = sum(num in h for h in past[-10:])

            gap = 0
            for j in range(len(past)-1, -1, -1):
                if num in past[j]:
                    gap = len(past) - j
                    break

            X.append([num, f, r5, r10, gap])
            y.append(1 if num in current else 0)

    return np.array(X), np.array(y)

def train_model(X, y):
    model = lgb.LGBMClassifier(n_estimators=200)
    model.fit(X, y)
    return model

def predict_probs(model, history):
    past = history[-20:]
    freq = Counter(sum(past, []))

    features = []

    for num in range(1,46):
        f = freq[num]
        r5 = sum(num in h for h in past[-5:])
        r10 = sum(num in h for h in past[-10:])

        gap = 0
        for j in range(len(past)-1, -1, -1):
            if num in past[j]:
                gap = len(past) - j
                break

        features.append([num, f, r5, r10, gap])

    probs = model.predict_proba(features)[:,1]
    return {i+1: probs[i] for i in range(45)}

def generate_combos(prob_map):
    nums = sorted(prob_map, key=prob_map.get, reverse=True)[:20]
    results = []
    for _ in range(5):
        results.append(sorted(random.sample(nums,6)))
    return results

if uploaded_file:
    history = load_data(uploaded_file)

    if st.button("🚀 추천 생성"):
        X, y = build_features(history)
        model = train_model(X, y)

        prob_map = predict_probs(model, history)

        st.subheader("🔥 TOP 번호")
        for k,v in sorted(prob_map.items(), key=lambda x:x[1], reverse=True)[:10]:
            st.write(k, round(v,4))

        st.subheader("🎯 추천 조합")
        combos = generate_combos(prob_map)
        for i,c in enumerate(combos):
            st.write(i+1, c)