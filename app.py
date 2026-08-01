from __future__ import annotations

import json, math, random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

MAIN = ["n1","n2","n3","n4","n5","n6"]
REQUIRED = ["round","date",*MAIN,"bonus"]
NUMBERS = list(range(1,46))
PRIMES = {2,3,5,7,11,13,17,19,23,29,31,37,41,43}
SEED = 20260720

WEIGHTS = {
    "long_freq":0.08, "short_freq":0.08, "gap":0.12, "egr":0.10,
    "carry":0.07, "neighbor":0.08, "bonus":0.05, "pair":0.05,
    "zone":0.07, "dna":0.15, "state":0.10, "hot_cold":0.05,
}

def normalize_columns(df):
    mapping = {
        "회차":"round","날짜":"date","추첨일":"date",
        "번호1":"n1","번호2":"n2","번호3":"n3",
        "번호4":"n4","번호5":"n5","번호6":"n6",
        "보너스":"bonus","보너스번호":"bonus",
        "drwNo":"round","drwNoDate":"date",
        "drwtNo1":"n1","drwtNo2":"n2","drwtNo3":"n3",
        "drwtNo4":"n4","drwtNo5":"n5","drwtNo6":"n6","bnusNo":"bonus",
    }
    out=df.copy()
    out.columns=[str(c).strip() for c in out.columns]
    return out.rename(columns=mapping)

def read_data(uploaded=None):
    if uploaded is not None:
        suffix=Path(uploaded.name).suffix.lower()
        if suffix==".csv":
            return pd.read_csv(uploaded), f"업로드: {uploaded.name}"
        if suffix==".json":
            raw=json.loads(uploaded.getvalue().decode("utf-8-sig"))
            if isinstance(raw,dict):
                for key in ("data","rows","results","lotto"):
                    if isinstance(raw.get(key),list):
                        raw=raw[key]; break
            return pd.DataFrame(raw), f"업로드: {uploaded.name}"
        raise ValueError("CSV 또는 JSON만 지원합니다.")
    path=Path("data/lotto_all.csv")
    if path.exists():
        df=pd.read_csv(path)
        if len(df)>0:
            return df, str(path)
    return None, ""

def validate(df):
    if df is None or df.empty:
        raise ValueError("데이터가 비어 있습니다.")
    out=normalize_columns(df)
    missing=[c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError("필수 컬럼 누락: "+", ".join(missing))
    out=out[REQUIRED].copy()
    notes=[]
    out["date"]=pd.to_datetime(out["date"],errors="coerce")
    for c in ["round",*MAIN,"bonus"]:
        out[c]=pd.to_numeric(out[c],errors="coerce")
    before=len(out)
    out=out.dropna(subset=["round",*MAIN,"bonus"])
    if len(out)<before: notes.append(f"누락값 행 {before-len(out)}개 제거")
    out[["round",*MAIN,"bonus"]]=out[["round",*MAIN,"bonus"]].astype(int)
    def sorter(row):
        nums=sorted(int(row[c]) for c in MAIN)
        for i,c in enumerate(MAIN): row[c]=nums[i]
        return row
    out=out.apply(sorter,axis=1)
    # 본번호는 1~45, 보너스는 미입력 데이터 호환을 위해 0~45 허용
    mask=np.ones(len(out),dtype=bool)
    for c in MAIN:
        mask &= out[c].between(1,45).to_numpy()
    mask &= out["bonus"].between(0,45).to_numpy()

    invalid_count=int((~mask).sum())
    if invalid_count>0:
        notes.append(f"범위 오류 행 {invalid_count}개 제거")
        out=out.loc[mask].copy()

    if out.empty:
        raise ValueError(
            "유효한 데이터가 없습니다. 본번호는 1~45, 보너스는 0~45인지 확인해 주세요."
        )

    unique=out.apply(
        lambda r: len({int(r[c]) for c in MAIN})==6,
        axis=1,
    ).astype(bool)

    duplicate_number_count=int((~unique).sum())
    if duplicate_number_count>0:
        notes.append(f"중복 본번호 행 {duplicate_number_count}개 제거")
        out=out.loc[unique].copy()
    before=len(out)
    out=out.drop_duplicates("round",keep="last").sort_values("round").reset_index(drop=True)
    if len(out)<before: notes.append(f"중복 회차 {before-len(out)}개 제거")
    if not notes: notes.append("기본 데이터 검증 통과")
    return out,notes

def nums(row): return [int(row[c]) for c in MAIN]
def odd_count(x): return sum(n%2==1 for n in x)
def low_count(x): return sum(n<=22 for n in x)
def prime_count(x): return sum(n in PRIMES for n in x)
def end_sum(x): return sum(n%10 for n in x)
def zidx(n): return 0 if n<=10 else 1 if n<=20 else 2 if n<=30 else 3 if n<=40 else 4
def zones(x):
    r=[0]*5
    for n in x: r[zidx(n)]+=1
    return tuple(r)
def ac(x):
    v=sorted(x)
    d={v[j]-v[i] for i in range(len(v)) for j in range(i+1,len(v))}
    return len(d)-(len(v)-1)
def consec_pairs(x):
    v=sorted(x); return sum(v[i]==v[i-1]+1 for i in range(1,len(v)))
def max_run(x):
    v=sorted(x); best=run=1
    for i in range(1,len(v)):
        if v[i]==v[i-1]+1: run+=1; best=max(best,run)
        else: run=1
    return best
def neighbors(x):
    r=set()
    for n in x:
        if n>1:r.add(n-1)
        if n<45:r.add(n+1)
    return r
def minmax(s):
    lo,hi=float(s.min()),float(s.max())
    return pd.Series(0.5,index=s.index) if math.isclose(lo,hi) else (s-lo)/(hi-lo)

def gap_tables(df):
    last={n:None for n in NUMBERS}; matrix=[]; rounds=[]
    for _,row in df.iterrows():
        r=int(row["round"])
        g={n:(r-last[n]-1 if last[n] is not None else np.nan) for n in NUMBERS}
        matrix.append({"round":r,**{f"gap_{n}":g[n] for n in NUMBERS}})
        selected=[g[n] for n in nums(row)]
        valid=[x for x in selected if pd.notna(x)]
        rounds.append({
            "round":r,"gap_values":selected,
            "gap_sum":float(np.sum(valid)) if valid else np.nan,
            "gap_mean":float(np.mean(valid)) if valid else np.nan,
            "gap_max":float(np.max(valid)) if valid else np.nan,
            "gap_std":float(np.std(valid)) if valid else np.nan,
            "gap_le_5":sum(x<=5 for x in valid),
            "gap_le_10":sum(x<=10 for x in valid),
            "gap_gt_10":sum(x>10 for x in valid),
            "gap_ge_17":sum(x>=17 for x in valid),
        })
        for n in nums(row): last[n]=r
    return pd.DataFrame(matrix),pd.DataFrame(rounds)

def current_gaps(df):
    latest=int(df["round"].max()); last={n:None for n in NUMBERS}; hist={n:[] for n in NUMBERS}
    for _,row in df.iterrows():
        r=int(row["round"])
        for n in nums(row):
            if last[n] is not None: hist[n].append(r-last[n]-1)
            last[n]=r
    rows=[]
    for n in NUMBERS:
        cg=latest-last[n] if last[n] is not None else len(df)
        rows.append({"number":n,"current_gap":cg,"average_gap":np.mean(hist[n]) if hist[n] else np.nan})
    out=pd.DataFrame(rows); out["gap_percentile"]=out["current_gap"].rank(pct=True)
    return out

def gap_distribution(df):
    _,r=gap_tables(df)
    values=[int(g) for item in r["gap_values"] for g in item if pd.notna(g)]
    c=pd.Series(values).value_counts().sort_index()
    out=c.rename_axis("gap").reset_index(name="count")
    out["ratio"]=out["count"]/out["count"].sum()
    return out

def egr_backtest(df,threshold=17,horizon=4):
    matrix,_=gap_tables(df); order=matrix["round"].astype(int).tolist()
    by_round={int(r["round"]):set(nums(r)) for _,r in df.iterrows()}
    active={n:False for n in NUMBERS}; events=[]
    for idx,row in matrix.iterrows():
        current=int(row["round"])
        for n in NUMBERS:
            gap=row[f"gap_{n}"]
            if pd.isna(gap): continue
            if gap>=threshold and not active[n]:
                recovery=None
                for future in order[idx:idx+horizon+1]:
                    if n in by_round.get(int(future),set()):
                        recovery=int(future)-current; break
                events.append({
                    "number":n,"entry_round":current,"entry_gap":int(gap),
                    "recovered_within_horizon":recovery is not None,
                    "recovery_after_rounds":recovery,
                })
                active[n]=True
            if n in by_round.get(current,set()): active[n]=False
    return pd.DataFrame(events)

def build_dna(df):
    _,gr=gap_tables(df); rows=[]
    for i,row in df.iterrows():
        x=nums(row); prev=nums(df.iloc[i-1]) if i>0 else []
        prev_bonus=int(df.iloc[i-1]["bonus"]) if i>0 else None
        z=zones(x); g=gr.iloc[i]
        rows.append({
            "round":int(row["round"]),"sum":sum(x),"end_digit_sum":end_sum(x),
            "odd_count":odd_count(x),"low_count":low_count(x),"prime_count":prime_count(x),
            "ac":ac(x),"consecutive_pairs":consec_pairs(x),
            "used_zones":sum(v>0 for v in z),"missing_zones":sum(v==0 for v in z),
            "zone_1_10":z[0],"zone_11_20":z[1],"zone_21_30":z[2],
            "zone_31_40":z[3],"zone_41_45":z[4],
            "carryover_count":len(set(x)&set(prev)),
            "neighbor_count":len(set(x)&neighbors(prev)) if prev else 0,
            "bonus_window_count":sum(abs(n-prev_bonus)<=2 for n in x) if prev_bonus is not None else 0,
            "gap_sum":g["gap_sum"],"gap_mean":g["gap_mean"],"gap_max":g["gap_max"],
            "gap_std":g["gap_std"],"gap_le_5":g["gap_le_5"],"gap_le_10":g["gap_le_10"],
            "gap_gt_10":g["gap_gt_10"],"gap_ge_17":g["gap_ge_17"],
        })
    out=pd.DataFrame(rows)
    out["gap_sum_ma5"]=out["gap_sum"].rolling(5,min_periods=3).mean()
    out["gap_sum_ma10"]=out["gap_sum"].rolling(10,min_periods=5).mean()
    return out

DNA_FEATURES=[
    "sum","end_digit_sum","odd_count","low_count","prime_count","ac","consecutive_pairs",
    "used_zones","missing_zones","zone_1_10","zone_11_20","zone_21_30","zone_31_40","zone_41_45",
    "carryover_count","neighbor_count","bonus_window_count","gap_sum","gap_mean","gap_max","gap_std",
    "gap_le_5","gap_le_10","gap_gt_10","gap_ge_17",
]

def classify_states(dna):
    out=dna.copy(); q25=out["gap_sum"].quantile(.25); q75=out["gap_sum"].quantile(.75); q90=out["gap_sum"].quantile(.90)
    def f(r):
        if r["gap_sum"]<=q25 and r["gap_max"]<=10:return "COMPRESSION"
        if r["gap_sum"]>=q90 or r["gap_max"]>=17:return "EXTREME_EXPANSION"
        if r["gap_sum"]>=q75:return "EXPANSION"
        return "NORMAL"
    out["gap_state"]=out.apply(f,axis=1); return out

def cec_drc(dna):
    s=classify_states(dna); rows=[]
    for i in range(len(s)-1):
        a,b=s.iloc[i],s.iloc[i+1]
        ce=a["gap_state"]=="COMPRESSION"; de=a["gap_state"]=="EXTREME_EXPANSION"
        rows.append({
            "round":int(a["round"]),"next_round":int(b["round"]),"state":a["gap_state"],
            "cec_event":ce,"cec_success":bool(b["gap_sum"]>a["gap_sum"] or b["gap_max"]>10) if ce else np.nan,
            "drc_event":de,"drc_success":bool(b["gap_sum"]<a["gap_sum"] or b["gap_le_10"]>a["gap_le_10"]) if de else np.nan,
        })
    return pd.DataFrame(rows)

def similar_rounds(dna,k=15):
    clean=dna.dropna(subset=DNA_FEATURES).reset_index(drop=True)
    if len(clean)<8:return pd.DataFrame()
    hist,target=clean.iloc[:-1],clean.iloc[[-1]]
    scaler=StandardScaler(); x=scaler.fit_transform(hist[DNA_FEATURES]); y=scaler.transform(target[DNA_FEATURES])
    model=NearestNeighbors(n_neighbors=min(k,len(hist)),metric="euclidean").fit(x)
    dist,idx=model.kneighbors(y); rows=[]
    for d,i in zip(dist[0],idx[0]):
        if int(i)+1>=len(clean):continue
        a=hist.iloc[int(i)]; b=clean.iloc[int(i)+1]
        rows.append({
            "similar_round":int(a["round"]),"distance":float(d),"next_round":int(b["round"]),
            "next_sum":float(b["sum"]),"next_gap_sum":float(b["gap_sum"]),
            "next_carryover":int(b["carryover_count"]),"next_neighbor":int(b["neighbor_count"]),
            "next_missing_zones":int(b["missing_zones"]),"next_ac":int(b["ac"]),
        })
    return pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)

def pair_counts(df):
    out={}
    for _,row in df.iterrows():
        for p in combinations(sorted(nums(row)),2): out[p]=out.get(p,0)+1
    return out

def number_scores(df,weights=None,egr_threshold=17,similarity_k=15):
    weights=weights or WEIGHTS
    recent30=df.tail(min(30,len(df))); recent100=df.tail(min(100,len(df)))
    long=pd.Series(0,index=NUMBERS,dtype=float); short=pd.Series(0,index=NUMBERS,dtype=float)
    for _,row in df.iterrows():
        for n in nums(row): long[n]+=1
    for _,row in recent30.iterrows():
        for n in nums(row): short[n]+=1
    long_n,short_n=minmax(long),minmax(short)
    gaps=current_gaps(df).set_index("number")
    latest=set(nums(df.iloc[-1])); neigh=neighbors(latest); bonus=int(df.iloc[-1]["bonus"])
    pc=pair_counts(recent100)
    pair_raw={n:sum(v for p,v in pc.items() if n in p) for n in NUMBERS}; max_pair=max(pair_raw.values()) or 1
    dna=build_dna(df); sim=similar_rounds(dna,similarity_k); follow={n:0. for n in NUMBERS}
    if not sim.empty:
        by_round={int(r["round"]):nums(r) for _,r in df.iterrows()}
        for _,item in sim.iterrows():
            w=1/(1+float(item["distance"]))
            for n in by_round.get(int(item["next_round"]),[]): follow[n]+=w
    max_follow=max(follow.values()) or 1
    state=classify_states(dna).iloc[-1]["gap_state"]; latest_z=zones(list(latest))
    rows=[]
    for n in NUMBERS:
        cg=int(gaps.loc[n,"current_gap"])
        egr=1. if cg>=egr_threshold else .6 if cg>=egr_threshold-5 else .2
        state_score=.5
        if state=="COMPRESSION" and cg>10:state_score=1.
        if state=="EXTREME_EXPANSION" and cg<=10:state_score=1.
        comp={
            "long_freq":float(long_n.loc[n]),"short_freq":float(short_n.loc[n]),
            "gap":float(gaps.loc[n,"gap_percentile"]),"egr":egr,
            "carry":1. if n in latest else .25,"neighbor":1. if n in neigh else .25,
            "bonus":1. if abs(n-bonus)<=2 else .25,"pair":pair_raw[n]/max_pair,
            "zone":1. if latest_z[zidx(n)]==0 else .5,"dna":follow[n]/max_follow,
            "state":state_score,"hot_cold":.75 if short[n]<=3 else .45,
        }
        total=sum(weights.values()); score=sum(weights[k]*comp[k] for k in weights)/total
        rows.append({"number":n,"final_score":score,"current_gap":cg,**{f"score_{k}":v for k,v in comp.items()}})
    return pd.DataFrame(rows).sort_values(["final_score","number"],ascending=[False,True]).reset_index(drop=True)

def hard_filter(c):
    x=sorted(c); z=zones(x)
    return (
        80<=sum(x)<=190 and 10<=end_sum(x)<=48 and odd_count(x) not in (0,1,5,6)
        and low_count(x) not in (0,6) and 4<=ac(x)<=12 and max_run(x)<4
        and max(z)<5 and sum(v>0 for v in z)>=3
    )

def score_combo(c,scores,df):
    x=tuple(sorted(c)); smap=dict(zip(scores["number"],scores["final_score"]))
    z=zones(x); latest=set(nums(df.iloc[-1])); total=sum(x); es=end_sum(x); o=odd_count(x); l=low_count(x); av=ac(x)
    carry=len(set(x)&latest); neigh=len(set(x)&neighbors(latest)); missing=sum(v==0 for v in z)
    score=float(np.mean([smap[n] for n in x]))
    score+=.08 if 125<=total<=165 else .02
    score+=.06 if 22<=es<=36 else .01
    score+=.06 if o in (2,3,4) else 0
    score+=.06 if l in (2,3,4) else 0
    score+=.07 if av in (7,8,9,10) else .02
    score+=.05 if missing in (0,1) else .01
    score+=.05 if carry in (0,1,2) else .01
    score+=.05 if neigh in (1,2,3) else .01
    return {
        "combination":x,"final_score":score,"sum":total,"end_digit_sum":es,
        "odd_count":o,"low_count":l,"ac":av,"prime_count":prime_count(x),
        "consecutive_pairs":consec_pairs(x),"zone_pattern":"-".join(map(str,z)),
        "missing_zones":missing,"carryover_count":carry,"neighbor_count":neigh,
    }

def generate_ranked(df,scores,candidate_count=18,limit=180000,seed=SEED):
    cand=scores.head(candidate_count)["number"].astype(int).tolist()
    total=math.comb(len(cand),6)
    if total<=limit: combos=list(combinations(cand,6))
    else:
        rng=random.Random(seed); pool=set()
        while len(pool)<limit: pool.add(tuple(sorted(rng.sample(cand,6))))
        combos=list(pool)
    rows=[score_combo(c,scores,df) for c in combos if hard_filter(c)]
    return pd.DataFrame(rows).sort_values(["final_score","combination"],ascending=[False,True]).reset_index(drop=True) if rows else pd.DataFrame()

def jaccard(a,b):
    sa,sb=set(a),set(b); return len(sa&sb)/len(sa|sb)

def portfolio(ranked,size=20,max_j=.5):
    chosen=[]
    for _,row in ranked.iterrows():
        c=tuple(row["combination"])
        if all(jaccard(c,x["combination"])<=max_j for x in chosen): chosen.append(row.to_dict())
        if len(chosen)>=size:break
    if len(chosen)<size:
        seen={tuple(x["combination"]) for x in chosen}
        for _,row in ranked.iterrows():
            c=tuple(row["combination"])
            if c not in seen: chosen.append(row.to_dict()); seen.add(c)
            if len(chosen)>=size:break
    return pd.DataFrame(chosen)

def strategies(ranked,size=20):
    if ranked.empty:return {"안정형":ranked,"균형형":ranked,"공격형":ranked}
    stable=ranked[ranked["sum"].between(125,160)&ranked["ac"].between(7,10)&ranked["odd_count"].between(2,4)&ranked["low_count"].between(2,4)]
    aggressive=ranked.sort_values(["neighbor_count","carryover_count","final_score"],ascending=[False,True,False])
    return {
        "안정형":portfolio(stable if not stable.empty else ranked,size,.55),
        "균형형":portfolio(ranked,size,.50),
        "공격형":portfolio(aggressive,size,.45),
    }

def walk_forward(df,rounds=50,train_window=300,candidate_count=18,top_combos=10,seed=SEED,weights=None):
    if len(df)<80: raise ValueError("최소 80회 데이터가 필요합니다.")
    start=max(60,len(df)-rounds); targets=list(range(start,len(df))); rows=[]
    progress=st.progress(0.,text="백테스트 준비 중...")
    for pos,idx in enumerate(targets):
        train=df.iloc[max(0,idx-train_window):idx].reset_index(drop=True)
        actual=set(nums(df.iloc[idx])); r=int(df.iloc[idx]["round"])
        scores=number_scores(train,weights=weights,similarity_k=min(10,max(3,len(train)//20)))
        rec={"round":r}
        for count in (11,13,15):
            rec[f"candidate_{count}_hits"]=len(actual&set(scores.head(count)["number"].astype(int)))
        ranked=generate_ranked(train,scores,min(candidate_count,18),12000,seed+r)
        top=portfolio(ranked,top_combos,.55)
        hits=[len(actual&set(c)) for c in top.get("combination",[])]
        mh=max(hits) if hits else 0
        rec.update({"top_combo_max_hit":mh,"top_combo_3plus":int(mh>=3),"top_combo_4plus":int(mh>=4),"top_combo_5plus":int(mh>=5),"top_combo_6":int(mh>=6)})
        rows.append(rec); progress.progress((pos+1)/len(targets),text=f"{r}회 검증 중...")
    progress.empty(); return pd.DataFrame(rows)

def summary(result):
    m={
        "검증 회차 수":len(result),
        "후보 11수 평균 적중":result["candidate_11_hits"].mean(),
        "후보 13수 평균 적중":result["candidate_13_hits"].mean(),
        "후보 15수 평균 적중":result["candidate_15_hits"].mean(),
        "TOP조합 평균 최고 적중":result["top_combo_max_hit"].mean(),
        "TOP조합 3개 이상 비율":result["top_combo_3plus"].mean(),
        "TOP조합 4개 이상 비율":result["top_combo_4plus"].mean(),
        "TOP조합 5개 이상 비율":result["top_combo_5plus"].mean(),
        "TOP조합 6개 비율":result["top_combo_6"].mean(),
    }
    return pd.DataFrame([{"지표":k,"값":v} for k,v in m.items()])

def optimize(df,trials=5,rounds=30,seed=SEED):
    rng=np.random.default_rng(seed); best=dict(WEIGHTS); best_obj=-np.inf; hist=[]
    for t in range(trials):
        proposal={k:max(.001,v*float(rng.uniform(.75,1.25))) for k,v in WEIGHTS.items()}
        result=walk_forward(df,rounds=rounds,candidate_count=16,top_combos=5,seed=seed+t,weights=proposal)
        obj=result["candidate_11_hits"].mean()*.35+result["candidate_15_hits"].mean()*.25+result["top_combo_max_hit"].mean()*.40
        hist.append({"trial":t+1,"objective":obj,**proposal})
        if obj>best_obj: best_obj=obj; best=proposal
    return best,pd.DataFrame(hist).sort_values("objective",ascending=False)

def csv_bytes(df): return df.to_csv(index=False).encode("utf-8-sig")
def self_test():
    tests=[]
    for name,fn in [
        ("A/C 계산",lambda:ac([1,7,13,22,34,45])),
        ("구간 계산",lambda:zones([1,11,21,31,41,45])),
        ("연속수 계산",lambda:max_run([1,2,3,10,20,30])),
        ("끝수합 계산",lambda:end_sum([7,13,22,29,34,40])),
    ]:
        try: fn(); tests.append({"검사":name,"결과":"통과","상세":""})
        except Exception as e: tests.append({"검사":name,"결과":"실패","상세":str(e)})
    return pd.DataFrame(tests)

st.set_page_config(page_title="Lotto64 Ultimate AI",page_icon="🍀",layout="wide")
st.title("🍀 Lotto64 Ultimate AI")
st.caption("GAP·EGR·CEC·DRC·회차 DNA·유사 회차·Walk-forward 백테스트 통합")
st.warning("로또는 무작위 추첨입니다. 이 앱은 당첨을 보장하지 않는 연구용 분석 도구입니다.")

with st.sidebar:
    st.header("설정")
    recent_window=st.slider("분석 회차",100,500,300,10)
    backtest_rounds=st.slider("백테스트 회차",20,100,50,10)
    candidate_count=st.slider("후보 번호 수",15,24,18)
    similarity_k=st.slider("유사 회차 수",5,30,15)
    egr_threshold=st.slider("EGR 임계 GAP",12,25,17)
    egr_horizon=st.slider("EGR 관찰 기간",2,8,4)
    seed=st.number_input("난수 시드",value=SEED,step=1)
    uploaded=st.file_uploader("CSV 또는 JSON",type=["csv","json"])

try: raw,source=read_data(uploaded)
except Exception as e: st.error(f"파일 오류: {e}"); st.stop()
if raw is None:
    st.info("CSV/JSON을 업로드하거나 data/lotto_all.csv에 데이터를 넣어 주세요.")
    st.code("round,date,n1,n2,n3,n4,n5,n6,bonus\n1,2002-12-07,10,23,29,33,37,40,16",language="csv")
    st.stop()
try: cleaned,notes=validate(raw)
except Exception as e: st.error(f"검증 오류: {e}"); st.stop()

analysis=cleaned.tail(min(recent_window,len(cleaned))).reset_index(drop=True)
st.success(f"{source} / 전체 {len(cleaned)}회 / 분석 {len(analysis)}회")

tabs=st.tabs(["데이터 진단","GAP·EGR","CEC·DRC","회차 DNA","후보 번호","TOP20","Walk-forward","가중치 최적화"])

with tabs[0]:
    for n in notes: st.write(f"- {n}")
    a,b,c=st.columns(3); a.metric("최초 회차",int(cleaned["round"].min())); b.metric("최신 회차",int(cleaned["round"].max())); c.metric("회차 수",len(cleaned))
    st.dataframe(cleaned.tail(20),use_container_width=True)
    st.dataframe(self_test(),use_container_width=True)
    st.download_button("정리 데이터 다운로드",csv_bytes(cleaned),"lotto_all_cleaned.csv")

with tabs[1]:
    dist=gap_distribution(analysis); st.bar_chart(dist.set_index("gap")["count"])
    st.dataframe(current_gaps(analysis).sort_values("current_gap",ascending=False),use_container_width=True)
    egr=egr_backtest(analysis,egr_threshold,egr_horizon)
    if egr.empty: st.info("EGR 사건 없음")
    else:
        a,b,c=st.columns(3); a.metric("사건 수",len(egr)); b.metric("회복률",f"{egr['recovered_within_horizon'].mean():.1%}")
        mr=egr["recovery_after_rounds"].dropna().mean(); c.metric("평균 회복기간",f"{mr:.2f}회" if pd.notna(mr) else "자료 없음")
        st.dataframe(egr,use_container_width=True)

with tabs[2]:
    dna=build_dna(analysis); tr=cec_drc(dna); ce=tr[tr["cec_event"]]; de=tr[tr["drc_event"]]
    a,b=st.columns(2); a.metric("CEC 사건",len(ce)); a.metric("CEC 성공률",f"{ce['cec_success'].mean():.1%}" if len(ce) else "자료 없음")
    b.metric("DRC 사건",len(de)); b.metric("DRC 성공률",f"{de['drc_success'].mean():.1%}" if len(de) else "자료 없음")
    states=classify_states(dna); st.line_chart(states.set_index("round")[["gap_sum","gap_sum_ma5","gap_sum_ma10"]]); st.dataframe(states.tail(60),use_container_width=True)

with tabs[3]:
    dna=build_dna(analysis); st.dataframe(dna.tail(30),use_container_width=True)
    sim=similar_rounds(dna,similarity_k)
    if sim.empty: st.info("데이터 부족")
    else:
        st.dataframe(sim,use_container_width=True)
        st.dataframe(pd.DataFrame({"지표":["총합","GAP합","CarryOver","Neighbor","구간공백","A/C"],"평균":[sim["next_sum"].mean(),sim["next_gap_sum"].mean(),sim["next_carryover"].mean(),sim["next_neighbor"].mean(),sim["next_missing_zones"].mean(),sim["next_ac"].mean()]}),use_container_width=True)

with tabs[4]:
    scores=number_scores(analysis,egr_threshold=egr_threshold,similarity_k=similarity_k)
    a,b,c=st.columns(3); a.code(", ".join(map(str,scores.head(15)["number"]))); b.code(", ".join(map(str,scores.head(13)["number"]))); c.code(", ".join(map(str,scores.head(11)["number"])))
    scores["grade"]=pd.qcut(scores["final_score"].rank(method="first"),5,labels=["E","D","C","B","A"])
    st.dataframe(scores,use_container_width=True); st.download_button("점수 다운로드",csv_bytes(scores),"number_scores.csv")

with tabs[5]:
    scores=number_scores(analysis,egr_threshold=egr_threshold,similarity_k=similarity_k)
    with st.spinner("조합 생성 중..."):
        ranked=generate_ranked(analysis,scores,candidate_count,180000,int(seed)); sets=strategies(ranked,20)
    for name,data in sets.items():
        st.subheader(name)
        if data.empty: st.warning("조합 없음"); continue
        d=data.copy(); d["combination"]=d["combination"].apply(lambda x:" ".join(map(str,x)))
        st.dataframe(d,use_container_width=True); st.download_button(f"{name} 다운로드",csv_bytes(d),f"top20_{name}.csv",key=name)
    if not sets["균형형"].empty:
        for count in (5,10,20):
            d=sets["균형형"].head(count).copy(); d["combination"]=d["combination"].apply(lambda x:" ".join(map(str,x)))
            st.write(f"**{count}게임**"); st.dataframe(d[["combination","final_score"]],use_container_width=True)

with tabs[6]:
    if st.button("Walk-forward 실행",type="primary"):
        try:
            result=walk_forward(cleaned,min(backtest_rounds,max(1,len(cleaned)-60)),recent_window,min(candidate_count,18),10,int(seed))
            st.dataframe(summary(result),use_container_width=True); st.dataframe(result,use_container_width=True)
            st.download_button("백테스트 다운로드",csv_bytes(result),"walk_forward.csv")
        except Exception as e: st.error(str(e))

with tabs[7]:
    trials=st.slider("탐색 횟수",3,12,5)
    if st.button("가중치 최적화 실행"):
        try:
            best,history=optimize(cleaned,trials,min(30,backtest_rounds),int(seed))
            st.dataframe(pd.DataFrame([{"feature":k,"weight":v} for k,v in best.items()]),use_container_width=True)
            st.dataframe(history,use_container_width=True)
            st.download_button("가중치 JSON",json.dumps(best,ensure_ascii=False,indent=2).encode(),"best_weights.json","application/json")
        except Exception as e: st.error(str(e))
