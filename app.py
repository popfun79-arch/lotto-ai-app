from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lotto64.analysis.dna import build_dna, classify_states
from lotto64.analysis.egr import egr_backtest
from lotto64.analysis.gap import current_gap_table, gap_distribution
from lotto64.analysis.similarity import similar_rounds
from lotto64.analysis.state import cec_drc_backtest
from lotto64.analysis.sum_series import build_sum_series, compare_sum_windows, forecast_next_sum
from lotto64.backtest.walk_forward import summarize, walk_forward
from lotto64.config import RunConfig
from lotto64.data.storage import load_best_available, sync_sqlite, write_csv
from lotto64.data.updater import update_latest
from lotto64.models.bayesian import bayesian_style_weight_search
from lotto64.models.scoring import explain_number, number_scores
from lotto64.recommend.ga import ga_optimize
from lotto64.recommend.portfolio import build_portfolio
from lotto64.recommend.final_pattern import final_recommendation_bundle
from lotto64.recommend.top_of_best import build_top_of_best_sets
from lotto64.reports.reporting import build_backtest_report, csv_bytes, json_bytes

st.set_page_config(page_title="Lotto64 v4.1 Top of the Best", page_icon="🍀", layout="wide")
st.title("🍀 Lotto64 Ultimate AI v4.1 · Top of the Best")
st.caption("데이터 · 합계 시계열 · GAP · EGR · CEC/DRC · 회차 DNA · GA · Walk-forward")
st.warning("로또는 무작위 추첨입니다. 이 앱은 당첨을 보장하지 않는 연구·검증 도구입니다.")

with st.sidebar:
    st.header("설정")
    recent_window = st.slider("분석 회차", 100, 500, 300, 10)
    backtest_rounds = st.slider("백테스트 회차", 20, 100, 50, 10)
    candidate_count = st.slider("후보 번호 수", 15, 24, 18)
    similarity_k = st.slider("유사 회차 수", 5, 30, 15)
    egr_threshold = st.slider("EGR 임계 GAP", 12, 25, 17)
    egr_horizon = st.slider("EGR 관찰 기간", 2, 8, 4)
    seed = st.number_input("난수 시드", value=20260720, step=1)
    uploaded = st.file_uploader("CSV 또는 JSON", type=["csv", "json"])

    st.divider()
    if st.button("최신 회차 자동 업데이트"):
        with st.spinner("공식 데이터 업데이트를 시도합니다..."):
            updated_df, new_count = update_latest()
        st.success(f"신규 {new_count}회 반영 / 현재 {len(updated_df)}회")
        st.cache_data.clear()
        st.rerun()

config = RunConfig(
    recent_window=recent_window,
    backtest_rounds=backtest_rounds,
    candidate_count=candidate_count,
    similarity_k=similarity_k,
    egr_threshold=egr_threshold,
    egr_horizon=egr_horizon,
    seed=int(seed),
)

try:
    cleaned, source, notes = load_best_available(uploaded)
except Exception as exc:
    st.error(f"데이터 로딩 오류: {exc}")
    st.stop()

analysis = cleaned.tail(min(recent_window, len(cleaned))).reset_index(drop=True)
st.success(f"{source} / 전체 {len(cleaned)}회 / 분석 {len(analysis)}회")

# Final Pattern 통합 엔진은 앱 전체에서 한 번만 계산합니다.
# Top of the Best와 Final Pattern이 동일한 최종 포트폴리오를 공유합니다.
with st.spinner("통합 Final Pattern 엔진 계산 중..."):
    final_bundle = final_recommendation_bundle(analysis)

top_of_best_sets = build_top_of_best_sets(final_bundle["portfolio"])

tabs = st.tabs([
    "데이터 진단", "합계 시계열", "GAP·EGR", "CEC·DRC", "회차 DNA",
    "후보 번호", "Top of the Best", "GA 최적화", "Walk-forward", "가중치 탐색",
    "설명", "Final Pattern",
])

with tabs[0]:
    for note in notes:
        st.write(f"- {note}")
    a, b, c = st.columns(3)
    a.metric("최초 회차", int(cleaned["round"].min()))
    b.metric("최신 회차", int(cleaned["round"].max()))
    c.metric("회차 수", len(cleaned))
    st.dataframe(cleaned.tail(20), use_container_width=True)

    col1, col2 = st.columns(2)
    if col1.button("CSV 정리본 저장"):
        write_csv(cleaned)
        st.success("data/lotto_all.csv 저장 완료")
    if col2.button("SQLite 동기화"):
        sync_sqlite(cleaned)
        st.success("data/lotto64.db 동기화 완료")

    st.download_button("정리 데이터 다운로드", csv_bytes(cleaned), "lotto_all_cleaned.csv")

with tabs[1]:
    st.subheader("회차별 조합 번호 합계 시계열")
    sum_series = build_sum_series(analysis, state_window=50)
    forecast = forecast_next_sum(
        analysis,
        state_window=50,
        transition_lookback=min(100, max(30, len(analysis) - 1)),
    )

    a, b, c, d = st.columns(4)
    a.metric("최근 당첨 합계", forecast.current_sum)
    b.metric("현재 합계 상태", forecast.current_state)
    c.metric("다음 합계 중심값", f"{forecast.target_center:.1f}")
    d.metric(
        "다음 핵심 구간",
        f"{forecast.target_low:.0f}~{forecast.target_high:.0f}",
    )

    st.caption(
        f"확장 구간 {forecast.wide_low:.0f}~{forecast.wide_high:.0f} · "
        f"동일 상태 전이 표본 {forecast.matched_transitions}회 · "
        "미래 데이터 없이 이전 회차만 사용"
    )

    chart = sum_series.tail(min(100, len(sum_series))).set_index("round")[
        ["sum", "ma5", "ma10", "ma20"]
    ]
    st.line_chart(chart)

    compare = compare_sum_windows(analysis, window=50)
    if not compare.empty:
        st.markdown("#### 최근 50회 vs 이전 50회")
        st.dataframe(compare, use_container_width=True)

    st.markdown("#### 최근 합계 상태")
    st.dataframe(
        sum_series.tail(30)[
            [
                "round", "sum", "delta1", "ma5", "ma10", "ma20",
                "std20", "z20", "sum_state",
            ]
        ],
        use_container_width=True,
    )

with tabs[2]:
    dist = gap_distribution(analysis)
    st.bar_chart(dist.set_index("gap")["count"])
    st.dataframe(current_gap_table(analysis).sort_values("current_gap", ascending=False), use_container_width=True)

    egr = egr_backtest(analysis, egr_threshold, egr_horizon)
    if egr.empty:
        st.info("EGR 사건 없음")
    else:
        a, b, c = st.columns(3)
        a.metric("EGR 사건", len(egr))
        b.metric("회복률", f"{egr['recovered_within_horizon'].mean():.1%}")
        mean_recovery = egr["recovery_after_rounds"].dropna().mean()
        c.metric("평균 회복 기간", f"{mean_recovery:.2f}회" if pd.notna(mean_recovery) else "자료 없음")
        st.dataframe(egr, use_container_width=True)

with tabs[3]:
    dna = build_dna(analysis)
    transitions = cec_drc_backtest(dna)
    cec = transitions[transitions["cec_event"]]
    drc = transitions[transitions["drc_event"]]

    a, b = st.columns(2)
    a.metric("CEC 사건", len(cec))
    a.metric("CEC 성공률", f"{cec['cec_success'].mean():.1%}" if len(cec) else "자료 없음")
    b.metric("DRC 사건", len(drc))
    b.metric("DRC 성공률", f"{drc['drc_success'].mean():.1%}" if len(drc) else "자료 없음")

    states = classify_states(dna)
    st.line_chart(states.set_index("round")[["gap_sum", "gap_sum_ma5", "gap_sum_ma10"]])
    st.dataframe(states.tail(60), use_container_width=True)

with tabs[4]:
    dna = build_dna(analysis)
    st.dataframe(dna.tail(30), use_container_width=True)
    sim = similar_rounds(dna, similarity_k)
    if sim.empty:
        st.info("유사 회차 데이터 부족")
    else:
        st.dataframe(sim, use_container_width=True)

with tabs[5]:
    scores = number_scores(
        analysis,
        egr_threshold=egr_threshold,
        similarity_k=similarity_k,
    )
    a, b, c = st.columns(3)
    a.markdown("#### 후보 15수")
    a.code(", ".join(map(str, scores.head(15)["number"])))
    b.markdown("#### 후보 13수")
    b.code(", ".join(map(str, scores.head(13)["number"])))
    c.markdown("#### 후보 11수")
    c.code(", ".join(map(str, scores.head(11)["number"])))

    st.caption("상대확률은 모델 내부 점수의 상대 비중이며 실제 당첨확률이 아닙니다.")
    st.dataframe(scores, use_container_width=True)
    st.download_button("번호 점수 다운로드", csv_bytes(scores), "number_scores.csv")

with tabs[6]:
    st.subheader("🏆 Top of the Best")
    st.caption(
        "기존 안정형·균형형·공격형 3종 TOP20(최대 60개)을 사용하지 않습니다. "
        "Final Pattern의 단일 통합 점수로 최종 20조합만 선정합니다."
    )
    st.info(
        "BEST 5 ⊂ BEST 10 ⊂ BEST 15 ⊂ BEST 20 입니다. "
        "실제 고유 추천은 최대 20조합이며, 5·10·15는 같은 Top20의 상위 구간입니다."
    )

    final_context = final_bundle["context"]
    sum_fc = final_context["sum_forecast"]
    gap_fc = final_context["gap_sum_forecast"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "번호합 핵심구간",
        f"{sum_fc['target_low']:.0f}~{sum_fc['target_high']:.0f}",
    )
    m2.metric("번호합 중심", f"{sum_fc['target_center']:.0f}")
    m3.metric(
        "GAP합 핵심구간",
        f"{gap_fc['target_low']:.0f}~{gap_fc['target_high']:.0f}",
    )
    m4.metric("GAP합 중심", f"{gap_fc['target_center']:.0f}")

    size_tabs = st.tabs(["BEST 5", "BEST 10", "BEST 15", "BEST 20"])

    for view_tab, size in zip(size_tabs, (5, 10, 15, 20)):
        with view_tab:
            selected = top_of_best_sets.get(
                size,
                pd.DataFrame(),
            ).copy()

            if selected.empty:
                st.warning("추천 가능한 조합이 없습니다.")
                continue

            selected["combination"] = selected["combination"].apply(
                lambda values: " ".join(map(str, values))
            )

            st.markdown(f"### Top of the Best {size}조합")
            st.dataframe(
                selected,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                f"Top of the Best {size}조합 CSV",
                csv_bytes(selected),
                f"top_of_the_best_{size}.csv",
                key=f"top_of_best_{size}",
            )

    st.markdown("#### 최종 점수에 반영되는 주요 구조")
    st.write(
        "Pattern Master 번호점수 + 번호합 시계열 + GAP합 시계열 + "
        "GAP구간 구성 + 홀짝/저고 + 번호구간 + 끝수 + AC + "
        "이월수 + 조합 간 중복/번호 노출 분산을 함께 반영합니다."
    )

with tabs[7]:
    st.subheader("유전자 알고리즘 조합 최적화")
    population = st.slider("개체 수", 100, 1000, 400, 100)
    generations = st.slider("세대 수", 5, 60, 25, 5)

    if st.button("GA 실행"):
        scores = number_scores(analysis, egr_threshold=egr_threshold, similarity_k=similarity_k)
        with st.spinner("GA 진화 중..."):
            ga_ranked = ga_optimize(
                analysis,
                scores,
                candidate_count=max(candidate_count, 18),
                population_size=population,
                generations=generations,
                seed=int(seed),
            )
        ga_top = build_portfolio(ga_ranked, size=20, max_jaccard=0.50)
        display = ga_top.copy()
        if not display.empty:
            display["combination"] = display["combination"].apply(lambda x: " ".join(map(str, x)))
            st.dataframe(display, use_container_width=True)
            st.download_button("GA TOP20 다운로드", csv_bytes(display), "ga_top20.csv")

with tabs[8]:
    st.subheader("Walk-forward 백테스트")
    if "walk_result" not in st.session_state:
        st.session_state["walk_result"] = None

    if st.button("Walk-forward 실행", type="primary"):
        progress = st.progress(0.0, text="백테스트 준비 중...")
        def callback(value: float, text: str):
            progress.progress(value, text=text)

        try:
            result = walk_forward(
                cleaned,
                rounds=min(backtest_rounds, max(1, len(cleaned) - 60)),
                train_window=recent_window,
                candidate_count=min(candidate_count, 18),
                top_combos=10,
                seed=int(seed),
                progress_callback=callback,
            )
            st.session_state["walk_result"] = result
        finally:
            progress.empty()

    result = st.session_state.get("walk_result")
    if result is not None and not result.empty:
        st.dataframe(summarize(result), use_container_width=True)
        st.line_chart(result.set_index("round")[[
            "candidate_11_hits", "candidate_13_hits",
            "candidate_15_hits", "top_combo_max_hit",
        ]])

        st.markdown("#### 합계 시계열 예측 검증")
        st.line_chart(result.set_index("round")[[
            "actual_sum", "predicted_sum_center",
            "predicted_sum_low", "predicted_sum_high",
        ]])
        sa, sb, sc = st.columns(3)
        sa.metric("합계 중심 MAE", f"{result['sum_abs_error'].mean():.2f}")
        sb.metric("핵심 구간 적중률", f"{result['sum_in_core_band'].mean():.1%}")
        sc.metric("확장 구간 적중률", f"{result['sum_in_wide_band'].mean():.1%}")
        cumulative = result[[
            "top_combo_3plus", "top_combo_4plus",
            "top_combo_5plus", "top_combo_6",
        ]].expanding().mean()
        cumulative.index = result["round"]
        st.line_chart(cumulative)
        st.dataframe(result, use_container_width=True)

        report = build_backtest_report(result, {
            **config.to_dict(),
            "latest_round": int(cleaned["round"].max()),
            "data_rows": len(cleaned),
        })
        a, b = st.columns(2)
        a.download_button("백테스트 CSV", csv_bytes(result), "walk_forward.csv")
        b.download_button("요약 JSON", json_bytes(report), "walk_forward_report.json", "application/json")

with tabs[9]:
    st.subheader("가중치 확률적 탐색")
    trials = st.slider("탐색 횟수", 3, 20, 6)

    if st.button("가중치 탐색 실행"):
        def objective(weights):
            result = walk_forward(
                cleaned,
                rounds=min(20, backtest_rounds),
                train_window=recent_window,
                candidate_count=16,
                top_combos=5,
                seed=int(seed),
                weights=weights,
            )
            return (
                result["candidate_11_hits"].mean() * 0.35
                + result["candidate_15_hits"].mean() * 0.25
                + result["top_combo_max_hit"].mean() * 0.40
            )

        with st.spinner("가중치 탐색 중..."):
            best, history = bayesian_style_weight_search(objective, trials=trials, seed=int(seed))

        st.dataframe(pd.DataFrame([{"feature": k, "weight": v} for k, v in best.items()]), use_container_width=True)
        st.dataframe(history, use_container_width=True)
        st.download_button(
            "최적 가중치 JSON",
            json.dumps(best, ensure_ascii=False, indent=2).encode("utf-8"),
            "best_weights.json",
            "application/json",
        )

with tabs[10]:
    st.subheader("번호 점수 설명")
    scores = number_scores(analysis, egr_threshold=egr_threshold, similarity_k=similarity_k)
    selected_number = st.selectbox("번호 선택", scores["number"].astype(int).tolist())
    row = scores[scores["number"] == selected_number].iloc[0]
    st.metric("최종 점수", f"{row['final_score']:.5f}")
    st.metric("상대확률 지표", f"{row['relative_probability_pct']:.2f}%")
    st.dataframe(explain_number(row), use_container_width=True)

st.divider()
st.caption("Lotto64 Ultimate AI v4.1 · Top of the Best 단일 통합 포트폴리오")



with tabs[11]:
    st.subheader("Final Pattern — Python + Games-Out/Skip + 합계 시계열")
    st.caption(
        "Drawings Since Hit·Skip/Hit·Skips Due·Number Groups·Last Digits·"
        "Odd/Even·High/Low 성격의 패턴과 Python 시계열/GAP/DNA를 결합합니다."
    )

    final_candidates = final_bundle["candidate_sets"]
    final_context = final_bundle["context"]
    final_portfolio = final_bundle["portfolio"].copy()

    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 후보 11수")
    c1.code(", ".join(map(str, final_candidates[11])))
    c2.markdown("#### 후보 13수")
    c2.code(", ".join(map(str, final_candidates[13])))
    c3.markdown("#### 후보 15수")
    c3.code(", ".join(map(str, final_candidates[15])))

    sum_fc = final_context["sum_forecast"]
    gap_fc = final_context["gap_sum_forecast"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("최근 번호합", f"{sum_fc['current_sum']:.0f}")
    s2.metric(
        "다음 번호합 핵심구간",
        f"{sum_fc['target_low']:.0f}~{sum_fc['target_high']:.0f}",
    )
    s3.metric("최근 GAP합", f"{gap_fc['current_gap_sum']:.0f}")
    s4.metric(
        "다음 GAP합 핵심구간",
        f"{gap_fc['target_low']:.0f}~{gap_fc['target_high']:.0f}",
    )

    if not final_portfolio.empty:
        final_portfolio["combination"] = final_portfolio["combination"].apply(
            lambda values: " ".join(map(str, values))
        )
        st.markdown("#### Top of the Best 20 조합")
        st.dataframe(final_portfolio, use_container_width=True)

        for size in (5, 10, 15, 20):
            st.download_button(
                f"Top of the Best {size}조합 CSV",
                csv_bytes(final_portfolio.head(size)),
                f"top_of_the_best_{size}_final_pattern.csv",
                key=f"final_top_{size}",
            )
