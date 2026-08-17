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
from lotto64.backtest.seed_stability import (
    ga_seed_stability,
    walk_forward_seed_stability,
)
from lotto64.config import RunConfig

try:
    from lotto64.config import DEFAULT_MULTI_SEED_COUNT, FIXED_SEED
except ImportError:
    FIXED_SEED = 20260720
    DEFAULT_MULTI_SEED_COUNT = 5
from lotto64.data.storage import load_best_available, sync_sqlite, write_csv
from lotto64.data.updater import update_latest
from lotto64.models.bayesian import bayesian_style_weight_search
from lotto64.models.scoring import explain_number, number_scores
from lotto64.recommend.ga import ga_optimize
from lotto64.recommend.portfolio import build_portfolio
from lotto64.recommend.final_pattern import final_recommendation_bundle
from lotto64.recommend.top_of_best import build_top_of_best_sets
from lotto64.reports.reporting import build_backtest_report, csv_bytes, json_bytes
from lotto64.backtest.historical_ledger import (
    build_historical_ledger,
    compare_previous_recent,
    cumulative_metrics,
    failure_counts,
    load_saved_ledger,
    max_strict_validation_rounds,
    required_history_start,
    save_ledger,
    summarize_ledger,
)

try:
    from lotto64.utils.lotto_math import NUMBER_GROUP_LABELS
except ImportError:
    # Partial-deployment safety.
    NUMBER_GROUP_LABELS = (
        "1~9", "10~19", "20~29", "30~39", "40~45"
    )

st.set_page_config(page_title="Lotto64 v4.5 Historical Validation Ledger", page_icon="🍀", layout="wide")
st.title("🍀 Lotto64 Ultimate AI v4.5 · Historical Validation Ledger")
st.caption("데이터 · 합계 시계열 · GAP · EGR · CEC/DRC · 회차 DNA · GA · Walk-forward")
st.warning("로또는 무작위 추첨입니다. 이 앱은 당첨을 보장하지 않는 연구·검증 도구입니다.")

with st.sidebar:
    st.header("설정")
    recent_window = st.slider(
        "분석 회차 (최신)",
        min_value=100,
        max_value=300,
        value=200,
        step=10,
        help="기본값은 최신 200회입니다. 필요할 때만 범위를 조정하세요.",
    )
    backtest_rounds = st.slider("백테스트 회차", 20, 100, 50, 10)
    candidate_count = st.slider("후보 번호 수", 15, 24, 18)
    similarity_k = st.slider("유사 회차 수", 5, 30, 15)
    egr_threshold = st.slider("EGR 임계 GAP", 12, 25, 17)
    egr_horizon = st.slider("EGR 관찰 기간", 2, 8, 4)
    seed = st.number_input(
        "고정 기준 Seed",
        value=FIXED_SEED,
        step=1,
        disabled=True,
        help=(
            "실전 추천과 기본 백테스트를 동일 조건으로 재현하기 위한 "
            "고정 Seed입니다. Seed 값 자체에는 예측력이 없습니다."
        ),
    )
    multi_seed_count = st.select_slider(
        "다중 Seed 검증 수",
        options=[3, 5, 7],
        value=DEFAULT_MULTI_SEED_COUNT,
        help=(
            "동일 데이터와 모델을 여러 Seed에서 반복해 랜덤 탐색의 "
            "민감도와 안정성을 확인합니다."
        ),
    )
    uploaded = st.file_uploader("CSV 또는 JSON", type=["csv", "json"])

    st.divider()
    if st.button("최신 회차 자동 업데이트"):
        with st.spinner("최신 데이터 업데이트를 확인합니다..."):
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
    seed=FIXED_SEED,
)

try:
    cleaned, source, notes = load_best_available(uploaded)
except Exception as exc:
    st.error(f"데이터 로딩 오류: {exc}")
    st.stop()

analysis = cleaned.tail(min(recent_window, len(cleaned))).reset_index(drop=True)
st.success(
    f"{source} / 전체 데이터 {len(cleaned)}회 / "
    f"현재 분석 최신 {len(analysis)}회"
)

# Final Pattern 통합 엔진은 앱 전체에서 한 번만 계산합니다.
# Top of the Best와 Final Pattern이 동일한 최종 포트폴리오를 공유합니다.
with st.spinner("통합 Final Pattern 엔진 계산 중..."):
    final_bundle = final_recommendation_bundle(analysis)

top_of_best_sets = build_top_of_best_sets(final_bundle["portfolio"])

tabs = st.tabs([
    "데이터 진단", "합계 시계열", "GAP·EGR", "CEC·DRC", "회차 DNA",
    "후보 번호", "Top of the Best", "GA 최적화", "Walk-forward", "가중치 탐색",
    "설명", "Final Pattern", "검증 원장",
])

with tabs[0]:
    for note in notes:
        st.write(f"- {note}")
    a, b, c = st.columns(3)
    a.metric("분석 시작 회차", int(analysis["round"].min()))
    b.metric("분석 최신 회차", int(analysis["round"].max()))
    c.metric("분석 회차 수", len(analysis))
    st.dataframe(analysis.tail(20), use_container_width=True)

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

    st.caption("Number Groups 적용 구간: " + " · ".join(NUMBER_GROUP_LABELS))

    st.caption(
        f"Top of the Best Final Pattern은 결정론적 계산이므로 Seed에 직접 "
        f"영향받지 않습니다. 고정 Seed {FIXED_SEED}와 다중 Seed 검증은 "
        "GA/랜덤 조합 탐색의 안정성 확인에 사용됩니다."
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
    st.caption(
        f"고정 기준 Seed = {FIXED_SEED}. 같은 데이터/설정이면 같은 "
        "GA 결과를 재현합니다."
    )

    population = st.slider("개체 수", 100, 1000, 400, 100)
    generations = st.slider("세대 수", 5, 60, 25, 5)

    if "ga_result" not in st.session_state:
        st.session_state["ga_result"] = None
    if "ga_seed_stability" not in st.session_state:
        st.session_state["ga_seed_stability"] = None

    if st.button("고정 Seed GA 실행"):
        scores = number_scores(
            analysis,
            egr_threshold=egr_threshold,
            similarity_k=similarity_k,
        )
        with st.spinner("고정 Seed GA 진화 중..."):
            ga_ranked = ga_optimize(
                analysis,
                scores,
                candidate_count=max(candidate_count, 18),
                population_size=population,
                generations=generations,
                seed=FIXED_SEED,
            )
        ga_top = build_portfolio(
            ga_ranked,
            size=20,
            max_jaccard=0.50,
        )
        st.session_state["ga_result"] = ga_top

    ga_top = st.session_state.get("ga_result")
    if ga_top is not None and not ga_top.empty:
        display = ga_top.copy()
        display["combination"] = display["combination"].apply(
            lambda x: " ".join(map(str, x))
        )
        st.markdown("#### 고정 Seed GA TOP20")
        st.dataframe(display, use_container_width=True)
        st.download_button(
            "고정 Seed GA TOP20 다운로드",
            csv_bytes(display),
            "ga_fixed_seed_top20.csv",
        )

    st.divider()
    st.markdown("### 다중 Seed 안정성 검증")
    st.info(
        "좋은 결과가 나온 Seed를 골라내는 기능이 아닙니다. "
        "같은 설정을 여러 Seed에서 반복해 어떤 조합/번호가 반복적으로 "
        "살아남는지 확인하는 검증 기능입니다."
    )

    if st.button("GA 다중 Seed 안정성 실행", type="primary"):
        scores = number_scores(
            analysis,
            egr_threshold=egr_threshold,
            similarity_k=similarity_k,
        )
        progress = st.progress(0.0, text="Seed 안정성 준비 중...")

        def ga_seed_callback(value: float, text: str):
            progress.progress(value, text=text)

        try:
            stability = ga_seed_stability(
                analysis,
                scores,
                candidate_count=max(candidate_count, 18),
                population_size=population,
                generations=generations,
                seed_count=int(multi_seed_count),
                top_n=20,
                base_seed=FIXED_SEED,
                progress_callback=ga_seed_callback,
            )
            st.session_state["ga_seed_stability"] = stability
        finally:
            progress.empty()

    stability = st.session_state.get("ga_seed_stability")
    if stability:
        metrics = stability.get("metrics", {})
        if metrics:
            a, b, c, d = st.columns(4)
            a.metric("검증 Seed 수", metrics["seed_count"])
            b.metric(
                "2개 Seed 이상 반복 조합",
                metrics["stable_combo_2plus"],
            )
            c.metric(
                "3개 Seed 이상 반복 조합",
                metrics["stable_combo_3plus"],
            )
            d.metric(
                "번호 노출 상관",
                f"{metrics['number_exposure_correlation']:.3f}",
            )

            st.caption(
                "조합 Jaccard가 낮아도 번호 노출 상관이 높으면 "
                "정확한 6개 조합은 달라도 핵심 번호 구조는 비슷하다는 뜻입니다."
            )

        consensus = stability.get("combo_consensus")
        if consensus is not None and not consensus.empty:
            st.markdown("#### Seed Consensus 조합")
            display = consensus.head(30).copy()
            display["combination"] = display["combination"].apply(
                lambda x: " ".join(map(str, x))
            )
            st.dataframe(display, use_container_width=True)
            st.download_button(
                "Seed Consensus 조합 CSV",
                csv_bytes(display),
                "ga_seed_consensus.csv",
            )

        number_stability = stability.get("number_stability")
        if number_stability is not None and not number_stability.empty:
            st.markdown("#### 번호별 Seed 안정성")
            st.dataframe(
                number_stability.head(45),
                use_container_width=True,
            )
            st.download_button(
                "번호 Seed 안정성 CSV",
                csv_bytes(number_stability),
                "number_seed_stability.csv",
            )

with tabs[8]:
    st.subheader("Walk-forward 백테스트")
    st.caption(
        f"기본 백테스트는 고정 Seed {FIXED_SEED}를 사용해 재현성을 유지합니다."
    )
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
                seed=FIXED_SEED,
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

    st.divider()
    st.markdown("### 다중 Seed Walk-forward 안정성")
    seed_validation_rounds = st.slider(
        "Seed 안정성 검증 회차",
        min_value=10,
        max_value=30,
        value=min(20, backtest_rounds),
        step=5,
        help=(
            "계산량을 고려해 최근 10~30회에서 여러 Seed를 반복합니다. "
            "기본 Walk-forward 50회와는 별도의 민감도 검증입니다."
        ),
    )

    if "multi_seed_walk" not in st.session_state:
        st.session_state["multi_seed_walk"] = None

    if st.button("다중 Seed Walk-forward 실행"):
        progress = st.progress(0.0, text="다중 Seed 백테스트 준비 중...")

        def seed_walk_callback(value: float, text: str):
            progress.progress(value, text=text)

        try:
            multi_result = walk_forward_seed_stability(
                cleaned,
                rounds=min(
                    seed_validation_rounds,
                    max(1, len(cleaned) - 60),
                ),
                train_window=recent_window,
                candidate_count=min(candidate_count, 18),
                top_combos=10,
                seed_count=int(multi_seed_count),
                base_seed=FIXED_SEED,
                progress_callback=seed_walk_callback,
            )
            st.session_state["multi_seed_walk"] = multi_result
        finally:
            progress.empty()

    multi_result = st.session_state.get("multi_seed_walk")
    if multi_result:
        metrics = multi_result.get("metrics", {})
        if metrics:
            a, b, c, d = st.columns(4)
            a.metric("Seed 수", metrics["seed_count"])
            b.metric(
                "평균 TOP조합 최고 적중",
                f"{metrics['mean_top_combo_hit']:.3f}",
            )
            c.metric(
                "회차별 Seed 완전 일치율",
                f"{metrics['all_seed_agreement_pct']:.1f}%",
            )
            d.metric(
                "평균 Seed 표준편차",
                f"{metrics['mean_seed_std']:.3f}",
            )

        by_seed = multi_result.get("by_seed")
        if by_seed is not None and not by_seed.empty:
            st.markdown("#### Seed별 성능")
            st.dataframe(by_seed, use_container_width=True)

        by_round = multi_result.get("by_round")
        if by_round is not None and not by_round.empty:
            st.markdown("#### 회차별 Seed 민감도")
            st.dataframe(by_round, use_container_width=True)
            st.line_chart(
                by_round.set_index("round")[
                    [
                        "top_combo_hit_mean",
                        "top_combo_hit_std",
                    ]
                ]
            )

        runs = multi_result.get("runs")
        if runs is not None and not runs.empty:
            st.download_button(
                "다중 Seed Walk-forward 전체 CSV",
                csv_bytes(runs),
                "walk_forward_multi_seed.csv",
            )

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
                seed=FIXED_SEED,
                weights=weights,
            )
            return (
                result["candidate_11_hits"].mean() * 0.35
                + result["candidate_15_hits"].mean() * 0.25
                + result["top_combo_max_hit"].mean() * 0.40
            )

        with st.spinner("가중치 탐색 중..."):
            best, history = bayesian_style_weight_search(objective, trials=trials, seed=FIXED_SEED)

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
st.caption("Lotto64 Ultimate AI v4.5 · Historical Validation Ledger · 엄격 200회 Rolling Walk-forward")



with tabs[11]:
    st.subheader("Final Pattern — Python + Games-Out/Skip + 합계 시계열")
    st.caption(
        "Drawings Since Hit·Skip/Hit·Skips Due·Number Groups·Last Digits·"
        "Odd/Even·High/Low 성격의 패턴과 Python 시계열/GAP/DNA를 결합합니다."
    )

    st.caption("Number Groups: " + " · ".join(NUMBER_GROUP_LABELS))

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


with tabs[12]:
    st.subheader("📒 Historical Validation Ledger")
    st.caption(
        "각 검증 회차의 실제 결과를 절대 학습에 포함하지 않고, "
        "직전 200회만 사용해 Final Pattern을 다시 계산합니다."
    )

    latest_round = int(cleaned["round"].max())
    earliest_round = int(cleaned["round"].min())
    strict_available = max_strict_validation_rounds(
        cleaned,
        train_window=200,
    )
    required_50 = required_history_start(
        latest_round,
        validation_rounds=50,
        train_window=200,
    )
    required_100 = required_history_start(
        latest_round,
        validation_rounds=100,
        train_window=200,
    )

    a, b, c, d = st.columns(4)
    a.metric("현재 최초 회차", earliest_round)
    b.metric("엄격 검증 가능", f"{strict_available}회")
    c.metric("최근 50회 필요 시작", required_50)
    d.metric("최근 100회 필요 시작", required_100)

    if strict_available < 100:
        st.warning(
            "현재 데이터만으로는 200회 학습창을 유지한 최근 100회 "
            "Historical Ledger를 만들 수 없습니다. "
            f"100회 검증에는 {required_100}회부터 데이터가 필요합니다. "
            "GitHub의 Weekly Lotto Data Update를 수동 실행하면 "
            "v4.5 workflow가 937회까지 과거 데이터를 백필하도록 설정되어 있습니다."
        )
    else:
        st.success(
            "최근 100회 엄격 Walk-forward를 실행할 충분한 과거 데이터가 있습니다."
        )

    saved_ledger = load_saved_ledger()

    if "historical_ledger" not in st.session_state:
        st.session_state["historical_ledger"] = (
            saved_ledger if not saved_ledger.empty else None
        )

    run_col1, run_col2 = st.columns(2)

    def run_ledger(rounds: int):
        progress = st.progress(
            0.0,
            text=f"최근 {rounds}회 Historical Ledger 준비 중...",
        )

        def ledger_callback(value: float, text: str):
            progress.progress(value, text=text)

        try:
            ledger_result = build_historical_ledger(
                cleaned,
                validation_rounds=rounds,
                train_window=200,
                progress_callback=ledger_callback,
            )
            st.session_state["historical_ledger"] = ledger_result
            save_ledger(ledger_result)
        except Exception as exc:
            st.error(f"Historical Ledger 실행 오류: {exc}")
        finally:
            progress.empty()

    if run_col1.button(
        "최근 50회 검증 원장 실행",
        disabled=strict_available < 50,
    ):
        run_ledger(50)

    if run_col2.button(
        "최근 100회 검증 원장 실행",
        type="primary",
        disabled=strict_available < 100,
    ):
        run_ledger(100)

    ledger = st.session_state.get("historical_ledger")

    if ledger is not None and not ledger.empty:
        summary = summarize_ledger(ledger)

        st.markdown(
            f"### 저장/현재 원장 · "
            f"{summary['start_round']}~{summary['end_round']}회 "
            f"({summary['rounds']}회)"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "후보15 평균 적중",
            f"{summary['candidate_15_mean']:.2f}",
        )
        m2.metric(
            "BEST20 평균 최고 적중",
            f"{summary['best20_mean']:.2f}",
        )
        m3.metric(
            "번호합 핵심 적중률",
            f"{summary['sum_core_rate']:.1%}",
        )
        m4.metric(
            "GAP합 핵심 적중률",
            f"{summary['gap_sum_core_rate']:.1%}",
        )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "BEST20 3+ 비율",
            f"{summary['best20_3plus_rate']:.1%}",
        )
        s2.metric(
            "BEST20 4+ 비율",
            f"{summary['best20_4plus_rate']:.1%}",
        )
        s3.metric(
            "번호합 MAE",
            f"{summary['sum_mae']:.2f}",
        )
        s4.metric(
            "GAP합 MAE",
            f"{summary['gap_sum_mae']:.2f}",
        )

        st.markdown("#### 이전 50회 vs 최근 50회")
        comparison = compare_previous_recent(
            ledger,
            window=50,
        )
        if not comparison.empty:
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### 실패 원인 누적")
        failures = failure_counts(ledger)
        if not failures.empty:
            f1, f2 = st.columns([1, 2])
            with f1:
                st.dataframe(
                    failures,
                    use_container_width=True,
                    hide_index=True,
                )
            with f2:
                st.bar_chart(
                    failures.set_index("failure")["count"]
                )

        st.markdown("#### 누적 검증 추세")
        cumulative = cumulative_metrics(ledger)
        if not cumulative.empty:
            st.line_chart(
                cumulative.set_index("round")[
                    [
                        "candidate_15_mean",
                        "best20_mean",
                        "sum_core_rate",
                        "gap_sum_core_rate",
                    ]
                ]
            )

        st.markdown("#### 회차별 검증 원장")
        display_columns = [
            "round",
            "date",
            "actual_numbers",
            "candidate_11_hits",
            "candidate_13_hits",
            "candidate_15_hits",
            "best5_max_hit",
            "best10_max_hit",
            "best15_max_hit",
            "best20_max_hit",
            "actual_sum",
            "predicted_sum_center",
            "sum_core_hit",
            "actual_gap_sum",
            "predicted_gap_sum_center",
            "gap_sum_core_hit",
            "master_top20_hits",
            "primary_failure",
            "failure_reasons",
        ]
        available_columns = [
            col for col in display_columns
            if col in ledger.columns
        ]
        st.dataframe(
            ledger[available_columns].sort_values(
                "round",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Historical Ledger CSV",
            csv_bytes(ledger),
            "historical_validation_ledger.csv",
        )
        dl2.download_button(
            "Historical Summary JSON",
            json.dumps(
                {
                    "summary": summary,
                    "failure_counts": failures.to_dict(
                        orient="records"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            "historical_validation_summary.json",
            "application/json",
        )

        st.info(
            "실패 원장은 과거 100회를 모델 개선에 맞춰 다시 골라내는 용도가 아니라, "
            "어느 단계에서 손실이 반복되는지 진단하는 기준선입니다. "
            "v4.5 규칙을 고정한 뒤 이후 실제 회차에서도 같은 지표를 계속 누적하는 "
            "방식으로 사용하는 것이 핵심입니다."
        )
