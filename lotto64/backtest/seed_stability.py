from __future__ import annotations

from itertools import combinations
import random
from typing import Callable, Optional

import numpy as np
import pandas as pd

from lotto64.backtest.walk_forward import walk_forward
from lotto64.config import FIXED_SEED
from lotto64.recommend.ga import ga_optimize
from lotto64.recommend.portfolio import build_portfolio


def build_seed_set(
    base_seed: int = FIXED_SEED,
    count: int = 5,
) -> list[int]:
    """
    기준 Seed에서 항상 동일한 다중 Seed 세트를 생성합니다.

    첫 번째 값은 고정 기준 Seed 자체이며, 나머지는 기준 Seed에서
    결정적으로 파생되므로 다시 실행해도 같은 검증 Seed들이 사용됩니다.
    """
    if count < 1:
        raise ValueError("Seed 개수는 1 이상이어야 합니다.")

    rng = random.Random(int(base_seed) ^ 0x5EED64)
    seeds = [int(base_seed)]

    while len(seeds) < int(count):
        candidate = rng.randrange(1, 2_147_483_647)
        if candidate not in seeds:
            seeds.append(candidate)

    return seeds


def _grade(score: float) -> str:
    if score >= 80:
        return "S"
    if score >= 65:
        return "A"
    if score >= 50:
        return "B"
    if score >= 35:
        return "C"
    return "D"


def _pairwise_combo_jaccard(runs: pd.DataFrame) -> float:
    if runs.empty or runs["seed"].nunique() < 2:
        return 1.0

    sets = {
        int(seed): {
            tuple(combo)
            for combo in frame["combination"].tolist()
        }
        for seed, frame in runs.groupby("seed")
    }

    values = []
    for left, right in combinations(sorted(sets), 2):
        a, b = sets[left], sets[right]
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)

    return float(np.mean(values)) if values else 1.0


def _number_exposure_correlation(
    runs: pd.DataFrame,
    top_n: int,
) -> float:
    if runs.empty or runs["seed"].nunique() < 2:
        return 1.0

    rows = []
    for seed, frame in runs.groupby("seed"):
        counts = {n: 0 for n in range(1, 46)}
        for combo in frame["combination"]:
            for number in combo:
                counts[int(number)] += 1
        rows.append({
            "seed": int(seed),
            **{f"n{n}": counts[n] / max(1, top_n) for n in range(1, 46)},
        })

    exposure = pd.DataFrame(rows).set_index("seed")
    corr = exposure.T.corr()

    values = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                values.append(float(value))

    return float(np.mean(values)) if values else 1.0


def ga_seed_stability(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_count: int = 20,
    population_size: int = 400,
    generations: int = 25,
    seed_count: int = 5,
    top_n: int = 20,
    base_seed: int = FIXED_SEED,
    progress_callback: Optional[
        Callable[[float, str], None]
    ] = None,
) -> dict:
    """
    같은 데이터/모델 설정으로 GA만 여러 Seed에서 반복하여
    조합과 번호 노출의 Seed 민감도를 측정합니다.

    결과가 좋은 Seed를 골라내는 기능이 아니라, 같은 결론이 여러
    Seed에서 반복되는지를 검증하는 기능입니다.
    """
    seeds = build_seed_set(base_seed, seed_count)
    run_frames: list[pd.DataFrame] = []

    for index, seed in enumerate(seeds, start=1):
        if progress_callback:
            progress_callback(
                (index - 1) / len(seeds),
                f"GA Seed {index}/{len(seeds)} 실행 중 · {seed}",
            )

        ranked = ga_optimize(
            df,
            scores,
            candidate_count=candidate_count,
            population_size=population_size,
            generations=generations,
            seed=seed,
        )
        portfolio = build_portfolio(
            ranked,
            size=top_n,
            max_jaccard=0.50,
        )

        if portfolio.empty:
            continue

        frame = portfolio.head(top_n).copy().reset_index(drop=True)
        frame.insert(0, "rank", range(1, len(frame) + 1))
        frame.insert(0, "seed", int(seed))
        run_frames.append(frame)

    if progress_callback:
        progress_callback(1.0, "GA 다중 Seed 집계 중...")

    if not run_frames:
        return {
            "seeds": seeds,
            "runs": pd.DataFrame(),
            "combo_consensus": pd.DataFrame(),
            "number_stability": pd.DataFrame(),
            "metrics": {},
        }

    runs = pd.concat(run_frames, ignore_index=True)
    executed_seed_count = int(runs["seed"].nunique())

    # Exact-combination consensus.
    combo = (
        runs.groupby("combination", as_index=False)
        .agg(
            seed_appearances=("seed", "nunique"),
            appearances=("seed", "size"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            mean_final_score=("final_score", "mean"),
        )
    )
    combo["seed_coverage_pct"] = (
        combo["seed_appearances"] / executed_seed_count * 100
    )
    combo["rank_quality"] = (
        (top_n + 1 - combo["mean_rank"]) / max(1, top_n)
    ).clip(0, 1)
    combo["robustness_score"] = (
        0.70 * (combo["seed_coverage_pct"] / 100)
        + 0.30 * combo["rank_quality"]
    ) * 100
    combo["stability_grade"] = combo["robustness_score"].map(_grade)

    combo = combo.sort_values(
        [
            "robustness_score",
            "seed_appearances",
            "mean_final_score",
            "best_rank",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # Number exposure stability.
    exposure_rows = []
    for seed, frame in runs.groupby("seed"):
        counts = {n: 0 for n in range(1, 46)}
        for values in frame["combination"]:
            for number in values:
                counts[int(number)] += 1

        for number in range(1, 46):
            exposure_rows.append({
                "seed": int(seed),
                "number": number,
                "exposure_count": counts[number],
                "exposure_pct": counts[number] / max(1, top_n) * 100,
                "present": int(counts[number] > 0),
            })

    exposure = pd.DataFrame(exposure_rows)
    number_stability = (
        exposure.groupby("number", as_index=False)
        .agg(
            seed_presence_pct=("present", "mean"),
            mean_exposure_pct=("exposure_pct", "mean"),
            exposure_std_pct=("exposure_pct", "std"),
            total_appearances=("exposure_count", "sum"),
        )
    )
    number_stability["seed_presence_pct"] *= 100
    number_stability["exposure_std_pct"] = (
        number_stability["exposure_std_pct"].fillna(0.0)
    )

    mean = number_stability["mean_exposure_pct"]
    std = number_stability["exposure_std_pct"]
    consistency = (1 - std / (mean + 1e-9)).clip(0, 1) * 100

    number_stability["robustness_score"] = (
        0.60 * number_stability["seed_presence_pct"]
        + 0.40 * consistency
    )
    number_stability["stability_grade"] = (
        number_stability["robustness_score"].map(_grade)
    )
    number_stability = number_stability.sort_values(
        [
            "robustness_score",
            "mean_exposure_pct",
            "number",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    metrics = {
        "base_seed": int(base_seed),
        "seed_count": executed_seed_count,
        "stable_combo_2plus": int(
            (combo["seed_appearances"] >= 2).sum()
        ),
        "stable_combo_3plus": int(
            (combo["seed_appearances"] >= 3).sum()
        ),
        "stable_combo_majority": int(
            (
                combo["seed_appearances"]
                >= max(2, (executed_seed_count + 1) // 2)
            ).sum()
        ),
        "pairwise_combo_jaccard": _pairwise_combo_jaccard(runs),
        "number_exposure_correlation": _number_exposure_correlation(
            runs,
            top_n,
        ),
    }

    return {
        "seeds": seeds,
        "runs": runs,
        "combo_consensus": combo,
        "number_stability": number_stability,
        "metrics": metrics,
    }


def walk_forward_seed_stability(
    df: pd.DataFrame,
    rounds: int = 20,
    train_window: int = 200,
    candidate_count: int = 18,
    top_combos: int = 10,
    seed_count: int = 5,
    base_seed: int = FIXED_SEED,
    weights: Optional[dict[str, float]] = None,
    progress_callback: Optional[
        Callable[[float, str], None]
    ] = None,
) -> dict:
    """
    Walk-forward를 여러 Seed에서 반복하여 조합 단계의 Seed 민감도를
    회차별로 측정합니다.

    후보번호와 합계 시계열 등 결정론적 부분은 동일하고, Seed의 영향을
    받는 랜덤 조합 탐색 결과의 변동성을 확인하는 용도입니다.
    """
    seeds = build_seed_set(base_seed, seed_count)
    frames: list[pd.DataFrame] = []

    for index, seed in enumerate(seeds, start=1):
        def inner_progress(value: float, text: str) -> None:
            if progress_callback:
                overall = (
                    (index - 1) + float(value)
                ) / len(seeds)
                progress_callback(
                    min(1.0, overall),
                    f"Seed {index}/{len(seeds)} · {text}",
                )

        result = walk_forward(
            df,
            rounds=rounds,
            train_window=train_window,
            candidate_count=candidate_count,
            top_combos=top_combos,
            seed=seed,
            weights=weights,
            progress_callback=inner_progress,
        )

        if not result.empty:
            frame = result.copy()
            frame["seed"] = int(seed)
            frames.append(frame)

    if progress_callback:
        progress_callback(1.0, "다중 Seed Walk-forward 집계 중...")

    if not frames:
        return {
            "seeds": seeds,
            "runs": pd.DataFrame(),
            "by_seed": pd.DataFrame(),
            "by_round": pd.DataFrame(),
            "metrics": {},
        }

    runs = pd.concat(frames, ignore_index=True)

    by_seed = (
        runs.groupby("seed", as_index=False)
        .agg(
            top_combo_mean_hit=("top_combo_max_hit", "mean"),
            top_combo_3plus_rate=("top_combo_3plus", "mean"),
            top_combo_4plus_rate=("top_combo_4plus", "mean"),
            top_combo_5plus_rate=("top_combo_5plus", "mean"),
            candidate_15_mean_hit=("candidate_15_hits", "mean"),
            sum_mae=("sum_abs_error", "mean"),
        )
    )

    by_round = (
        runs.groupby("round", as_index=False)
        .agg(
            top_combo_hit_mean=("top_combo_max_hit", "mean"),
            top_combo_hit_std=("top_combo_max_hit", "std"),
            top_combo_hit_min=("top_combo_max_hit", "min"),
            top_combo_hit_max=("top_combo_max_hit", "max"),
            top_combo_3plus_seed_rate=("top_combo_3plus", "mean"),
            distinct_hit_results=("top_combo_max_hit", "nunique"),
        )
    )
    by_round["top_combo_hit_std"] = (
        by_round["top_combo_hit_std"].fillna(0.0)
    )
    by_round["all_seed_agree"] = (
        by_round["distinct_hit_results"] == 1
    ).astype(int)

    metrics = {
        "base_seed": int(base_seed),
        "seed_count": int(runs["seed"].nunique()),
        "rounds": int(by_round["round"].nunique()),
        "mean_top_combo_hit": float(
            runs["top_combo_max_hit"].mean()
        ),
        "mean_seed_std": float(
            by_round["top_combo_hit_std"].mean()
        ),
        "all_seed_agreement_pct": float(
            by_round["all_seed_agree"].mean() * 100
        ),
        "top_combo_3plus_rate": float(
            runs["top_combo_3plus"].mean() * 100
        ),
        "top_combo_4plus_rate": float(
            runs["top_combo_4plus"].mean() * 100
        ),
    }

    return {
        "seeds": seeds,
        "runs": runs,
        "by_seed": by_seed,
        "by_round": by_round,
        "metrics": metrics,
    }
