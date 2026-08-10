from __future__ import annotations

import random
from typing import Sequence

import pandas as pd

from lotto64.analysis.sum_series import forecast_next_sum
from lotto64.recommend.combination import score_combo
from lotto64.recommend.filters import hard_filter

def _repair(combo: Sequence[int], candidate_pool: Sequence[int], rng: random.Random) -> tuple[int, ...]:
    values = set(int(n) for n in combo if 1 <= int(n) <= 45)
    while len(values) < 6:
        values.add(rng.choice(list(candidate_pool)))
    while len(values) > 6:
        values.remove(rng.choice(list(values)))
    return tuple(sorted(values))

def ga_optimize(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_count: int = 20,
    population_size: int = 500,
    generations: int = 30,
    mutation_rate: float = 0.15,
    seed: int = 20260720,
) -> pd.DataFrame:
    rng = random.Random(seed)
    pool = scores.head(candidate_count)["number"].astype(int).tolist()

    sum_forecast = forecast_next_sum(df)

    population = {
        tuple(sorted(rng.sample(pool, 6)))
        for _ in range(population_size * 2)
    }
    population = list(population)[:population_size]

    for _ in range(generations):
        evaluated = [
            score_combo(combo, scores, df, sum_forecast=sum_forecast)
            for combo in population
            if hard_filter(combo)
        ]
        evaluated.sort(key=lambda row: row["final_score"], reverse=True)
        elites = evaluated[: max(20, population_size // 5)]
        if not elites:
            population = [tuple(sorted(rng.sample(pool, 6))) for _ in range(population_size)]
            continue

        next_population = {tuple(row["combination"]) for row in elites}
        while len(next_population) < population_size:
            parent1 = tuple(rng.choice(elites)["combination"])
            parent2 = tuple(rng.choice(elites)["combination"])
            split = rng.randint(1, 5)
            child = _repair(parent1[:split] + parent2[split:], pool, rng)

            if rng.random() < mutation_rate:
                child_list = list(child)
                child_list[rng.randrange(6)] = rng.choice(pool)
                child = _repair(child_list, pool, rng)

            next_population.add(child)

        population = list(next_population)

    final_rows = [
        score_combo(combo, scores, df, sum_forecast=sum_forecast)
        for combo in population
        if hard_filter(combo)
    ]
    return pd.DataFrame(final_rows).sort_values(
        ["final_score", "combination"],
        ascending=[False, True],
    ).reset_index(drop=True)
