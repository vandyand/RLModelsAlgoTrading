#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

from backtest_simple_strategy import load_candles, evaluate_strategy, StrategyParams  # type: ignore


@dataclass
class GAConfig:
    population: int = 50
    generations: int = 50
    elite_frac: float = 0.1
    mutation_prob: float = 0.2
    mutation_scale: Dict[str, float] = None  # set in code
    crossover_prob: float = 0.8
    cost_bps: float = 0.0


def random_individual() -> StrategyParams:
    return StrategyParams(
        rsi_buy_threshold=random.uniform(50.0, 90.0),
        rsi_sell_threshold=random.uniform(10.0, 50.0),
        ema_fast_period=random.randint(5, 20),
        ema_slow_period=random.randint(21, 100),
        use_macd_filter=random.choice([0, 1]),
    )


action_bounds = {
    "rsi_buy_threshold": (50.0, 90.0),
    "rsi_sell_threshold": (10.0, 50.0),
    "ema_fast_period": (5, 20),
    "ema_slow_period": (21, 100),
}


def clamp_params(p: StrategyParams) -> StrategyParams:
    rsi_buy = float(np.clip(p.rsi_buy_threshold, *action_bounds["rsi_buy_threshold"]))
    rsi_sell = float(np.clip(p.rsi_sell_threshold, *action_bounds["rsi_sell_threshold"]))
    ema_fast = int(np.clip(int(p.ema_fast_period), *action_bounds["ema_fast_period"]))
    ema_slow = int(np.clip(int(p.ema_slow_period), *action_bounds["ema_slow_period"]))
    if ema_slow <= ema_fast:
        ema_slow = max(ema_fast + 1, ema_fast + 1)
    use_macd = 1 if int(p.use_macd_filter) == 1 else 0
    return StrategyParams(rsi_buy, rsi_sell, ema_fast, ema_slow, use_macd)


def mutate(p: StrategyParams, cfg: GAConfig) -> StrategyParams:
    q = StrategyParams(**p.__dict__)
    if random.random() < cfg.mutation_prob:
        q.rsi_buy_threshold += random.gauss(0, cfg.mutation_scale["rsi"])
    if random.random() < cfg.mutation_prob:
        q.rsi_sell_threshold += random.gauss(0, cfg.mutation_scale["rsi"])
    if random.random() < cfg.mutation_prob:
        q.ema_fast_period += int(round(random.gauss(0, cfg.mutation_scale["ema_fast"])) )
    if random.random() < cfg.mutation_prob:
        q.ema_slow_period += int(round(random.gauss(0, cfg.mutation_scale["ema_slow"])) )
    if random.random() < cfg.mutation_prob:
        q.use_macd_filter = 1 - q.use_macd_filter
    return clamp_params(q)


def crossover(a: StrategyParams, b: StrategyParams, cfg: GAConfig) -> Tuple[StrategyParams, StrategyParams]:
    if random.random() > cfg.crossover_prob:
        return clamp_params(a), clamp_params(b)
    child1 = StrategyParams(
        rsi_buy_threshold=random.choice([a.rsi_buy_threshold, b.rsi_buy_threshold]),
        rsi_sell_threshold=random.choice([a.rsi_sell_threshold, b.rsi_sell_threshold]),
        ema_fast_period=random.choice([a.ema_fast_period, b.ema_fast_period]),
        ema_slow_period=random.choice([a.ema_slow_period, b.ema_slow_period]),
        use_macd_filter=random.choice([a.use_macd_filter, b.use_macd_filter]),
    )
    child2 = StrategyParams(
        rsi_buy_threshold=random.choice([a.rsi_buy_threshold, b.rsi_buy_threshold]),
        rsi_sell_threshold=random.choice([a.rsi_sell_threshold, b.rsi_sell_threshold]),
        ema_fast_period=random.choice([a.ema_fast_period, b.ema_fast_period]),
        ema_slow_period=random.choice([a.ema_slow_period, b.ema_slow_period]),
        use_macd_filter=random.choice([a.use_macd_filter, b.use_macd_filter]),
    )
    return clamp_params(child1), clamp_params(child2)


def build_fitness_fn(
    w_sharpe: float,
    w_return: float,
    w_dd: float,
    w_trades: float,
    neg_sharpe_penalty: float,
    neg_return_penalty: float,
) -> Callable[[Dict[str, float]], float]:
    def _fitness(metrics: Dict[str, float]) -> float:
        sharpe = float(metrics.get("sharpe", 0.0))
        cum_return = float(metrics.get("cum_return", 0.0))  # fraction
        dd = float(metrics.get("max_drawdown", 0.0))  # negative
        trades = float(metrics.get("trades", 0.0))

        # Heavy penalty for negative Sharpe and negative returns
        if sharpe < 0.0:
            return -abs(sharpe) * float(neg_sharpe_penalty)
        if cum_return <= 0.0:
            return -abs(cum_return) * 100.0 * float(neg_return_penalty)

        # Convert drawdown to stability term (larger is better)
        dd_term = 1.0 / max(1e-6, -dd)  # if dd=-0.1 => 10; dd=-0.5 => 2
        # Encourage a reasonable number of trades (0 best at 200 trades)
        trade_term = -abs(trades - 200.0) / 200.0

        score = (
            float(w_sharpe) * sharpe
            + float(w_return) * (cum_return * 100.0)
            + float(w_dd) * dd_term
            + float(w_trades) * trade_term
        )
        return float(score)

    return _fitness


def run_ga(
    df: pd.DataFrame,
    cfg: GAConfig,
    df_val: Optional[pd.DataFrame] = None,
    fitness_fn: Optional[Callable[[Dict[str, float]], float]] = None,
) -> Dict[str, object]:
    if cfg.mutation_scale is None:
        cfg.mutation_scale = {"rsi": 3.0, "ema_fast": 2.0, "ema_slow": 5.0}
    if fitness_fn is None:
        fitness_fn = build_fitness_fn(0.4, 0.3, 0.2, 0.1, 10.0, 5.0)
    # Initialize population
    pop: List[StrategyParams] = [clamp_params(random_individual()) for _ in range(cfg.population)]
    best_hist: List[Dict[str, object]] = []

    for gen in range(cfg.generations):
        scored: List[Tuple[float, StrategyParams, Dict[str, float]]] = []
        for p in pop:
            m = evaluate_strategy(df, p, cost_bps=cfg.cost_bps)
            f = fitness_fn(m)
            scored.append((f, p, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_f, best_p, best_m = scored[0]
        # Optional validation metrics for the generation leader
        val_m: Optional[Dict[str, float]] = None
        if df_val is not None and len(df_val) > 0:
            try:
                val_m = evaluate_strategy(df_val, best_p, cost_bps=cfg.cost_bps)
            except Exception:
                val_m = None
        best_hist.append({"gen": gen, "fitness": best_f, "params": best_p.__dict__, "metrics": best_m, "val_metrics": val_m})
        print(json.dumps({
            "event": "gen",
            "gen": gen,
            "best_fitness": best_f,
            "best_metrics": best_m,
            "best_params": best_p.__dict__,
            "val_metrics": val_m,
        }), flush=True)

        # Selection: elitism + tournament for rest
        elite_n = max(1, int(cfg.elite_frac * cfg.population))
        next_pop: List[StrategyParams] = [scored[i][1] for i in range(elite_n)]

        # Fill the rest by tournament and crossover
        while len(next_pop) < cfg.population:
            # tournament
            k = 3
            cand = random.sample(scored[:max(10, cfg.population)], k)
            cand.sort(key=lambda x: x[0], reverse=True)
            parent1 = cand[0][1]
            parent2 = cand[1][1]
            c1, c2 = crossover(parent1, parent2, cfg)
            c1 = mutate(c1, cfg)
            if len(next_pop) < cfg.population:
                next_pop.append(c1)
            c2 = mutate(c2, cfg)
            if len(next_pop) < cfg.population:
                next_pop.append(c2)

        pop = next_pop

    # Final evaluation
    final_scores: List[Tuple[float, StrategyParams, Dict[str, float]]] = []
    for p in pop:
        m = evaluate_strategy(df, p, cost_bps=cfg.cost_bps)
        f = fitness_fn(m)
        final_scores.append((f, p, m))
    final_scores.sort(key=lambda x: x[0], reverse=True)
    bf, bp, bm = final_scores[0]
    return {"best_fitness": bf, "best_params": bp.__dict__, "best_metrics": bm, "history": best_hist}


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA optimize simple 5-parameter FX strategy with splits")
    p.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "EUR_USD_M5.csv"))
    # Legacy single-range args (used as train if splits omitted)
    p.add_argument("--start")
    p.add_argument("--end")
    # Train/Val/Test splits
    p.add_argument("--train-start")
    p.add_argument("--train-end")
    p.add_argument("--val-start")
    p.add_argument("--val-end")
    p.add_argument("--test-start")
    p.add_argument("--test-end")
    # GA params
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--mutation-prob", type=float, default=0.3)
    p.add_argument("--crossover-prob", type=float, default=0.8)
    p.add_argument("--cost-bps", type=float, default=1.0)
    # Fitness weights and penalties
    p.add_argument("--w-sharpe", type=float, default=0.4)
    p.add_argument("--w-return", type=float, default=0.3)
    p.add_argument("--w-dd", type=float, default=0.2)
    p.add_argument("--w-trades", type=float, default=0.1)
    p.add_argument("--neg-sharpe-penalty", type=float, default=10.0)
    p.add_argument("--neg-return-penalty", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    # Determine splits
    train_start = a.train_start or a.start
    train_end = a.train_end or a.end
    val_start = a.val_start
    val_end = a.val_end
    test_start = a.test_start
    test_end = a.test_end

    df_train = load_candles(a.csv, start=train_start, end=train_end)
    df_val = load_candles(a.csv, start=val_start, end=val_end) if val_start or val_end else None
    df_test = load_candles(a.csv, start=test_start, end=test_end) if test_start or test_end else None

    cfg = GAConfig(
        population=int(a.population),
        generations=int(a.generations),
        elite_frac=float(a.elite_frac),
        mutation_prob=float(a.mutation_prob),
        crossover_prob=float(a.crossover_prob),
        cost_bps=float(a.cost_bps),
    )
    fitness_fn = build_fitness_fn(
        w_sharpe=float(a.w_sharpe),
        w_return=float(a.w_return),
        w_dd=float(a.w_dd),
        w_trades=float(a.w_trades),
        neg_sharpe_penalty=float(a.neg_sharpe_penalty),
        neg_return_penalty=float(a.neg_return_penalty),
    )
    result = run_ga(df_train, cfg, df_val=df_val, fitness_fn=fitness_fn)

    # Evaluate best on validation and test
    best_params = StrategyParams(**result["history"][-1]["params"]) if len(result.get("history", [])) > 0 else None
    val_metrics = None
    test_metrics = None
    if best_params is not None:
        if df_val is not None:
            val_metrics = evaluate_strategy(df_val, best_params, cost_bps=cfg.cost_bps)
        if df_test is not None:
            test_metrics = evaluate_strategy(df_test, best_params, cost_bps=cfg.cost_bps)

    print(json.dumps({
        "event": "result",
        "best": {
            "fitness": result["best_fitness"],
            "params": result["best_params"],
            "metrics_train": result["best_metrics"],
            "metrics_val": val_metrics,
            "metrics_test": test_metrics,
        }
    }), flush=True)


if __name__ == "__main__":
    main()
