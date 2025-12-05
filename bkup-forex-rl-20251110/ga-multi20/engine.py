from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from config import Config
from model import MultiHeadGenome
from backtester import evaluate_genome
from fitness import PortfolioMetrics

# Global pool context for worker processes
_POOL_CTX: dict = {}


def _pool_init(X_use, closes_use, trade_cost: float, fitness_fn, thresholds, mode: str, band_enter: float, band_exit: float) -> None:
    global _POOL_CTX
    _POOL_CTX = {
        "X": X_use,
        "closes": closes_use,
        "trade_cost": float(trade_cost),
        "fitness_fn": fitness_fn,
        "thresholds": thresholds,
        "mode": mode,
        "band_enter": float(band_enter),
        "band_exit": float(band_exit),
    }


def _eval_one(g: MultiHeadGenome) -> Tuple[float, MultiHeadGenome]:
    ctx = _POOL_CTX
    s, _ = evaluate_genome(
        g,
        ctx["X"],
        ctx["closes"],
        ctx["trade_cost"],
        ctx["fitness_fn"],
        thresholds=ctx["thresholds"],
        mode=ctx["mode"],
        band_enter=ctx["band_enter"],
        band_exit=ctx["band_exit"],
    )
    return float(s), g


@dataclass
class GAState:
    best_score: float
    best_genome: MultiHeadGenome


def run_ga(
    cfg: Config,
    X: pd.DataFrame,
    closes: pd.DataFrame,
    fitness_fn=None,
    seed_genomes: Optional[List[MultiHeadGenome]] = None,
    on_generation: Optional[Callable[[int, float, PortfolioMetrics, int], None]] = None,
) -> GAState:
    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    input_dim = int(X.shape[1])
    num_heads = int(closes.shape[1])
    pop: List[MultiHeadGenome] = []
    if seed_genomes:
        # Include provided seeds, then fill remainder with mutated seeds and randoms
        for g in seed_genomes:
            pop.append(g.clone())
        while len(pop) < cfg.population and len(seed_genomes) > 0:
            base = seed_genomes[len(pop) % len(seed_genomes)]
            pop.append(base.mutate(cfg.mutation_prob, cfg.weight_sigma, cfg.affine_sigma, rng=rng))
    while len(pop) < cfg.population:
        pop.append(MultiHeadGenome.init(input_dim, cfg.hidden_layers, num_heads, use_affine=(cfg.input_norm == "affine"), rng=rng))
    elite_k = max(1, int(round(cfg.elite_frac * cfg.population)))
    crossover_k = max(0, int(round(cfg.crossover_frac * cfg.population)))
    random_k = max(0, int(round(cfg.random_frac * cfg.population)))
    best_score = -1e30
    best_genome = pop[0]
    import time
    for gen in range(cfg.generations):
        t0 = time.time()
        scored: List[Tuple[float, MultiHeadGenome]] = []
        # Vectorized-ish: loop but avoid repeated attribute lookups
        mode = ("absolute" if cfg.threshold_mode=="absolute" else "band")
        th = (cfg.enter_long, cfg.exit_long, cfg.enter_short, cfg.exit_short) if cfg.threshold_mode=="absolute" else None
        b_enter = cfg.band_enter; b_exit = cfg.band_exit
        # Optional downsampling
        X_use = X
        closes_use = closes
        if getattr(cfg, 'downsample', 1) and int(cfg.downsample) > 1:
            step = int(cfg.downsample)
            X_use = X.iloc[::step, :]
            closes_use = closes.iloc[::step, :]

        n_jobs = int(cfg.n_jobs) if getattr(cfg, 'n_jobs', 0) is not None else 0
        if n_jobs == 0:
            try:
                n_jobs = os.cpu_count() or 1
            except Exception:
                n_jobs = 1
        n_jobs = max(1, n_jobs)

        def _coerce_score(val) -> float:
            try:
                s = float(val)
                if not np.isfinite(s):
                    return -1e12
                return s
            except Exception:
                return -1e12

        if n_jobs == 1:
            for g in pop:
                s, _ = evaluate_genome(g, X_use, closes_use, cfg.trade_cost, fitness_fn, thresholds=th, mode=mode, band_enter=b_enter, band_exit=b_exit)
                scored.append((_coerce_score(s), g))
        else:
            # Threaded evaluation to avoid large copies/IPC; relies on numpy releasing the GIL
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                future_to_g = {
                    ex.submit(
                        evaluate_genome, g, X_use, closes_use, cfg.trade_cost, fitness_fn, th, mode, b_enter, b_exit
                    ): g for g in pop
                }
                for fut in as_completed(future_to_g):
                    s, _ = fut.result()
                    scored.append((_coerce_score(s), future_to_g[fut]))
        scored.sort(key=lambda x: x[0], reverse=True)
        avg_score = float(np.mean([s for s, _ in scored])) if scored else float('nan')
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_genome = scored[0][1]
        # Per-generation summary on the current best (single one-liner)
        _, best_result = evaluate_genome(
            scored[0][1], X, closes, cfg.trade_cost, fitness_fn,
            thresholds=(cfg.enter_long, cfg.exit_long, cfg.enter_short, cfg.exit_short) if cfg.threshold_mode=="absolute" else None,
            mode=("absolute" if cfg.threshold_mode=="absolute" else "band"),
            band_enter=cfg.band_enter, band_exit=cfg.band_exit,
        )
        dt = time.time() - t0
        if on_generation is not None:
            on_generation(gen, float(scored[0][0]), best_result.metrics, int(len(best_result.trades)))
        tim = getattr(best_result.metrics, "time_in_market", 0.0)
        pf = getattr(best_result.metrics, "profit_factor", 0.0)
        wr = getattr(best_result.metrics, "win_rate", 0.0)
        wl = getattr(best_result.metrics, "win_loss_ratio", 0.0)
        r2 = getattr(best_result.metrics, "equity_r2", 0.0)
        print(
            f"Gen {gen:03d} | score={scored[0][0]:.6f} | pop_avg={avg_score:.6f} | time={dt:.2f}s | "
            f"cum={best_result.metrics.cum_return:.4f} sharpe={best_result.metrics.sharpe:.4f} "
            f"sortino={best_result.metrics.sortino:.4f} maxDD={best_result.metrics.max_drawdown:.4f} trades={int(best_result.metrics.trades)} tim={tim:.4f} PF={pf:.3f} WR={wr:.3f} WL={wl:.3f} R2={r2:.3f}"
        )
        elites = [g for _, g in scored[:elite_k]]
        # Refill population: elites + crossovers + mutants + randoms
        new_pop: List[MultiHeadGenome] = []
        # Keep elites
        new_pop.extend(elites)
        # Crossovers between top half
        parents = [g for _, g in scored[: max(2, cfg.population // 2)]]
        for _ in range(crossover_k):
            a = rng.integers(0, len(parents))
            b = rng.integers(0, len(parents))
            if b == a:
                b = (a + 1) % len(parents)
            child = parents[a].crossover(parents[b], rng=rng)
            child = child.mutate(cfg.mutation_prob, cfg.weight_sigma, cfg.affine_sigma, rng=rng)
            new_pop.append(child)
        # Mutated elites fill until near target
        while len(new_pop) < cfg.population - random_k:
            base = elites[len(new_pop) % elite_k]
            new_pop.append(base.mutate(cfg.mutation_prob, cfg.weight_sigma, cfg.affine_sigma, rng=rng))
        # Random immigrants
        while len(new_pop) < cfg.population:
            new_pop.append(MultiHeadGenome.init(input_dim, cfg.hidden_layers, num_heads, use_affine=(cfg.input_norm == "affine"), rng=rng))
        pop = new_pop
    return GAState(best_score=best_score, best_genome=best_genome)
