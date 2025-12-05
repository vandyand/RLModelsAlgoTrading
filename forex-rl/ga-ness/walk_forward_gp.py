#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

from multi_features import build_features
from gp_optimize import (
    GPConfig,
    Node,
    random_tree,
    mutate,
    crossover,
    fitness,
    node_size,
    node_depth,
    simplify_const,
    clone_individual,
    individual_key,
    positions_from_tree_three_state,
    metrics_from_positions,
    pretty,
    trade_frequency_score,
    # NN strategy support
    NNGenome,
    _nn_init,
    _nn_mutate,
    _nn_crossover,
    fitness_nn,
    positions_from_nn_hysteresis,
    _nn_forward,
)


def compute_extras(close: pd.Series, pos: pd.Series, cost_bps: float = 1.0) -> Dict[str, float]:
    """Lightweight diagnostics from (close, pos): equity shape, trade/idle stats, distributions.
    Avoids heavy libs; uses numpy/pandas only.
    """
    import numpy as _np
    import pandas as _pd

    def _skew_np(x: _np.ndarray) -> float:
        x = x[_np.isfinite(x)]
        n = x.size
        if n < 3:
            return float("nan")
        mu = _np.mean(x)
        s = _np.std(x, ddof=1)
        if s <= 1e-12:
            return 0.0
        m3 = _np.mean((x - mu) ** 3)
        g1 = m3 / (s ** 3)
        return float(g1)

    def _kurtosis_np(x: _np.ndarray) -> float:
        x = x[_np.isfinite(x)]
        n = x.size
        if n < 4:
            return float("nan")
        mu = _np.mean(x)
        s2 = _np.var(x, ddof=1)
        if s2 <= 1e-12:
            return 0.0
        m4 = _np.mean((x - mu) ** 4)
        g2 = m4 / (s2 ** 2) - 3.0  # excess kurtosis
        return float(g2)

    ret = close.pct_change().fillna(0.0)
    strat_ret = (pos * ret).astype(float)
    cost = float(cost_bps) * 1e-4
    if cost > 0:
        trade_flag = pos.ne(pos.shift(1)).fillna(False)
        strat_ret.loc[trade_flag] = strat_ret.loc[trade_flag] - cost
    equity = (1.0 + strat_ret).cumprod()

    # R^2 of equity vs linear time
    n = len(equity)
    if n >= 2:
        t = _np.arange(n, dtype=float)
        X = _np.vstack([_np.ones(n), t]).T
        beta, *_ = _np.linalg.lstsq(X, equity.values, rcond=None)
        fitted = X @ beta
        ss_res = float(_np.sum((equity.values - fitted) ** 2))
        ss_tot = float(_np.sum((equity.values - _np.mean(equity.values)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    else:
        r2 = float("nan")

    # Exposure and turnover
    abs_pos = pos.abs().astype(float)
    exposure = float(abs_pos.mean()) if len(abs_pos) > 0 else float("nan")
    long_frac = float((pos > 0).mean()) if len(pos) > 0 else float("nan")
    short_frac = float((pos < 0).mean()) if len(pos) > 0 else float("nan")
    flat_frac = float((pos == 0).mean()) if len(pos) > 0 else float("nan")
    trades = int(pos.ne(pos.shift(1)).fillna(False).sum())
    turnover_per_bar = float(trades) / float(n) if n > 0 else float("nan")
    sign = _np.sign(pos.values)
    sign_prev = _np.roll(sign, 1)
    sign_prev[0] = 0.0
    flips = _np.logical_and(sign != 0.0, _np.logical_and(sign_prev != 0.0, sign * sign_prev < 0.0)).sum()
    flip_rate = float(flips) / float(n) if n > 0 else float("nan")

    # Drawdown stats
    running_max = equity.cummax()
    dd = (equity / running_max - 1.0).fillna(0.0)
    # Episode durations
    dd_active = dd < 0.0
    dd_id = dd_active.ne(dd_active.shift(1)).cumsum()
    dd_lengths = dd_id[dd_active].value_counts().astype(float) if dd_active.any() else _pd.Series(dtype=float)
    avg_dd_dur = float(dd_lengths.mean()) if len(dd_lengths) > 0 else float("nan")
    max_dd_dur = float(dd_lengths.max()) if len(dd_lengths) > 0 else float("nan")
    avg_dd_depth = float(dd[dd < 0.0].mean()) if (dd < 0.0).any() else float("nan")

    # Per-trade returns (approximate using equity within holding segments)
    change = pos.ne(pos.shift(1)).fillna(False)
    seg_id = change.cumsum()
    hold_mask = pos != 0.0
    hold_groups = seg_id[hold_mask]
    hold_sizes = hold_groups.value_counts().astype(float) if len(hold_groups) > 0 else _pd.Series(dtype=float)
    trade_eq = []
    if len(hold_sizes) > 0:
        for sid, cnt in hold_sizes.items():
            idx = (seg_id == sid)
            eq_seg = equity[idx]
            if len(eq_seg) > 1:
                trade_eq.append(float(eq_seg.iloc[-1] / eq_seg.iloc[0] - 1.0))
    trade_eq = _np.array(trade_eq, dtype=float) if len(trade_eq) > 0 else _np.array([], dtype=float)
    avg_hold = float(hold_sizes.mean()) if len(hold_sizes) > 0 else float("nan")
    std_hold = float(_np.std(hold_sizes.values, ddof=1)) if len(hold_sizes) > 1 else float("nan")
    skew_hold = _skew_np(hold_sizes.values) if len(hold_sizes) > 2 else float("nan")
    # Idle durations
    idle_mask = pos == 0.0
    idle_groups = seg_id[idle_mask]
    idle_sizes = idle_groups.value_counts().astype(float) if len(idle_groups) > 0 else _pd.Series(dtype=float)
    avg_idle = float(idle_sizes.mean()) if len(idle_sizes) > 0 else float("nan")
    std_idle = float(_np.std(idle_sizes.values, ddof=1)) if len(idle_sizes) > 1 else float("nan")
    skew_idle = _skew_np(idle_sizes.values) if len(idle_sizes) > 2 else float("nan")

    # Trade-level stats
    avg_trade_ret = float(_np.nanmean(trade_eq)) if trade_eq.size > 0 else float("nan")
    std_trade_ret = float(_np.nanstd(trade_eq, ddof=1)) if trade_eq.size > 1 else float("nan")
    skew_trade_ret = _skew_np(trade_eq) if trade_eq.size > 2 else float("nan")
    profit_factor = float((_np.sum(trade_eq[trade_eq > 0.0]) / max(1e-12, -_np.sum(trade_eq[trade_eq < 0.0])))) if trade_eq.size > 0 else float("nan")
    win_rate = float(_np.mean(trade_eq > 0.0)) if trade_eq.size > 0 else float("nan")
    # Max losing streak
    max_lose_streak = 0
    cur = 0
    for v in trade_eq:
        if v < 0.0:
            cur += 1
            if cur > max_lose_streak:
                max_lose_streak = cur
        else:
            cur = 0

    # Return-series stats
    ret_std = float(_np.std(strat_ret.values, ddof=1)) if n > 1 else float("nan")
    downside_std = float(_np.std(strat_ret.values[strat_ret.values < 0.0], ddof=1)) if (strat_ret.values < 0.0).any() else float("nan")
    ret_skew = _skew_np(strat_ret.values)
    ret_kurt = _kurtosis_np(strat_ret.values)
    # Lag-1 autocorrelation (safe computation without np.corrcoef warnings)
    if n > 1:
        x = strat_ret.values
        x = x[_np.isfinite(x)]
        if x.size > 2:
            x1 = x[1:]
            x0 = x[:-1]
            m1 = _np.mean(x1)
            m0 = _np.mean(x0)
            s1 = _np.std(x1)
            s0 = _np.std(x0)
            if s1 > 1e-12 and s0 > 1e-12:
                ac = float(_np.mean((x1 - m1) * (x0 - m0)) / (s1 * s0))
            else:
                ac = float("nan")
        else:
            ac = float("nan")
    else:
        ac = float("nan")

    # Daily aggregated returns volatility (stability proxy)
    try:
        daily = strat_ret.replace([_np.inf, -_np.inf], _np.nan).resample('D').sum(min_count=1)
        daily = daily.replace([_np.inf, -_np.inf], _np.nan).dropna()
        daily_std = float(_np.std(daily.values, ddof=1)) if len(daily) > 1 else float("nan")
    except Exception:
        daily_std = float("nan")

    return {
        "equity_r2": float(r2),
        "exposure": exposure,
        "frac_long": long_frac,
        "frac_short": short_frac,
        "frac_flat": flat_frac,
        "turnover_per_bar": turnover_per_bar,
        "flip_rate": flip_rate,
        "avg_dd_depth": avg_dd_depth,
        "avg_dd_dur": avg_dd_dur,
        "max_dd_dur": max_dd_dur,
        "avg_hold": avg_hold,
        "std_hold": std_hold,
        "skew_hold": skew_hold,
        "avg_idle": avg_idle,
        "std_idle": std_idle,
        "skew_idle": skew_idle,
        "avg_trade_ret": avg_trade_ret,
        "std_trade_ret": std_trade_ret,
        "skew_trade_ret": skew_trade_ret,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "max_lose_streak": float(max_lose_streak),
        "ret_std": ret_std,
        "downside_std": downside_std,
        "ret_skew": ret_skew,
        "ret_kurt": ret_kurt,
        "autocorr_lag1": ac,
        "daily_ret_std": daily_std,
    }


# ---- Fitness Function (FF) models: lightweight online trainers over past windows ----

FFFeature = List[str]


def _ff_extract_features_from_train_metrics(
    m: Dict[str, float], size: int, depth: int, tf: float
) -> List[float]:
    # Keep this feature set minimal and cheap: available inside fitness()
    sh = float(m.get("sharpe", 0.0))
    so = float(m.get("sortino", 0.0))
    cr = float(m.get("cum_return", 0.0))
    dd = float(m.get("max_drawdown", 0.0))
    return [
        sh,
        so,
        cr,
        dd,
        float(tf),
        float(size),
        float(depth),
    ]


def _ff_feature_names() -> FFFeature:
    return [
        "train_sharpe",
        "train_sortino",
        "train_cumret",
        "train_dd",
        "tf",
        "tree_size",
        "tree_depth",
    ]


class _FFLinearLogistic:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray, b: float):
        self.mu = mu
        self.sigma = sigma
        self.w = w
        self.b = float(b)

    def score(self, features: List[float]) -> float:
        x = np.array(features, dtype=float)
        z = (x - self.mu) / (self.sigma + 1e-12)
        p = 1.0 / (1.0 + np.exp(-(z @ self.w + self.b)))
        return float(p)


def _train_ff_linear_logreg(X: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.1, reg: float = 1e-3) -> _FFLinearLogistic:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xn = (X - mu) / sigma
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    yv = y.astype(float)
    for _ in range(int(iters)):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = (Xn.T @ (p - yv)) / Xn.shape[0] + reg * w
        grad_b = float(p.mean() - yv.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return _FFLinearLogistic(mu, sigma, w, b)


class _FFNumpyMLP:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def score(self, features: List[float]) -> float:
        x = np.array(features, dtype=float)
        z = (x - self.mu) / (self.sigma + 1e-12)
        Z1 = z @ self.W1 + self.b1
        H = np.maximum(0.0, Z1)
        Z2 = H @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-Z2))
        return float(p.squeeze())


def _train_ff_numpy_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 16,
    iters: int = 500,
    lr: float = 0.05,
    reg: float = 1e-4,
) -> _FFNumpyMLP:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xn = (X - mu) / sigma
    rng = np.random.default_rng(42)
    n_in = Xn.shape[1]
    W1 = 0.1 * rng.standard_normal((n_in, int(hidden)))
    b1 = np.zeros((int(hidden),), dtype=float)
    W2 = 0.1 * rng.standard_normal((int(hidden), 1))
    b2 = np.zeros((1,), dtype=float)
    yv = y.reshape(-1, 1).astype(float)
    for _ in range(int(iters)):
        Z1 = Xn @ W1 + b1
        H = np.maximum(0.0, Z1)
        Z2 = H @ W2 + b2
        P = 1.0 / (1.0 + np.exp(-Z2))
        dZ2 = (P - yv)
        dW2 = (H.T @ dZ2) / Xn.shape[0] + reg * W2
        db2 = dZ2.mean(axis=0)
        dH = dZ2 @ W2.T
        dZ1 = dH * (Z1 > 0)
        dW1 = (Xn.T @ dZ1) / Xn.shape[0] + reg * W1
        db1 = dZ1.mean(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    return _FFNumpyMLP(mu, sigma, W1, b1, W2, b2)


def _build_ff_model_from_past(
    past_windows: List[Dict[str, Any]],
    mode: str,
    min_rows: int = 100,
) -> Optional[Callable[[Dict[str, float], Dict[str, float], int, int], float]]:
    # Build dataset from final_population across past windows
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    for rec in past_windows:
        pop = rec.get("final_population", []) or []
        for indiv in pop:
            tm = indiv.get("metrics_train", {}) or {}
            y = 1 if float((indiv.get("metrics_val", {}) or {}).get("sharpe", 0.0)) > 0.0 else 0
            tf_tr = float(indiv.get("tf_train", float("nan")))
            size = int(indiv.get("tree_size", 0))
            depth = int(indiv.get("tree_depth", 0))
            if not np.isfinite(tf_tr):
                tf_tr = 0.0
            x = _ff_extract_features_from_train_metrics(tm, size, depth, tf_tr)
            if any((not np.isfinite(v)) for v in x):
                continue
            X_rows.append(x)
            y_rows.append(int(y))

    if len(X_rows) < int(min_rows):
        return None
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    # Train model
    if mode == "linear":
        model = _train_ff_linear_logreg(X, y, iters=400, lr=0.1, reg=1e-3)

        def score_fn(metrics: Dict[str, float], extras: Dict[str, float], size: int, depth: int) -> float:
            tf = float(extras.get("tf", 0.0))
            feats = _ff_extract_features_from_train_metrics(metrics, int(size), int(depth), tf)
            return model.score(feats)

        return score_fn
    elif mode == "nn":
        model = _train_ff_numpy_mlp(X, y, hidden=16, iters=500, lr=0.05, reg=1e-4)

        def score_fn(metrics: Dict[str, float], extras: Dict[str, float], size: int, depth: int) -> float:
            tf = float(extras.get("tf", 0.0))
            feats = _ff_extract_features_from_train_metrics(metrics, int(size), int(depth), tf)
            return model.score(feats)

        return score_fn
    else:
        return None


@dataclass
class WFOConfig:
    train_months: int = 1
    val_months: int = 1
    step_months: int = 1
    population: int = 12
    generations: int = 6
    max_depth: int = 3
    cost_bps: float = 1.0
    min_hold: int = 3
    cooldown: int = 12
    subsample: int = 20
    seed: int = 123


def _add_offset(ts: pd.Timestamp, n: int, unit: str) -> pd.Timestamp:
    # Add offset in months or weeks and normalize; preserve tz
    if unit == "weeks":
        return (ts + pd.DateOffset(weeks=int(n))).normalize()
    return (ts + pd.DateOffset(months=int(n))).normalize()


def generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_n: int,
    val_n: int,
    step_n: int,
    unit: str = "months",
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur_train_start = start
    while True:
        cur_train_end = _add_offset(cur_train_start, train_n, unit) - pd.Timedelta(seconds=1)
        cur_val_start = cur_train_end + pd.Timedelta(seconds=1)
        cur_val_end = _add_offset(cur_val_start, val_n, unit) - pd.Timedelta(seconds=1)
        if cur_val_end > end:
            break
        windows.append((cur_train_start, cur_train_end, cur_val_start, cur_val_end))
        cur_train_start = _add_offset(cur_train_start, step_n, unit)
    return windows


def run_gp_once(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    close_col: str,
    cfg: GPConfig,
    seed: int,
    log_gens: bool = True,
    final_top_k: int = 0,
    select_top_k: int = 0,
    select_by: str = "ff",
    select_ff_threshold: float = 0.0,
    select_train_sharpe_min: float = -1e9,
    select_train_trades_min: int = 0,
) -> Dict[str, Any]:
    # Seed per window for determinism
    np.random.seed(int(seed))
    import random as _random
    _random.seed(int(seed))

    feature_names = [c for c in X_train.columns if c != close_col and not c.endswith("_close")]
    is_nn = (str(cfg.strategy_type) == "nn")
    if not is_nn:
        pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = [
            (
                random_tree(feature_names, cfg.max_depth),
                random_tree(feature_names, cfg.max_depth),
                random_tree(feature_names, cfg.max_depth - 1),
                random_tree(feature_names, cfg.max_depth - 1),
            )
            for _ in range(cfg.population)
        ]
    else:
        pop_nn: List[NNGenome] = [
            _nn_init(len(feature_names), list(cfg.nn_arch), use_affine=(str(cfg.input_norm) == "affine"))
            for _ in range(cfg.population)
        ]

    best_hist: List[Dict[str, Any]] = []
    for gen in range(cfg.generations):
        t0 = time.time()
        if not is_nn:
            scored: List[Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]] = []
            for ind in pop:
                fval, met, tr = fitness(ind[0], ind[1], ind[2], ind[3], X_train, close_col, cfg)
                scored.append((fval, ind, met, tr))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_f, best_ind, best_m, best_tr = scored[0]
            if log_gens:
                best_tf = trade_frequency_score(int(best_tr), len(X_train))
                med = float(np.median([s[0] for s in scored])) if scored else float("nan")
                evt = {
                    "event": "gen_summary",
                    "gen": int(gen),
                    "med": round(med, 4) if np.isfinite(med) else med,
                    "sec": round(time.time() - t0, 3),
                    "tree": int(
                        node_size(best_ind[0])
                        + node_size(best_ind[1])
                        + (node_size(best_ind[2]) if best_ind[2] else 0)
                        + (node_size(best_ind[3]) if best_ind[3] else 0)
                    ),
                    "best": {
                        "fit": round(float(best_f), 4),
                        "trd": int(best_tr),
                        "shrp": round(float(best_m.get("sharpe", 0.0)), 4),
                        "srtn": round(float(best_m.get("sortino", 0.0)), 4),
                        "cum": round(float(best_m.get("cum_return", 0.0)), 4),
                        "dd": round(float(best_m.get("max_drawdown", 0.0)), 4),
                        "tf": round(float(best_tf), 4),
                    },
                }
                print(json.dumps(evt), flush=True)
            # evolve logic trees
            elite_n = int(cfg.elite_count) if int(cfg.elite_count) > 0 else max(1, int(cfg.elite_frac * cfg.population))
            elite_n = min(max(1, elite_n), cfg.population - 1 if cfg.population > 1 else 1)
            next_pop: List[Tuple[Node, Node, Optional[Node], Optional[Node]]] = []
            seen: set[str] = set()
            for i in range(elite_n):
                raw = scored[i][1]
                elite = (
                    simplify_const(raw[0]),
                    simplify_const(raw[1]),
                    simplify_const(raw[2]) if raw[2] else None,
                    simplify_const(raw[3]) if raw[3] else None,
                )
                k = individual_key(elite)
                if k not in seen:
                    next_pop.append(elite)  # type: ignore[arg-type]
                    seen.add(k)
            while len(next_pop) < cfg.population:
                parents = _random.sample(scored[: max(10, cfg.population)], 2)
                a_ind = parents[0][1]
                b_ind = parents[1][1]
                pa = clone_individual(a_ind)
                pb = clone_individual(b_ind)
                e1, e2 = crossover(pa[0], pb[0])
                s1, s2 = crossover(pa[1], pb[1])
                xl1, xl2 = crossover(pa[2], pb[2]) if (pa[2] and pb[2]) else (pa[2] or pb[2], None)
                xs1, xs2 = crossover(pa[3], pb[3]) if (pa[3] and pb[3]) else (pa[3] or pb[3], None)
                e1 = mutate(e1, feature_names, cfg.max_depth, 0.3)
                s1 = mutate(s1, feature_names, cfg.max_depth, 0.3)
                if xl1 is not None:
                    xl1 = mutate(xl1, feature_names, cfg.max_depth - 1, 0.3)
                if xs1 is not None:
                    xs1 = mutate(xs1, feature_names, cfg.max_depth - 1, 0.3)
                child1 = (
                    simplify_const(e1),
                    simplify_const(s1),
                    simplify_const(xl1) if xl1 else None,
                    simplify_const(xs1) if xs1 else None,
                )
                k1 = individual_key(child1)
                if k1 not in seen:
                    next_pop.append(child1)  # type: ignore[arg-type]
                    seen.add(k1)
                if len(next_pop) < cfg.population:
                    e2m = mutate(e2, feature_names, cfg.max_depth, 0.3)
                    s2m = mutate(s2, feature_names, cfg.max_depth, 0.3)
                    if xl2 is not None:
                        xl2 = mutate(xl2, feature_names, cfg.max_depth - 1, 0.3)
                    if xs2 is not None:
                        xs2 = mutate(xs2, feature_names, cfg.max_depth - 1, 0.3)
                    child2 = (
                        simplify_const(e2m),
                        simplify_const(s2m),
                        simplify_const(xl2) if xl2 else None,
                        simplify_const(xs2) if xs2 else None,
                    )
                    k2 = individual_key(child2)
                    if k2 not in seen:
                        next_pop.append(child2)  # type: ignore[arg-type]
                        seen.add(k2)
            pop = next_pop
        else:
            # NN strategy branch
            scored_nn: List[Tuple[float, NNGenome, Dict[str, float], int, int, int]] = []
            for g in pop_nn:
                fval, met, tr, sz, dp = fitness_nn(g, X_train, feature_names, close_col, cfg)
                scored_nn.append((fval, g, met, tr, sz, dp))
            scored_nn.sort(key=lambda x: x[0], reverse=True)
            best_f, best_g, best_m, best_tr, best_sz, best_dp = scored_nn[0]
            if log_gens:
                best_tf = trade_frequency_score(int(best_tr), len(X_train))
                med = float(np.median([s[0] for s in scored_nn])) if scored_nn else float("nan")
                evt = {
                    "event": "gen_summary",
                    "gen": int(gen),
                    "med": round(med, 4) if np.isfinite(med) else med,
                    "sec": round(time.time() - t0, 3),
                    "tree": int(best_sz),  # reuse field name for size
                    "best": {
                        "fit": round(float(best_f), 4),
                        "trd": int(best_tr),
                        "shrp": round(float(best_m.get("sharpe", 0.0)), 4),
                        "srtn": round(float(best_m.get("sortino", 0.0)), 4),
                        "cum": round(float(best_m.get("cum_return", 0.0)), 4),
                        "dd": round(float(best_m.get("max_drawdown", 0.0)), 4),
                        "tf": round(float(best_tf), 4),
                    },
                }
                print(json.dumps(evt), flush=True)
            # Evolve NN population
            elite_n = int(cfg.elite_count) if int(cfg.elite_count) > 0 else max(1, int(cfg.elite_frac * cfg.population))
            elite_n = min(max(1, elite_n), cfg.population - 1 if cfg.population > 1 else 1)
            next_pop_nn: List[NNGenome] = []
            for i in range(elite_n):
                next_pop_nn.append(scored_nn[i][1])
            while len(next_pop_nn) < cfg.population:
                parents = _random.sample(scored_nn[: max(10, cfg.population)], 2)
                pa = parents[0][1]; pb = parents[1][1]
                c1, c2 = _nn_crossover(pa, pb)
                c1 = _nn_mutate(c1, p_mut=float(cfg.nn_mutation_prob), w_sigma=float(cfg.nn_mutation_sigma), aff_sigma=float(cfg.nn_affine_sigma))
                next_pop_nn.append(c1)
                if len(next_pop_nn) < cfg.population:
                    c2 = _nn_mutate(c2, p_mut=float(cfg.nn_mutation_prob), w_sigma=float(cfg.nn_mutation_sigma), aff_sigma=float(cfg.nn_affine_sigma))
                    next_pop_nn.append(c2)
            pop_nn = next_pop_nn

    # Final evaluation and return
    val_metrics: Optional[Dict[str, float]] = None
    final_population: List[Dict[str, Any]] = []
    selection: Optional[Dict[str, Any]] = None
    if not is_nn:
        final_scores: List[
            Tuple[float, Tuple[Node, Node, Optional[Node], Optional[Node]], Dict[str, float], int]
        ] = []
        for ind in pop:
            fval, met, tr = fitness(ind[0], ind[1], ind[2], ind[3], X_train, close_col, cfg)
            final_scores.append((fval, ind, met, tr))
        final_scores.sort(key=lambda x: x[0], reverse=True)
        bf, bind, bm, btr = final_scores[0]
        if X_val is not None and len(X_val) > 0:
            pos_val_best = positions_from_tree_three_state(bind[0], bind[1], bind[2], bind[3], X_val, cfg.min_hold_candles, cfg.cooldown_candles)
            val_metrics = metrics_from_positions(X_val[close_col], pos_val_best, cost_bps=cfg.cost_bps)
            eval_scores = final_scores if int(final_top_k) <= 0 else final_scores[: int(final_top_k)]
            cand_list: List[Dict[str, Any]] = []
            for rank, (fit_val, ind, train_m, train_tr) in enumerate(eval_scores, start=1):
                pos_val_i = positions_from_tree_three_state(ind[0], ind[1], ind[2], ind[3], X_val, cfg.min_hold_candles, cfg.cooldown_candles)
                val_m_i = metrics_from_positions(X_val[close_col], pos_val_i, cost_bps=cfg.cost_bps)
                pos_train_i = positions_from_tree_three_state(ind[0], ind[1], ind[2], ind[3], X_train, cfg.min_hold_candles, cfg.cooldown_candles)
                extras_train = compute_extras(X_train[close_col], pos_train_i, cost_bps=cfg.cost_bps)
                extras_val = compute_extras(X_val[close_col], pos_val_i, cost_bps=cfg.cost_bps)
                tsize = node_size(ind[0]) + node_size(ind[1]) + (node_size(ind[2]) if ind[2] else 0) + (node_size(ind[3]) if ind[3] else 0)
                tdepth = max(
                    node_depth(ind[0]), node_depth(ind[1]),
                    node_depth(ind[2]) if ind[2] else 0,
                    node_depth(ind[3]) if ind[3] else 0,
                )
                tf_train = trade_frequency_score(int(train_tr), len(X_train))
                tf_val = trade_frequency_score(int(val_m_i.get("trades", 0.0)), len(X_val)) if X_val is not None else float("nan")
                ff_score = None
                if cfg.score_fn is not None:
                    try:
                        ff_score = float(cfg.score_fn(train_m, {"tf": float(tf_train)}, int(tsize), int(tdepth)))
                    except Exception:
                        ff_score = None
                rec_i = {
                    "rank": int(rank),
                    "fitness_train": float(fit_val),
                    "metrics_train": train_m,
                    "trades_train": int(train_tr),
                    "metrics_val": val_m_i,
                    "trades_val": int(val_m_i.get("trades", 0.0)),
                    "tree_size": int(tsize),
                    "tree_depth": int(tdepth),
                    "tf_train": float(tf_train),
                    "tf_val": float(tf_val),
                    "ff_score": (float(ff_score) if ff_score is not None else None),
                    "extras_train": extras_train,
                    "extras_val": extras_val,
                    "tree": {
                        "entry_long": pretty(ind[0]),
                        "entry_short": pretty(ind[1]),
                        "exit_long": pretty(ind[2]) if ind[2] else None,
                        "exit_short": pretty(ind[3]) if ind[3] else None,
                    },
                }
                final_population.append(rec_i)
                cand_list.append({
                    "pos_train": pos_train_i,
                    "pos_val": pos_val_i,
                    "train_m": train_m,
                    "val_m": val_m_i,
                    "train_tr": int(train_tr),
                    "fit": float(fit_val),
                    "ff": (float(ff_score) if ff_score is not None else float("nan")),
                    "size": int(tsize),
                    "depth": int(tdepth),
                })
            if int(select_top_k) > 0:
                def eligible(c: Dict[str, Any]) -> bool:
                    ok = (float(c["train_m"].get("sharpe", 0.0)) >= float(select_train_sharpe_min)) and (int(c["train_tr"]) >= int(select_train_trades_min))
                    if not ok:
                        return False
                    if str(select_by) == "ff" and np.isfinite(c["ff"]):
                        return c["ff"] >= float(select_ff_threshold)
                    return True
                pool = [c for c in cand_list if eligible(c)]
                if pool:
                    key = (lambda c: c["ff"]) if (str(select_by) == "ff" and all(np.isfinite(x["ff"]) for x in pool)) else (lambda c: c["fit"])
                    pool.sort(key=key, reverse=True)
                    take = pool[: int(select_top_k)]
                    sum_train = np.sum([t["pos_train"].to_numpy() for t in take], axis=0)
                    sum_val = np.sum([t["pos_val"].to_numpy() for t in take], axis=0)
                    ens_train = pd.Series(np.sign(sum_train), index=X_train.index)
                    ens_val = pd.Series(np.sign(sum_val), index=X_val.index)
                    sel_train_m = metrics_from_positions(X_train[close_col], ens_train, cost_bps=cfg.cost_bps)
                    sel_val_m = metrics_from_positions(X_val[close_col], ens_val, cost_bps=cfg.cost_bps)
                    selection = {
                        "active": True,
                        "k": int(len(take)),
                        "by": str(select_by),
                        "ff_threshold": float(select_ff_threshold),
                        "train_metrics": sel_train_m,
                        "val_metrics": sel_val_m,
                    }
                else:
                    selection = {"active": False, "k": 0, "by": str(select_by), "ff_threshold": float(select_ff_threshold)}
        return {
            "best_fitness": float(bf),
            "best_tree": {
                "entry_long": pretty(bind[0]),
                "entry_short": pretty(bind[1]),
                "exit_long": pretty(bind[2]) if bind[2] else None,
                "exit_short": pretty(bind[3]) if bind[3] else None,
            },
            "train_metrics": bm,
            "train_trades": int(btr),
            "val_metrics": val_metrics,
            "final_population": final_population,
            "selection": selection,
        }
    else:
        final_scores_nn: List[Tuple[float, NNGenome, Dict[str, float], int, int, int]] = []
        for g in pop_nn:
            fval, met, tr, sz, dp = fitness_nn(g, X_train, feature_names, close_col, cfg)
            final_scores_nn.append((fval, g, met, tr, sz, dp))
        final_scores_nn.sort(key=lambda x: x[0], reverse=True)
        bf, bg, bm, btr, bsz, bdp = final_scores_nn[0]
        if X_val is not None and len(X_val) > 0:
            out_best = _nn_forward(bg, X_val, feature_names, input_norm=str(cfg.input_norm))
            pos_val_best = positions_from_nn_hysteresis(out_best)
            pos_val_best.index = X_val.index
            val_metrics = metrics_from_positions(X_val[close_col], pos_val_best, cost_bps=cfg.cost_bps)
            eval_scores = final_scores_nn if int(final_top_k) <= 0 else final_scores_nn[: int(final_top_k)]
            cand_list_nn: List[Dict[str, Any]] = []
            for rank, (fit_val, g, train_m, train_tr, sz, dp) in enumerate(eval_scores, start=1):
                out_i = _nn_forward(g, X_val, feature_names, input_norm=str(cfg.input_norm))
                pos_val_i = positions_from_nn_hysteresis(out_i)
                pos_val_i.index = X_val.index
                val_m_i = metrics_from_positions(X_val[close_col], pos_val_i, cost_bps=cfg.cost_bps)
                # Train extras
                out_tr = _nn_forward(g, X_train, feature_names, input_norm=str(cfg.input_norm))
                pos_tr = positions_from_nn_hysteresis(out_tr)
                pos_tr.index = X_train.index
                extras_train = compute_extras(X_train[close_col], pos_tr, cost_bps=cfg.cost_bps)
                extras_val = compute_extras(X_val[close_col], pos_val_i, cost_bps=cfg.cost_bps)
                tf_train = trade_frequency_score(int(train_tr), len(X_train))
                tf_val = trade_frequency_score(int(val_m_i.get("trades", 0.0)), len(X_val)) if X_val is not None else float("nan")
                tsize = int(sz)
                tdepth = int(dp)
                ff_score = None
                if cfg.score_fn is not None:
                    try:
                        ff_score = float(cfg.score_fn(train_m, {"tf": float(tf_train)}, int(tsize), int(tdepth)))
                    except Exception:
                        ff_score = None
                rec_i = {
                    "rank": int(rank),
                    "fitness_train": float(fit_val),
                    "metrics_train": train_m,
                    "trades_train": int(train_tr),
                    "metrics_val": val_m_i,
                    "trades_val": int(val_m_i.get("trades", 0.0)),
                    "tree_size": int(sz),  # reuse keys for compatibility
                    "tree_depth": int(dp),
                    "tf_train": float(tf_train),
                    "tf_val": float(tf_val),
                    "ff_score": (float(ff_score) if ff_score is not None else None),
                    "extras_train": extras_train,
                    "extras_val": extras_val,
                    "nn": {
                        "arch": list(g.arch),
                        "params": int(sz),
                    },
                }
                final_population.append(rec_i)
                cand_list_nn.append({
                    "pos_train": pos_tr,
                    "pos_val": pos_val_i,
                    "train_m": train_m,
                    "val_m": val_m_i,
                    "train_tr": int(train_tr),
                    "fit": float(fit_val),
                    "ff": (float(ff_score) if ff_score is not None else float("nan")),
                    "size": int(tsize),
                    "depth": int(tdepth),
                })
            if int(select_top_k) > 0:
                def eligible(c: Dict[str, Any]) -> bool:
                    ok = (float(c["train_m"].get("sharpe", 0.0)) >= float(select_train_sharpe_min)) and (int(c["train_tr"]) >= int(select_train_trades_min))
                    if not ok:
                        return False
                    if str(select_by) == "ff" and np.isfinite(c["ff"]):
                        return c["ff"] >= float(select_ff_threshold)
                    return True
                pool = [c for c in cand_list_nn if eligible(c)]
                if pool:
                    key = (lambda c: c["ff"]) if (str(select_by) == "ff" and all(np.isfinite(x["ff"]) for x in pool)) else (lambda c: c["fit"])
                    pool.sort(key=key, reverse=True)
                    take = pool[: int(select_top_k)]
                    sum_train = np.sum([t["pos_train"].to_numpy() for t in take], axis=0)
                    sum_val = np.sum([t["pos_val"].to_numpy() for t in take], axis=0)
                    ens_train = pd.Series(np.sign(sum_train), index=X_train.index)
                    ens_val = pd.Series(np.sign(sum_val), index=X_val.index)
                    sel_train_m = metrics_from_positions(X_train[close_col], ens_train, cost_bps=cfg.cost_bps)
                    sel_val_m = metrics_from_positions(X_val[close_col], ens_val, cost_bps=cfg.cost_bps)
                    selection = {
                        "active": True,
                        "k": int(len(take)),
                        "by": str(select_by),
                        "ff_threshold": float(select_ff_threshold),
                        "train_metrics": sel_train_m,
                        "val_metrics": sel_val_m,
                    }
                else:
                    selection = {"active": False, "k": 0, "by": str(select_by), "ff_threshold": float(select_ff_threshold)}
        return {
            "best_fitness": float(bf),
            "best_tree": {"nn_summary": f"NN[inputs={len(feature_names)}, arch={'-'.join(map(str, list(cfg.nn_arch)+[1]))}, params={int(bsz)}]"},
            "train_metrics": bm,
            "train_trades": int(btr),
            "val_metrics": val_metrics,
            "final_population": final_population,
            "selection": selection,
        }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-Forward Optimization driver for gp_optimize")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--start", required=True, help="Global start timestamp (e.g., 2020-01-01)")
    p.add_argument("--end", required=True, help="Global end timestamp (e.g., 2024-12-31)")
    # Window sizing (months or weeks)
    p.add_argument("--train-months", type=int, default=0)
    p.add_argument("--val-months", type=int, default=0)
    p.add_argument("--step-months", type=int, default=0)
    p.add_argument("--train-weeks", type=int, default=6)
    p.add_argument("--val-weeks", type=int, default=1)
    p.add_argument("--step-weeks", type=int, default=1)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--min-hold", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=12)
    p.add_argument("--subsample", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--windows-limit", type=int, default=0, help="Optional: limit number of windows for quick runs")
    p.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(__file__), "runs", "wfo", "gp"))
    p.add_argument("--no-gen-logs", action="store_true", help="Do not print per-generation summaries")
    p.add_argument("--final-top-k", type=int, default=0, help="Save only top-K of final population per window (0=all)")
    # Selection/ensemble options
    p.add_argument("--select-top-k", type=int, default=0, help="If >0, build ensemble from top-K candidates per window")
    p.add_argument("--select-by", default="ff", choices=["ff", "fitness"], help="Ranking metric for selection: learned FF or train fitness")
    p.add_argument("--select-ff-threshold", type=float, default=0.0, help="Minimum FF score to include in ensemble (0-1 for FF models)")
    p.add_argument("--select-train-sharpe-min", type=float, default=0.0, help="Minimum train Sharpe for selection eligibility")
    p.add_argument("--select-train-trades-min", type=int, default=5, help="Minimum train trades for selection eligibility")
    # Strategy representation
    p.add_argument("--strategy-type", default="logic", choices=["logic", "nn"], help="Strategy representation: logic trees or neural-net genome")
    p.add_argument("--nn-arch", default="64,32", help="Hidden layer sizes for NN, comma-separated (e.g., 256,128,64)")
    p.add_argument("--input-norm", default="rolling", choices=["rolling", "affine"], help="Input normalization for NN inputs")
    p.add_argument("--nn-mutation-prob", type=float, default=0.3)
    p.add_argument("--nn-mutation-sigma", type=float, default=0.05)
    p.add_argument("--nn-affine-sigma", type=float, default=0.02)
    # Fitness Function learning (FF) options
    p.add_argument("--ff-enable", action="store_true", help="Enable learned fitness function scoring from past windows")
    p.add_argument("--ff-mode", default="linear", choices=["linear", "nn"], help="FF model type: linear logistic or small NN")
    p.add_argument("--ff-min-rows", type=int, default=200, help="Minimum rows from past windows to train FF")
    p.add_argument("--ff-warmup-windows", type=int, default=5, help="Use first N windows to collect data before training FF")
    p.add_argument("--ff-freeze", action="store_true", help="Use a frozen FF model (no updates during this run)")
    p.add_argument("--ff-save", action="store_true", help="Save the trained FF model to disk at end (default true)")
    p.add_argument("--ff-load", default="", help="Path to load a previously saved FF model")
    # Memory controls
    p.add_argument("--per-window-build", action="store_true", help="Build features per window to reduce RAM")
    p.add_argument("--warmup-days", type=int, default=10, help="Warmup days before train start for indicators/z-scores")
    # Reporting scaling
    p.add_argument("--account", type=float, default=10000.0, help="Account base currency size for scaling PnL")
    p.add_argument("--leverage", type=float, default=50.0, help="Leverage multiplier for scaled returns")
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    # Suppress noisy numpy RuntimeWarnings that don't affect control flow
    import warnings as _warnings
    _warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
    _warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
    _warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    start_ts = pd.to_datetime(a.start, utc=True)
    end_ts = pd.to_datetime(a.end, utc=True)
    
    def _align_to_next_saturday(ts: pd.Timestamp) -> pd.Timestamp:
        # Saturday=5 in weekday() where Monday=0
        delta = (5 - int(ts.weekday())) % 7
        return (ts + pd.Timedelta(days=delta)).normalize()

    close_col = "EUR_USD_close"
    X_all = None
    feats = None
    if not a.per_window_build:
        # Build features once for the entire range
        feats = build_features(
            a.data_dir,
            start=a.start,
            end=a.end,
            subsample=int(a.subsample) if a.subsample else 1,
            max_rows=None,
        )
        X_all = feats["X"]

    # Decide granularity
    if int(a.train_months) > 0 and int(a.val_months) > 0 and int(a.step_months) > 0:
        unit = "months"
        n_train, n_val, n_step = int(a.train_months), int(a.val_months), int(a.step_months)
    else:
        unit = "weeks"
        n_train, n_val, n_step = int(a.train_weeks), int(a.val_weeks), int(a.step_weeks)
        # Align the starting timestamp to the nearest future Saturday
        start_ts = _align_to_next_saturday(start_ts)
    windows = generate_windows(start_ts, end_ts, n_train, n_val, n_step, unit=unit)
    if int(a.windows_limit) > 0:
        windows = windows[: int(a.windows_limit)]

    run_dir = os.path.join(str(a.checkpoint_dir), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    meta: Dict[str, Any] = {"args": vars(a), "windows": len(windows)}
    if X_all is not None and feats is not None:
        meta.update({
            "features": int(X_all.shape[1]),
            "loaded_instruments": feats.get("loaded"),
            "missing_instruments": feats.get("missing"),
        })
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    start_evt = {"event": "wfo_start", "run_dir": run_dir, "windows": len(windows), "unit": unit, "train_n": n_train, "val_n": n_val, "step_n": n_step}
    if unit == "weeks":
        start_evt["start_aligned"] = str(start_ts)
    if X_all is not None:
        start_evt["features"] = int(X_all.shape[1])
    print(json.dumps(start_evt), flush=True)

    # Prepare GA/NN config template
    cfg = GPConfig(
        population=int(a.population),
        generations=int(a.generations),
        max_depth=int(a.max_depth),
        cost_bps=float(a.cost_bps),
        min_hold_candles=(0 if str(a.strategy_type) == "nn" else int(a.min_hold)),
        cooldown_candles=(0 if str(a.strategy_type) == "nn" else int(a.cooldown)),
        strategy_type=str(a.strategy_type),
        nn_arch=[int(x) for x in str(a.nn_arch).split(',') if str(x).strip()],
        input_norm=str(a.input_norm),
        nn_mutation_prob=float(a.nn_mutation_prob),
        nn_mutation_sigma=float(a.nn_mutation_sigma),
        nn_affine_sigma=float(a.nn_affine_sigma),
    )

    # Iterate windows
    out_path = os.path.join(run_dir, "windows.jsonl")
    train_sharpes: List[float] = []
    val_sharpes: List[float] = []
    train_cumrets: List[float] = []
    val_cumrets: List[float] = []

    # Optional: load a saved FF model
    ff_model: Optional[Callable[[Dict[str, float], Dict[str, float], int, int], float]] = None
    if a.ff_enable and a.ff_load:
        try:
            with open(a.ff_load, "r", encoding="utf-8") as lf:
                blob = json.load(lf)
            mode = str(blob.get("mode", "linear"))
            data = blob.get("data", {})
            if mode == "linear":
                mu = np.array(data["mu"], dtype=float)
                sigma = np.array(data["sigma"], dtype=float)
                w = np.array(data["w"], dtype=float)
                b = float(data["b"])
                model = _FFLinearLogistic(mu, sigma, w, b)
                ff_model = lambda m, e, s, d: model.score(_ff_extract_features_from_train_metrics(m, int(s), int(d), float(e.get("tf", 0.0))))
            elif mode == "nn":
                mu = np.array(data["mu"], dtype=float)
                sigma = np.array(data["sigma"], dtype=float)
                W1 = np.array(data["W1"], dtype=float)
                b1 = np.array(data["b1"], dtype=float)
                W2 = np.array(data["W2"], dtype=float)
                b2 = np.array(data["b2"], dtype=float)
                class _Loaded:
                    def __init__(self, mu, sigma, W1, b1, W2, b2):
                        self.mu, self.sigma, self.W1, self.b1, self.W2, self.b2 = mu, sigma, W1, b1, W2, b2
                    def score(self, feats):
                        z = (np.array(feats) - self.mu) / (self.sigma + 1e-12)
                        Z1 = z @ self.W1 + self.b1
                        H = np.maximum(0.0, Z1)
                        Z2 = H @ self.W2 + self.b2
                        return float(1.0 / (1.0 + np.exp(-Z2)))
                model = _Loaded(mu, sigma, W1, b1, W2, b2)
                ff_model = lambda m, e, s, d: model.score(_ff_extract_features_from_train_metrics(m, int(s), int(d), float(e.get("tf", 0.0))))
            start_evt["ff_loaded"] = True
            start_evt["ff_mode"] = mode
        except Exception:
            ff_model = None
            start_evt["ff_loaded"] = False
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (ts_train_s, ts_train_e, ts_val_s, ts_val_e) in enumerate(windows):
            # Build/slice features for this window
            if a.per_window_build:
                build_start = ts_train_s - pd.Timedelta(days=int(a.warmup_days))
                if build_start < start_ts:
                    build_start = start_ts
                build_end = ts_val_e
                try:
                    feats_win = build_features(
                        a.data_dir,
                        start=str(build_start),
                        end=str(build_end),
                        subsample=int(a.subsample) if a.subsample else 1,
                        max_rows=None,
                    )
                except Exception as e:
                    print(json.dumps({
                        "event": "wfo_build_error",
                        "i": i,
                        "error": str(e),
                        "build_start": str(build_start),
                        "build_end": str(build_end),
                    }), flush=True)
                    continue
                X_win = feats_win["X"]
                X_train = X_win[(X_win.index >= ts_train_s) & (X_win.index <= ts_train_e)]
                X_val = X_win[(X_win.index >= ts_val_s) & (X_win.index <= ts_val_e)]
            else:
                X_train = X_all[(X_all.index >= ts_train_s) & (X_all.index <= ts_train_e)]  # type: ignore[index]
                X_val = X_all[(X_all.index >= ts_val_s) & (X_all.index <= ts_val_e)]  # type: ignore[index]
            if len(X_train) == 0 or len(X_val) == 0:
                continue

            window_seed = int(a.seed) + i
            t0 = time.time()
            try:
                # Optional: build/update learned fitness function from past windows
                score_fn: Optional[Callable[[Dict[str, float], Dict[str, float], int, int], float]] = None
                if a.ff_enable and i >= int(a.ff_warmup_windows):
                    # Load past windows' final_population from the output file itself for simplicity
                    # We reopen and read existing lines (past windows only)
                    try:
                        if a.ff_freeze and ff_model is not None:
                            cfg.score_fn = ff_model
                        else:
                            with open(out_path, "r", encoding="utf-8") as rf:
                                past: List[Dict[str, Any]] = []
                                k = 0
                                for line_p in rf:
                                    line_p = line_p.strip()
                                    if not line_p:
                                        continue
                                    past.append(json.loads(line_p))
                                    k += 1
                                    if k >= i:  # only past entries
                                        break
                            score_fn = _build_ff_model_from_past(past, mode=str(a.ff_mode), min_rows=int(a.ff_min_rows))
                            if score_fn is not None:
                                cfg.score_fn = score_fn
                                if a.ff_freeze:
                                    ff_model = score_fn
                            else:
                                cfg.score_fn = ff_model if a.ff_freeze else None
                    except Exception:
                        cfg.score_fn = ff_model if a.ff_freeze else None
                result = run_gp_once(
                    X_train,
                    X_val,
                    close_col,
                    cfg,
                    seed=window_seed,
                    log_gens=(not a.no_gen_logs),
                    final_top_k=int(a.final_top_k),
                    select_top_k=int(a.select_top_k),
                    select_by=str(a.select_by),
                    select_ff_threshold=float(a.select_ff_threshold),
                    select_train_sharpe_min=float(a.select_train_sharpe_min),
                    select_train_trades_min=int(a.select_train_trades_min),
                )
            except Exception as e:
                # Emit a debug event and continue to next window
                print(json.dumps({
                    "event": "wfo_error",
                    "i": i,
                    "error": str(e),
                    "train_rows": int(len(X_train)),
                    "val_rows": int(len(X_val)) if X_val is not None else 0,
                }), flush=True)
                continue
            dt = round(time.time() - t0, 3)

            tm = result.get("train_metrics", {}) or {}
            vm = result.get("val_metrics", {}) or {}
            train_sh = float(tm.get("sharpe", 0.0))
            val_sh = float(vm.get("sharpe", 0.0)) if vm else float("nan")
            train_cr = float(tm.get("cum_return", 0.0))
            val_cr = float(vm.get("cum_return", 0.0)) if vm else float("nan")
            deg_sh = (val_sh / train_sh) if (not math.isnan(val_sh) and abs(train_sh) > 1e-12) else float("nan")
            deg_cr = (val_cr / train_cr) if (not math.isnan(val_cr) and abs(train_cr) > 1e-12) else float("nan")

            # Scale cum returns into PnL estimates for readability (linear scaling)
            def scaled_fields(m: Dict[str, float]) -> Dict[str, float]:
                cr = float(m.get("cum_return", 0.0))
                scaled_cr = cr * float(a.leverage)
                pnl = scaled_cr * float(a.account)
                return {"cumret_scaled": scaled_cr, "pnl": pnl}

            sel = result.get("selection") or {}
            rec = {
                "window": int(i),
                "train": {"start": str(ts_train_s), "end": str(ts_train_e)},
                "val": {"start": str(ts_val_s), "end": str(ts_val_e)},
                "time_sec": float(dt),
                "best_tree": result["best_tree"],
                "train_metrics": {
                    **tm,
                    "sharpe": round(float(tm.get("sharpe", 0.0)), 4),
                    "cum_return": round(float(tm.get("cum_return", 0.0)), 4),
                    "max_drawdown": round(float(tm.get("max_drawdown", 0.0)), 4),
                },
                "val_metrics": (
                    {
                        **vm,
                        "sharpe": round(float(vm.get("sharpe", 0.0)), 4),
                        "cum_return": round(float(vm.get("cum_return", 0.0)), 4),
                        "max_drawdown": round(float(vm.get("max_drawdown", 0.0)), 4),
                    } if vm else None
                ),
                "deg": {"sharpe": (round(float(deg_sh), 4) if np.isfinite(deg_sh) else deg_sh), "cumret": (round(float(deg_cr), 4) if np.isfinite(deg_cr) else deg_cr)},
                "final_population": result.get("final_population", []),
                "selection": sel,
                "ff": {
                    "enabled": bool(a.ff_enable),
                    "mode": str(a.ff_mode) if a.ff_enable else None,
                    "trained": bool(cfg.score_fn is not None) if a.ff_enable else False,
                    "min_rows": int(a.ff_min_rows) if a.ff_enable else None,
                    "warmup_windows": int(a.ff_warmup_windows) if a.ff_enable else None,
                },
                "scaled": {
                    "train": {k: round(float(v), 4) for k, v in scaled_fields(tm).items()},
                    "val": ({k: round(float(v), 4) for k, v in scaled_fields(vm).items()} if vm else None),
                    "account": float(a.account),
                    "leverage": float(a.leverage),
                },
            }
            f.write(json.dumps(rec) + "\n")

            # Emit concise event with basic sanity fields
            # Colorized standout event for readability (yellow)
            Y = "\033[33m"; R = "\033[0m"
            evt = {
                "event": "wfo_window",
                "i": int(i),
                "train": {"shrp": round(train_sh, 4), "cum": round(train_cr, 4)},
                "val": {"shrp": round(val_sh, 4), "cum": round(val_cr, 4)},
                "deg": {"shrp": (round(deg_sh, 4) if np.isfinite(deg_sh) else deg_sh), "cum": (round(deg_cr, 4) if np.isfinite(deg_cr) else deg_cr)},
                "sec": float(dt),
                "rows": {"train": int(len(X_train)), "val": int(len(X_val))},
            }
            print(f"{Y}{json.dumps(evt)}{R}", flush=True)

            if not math.isnan(train_sh) and not math.isnan(val_sh):
                train_sharpes.append(train_sh)
                val_sharpes.append(val_sh)
            if not math.isnan(train_cr) and not math.isnan(val_cr):
                train_cumrets.append(train_cr)
                val_cumrets.append(val_cr)

            # Free per-window features to reduce RAM
            if a.per_window_build:
                del X_win
                del feats_win
                del X_train
                del X_val
                import gc as _gc
                _gc.collect()

    # Correlations and aggregates
    def safe_corr(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(x) != len(y):
            return float("nan")
        return float(np.corrcoef(np.array(x), np.array(y))[0, 1])

    corr_sh = safe_corr(train_sharpes, val_sharpes)
    corr_cr = safe_corr(train_cumrets, val_cumrets)

    summary = {
        "event": "wfo_summary",
        "run_dir": run_dir,
        "windows": len(windows),
        "corr": {"sharpe": corr_sh, "cumret": corr_cr},
        "stats": {
            "train_sharpe_mean": float(np.nanmean(train_sharpes)) if train_sharpes else float("nan"),
            "val_sharpe_mean": float(np.nanmean(val_sharpes)) if val_sharpes else float("nan"),
            "train_cumret_mean": float(np.nanmean(train_cumrets)) if train_cumrets else float("nan"),
            "val_cumret_mean": float(np.nanmean(val_cumrets)) if val_cumrets else float("nan"),
        },
    }

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary), flush=True)

    # Save FF model if enabled
    if a.ff_enable and (a.ff_save or not hasattr(a, 'ff_save')) and (ff_model is not None):
        try:
            save = {"mode": str(a.ff_mode), "data": {}}
            if str(a.ff_mode) == "linear" and isinstance(ff_model, _FFLinearLogistic):
                save["data"] = {
                    "mu": ff_model.mu.tolist(),
                    "sigma": ff_model.sigma.tolist(),
                    "w": ff_model.w.tolist(),
                    "b": ff_model.b,
                }
            else:
                # Best-effort extract from closures; not guaranteed
                # Re-train a compact model from all windows to serialize
                try:
                    with open(out_path, "r", encoding="utf-8") as rf:
                        past: List[Dict[str, Any]] = [json.loads(l) for l in rf if l.strip()]
                    model = _build_ff_model_from_past(past, mode=str(a.ff_mode), min_rows=int(a.ff_min_rows))
                    if isinstance(model, _FFLinearLogistic):
                        save["mode"] = "linear"
                        save["data"] = {
                            "mu": model.mu.tolist(),
                            "sigma": model.sigma.tolist(),
                            "w": model.w.tolist(),
                            "b": model.b,
                        }
                except Exception:
                    pass
            ff_path = os.path.join(run_dir, "ff_model.json")
            with open(ff_path, "w", encoding="utf-8") as sf:
                json.dump(save, sf)
            print(json.dumps({"event": "ff_saved", "file": ff_path, "mode": save.get("mode")}), flush=True)
        except Exception as e:
            print(json.dumps({"event": "ff_save_error", "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
