from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from model import MultiHeadGenome, forward, hysteresis_map
from fitness import compute_portfolio_metrics, PortfolioMetrics, FitnessFn, ballast_score, TradeRecord


@dataclass
class BacktestResult:
    metrics: PortfolioMetrics
    positions: pd.DataFrame
    trades: List[TradeRecord]


def _extract_trades(positions: pd.DataFrame, closes: pd.DataFrame, trade_cost: float) -> List[TradeRecord]:
    trades: List[TradeRecord] = []
    instruments = list(positions.columns)
    for inst in instruments:
        pos = positions[inst].astype(float)
        px = closes[inst].astype(float)
        # Identify state changes
        prev = 0.0
        entry_idx = None
        entry_price = 0.0
        entry_side = 0
        for t, v in enumerate(pos.values):
            if v != prev:
                # closing trade if previously in market
                if prev != 0.0 and entry_idx is not None:
                    exit_time = positions.index[t]
                    exit_price = float(px.iloc[t])
                    side = int(entry_side)
                    size = 100 if side > 0 else -100
                    gross = (exit_price / entry_price - 1.0) * side
                    # two-side cost (enter + exit)
                    net = gross - (2.0 * float(trade_cost))
                    trades.append(TradeRecord(
                        instrument=inst,
                        side=side,
                        size=size,
                        entry_time=positions.index[entry_idx],
                        exit_time=exit_time,
                        entry_price=float(entry_price),
                        exit_price=float(exit_price),
                        gross_pnl=float(gross),
                        net_pnl=float(net),
                        bars=int(t - entry_idx),
                    ))
                # open new if v != 0
                if v != 0.0:
                    entry_idx = t
                    entry_price = float(px.iloc[t])
                    entry_side = int(v)
                else:
                    entry_idx = None
                    entry_price = 0.0
                    entry_side = 0
                prev = float(v)
        # If position still open at end, close at last bar
        if prev != 0.0 and entry_idx is not None and len(pos) > entry_idx:
            t = len(pos) - 1
            exit_time = positions.index[t]
            exit_price = float(px.iloc[t])
            side = int(entry_side)
            size = 100 if side > 0 else -100
            gross = (exit_price / entry_price - 1.0) * side
            net = gross - (2.0 * float(trade_cost))
            trades.append(TradeRecord(
                instrument=inst,
                side=side,
                size=size,
                entry_time=positions.index[entry_idx],
                exit_time=exit_time,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                gross_pnl=float(gross),
                net_pnl=float(net),
                bars=int(t - entry_idx),
            ))
    return trades


def evaluate_genome(
    genome: MultiHeadGenome,
    X: pd.DataFrame,
    closes: pd.DataFrame,
    trade_cost: float,
    fitness_fn: FitnessFn | None = None,
    thresholds: tuple[float,float,float,float] | None = None,
    mode: str = "absolute",
    band_enter: float = 0.05,
    band_exit: float = 0.02,
) -> Tuple[float, BacktestResult]:
    out_vals = forward(genome, X)
    if thresholds is None:
        enter_long, exit_long, enter_short, exit_short = 0.7, 0.6, 0.3, 0.4
    else:
        enter_long, exit_long, enter_short, exit_short = thresholds
    pos_mat = hysteresis_map(out_vals, enter_long, exit_long, enter_short, exit_short, mode=mode, band_enter=band_enter, band_exit=band_exit)
    positions = pd.DataFrame(pos_mat, index=X.index, columns=closes.columns)
    metrics = compute_portfolio_metrics(closes, positions, trade_cost)
    trades = _extract_trades(positions, closes, trade_cost)
    # Enrich metrics with trade-level stats
    if trades:
        profits = np.array([t.net_pnl for t in trades], dtype=float)
        gains = profits[profits > 0]
        losses = -profits[profits < 0]
        pf = float(gains.sum() / max(1e-12, losses.sum())) if losses.size > 0 else float('inf') if gains.size > 0 else 0.0
        n_win = int((profits > 0).sum())
        n_loss = int((profits < 0).sum())
        wl = float(n_win / max(1, n_loss)) if n_loss > 0 else float('inf') if n_win > 0 else 0.0
        wr = float(n_win / max(1, len(profits)))
        # R^2 of equity vs time (linear trend fit)
        # Build equity using per-bar returns used in compute_portfolio_metrics
        px = closes.to_numpy(dtype=np.float32, copy=False)
        pos = positions.to_numpy(dtype=np.float32, copy=False)
        ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
        strat_ret = (pos[1:, :] * ret).mean(axis=1)
        if strat_ret.size > 0:
            eq = np.cumprod(1.0 + strat_ret)
            x = np.arange(eq.size, dtype=np.float64)
            y = eq.astype(np.float64)
            x_mean = x.mean(); y_mean = y.mean()
            cov = float(((x - x_mean) * (y - y_mean)).sum())
            varx = float(((x - x_mean) ** 2).sum())
            if varx > 1e-12:
                beta = cov / varx
                alpha = y_mean - beta * x_mean
                yhat = alpha + beta * x
                ss_res = float(((y - yhat) ** 2).sum())
                ss_tot = float(((y - y_mean) ** 2).sum())
                r2 = float(1.0 - ss_res / max(1e-12, ss_tot)) if ss_tot > 1e-12 else 0.0
            else:
                r2 = 0.0
        else:
            r2 = 0.0
        # Update metrics (dataclasses are mutable objects; set attributes)
        metrics.profit_factor = pf
        metrics.num_profit_trades = n_win
        metrics.num_loss_trades = n_loss
        metrics.win_loss_ratio = wl
        metrics.win_rate = wr
        metrics.equity_r2 = r2
    # size/depth are rough descriptors (optional)
    size = 0
    for h, head in enumerate(genome.weights):
        for W, b in zip(head, genome.biases[h]):
            size += int(W.size + b.size)
    depth = len(genome.arch) + 1
    scorer = fitness_fn or ballast_score
    score = scorer(metrics, trades, size, depth)
    return score, BacktestResult(metrics=metrics, positions=positions, trades=trades)
