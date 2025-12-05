from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple, List, Any
import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    cum_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    trades: float
    time_in_market: float
    num_trade_days: int
    profit_factor: float
    num_profit_trades: int
    num_loss_trades: int
    win_loss_ratio: float
    win_rate: float
    equity_r2: float


def compute_portfolio_metrics(closes: pd.DataFrame, positions: pd.DataFrame, trade_cost: float) -> PortfolioMetrics:
    # Fast numpy path
    px = closes.to_numpy(dtype=np.float32, copy=False)
    pos = positions.to_numpy(dtype=np.float32, copy=False)
    # returns per bar per instrument
    ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
    # align positions (current bar decisions act on next bar)
    pos_aligned = pos[1:, :]
    strat_ret = pos_aligned * ret
    mean_across_inst = np.mean(strat_ret, axis=1)
    # apply cost per flip
    flips = (pos[1:, :] != pos[:-1, :])
    flips_count = flips.sum(axis=1).astype(np.float32)
    if trade_cost and float(trade_cost) != 0.0 and positions.shape[1] > 0:
        mean_across_inst = mean_across_inst - (flips_count * float(trade_cost) / float(positions.shape[1]))
    # equity curve
    equity = np.cumprod(1.0 + mean_across_inst, dtype=np.float64)
    cum_return = float((equity[-1] - 1.0) if equity.size > 0 else 0.0)
    vals = mean_across_inst.astype(np.float64)
    vol = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    mean_ret = float(np.mean(vals)) if vals.size > 0 else 0.0
    sharpe = float((mean_ret / vol) * np.sqrt(252 * 288)) if vol > 1e-12 else 0.0
    neg_vals = vals[vals < 0.0]
    dstd = float(np.std(neg_vals, ddof=1)) if neg_vals.size > 1 else 0.0
    sortino = float((mean_ret / dstd) * np.sqrt(252 * 288)) if dstd > 1e-12 else 0.0
    # drawdown
    if equity.size > 0:
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity / np.clip(running_max, 1e-12, None)) - 1.0
        max_dd = float(np.min(drawdown))
    else:
        max_dd = 0.0
    trades = float(flips.sum())
    # Time in market (average fraction across instruments)
    T = float(pos.shape[0])
    if T > 0 and pos.shape[1] > 0:
        in_mkt_per_inst = (pos != 0.0).sum(axis=0).astype(np.float32) / T
        tim = float(np.mean(in_mkt_per_inst))
    else:
        tim = 0.0
    # Number of trading days present in closes index (count weekdays with any bars)
    try:
        idx_days = pd.DatetimeIndex(closes.index).normalize().unique()
        num_days = int(sum(d.weekday() < 5 for d in idx_days))
    except Exception:
        num_days = 0
    return PortfolioMetrics(
        cum_return=cum_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        trades=trades,
        time_in_market=tim,
        num_trade_days=num_days,
        profit_factor=0.0,
        num_profit_trades=0,
        num_loss_trades=0,
        win_loss_ratio=0.0,
        win_rate=0.0,
        equity_r2=0.0,
    )


def ballast_score_old(metrics: PortfolioMetrics, trades: Optional[List[TradeRecord]] = None, size: int = 0, depth: int = 0) -> float:
    sharpe = float(metrics.sharpe)
    sortino = float(metrics.sortino)
    cumret = float(metrics.cum_return)
    max_dd = float(metrics.max_drawdown)
    if trades is not None and len(trades) > 0:
        unique_insts = {t.instrument for t in trades}
        bars_total = float(sum(t.bars for t in trades))
        bars_per_inst = bars_total / max(1.0, float(len(unique_insts)))
        days = bars_per_inst / 288.0 if bars_per_inst > 0 else 0.0
        tpd = float(len(trades)) / max(1e-6, days)
        numer = (sharpe + 100.0 if sharpe >= 0.0 else 100.0) * (sortino + 100.0 if sortino >= 0.0 else 100.0) * ((cumret * 100.0) + 100.0 if cumret >= 0.0 else 100.0) * ((len(trades)/20.0) + 100) 
        denom = (abs(sharpe) + 100.0 if sharpe < 0.0 else 100.0) * (abs(sortino) + 100.0 if sortino < 0.0 else 100.0) * (abs(cumret * 100.0) + 100.0 if cumret < 0.0 else 100.0) * (100.0 + abs(max_dd) * 100.0)
        return float(numer / max(1e-12, denom))
        # return tpd
        # return len(trades)
    return -9.9


def ballast_score(
    metrics: PortfolioMetrics,
    trades: Optional[List[TradeRecord]] = None,
    size: int = 0,
    depth: int = 0,
) -> float:
    """Ballast fitness with daily trade pacing, normalized by instruments.

    - Counts trade entries per UTC day across all instruments, then normalizes by
      the number of unique instruments so that the target trades/day is per-instrument.
    - Rewards mean daily trades close to target with low variance.
    - Blends with risk terms (cum_return, sharpe/sortino, drawdown).
    """
    if not trades:
        return 0.0

    # Determine inclusive daily window
    start_day = min(t.entry_time.normalize() for t in trades)
    end_day = max(t.exit_time.normalize() for t in trades)
    if pd.isna(start_day) or pd.isna(end_day) or end_day < start_day:
        return 0.0
    days_index = pd.date_range(start=start_day, end=end_day, freq="D")

    # Derive number of instruments from trades
    unique_instruments = {t.instrument for t in trades}
    num_instruments = float(max(1, len(unique_instruments)))

    # Count entries per day (across all instruments), then normalize per instrument
    entry_days = pd.to_datetime([t.entry_time.normalize() for t in trades])
    counts = pd.Series(1.0, index=entry_days).groupby(level=0).sum()
    counts = counts.reindex(days_index, fill_value=0.0).astype(float)
    counts_per_inst = counts / num_instruments
    
    num_trades = len(trades)
    mean_trades_per_day = float(counts_per_inst.mean()) if len(counts_per_inst) > 0 else 0.0
    std_trades_per_day = float(counts_per_inst.std(ddof=0)) if len(counts_per_inst) > 0 else 0.0

    # Target behavior per instrument: ~5 trades/day with tolerance ~2
    target = 5.0
    tol = 2.0
    proximity = float(np.exp(-0.5 * ((mean_trades_per_day - target) / max(1e-6, tol)) ** 2))
    consistency = float(np.exp(-std_trades_per_day / max(1e-6, target)))

    # Risk terms
    sharpe = float(metrics.sharpe)
    sortino = float(metrics.sortino)
    cumret = float(metrics.cum_return)
    max_dd = float(metrics.max_drawdown)
    tim = float(getattr(metrics, "time_in_market", 0.0))
    # Additional trade/equity quality terms (populated in evaluate_genome)
    pf = float(getattr(metrics, "profit_factor", 0.0))
    win_rate = float(getattr(metrics, "win_rate", 0.0))
    win_loss_ratio = float(getattr(metrics, "win_loss_ratio", 0.0))
    num_profit_trades = int(getattr(metrics, "num_profit_trades", 0))
    num_loss_trades = int(getattr(metrics, "num_loss_trades", 0))
    equity_r2 = float(getattr(metrics, "equity_r2", 0.0))

    # Trading days from metrics if available; fallback to days_index length filtered to weekdays
    num_days = int(getattr(metrics, "num_trade_days", 0))
    if num_days <= 0:
        num_days = int(sum(d.weekday() < 5 for d in days_index))

    # Trades score relative to expected trades = (num_days * num_instruments)
    target_num_trades = max(1e-6, float(num_days) * float(num_instruments))
    trdscr_raw = (float(num_trades) - target_num_trades) / target_num_trades
    trdscr = float(np.clip(trdscr_raw, 0.0, 10.0))

    # Compose score (numerator upweights good behavior; denominator penalizes risk)
    pf_cap = float(np.clip(pf, 0.0, 3.0))           # cap PF to 3
    wr_clamp = float(np.clip(win_rate, 0.0, 1.0))   # [0,1]
    r2_clamp = float(np.clip(equity_r2, 0.0, 1.0))  # [0,1]

    numer = (
        (sharpe + 100.0 if sharpe >= 0.0 else 100.0)
        * (sortino + 100.0 if sortino >= 0.0 else 100.0)
        * ((cumret * 100.0) + 100.0 if cumret >= 0.0 else 100.0)
        * ((trdscr * 10.0) + 100.0 if trdscr >= 0.0 else 100.0)
        * (100.0 + wr_clamp * 20.0)
        * (100.0 + pf_cap * 10.0)
        * 100.0
    )
    denom = (
        (abs(sharpe) + 100.0 if sharpe < 0.0 else 100.0)
        * (abs(sortino) + 100.0 if sortino < 0.0 else 100.0)
        * (abs(cumret * 100.0) + 100.0 if cumret < 0.0 else 100.0)
        * (abs(trdscr * 10.0) + 100.0 if trdscr < 0.0 else 100.0)
        * (abs(max_dd * 100.0) + 100.0)
        * ((tim * 100.0) + 100.0)
        * (100.0 + (1.0 - r2_clamp) * 50.0)
    )
    score = float(numer / max(1e-12, denom))
    return score

    # (Unreachable legacy code removed)



@dataclass
class TradeRecord:
    instrument: str
    side: int  # +1 long, -1 short
    size: int  # +100 or -100
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_pnl: float  # fraction (e.g., 0.01 = +1%)
    net_pnl: float    # includes costs (enter+exit)
    bars: int


# Allow user to plug different fitness functions
FitnessFn = Callable[[PortfolioMetrics, List[TradeRecord], int, int], float]
