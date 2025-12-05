#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from indicators import ema, rsi, macd, atr  # type: ignore
except Exception:  # pragma: no cover
    # Fallback when executed as a script via absolute path
    HERE = os.path.dirname(__file__)
    if HERE and HERE not in sys.path:
        sys.path.append(HERE)
    from indicators import ema, rsi, macd, atr  # type: ignore


@dataclass
class StrategyParams:
    rsi_buy_threshold: float = 55.0
    rsi_sell_threshold: float = 45.0
    ema_fast_period: int = 10
    ema_slow_period: int = 50
    use_macd_filter: int = 0  # 0 or 1


ANNUALIZATION_5M = 252 * 288  # ~ 72,576 bars/year


def load_candles(csv_path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df.index <= pd.to_datetime(end, utc=True)]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def evaluate_strategy(df: pd.DataFrame, p: StrategyParams, cost_bps: float = 0.0) -> Dict[str, float]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rsi14 = rsi(close, 14)
    ema_f = ema(close, p.ema_fast_period)
    ema_s = ema(close, p.ema_slow_period)
    m_line, m_sig, _ = macd(close)

    # Signal logic
    long_cond = (ema_f > ema_s) & (rsi14 > p.rsi_buy_threshold)
    short_cond = (ema_f < ema_s) & (rsi14 < p.rsi_sell_threshold)
    if int(p.use_macd_filter) == 1:
        long_cond &= (m_line > m_sig)
        short_cond &= (m_line < m_sig)

    position = pd.Series(0.0, index=df.index)
    position[long_cond] = 1.0
    position[short_cond] = -1.0

    # Use position decided at t-1 for return at t
    pos = position.shift(1).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    strat_ret = pos * ret

    # Transaction cost on position changes
    cost = float(cost_bps) * 1e-4
    if cost > 0:
        trade_flag = pos.ne(pos.shift(1)).fillna(False)
        strat_ret[trade_flag] = strat_ret[trade_flag] - cost

    equity = (1.0 + strat_ret).cumprod()
    cum_return = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0
    vol = float(strat_ret.std())
    sharpe = float((strat_ret.mean() / vol) * np.sqrt(ANNUALIZATION_5M)) if vol > 1e-12 else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0).fillna(0.0)
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    trades = int(pos.ne(pos.shift(1)).sum())

    return {
        "cum_return": cum_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": float(trades),
    }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest simple 5-parameter FX strategy on M5 candles")
    p.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "data", "EUR_USD_M5.csv"))
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--rsi-buy", type=float, default=55.0)
    p.add_argument("--rsi-sell", type=float, default=45.0)
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=50)
    p.add_argument("--use-macd-filter", action="store_true")
    p.add_argument("--cost-bps", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    df = load_candles(a.csv, start=a.start, end=a.end)
    # Ensure valid ordering constraints
    ema_fast = int(a.ema_fast)
    ema_slow = int(a.ema_slow)
    if ema_slow <= ema_fast:
        ema_slow = max(ema_fast + 1, 2)
    p = StrategyParams(
        rsi_buy_threshold=float(a.rsi_buy),
        rsi_sell_threshold=float(a.rsi_sell),
        ema_fast_period=ema_fast,
        ema_slow_period=ema_slow,
        use_macd_filter=1 if bool(a.use_macd_filter) else 0,
    )
    metrics = evaluate_strategy(df, p, cost_bps=float(a.cost_bps))
    print(json.dumps({"event": "backtest", "params": p.__dict__, "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
