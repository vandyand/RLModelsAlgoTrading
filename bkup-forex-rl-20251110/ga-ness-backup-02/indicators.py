from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False, min_periods=1).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / float(int(period))
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_rma(gain, int(period))
    avg_loss = _wilder_rma(loss, int(period)) + eps
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, int(fast))
    ema_slow = ema(close, int(slow))
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, int(signal))
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return _wilder_rma(tr, int(period))


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(int(period), min_periods=1).mean()


def roc(series: pd.Series, period: int) -> pd.Series:
    p = int(period)
    return (series / series.shift(p) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    p = int(period)
    dsm = int(d_smooth)
    ll = low.rolling(p, min_periods=1).min()
    hh = high.rolling(p, min_periods=1).max()
    k = 100.0 * (close - ll) / (hh - ll + 1e-12)
    d = k.rolling(dsm, min_periods=1).mean()
    return k, d


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    p = int(period)
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(p, min_periods=1).mean()
    md = (tp - sma_tp).abs().rolling(p, min_periods=1).mean()
    return (tp - sma_tp) / (0.015 * (md + 1e-12))


def bollinger_width(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    p = int(period)
    m = float(mult)
    sma_c = close.rolling(p, min_periods=1).mean()
    sd = close.rolling(p, min_periods=1).std().fillna(0.0)
    upper = sma_c + m * sd
    lower = sma_c - m * sd
    return (upper - lower)
