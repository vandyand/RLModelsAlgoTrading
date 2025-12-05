#!/usr/bin/env python3
"""
Expert baselines for trading signals used in offline pretraining and
DAgger-style supervision during live trading.

Functions return labels in [0, 1] where 1 = long preference, 0 = short preference.
When no clear signal exists (neutral zone), functions may return None to indicate
no supervision should be applied.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple
import numpy as np


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(series, dtype=float)
    if len(series) == 0:
        return out
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out


def ma_crossover_label(close: Iterable[float], fast: int = 12, slow: int = 26, neutral_band: float = 0.0) -> Optional[float]:
    """Return latest MA-crossover label.

    - close: iterable of recent close prices (most recent last)
    - fast, slow: EMA spans
    - neutral_band: if |fast - slow| / close < neutral_band => no label (None)

    Returns: 1.0 (long), 0.0 (short), or None if neutral.
    """
    arr = np.asarray(list(close), dtype=float)
    if arr.size < max(fast, slow) + 2:
        return None
    f = _ema(arr, fast)
    s = _ema(arr, slow)
    diff = f[-1] - s[-1]
    denom = abs(arr[-1]) if arr[-1] != 0 else 1.0
    if neutral_band > 0.0 and abs(diff) / denom < neutral_band:
        return None
    return 1.0 if diff > 0 else 0.0


def rsi_label(close: Iterable[float], period: int = 14, overbought: float = 70.0, oversold: float = 30.0, neutral_margin: float = 5.0) -> Optional[float]:
    """Return latest RSI-based label.

    - If RSI > overbought: short (0.0)
    - If RSI < oversold: long (1.0)
    - Else None, unless near extremes by neutral_margin then weak signals are ignored.
    """
    arr = np.asarray(list(close), dtype=float)
    if arr.size < period + 2:
        return None
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    # Wilder's smoothing approximation
    avg_gain = gains[-period:].mean() if len(gains) >= period else gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses[-period:].mean() if len(losses) >= period else losses.mean() if len(losses) > 0 else 0.0
    avg_loss = max(avg_loss, 1e-12)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if rsi >= (overbought + neutral_margin):
        return 0.0
    if rsi <= (oversold - neutral_margin):
        return 1.0
    return None


def rsi_signal(close: Iterable[float], period: int = 14) -> Optional[float]:
    """Return normalized RSI in [0,1] or None if insufficient data.
    Values > 0.5 imply bullish tilt, < 0.5 bearish.
    """
    arr = np.asarray(list(close), dtype=float)
    if arr.size < period + 2:
        return None
    deltas = np.diff(arr)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = gains[-period:].mean() if len(gains) >= period else gains.mean() if len(gains) > 0 else 0.0
    avg_loss = losses[-period:].mean() if len(losses) >= period else losses.mean() if len(losses) > 0 else 0.0
    avg_loss = max(avg_loss, 1e-12)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi / 100.0, 0.0, 1.0))


def ma_crossover_filtered(
    closes: Iterable[float],
    fast: int = 12,
    slow: int = 26,
    neutral_band: float = 0.0002,
    rsi_period: int = 14,
    min_strength: float = 0.0005,
) -> Optional[float]:
    """MA crossover with RSI confirmation and strength filtering.

    Returns moderate labels close to 0.75/0.25 only when confirmation is strong.
    Otherwise returns None to avoid supervising on noise.
    """
    arr = np.asarray(list(closes), dtype=float)
    if arr.size < max(fast, slow) + 2:
        return None
    f = _ema(arr, fast)
    s = _ema(arr, slow)
    fast_ma = f[-1]
    slow_ma = s[-1]
    denom = abs(slow_ma) if slow_ma != 0 else 1.0
    cross_dir = (fast_ma - slow_ma) / denom
    strength = abs(fast_ma - slow_ma) / denom
    if abs(cross_dir) <= neutral_band or strength <= min_strength:
        return None
    rsi_val = rsi_signal(arr, rsi_period)
    if rsi_val is None:
        return None
    # Long confirmation
    if cross_dir > 0 and rsi_val > 0.55:
        return 0.75
    # Short confirmation
    if cross_dir < 0 and rsi_val < 0.45:
        return 0.25
    return None


def combined_label(close: Iterable[float], method: str = "ma", **kwargs) -> Optional[float]:
    """Convenience wrapper to compute label by method name.

    method in {"ma", "rsi", "ma_filt", "ma_filtered"}
    """
    if method == "ma":
        return ma_crossover_label(close, **kwargs)
    if method == "rsi":
        return rsi_label(close, **kwargs)
    if method in {"ma_filt", "ma_filtered"}:
        return ma_crossover_filtered(close, **kwargs)
    raise ValueError(f"Unknown expert method: {method}")
