from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import os
import sys
# Ensure repo roots on path when executed directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore


@dataclass
class FeatureConfig:
    # Granularities to include and bar counts
    m5_bars: int = 600   # ~2 days
    h1_bars: int = 720   # 30 days
    d_bars: int = 1200   # ~5 years
    include_m5: bool = True
    include_h1: bool = True
    include_d: bool = True


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False, min_periods=1).mean()


def _rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / int(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / int(period), adjust=False).mean() + eps
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get('close', 'close')].astype(float)
    high = df[cols.get('high', 'high')].astype(float)
    low = df[cols.get('low', 'low')].astype(float)
    vol = df[cols.get('volume', 'volume')].astype(float)

    logc = np.log(np.maximum(close, 1e-12))
    r = logc.diff().fillna(0.0)

    def last_ret(k: int) -> pd.Series:
        return (logc - logc.shift(k)).fillna(0.0)

    def std_ret(k: int) -> pd.Series:
        return r.rolling(k, min_periods=1).std().fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20, min_periods=1).mean().fillna(0.0)

    roll_max20 = close.rolling(20, min_periods=1).max()
    roll_min20 = close.rolling(20, min_periods=1).min()
    roll_std20 = close.rolling(20, min_periods=1).std().fillna(1e-6)
    dist_max20 = (close - roll_max20) / (roll_std20 + 1e-12)
    dist_min20 = (close - roll_min20) / (roll_std20 + 1e-12)

    v_mean20 = vol.rolling(20, min_periods=1).mean()
    v_std20 = vol.rolling(20, min_periods=1).std().fillna(1e-6)
    v20 = (vol - v_mean20) / (v_std20 + 1e-12)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal

    rsi14 = _rsi(close, 14) / 100.0
    vol_of_vol = r.abs().rolling(20, min_periods=1).std().fillna(0.0)

    direction = close.diff()
    obv_step = np.where(direction > 0, vol, np.where(direction < 0, -vol, 0.0))
    obv = pd.Series(obv_step, index=close.index).cumsum()
    obv_mean20 = obv.rolling(20, min_periods=1).mean()
    obv_std20 = obv.rolling(20, min_periods=1).std().fillna(1e-6)
    obv_z = (obv - obv_mean20) / (obv_std20 + 1e-12)

    feats = pd.DataFrame({
        'ret_1': last_ret(1), 'ret_5': last_ret(5), 'ret_20': last_ret(20),
        'std_5': std_ret(5), 'std_20': std_ret(20), 'std_60': std_ret(60),
        'atr20': atr20,
        'dist_max20': dist_max20, 'dist_min20': dist_min20,
        'vol_z20': v20,
        'macd': macd, 'macd_signal': signal, 'macd_hist': hist,
        'rsi14_norm': rsi14,
        'vol_of_vol20': vol_of_vol,
        'obv_z20': obv_z,
    }, index=df.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats.astype(np.float32)


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / float(max(1, int(period)))
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, period: int, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    p = int(period)
    dsm = int(d_smooth)
    ll = low.rolling(p, min_periods=1).min()
    hh = high.rolling(p, min_periods=1).max()
    k = 100.0 * (close - ll) / (hh - ll + 1e-12)
    d = k.rolling(dsm, min_periods=1).mean()
    return k, d


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    return -100.0 * (hh - close) / (hh - ll + 1e-12)


def _keltner_width(high: pd.Series, low: pd.Series, close: pd.Series, period: int, mult: float = 2.0) -> pd.Series:
    ema_c = _ema(close, period)
    atr = _wilder_rma(_true_range(high, low, close), period)
    upper = ema_c + mult * atr
    lower = ema_c - mult * atr
    return upper - lower


def _bollinger_width(close: pd.Series, period: int, mult: float = 2.0) -> pd.Series:
    sma = close.rolling(period, min_periods=1).mean()
    sd = close.rolling(period, min_periods=1).std().fillna(0.0)
    upper = sma + mult * sd
    lower = sma - mult * sd
    return upper - lower


def _donchian_width(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    return hh - ll


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff()
    step = np.where(direction > 0, volume, np.where(direction < 0, -volume, 0.0))
    return pd.Series(step, index=close.index).cumsum()


def _hist_vol(close: pd.Series, period: int) -> pd.Series:
    logc = np.log(np.maximum(close, 1e-12))
    r = logc.diff().fillna(0.0)
    return r.rolling(period, min_periods=1).std().fillna(0.0)


def _zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().fillna(1e-6)
    return (series - mean) / (std + 1e-12)


def _adx_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    p = int(period)
    up = high.diff().fillna(0.0)
    down = (-low.diff()).fillna(0.0)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(high, low, close)
    atr = _wilder_rma(tr, p)
    plus_di = 100.0 * _wilder_rma(pd.Series(plus_dm, index=high.index), p) / (atr + 1e-12)
    minus_di = 100.0 * _wilder_rma(pd.Series(minus_dm, index=high.index), p) / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _wilder_rma(dx, p)
    return adx, plus_di, minus_di


def _ppo(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    ppo_line = 100.0 * (ema_fast - ema_slow) / (ema_slow + 1e-12)
    ppo_signal = _ema(ppo_line, signal)
    ppo_hist = ppo_line - ppo_signal
    return ppo_line, ppo_signal, ppo_hist


def _trix(close: pd.Series, period: int) -> pd.Series:
    p = int(period)
    e1 = _ema(close, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    trix = (e3 / (e3.shift(1) + 1e-12) - 1.0) * 100.0
    return trix.fillna(0.0)


def _aroon(high: pd.Series, low: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    p = int(period)
    def bars_since_extreme(series: pd.Series, win: int, is_high: bool) -> pd.Series:
        def f(x: np.ndarray) -> float:
            idx = np.argmax(x) if is_high else np.argmin(x)
            return float(len(x) - 1 - idx)
        return series.rolling(win, min_periods=1).apply(f, raw=True)
    bars_since_high = bars_since_extreme(high, p, True)
    bars_since_low = bars_since_extreme(low, p, False)
    up = 100.0 * (p - bars_since_high) / max(1, p)
    down = 100.0 * (p - bars_since_low) / max(1, p)
    osc = up - down
    return up, down, osc


def _bb_perc_b(close: pd.Series, period: int, mult: float = 2.0) -> pd.Series:
    p = int(period)
    sma = close.rolling(p, min_periods=1).mean()
    sd = close.rolling(p, min_periods=1).std().fillna(0.0)
    upper = sma + mult * sd
    lower = sma - mult * sd
    return (close - lower) / (upper - lower + 1e-12)


def _ichimoku(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    tenkan = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2.0
    kijun = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2.0
    span_a = (tenkan + kijun) / 2.0  # unshifted to avoid lookahead
    span_b = (high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2.0
    return tenkan, kijun, span_a, span_b


def _tema(close: pd.Series, period: int) -> pd.Series:
    p = int(period)
    e1 = _ema(close, p)
    e2 = _ema(e1, p)
    e3 = _ema(e2, p)
    return 3.0 * (e1 - e2) + e3


def _kama(close: pd.Series, period: int, fast: int = 2, slow: int = 30) -> pd.Series:
    p = int(period)
    change = (close - close.shift(p)).abs().fillna(0.0)
    volatility = close.diff().abs().rolling(p, min_periods=1).sum().fillna(0.0)
    er = change / (volatility + 1e-12)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.zeros(len(close), dtype=float)
    if len(close) > 0:
        out[0] = float(close.iloc[0])
    for i in range(1, len(close)):
        out[i] = out[i-1] + sc.iloc[i] * (float(close.iloc[i]) - out[i-1])
    return pd.Series(out, index=close.index)


def _tsi(close: pd.Series, r: int, s: int) -> pd.Series:
    delta = close.diff().fillna(0.0)
    ema1 = delta.ewm(span=int(r), adjust=False, min_periods=1).mean()
    ema2 = ema1.ewm(span=int(s), adjust=False, min_periods=1).mean()
    abs1 = delta.abs().ewm(span=int(r), adjust=False, min_periods=1).mean()
    abs2 = abs1.ewm(span=int(s), adjust=False, min_periods=1).mean()
    return 100.0 * ema2 / (abs2 + 1e-12)


def _wma(series: pd.Series, period: int) -> pd.Series:
    p = int(max(1, period))
    weights = np.arange(1, p + 1, dtype=float)
    def f(x: np.ndarray) -> float:
        w = weights[-len(x):]
        return float(np.dot(x, w) / (w.sum() + 1e-12))
    return series.rolling(p, min_periods=1).apply(f, raw=True)


def _hma(series: pd.Series, period: int) -> pd.Series:
    p = int(max(2, period))
    wma_half = _wma(series, p // 2)
    wma_full = _wma(series, p)
    diff = 2.0 * wma_half - wma_full
    return _wma(diff, int(np.sqrt(p)))


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns open, high, low, close
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = ha_close.copy()
    if len(df) > 0:
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
    ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close}, index=df.index)


def _psar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    # Simplified Parabolic SAR implementation
    af = step
    ep = high.iloc[0]
    psar = low.iloc[0]
    long = True
    out = [psar]
    for i in range(1, len(high)):
        prev_psar = out[-1]
        if long:
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = min(psar_i, low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < psar_i:
                long = False
                psar_i = ep
                ep = low.iloc[i]
                af = step
        else:
            psar_i = prev_psar + af * (ep - prev_psar)
            psar_i = max(psar_i, high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > psar_i:
                long = True
                psar_i = ep
                ep = high.iloc[i]
                af = step
        out.append(psar_i)
    return pd.Series(out, index=high.index)


def _kst(close: pd.Series, rocs: Tuple[int, int, int, int] = (10, 15, 20, 30), sma: Tuple[int, int, int, int] = (10, 10, 10, 15)) -> pd.Series:
    r1, r2, r3, r4 = rocs
    s1, s2, s3, s4 = sma
    roc1 = (close / close.shift(r1) - 1.0).fillna(0.0)
    roc2 = (close / close.shift(r2) - 1.0).fillna(0.0)
    roc3 = (close / close.shift(r3) - 1.0).fillna(0.0)
    roc4 = (close / close.shift(r4) - 1.0).fillna(0.0)
    kst = roc1.rolling(s1, min_periods=1).mean() + 2 * roc2.rolling(s2, min_periods=1).mean() + 3 * roc3.rolling(s3, min_periods=1).mean() + 4 * roc4.rolling(s4, min_periods=1).mean()
    return kst


def _vwap_bands(df: pd.DataFrame, period: int, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    p = int(period)
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = tp * df['volume']
    vol = df['volume']
    vwap = (pv.rolling(p, min_periods=1).sum()) / (vol.rolling(p, min_periods=1).sum() + 1e-12)
    std_tp = tp.rolling(p, min_periods=1).std().fillna(0.0)
    upper = vwap + mult * std_tp
    lower = vwap - mult * std_tp
    return vwap, upper, lower


def compute_indicator_grid(df: pd.DataFrame, prefix: str, periods: List[int]) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    open_ = df[cols.get('open', 'open')].astype(float)
    high = df[cols.get('high', 'high')].astype(float)
    low = df[cols.get('low', 'low')].astype(float)
    close = df[cols.get('close', 'close')].astype(float)
    volume = df[cols.get('volume', 'volume')].astype(float)

    feats: Dict[str, pd.Series] = {}

    # Baseline returns
    logc = np.log(np.maximum(close, 1e-12))
    ret1 = logc.diff().fillna(0.0)
    feats[f"{prefix}ret_1"] = ret1

    # MACD (fixed config)
    macd = _ema(close, 12) - _ema(close, 26)
    macds = _ema(macd, 9)
    macdh = macd - macds
    feats[f"{prefix}macd_12_26"] = macd
    feats[f"{prefix}macds_9"] = macds
    feats[f"{prefix}macdh_12_26_9"] = macdh

    # PPO (fixed config)
    ppo, ppos, ppoh = _ppo(close, 12, 26, 9)
    feats[f"{prefix}ppo_12_26"] = ppo
    feats[f"{prefix}ppos_9"] = ppos
    feats[f"{prefix}ppoh_12_26_9"] = ppoh

    # OBV family
    obv = _obv(close, volume)
    for p in periods:
        feats[f"{prefix}obv_z{p}"] = _zscore(obv, p)

    # Ichimoku (standard params, unshifted to avoid lookahead)
    tenkan, kijun, span_a, span_b = _ichimoku(high, low)
    feats[f"{prefix}ichimoku_tenkan_9"] = tenkan
    feats[f"{prefix}ichimoku_kijun_26"] = kijun
    feats[f"{prefix}ichimoku_spanA"] = span_a
    feats[f"{prefix}ichimoku_spanB"] = span_b

    # Heikin-Ashi OHLC and deviations
    ha = _heikin_ashi(df)
    feats[f"{prefix}ha_open"] = ha['ha_open']
    feats[f"{prefix}ha_high"] = ha['ha_high']
    feats[f"{prefix}ha_low"] = ha['ha_low']
    feats[f"{prefix}ha_close"] = ha['ha_close']
    feats[f"{prefix}ha_close_diff"] = (ha['ha_close'] - close)

    # Parabolic SAR (fixed params)
    feats[f"{prefix}psar"] = _psar(high, low)

    # KST indicator (fixed default params)
    feats[f"{prefix}kst"] = _kst(close)

    for p in periods:
        # Moving averages / momentum
        feats[f"{prefix}sma_{p}"] = close.rolling(p, min_periods=1).mean()
        feats[f"{prefix}ema_{p}"] = _ema(close, p)
        feats[f"{prefix}hma_{p}"] = _hma(close, p)
        feats[f"{prefix}roc_{p}"] = (close / close.shift(p) - 1.0).fillna(0.0)
        feats[f"{prefix}mom_{p}"] = (close - close.shift(p)).fillna(0.0)
        feats[f"{prefix}rsi_{p}"] = (_rsi(close, p) / 100.0)
        # Stochastics
        k, d = _stoch(high, low, close, p)
        feats[f"{prefix}stochk_{p}"] = k / 100.0
        feats[f"{prefix}stochd_{p}"] = d / 100.0
        # CCI / Williams R
        cci_tp = (high + low + close) / 3.0
        cci_ma = cci_tp.rolling(p, min_periods=1).mean()
        cci_md = (cci_tp - cci_ma).abs().rolling(p, min_periods=1).mean()
        feats[f"{prefix}cci_{p}"] = (cci_tp - cci_ma) / (0.015 * (cci_md + 1e-12))
        feats[f"{prefix}willr_{p}"] = (_willr(high, low, close, p) / 100.0)
        # Volatility
        tr = _true_range(high, low, close)
        feats[f"{prefix}atr_{p}"] = _wilder_rma(tr, p)
        feats[f"{prefix}hv_{p}"] = _hist_vol(close, p)
        feats[f"{prefix}bb_width_{p}"] = _bollinger_width(close, p)
        feats[f"{prefix}bb_percB_{p}"] = _bb_perc_b(close, p)
        feats[f"{prefix}kc_width_{p}"] = _keltner_width(high, low, close, p)
        feats[f"{prefix}dc_width_{p}"] = _donchian_width(high, low, p)
        # Volume family
        feats[f"{prefix}vroc_{p}"] = (volume / volume.shift(p) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Money Flow Index
        tp = (high + low + close) / 3.0
        mf = tp * volume
        pos_mf = (tp > tp.shift(1)).astype(float) * mf
        neg_mf = (tp < tp.shift(1)).astype(float) * mf
        pos_sum = pos_mf.rolling(p, min_periods=1).sum()
        neg_sum = neg_mf.rolling(p, min_periods=1).sum() + 1e-12
        mr = pos_sum / neg_sum
        feats[f"{prefix}mfi_{p}"] = 100.0 - (100.0 / (1.0 + mr)) / 100.0
        # Chaikin Money Flow
        mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
        mfv = mfm * volume
        feats[f"{prefix}cmf_{p}"] = (mfv.rolling(p, min_periods=1).sum()) / (volume.rolling(p, min_periods=1).sum() + 1e-12)
        # Z-score of close
        feats[f"{prefix}zclose_{p}"] = _zscore(close, p)
        # ADX & DI
        adx, dip, dim = _adx_di(high, low, close, p)
        feats[f"{prefix}adx_{p}"] = adx
        feats[f"{prefix}di_plus_{p}"] = dip
        feats[f"{prefix}di_minus_{p}"] = dim
        # TRIX
        feats[f"{prefix}trix_{p}"] = _trix(close, p)
        # Aroon
        ar_up, ar_dn, ar_osc = _aroon(high, low, p)
        feats[f"{prefix}aroon_up_{p}"] = ar_up
        feats[f"{prefix}aroon_down_{p}"] = ar_dn
        feats[f"{prefix}aroon_osc_{p}"] = ar_osc
        # TEMA
        feats[f"{prefix}tema_{p}"] = _tema(close, p)
        # KAMA
        feats[f"{prefix}kama_{p}"] = _kama(close, p)
        # VWAP bands
        vwap, vwap_u, vwap_l = _vwap_bands(df, p)
        feats[f"{prefix}vwap_{p}"] = vwap
        feats[f"{prefix}vwap_u_{p}"] = vwap_u
        feats[f"{prefix}vwap_l_{p}"] = vwap_l
        # TSI (derive r,s from p)
        r = max(2, int(p))
        s = max(2, int(max(2, p//2)))
        feats[f"{prefix}tsi_{p}"] = _tsi(close, r, s)

    out = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.astype(np.float32)


def _load_local_ohlcv(instrument: str, granularity: str) -> Optional[pd.DataFrame]:
    ct_dir = os.path.join(FX_ROOT, "continuous-trader", "data")
    safe = instrument.replace('/', '_')
    path = os.path.join(ct_dir, f"{safe}_{granularity}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
            cols = {c.lower(): c for c in df.columns}
            df = df[[cols.get("open","open"), cols.get("high","high"), cols.get("low","low"), cols.get("close","close"), cols.get("volume","volume")]]
            df.columns = ["open","high","low","close","volume"]
            if granularity in ("D", "W") and isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.normalize()
            return df.astype(float)
        except Exception:
            return None
    return None


def export_features(
    instruments: List[str],
    granularities: List[str],
    periods: Optional[List[int]] = None,
    out_dir: Optional[str] = None,
    environment: str = "practice",
    access_token: Optional[str] = None,
    max_rows: Optional[int] = None,
    chunk_rows: int = 100_000,
    overlap_rows: Optional[int] = None,
) -> None:
    """Export feature CSVs per instrument per granularity.

    File path: {out_dir}/features/{granularity}/{instrument}_{granularity}_features.csv
    """
    prds = periods or [5, 15, 45, 135, 405]
    base_out = out_dir or os.path.join(FX_ROOT, "continuous-trader", "data")
    feat_root = os.path.join(base_out, "features")
    os.makedirs(feat_root, exist_ok=True)

    for gran in [g.upper() for g in granularities]:
        gran_dir = os.path.join(feat_root, gran)
        os.makedirs(gran_dir, exist_ok=True)
        for inst in instruments:
            try:
                # Load from local candles; fallback to REST on demand
                df = _load_local_ohlcv(inst, gran)
                if df is None:
                    adapter = OandaRestCandlesAdapter(instrument=inst, granularity=gran, environment=environment, access_token=access_token)
                    rows = list(adapter.fetch_range(from_time=None, to_time=None, batch=5000))
                    if not rows:
                        print({"status": "skip_no_data", "instrument": inst, "granularity": gran})
                        continue
                    tmp = pd.DataFrame(rows)
                    tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], utc=True)
                    df = tmp.set_index('timestamp').sort_index()[['open','high','low','close','volume']].astype(float)
                    if gran in ("D", "W"):
                        df.index = df.index.normalize()
                # Limit rows to reduce memory usage if requested
                if max_rows is not None and len(df) > int(max_rows):
                    df = df.tail(int(max_rows))

                # Streaming chunked feature computation and append to CSV
                safe = inst.replace('/', '_')
                out_path = os.path.join(gran_dir, f"{safe}_{gran}_features.csv")
                # Remove old file if exists to avoid appending to stale content
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass

                max_p = max(prds) if len(prds) > 0 else 1
                ovl = overlap_rows if overlap_rows is not None else max(100, max_p)
                total_rows = len(df)
                start = 0
                written = 0
                while start < total_rows:
                    end = min(total_rows, start + int(chunk_rows))
                    ext_start = max(0, start - ovl)
                    df_ext = df.iloc[ext_start:end]
                    feats_ext = compute_indicator_grid(df_ext, prefix=f"{inst}_", periods=prds)
                    take = end - start
                    out_chunk = feats_ext.tail(take).astype(np.float32)
                    # Append to CSV
                    out_chunk.to_csv(out_path, mode='a', header=(written == 0), index=True)
                    written += out_chunk.shape[0]
                    print({"status": "chunk_saved", "instrument": inst, "granularity": gran, "rows": int(written), "chunk": [int(start), int(end)], "path": out_path})
                    # Free chunk memory
                    del feats_ext
                    del out_chunk
                    import gc
                    gc.collect()
                    start = end
                # Free instrument memory
                del df
                import gc
                gc.collect()
            except Exception as exc:
                print({"status": "error_features", "instrument": inst, "granularity": gran, "error": str(exc)})

def _fetch_fx_recent(instrument: str, granularity: str, count: int, environment: str, access_token: Optional[str]) -> pd.DataFrame:
    adapter = OandaRestCandlesAdapter(instrument=instrument, granularity=granularity, environment=environment, access_token=access_token)
    rows = list(adapter.fetch(count=count))
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])  # empty
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    return df[['open','high','low','close','volume']].astype(float)


def build_feature_grid(
    instruments: List[str],
    environment: str,
    access_token: Optional[str],
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Build a single wide feature row aligned to the latest available M5/H1/D bars per instrument.

    This returns the latest feature vector (one row DataFrame) to feed the policy/actor.
    """
    blocks: List[pd.DataFrame] = []
    for inst in instruments:
        if cfg.include_m5:
            df_m5 = _fetch_fx_recent(inst, "M5", cfg.m5_bars, environment, access_token)
            if not df_m5.empty:
                feats_m5 = _compute_ohlcv_features(df_m5).add_prefix(f"M5_{inst}_")
                blocks.append(feats_m5.tail(1))
        if cfg.include_h1:
            df_h1 = _fetch_fx_recent(inst, "H1", cfg.h1_bars, environment, access_token)
            if not df_h1.empty:
                feats_h1 = _compute_ohlcv_features(df_h1).add_prefix(f"H1_{inst}_")
                blocks.append(feats_h1.tail(1))
        if cfg.include_d:
            df_d = _fetch_fx_recent(inst, "D", cfg.d_bars, environment, access_token)
            if not df_d.empty:
                feats_d = _compute_ohlcv_features(df_d).add_prefix(f"D_{inst}_")
                blocks.append(feats_d.tail(1))

    if not blocks:
        return pd.DataFrame()

    # Align columns; forward-fill within the single row set is trivial; outer join across blocks then ffill
    X = pd.concat(blocks, axis=1)
    # Keep only last row after concat; fill NaNs with 0
    X = X.tail(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X.astype(np.float32)


if __name__ == "__main__":
    import argparse
    import json
    from instruments import load_68  # type: ignore
    p = argparse.ArgumentParser(description="Build latest-row feature grid across instruments and granularities")
    p.add_argument("--instruments-csv", default=None)
    p.add_argument("--environment", choices=["practice","live"], default="practice")
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--include-h1", action="store_true")
    p.add_argument("--include-d", action="store_true")
    p.add_argument("--export", action="store_true", help="Export full feature CSVs per instrument/granularity")
    p.add_argument("--granularities", default="M5,H1,D", help="Granularities to export when --export is set")
    p.add_argument("--periods", default="5,15,45,135,405", help="Indicator lookbacks, comma-separated")
    p.add_argument("--max-rows", type=int, default=None, help="Limit rows per instrument to cap memory (tail N)")
    p.add_argument("--chunk-rows", type=int, default=100000, help="Process this many rows per chunk")
    p.add_argument("--overlap-rows", type=int, default=None, help="Overlap rows between chunks (defaults to max period)")
    p.set_defaults(include_h1=True, include_d=True)
    args = p.parse_args()

    instruments = load_68(args.instruments_csv)
    fc = FeatureConfig(include_m5=True, include_h1=bool(args.include_h1), include_d=bool(args.include_d))
    if args.export:
        grans = [g.strip().upper() for g in (args.granularities or "M5,H1,D").split(',') if g.strip()]
        prds = [int(x) for x in (args.periods or "").split(',') if x.strip()]
        export_features(
            instruments,
            grans,
            prds,
            out_dir=None,
            environment=args.environment,
            access_token=args.access_token,
            max_rows=args.max_rows,
            chunk_rows=int(args.chunk_rows),
            overlap_rows=(int(args.overlap_rows) if args.overlap_rows is not None else None),
        )
    else:
        X = build_feature_grid(instruments, args.environment, args.access_token, fc)
        print(json.dumps({"cols": X.shape[1], "ok": not X.empty}))
