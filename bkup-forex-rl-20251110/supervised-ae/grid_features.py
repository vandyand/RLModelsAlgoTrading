#!/usr/bin/env python3
"""
Grid feature builder for FX (20 instruments) + ETFs (15 default or custom).

- Fetches daily OHLCV for FX via OANDA REST and ETFs via yfinance
- Computes a large grid of technical indicators across a shared parameter set
  periods = [5, 15, 45, 135, 405] (roughly 1w, 3w, 9w, ~6m, ~1.5y)
- Uses forward computation with min_periods=1 for early bars so features exist
- Aligns all features on the intersection of FX dates; ETF features forward-filled
- Saves:
  * features CSV (wide, columns per instrument-indicator-params)
  * returns CSV (next-day log returns of FX instruments)
  * dates CSV (ISO date index)

Usage example:
  python forex-rl/supervised-ae/grid_features.py \
    --start 2015-01-01 --end 2025-08-31 \
    --instruments EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD \
    --use-all-etfs \
    --out-features forex-rl/supervised-ae/data/multi_features.csv \
    --out-returns forex-rl/supervised-ae/data/fx_returns.csv \
    --out-dates forex-rl/supervised-ae/data/dates.csv

Notes:
- Requires env: OANDA_DEMO_KEY (and account id optional) to fetch OANDA candles
- yfinance is used for ETFs (auto-adjusted OHLCV)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None


DEFAULT_OANDA_20: List[str] = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF",
    "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
    "EUR_CHF", "EUR_AUD", "EUR_CAD", "GBP_CHF", "AUD_JPY",
    "AUD_CHF", "CAD_JPY", "NZD_JPY", "GBP_AUD", "AUD_NZD",
]


def get_etf_universe(use_all: bool) -> List[str]:
    all_tickers = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VTV', 'VUG', 'IJR', 'IJH', 'IVV', 'VOO', 'XLK', 'VGT',
        'FTEC', 'SOXX', 'ARKK', 'ARKQ', 'CLOU', 'CYBR', 'VEA', 'IEFA', 'EFA', 'VT', 'VXUS',
        'SCHF', 'EWJ', 'FEZ', 'VWO', 'EEM', 'IEMG', 'SCHE', 'EWZ', 'FXI', 'ASHR', 'KWEB', 'XLF',
        'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'VDE', 'DBC', 'PDBC',
        'USCI', 'DJP', 'GSG', 'COMT', 'GLD', 'IAU', 'SLV', 'PPLT', 'PALL', 'GLTR', 'GDX', 'GDXJ',
        'USO', 'UCO', 'UGA', 'BNO', 'USL', 'DBO', 'OIL', 'OILK', 'UNG', 'BOIL', 'KOLD', 'FCG',
        'CPER', 'DBB', 'PICK', 'REMX', 'DBA', 'CORN', 'WEAT', 'SOYB', 'JO', 'NIB', 'AGG', 'BND',
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'TIP', 'VCIT', 'VNQ', 'IYR', 'XLRE', 'SCHH',
        'USRT', 'RWR', 'VNQI', 'VXX', 'UVXY', 'VIXY', 'SVXY', 'VIXM', 'VXZ', 'UUP', 'UDN', 'FXE',
        'FXY', 'FXB', 'FXC', 'TAIL', 'PHDG', 'NUSI', 'QYLD', 'JEPI'
    ]
    if use_all:
        return all_tickers
    return [
        'SPY','QQQ','VEA','VWO','AGG','TLT','HYG','GLD','DBC','VNQ','UUP','XLE','XLV','XLF','XLK'
    ]


# ---------- Data fetch ----------

def fetch_fx_ohlcv(instrument: str, start: str, end: Optional[str], environment: str, access_token: Optional[str]) -> pd.DataFrame:
    if access_token is None:
        raise RuntimeError("Access token required for OANDA REST (OANDA_DEMO_KEY)")
    adapter = OandaRestCandlesAdapter(
        instrument=instrument,
        granularity="D",
        environment=environment,
        access_token=access_token,
    )
    if end is None:
        from datetime import datetime, timezone, timedelta
        end = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    rows: List[Dict[str, Any]] = list(adapter.fetch_range(from_time=f"{start}T00:00:00Z", to_time=f"{end}T23:59:59Z", batch=5000))
    if not rows:
        raise RuntimeError(f"No candles fetched for {instrument} between {start} and {end}")
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    df.index = df.index.normalize()
    return df[['open','high','low','close','volume']].astype(float)


def fetch_etf_ohlcv(tickers: List[str], start: str, end: Optional[str]) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Please install yfinance.")
    if end is None:
        from datetime import datetime, timezone, timedelta
        end_excl = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    else:
        try:
            end_excl = (pd.to_datetime(end).tz_localize('UTC') + pd.Timedelta(days=1)).date().isoformat()
        except Exception:
            end_excl = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()
    data = yf.download(tickers, start=start, end=end_excl, auto_adjust=True, progress=False, threads=True)
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for tkr in tickers:
            sub = pd.DataFrame({
                'open': data['Open'][tkr],
                'high': data['High'][tkr],
                'low': data['Low'][tkr],
                'close': data['Close'][tkr],
                'volume': data['Volume'][tkr],
            })
            sub.index = pd.to_datetime(sub.index, utc=True).normalize()
            out[tkr] = sub.dropna()
    else:
        tkr = tickers[0]
        sub = pd.DataFrame({
            'open': data['Open'], 'high': data['High'], 'low': data['Low'], 'close': data['Close'], 'volume': data['Volume']
        })
        sub.index = pd.to_datetime(sub.index, utc=True).normalize()
        out[tkr] = sub.dropna()
    return out


# ---------- Indicator helpers ----------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    alpha = 1.0 / float(period)
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period) + eps
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return _wilder_rma(tr, period)


def _stoch(high: pd.Series, low: pd.Series, close: pd.Series, period: int, d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    ll = low.rolling(period, min_periods=1).min()
    hh = high.rolling(period, min_periods=1).max()
    k = 100.0 * (close - ll) / (hh - ll + 1e-12)
    d = k.rolling(d_smooth, min_periods=1).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=1).mean()
    md = (tp - sma).abs().rolling(period, min_periods=1).mean()
    return (tp - sma) / (0.015 * (md + 1e-12))


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    return -100.0 * (hh - close) / (hh - ll + 1e-12)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff()
    step = np.where(direction > 0, volume, np.where(direction < 0, -volume, 0.0))
    return pd.Series(step, index=close.index).cumsum()


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    pos_mf = (tp > tp.shift(1)).astype(float) * mf
    neg_mf = (tp < tp.shift(1)).astype(float) * mf
    pos_sum = pos_mf.rolling(period, min_periods=1).sum()
    neg_sum = neg_mf.rolling(period, min_periods=1).sum() + 1e-12
    mr = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + mr))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
    mfv = mfm * volume
    return (mfv.rolling(period, min_periods=1).sum()) / (volume.rolling(period, min_periods=1).sum() + 1e-12)


def _donchian_width(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    return hh - ll


def _keltner_width(high: pd.Series, low: pd.Series, close: pd.Series, period: int, mult: float = 2.0) -> pd.Series:
    ema = _ema(close, period)
    atr = _atr(high, low, close, period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper - lower


def _bollinger_width(close: pd.Series, period: int, mult: float = 2.0) -> pd.Series:
    sma = close.rolling(period, min_periods=1).mean()
    sd = close.rolling(period, min_periods=1).std().fillna(0.0)
    upper = sma + mult * sd
    lower = sma - mult * sd
    return upper - lower


def _hist_vol(close: pd.Series, period: int) -> pd.Series:
    logc = np.log(np.maximum(close, 1e-12))
    r = logc.diff().fillna(0.0)
    return r.rolling(period, min_periods=1).std().fillna(0.0)


def _zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period, min_periods=1).mean()
    std = series.rolling(period, min_periods=1).std().fillna(1e-6)
    return (series - mean) / (std + 1e-12)


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
    macd, macds, macdh = _macd(close)
    feats[f"{prefix}macd_12_26"] = macd
    feats[f"{prefix}macds_9"] = macds
    feats[f"{prefix}macdh_12_26_9"] = macdh

    # OBV family
    obv = _obv(close, volume)
    for p in periods:
        feats[f"{prefix}obv_z{p}"] = _zscore(obv, p)

    for p in periods:
        # Moving averages / momentum
        feats[f"{prefix}sma_{p}"] = close.rolling(p, min_periods=1).mean()
        feats[f"{prefix}ema_{p}"] = _ema(close, p)
        feats[f"{prefix}roc_{p}"] = (close / close.shift(p) - 1.0).fillna(0.0)
        feats[f"{prefix}mom_{p}"] = (close - close.shift(p)).fillna(0.0)
        feats[f"{prefix}rsi_{p}"] = (_rsi(close, p) / 100.0)
        # Stochastics
        k, d = _stoch(high, low, close, p)
        feats[f"{prefix}stochk_{p}"] = k / 100.0
        feats[f"{prefix}stochd_{p}"] = d / 100.0
        # CCI / Williams R
        feats[f"{prefix}cci_{p}"] = _cci(high, low, close, p)
        feats[f"{prefix}willr_{p}"] = (_willr(high, low, close, p) / 100.0)
        # Volatility
        feats[f"{prefix}atr_{p}"] = _atr(high, low, close, p)
        feats[f"{prefix}hv_{p}"] = _hist_vol(close, p)
        feats[f"{prefix}bb_width_{p}"] = _bollinger_width(close, p)
        feats[f"{prefix}kc_width_{p}"] = _keltner_width(high, low, close, p)
        feats[f"{prefix}dc_width_{p}"] = _donchian_width(high, low, p)
        # Volume family
        feats[f"{prefix}vroc_{p}"] = (volume / volume.shift(p) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        feats[f"{prefix}mfi_{p}"] = (_mfi(high, low, close, volume, p) / 100.0)
        feats[f"{prefix}cmf_{p}"] = _cmf(high, low, close, volume, p)
        # Z-score of close
        feats[f"{prefix}zclose_{p}"] = _zscore(close, p)

    out = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.astype(np.float32)


def time_cyclical_features_from_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.to_datetime(index, utc=True).tz_convert('UTC')
    dow = idx.weekday.values
    dom = idx.day.values - 1
    moy = idx.month.values - 1
    def sc(vals: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
        ang = 2.0 * np.pi * (vals.astype(float) / period)
        return np.sin(ang), np.cos(ang)
    dw_s, dw_c = sc(dow, 7.0)
    dm_s, dm_c = sc(dom, 31.0)
    my_s, my_c = sc(moy, 12.0)
    df = pd.DataFrame({
        'time_dow_sin': dw_s, 'time_dow_cos': dw_c,
        'time_dom_sin': dm_s, 'time_dom_cos': dm_c,
        'time_moy_sin': my_s, 'time_moy_cos': my_c,
    }, index=index)
    return df.astype(np.float32)


@dataclass
class Args:
    instruments: List[str]
    environment: str
    account_id: Optional[str]
    access_token: Optional[str]
    start: str
    end: Optional[str]
    use_all_etfs: bool
    etf_tickers: Optional[List[str]]
    out_features: str
    out_returns: str
    out_dates: str


def build_and_save(args: Args) -> None:
    periods = [5, 15, 45, 135, 405]

    # FX features and returns
    fx_frames: List[pd.DataFrame] = []
    fx_close_map: Dict[str, pd.Series] = {}
    for inst in args.instruments:
        d_df = fetch_fx_ohlcv(inst, args.start, args.end, args.environment, args.access_token)
        fx_close_map[inst] = d_df['close']
        fx_feats = compute_indicator_grid(d_df, prefix=f"FX_{inst}_", periods=periods)
        fx_frames.append(fx_feats)

    # Base index = intersection of all FX dates
    base_index = fx_frames[0].index
    for f in fx_frames[1:]:
        base_index = base_index.intersection(f.index)

    # ETFs
    etf_frames: List[pd.DataFrame] = []
    etf_list = (args.etf_tickers if args.etf_tickers and len(args.etf_tickers) > 0 else get_etf_universe(args.use_all_etfs))
    if etf_list:
        etf_map = fetch_etf_ohlcv(etf_list, args.start, args.end)
        for tkr, df in etf_map.items():
            feats = compute_indicator_grid(df, prefix=f"ETF_{tkr}_", periods=periods)
            # forward fill to FX base index (ETFs can have holiday gaps)
            feats = feats.reindex(base_index).ffill().fillna(0.0)
            etf_frames.append(feats)

    blocks: List[pd.DataFrame] = []
    for frm in fx_frames:
        blocks.append(frm.reindex(base_index).fillna(0.0))
    blocks.extend(etf_frames)
    blocks.append(time_cyclical_features_from_index(base_index))

    X = pd.concat(blocks, axis=1).astype(np.float32)

    # Returns (next-day log returns per FX instrument)
    r_cols: Dict[str, pd.Series] = {}
    for inst, cls in fx_close_map.items():
        close = cls.reindex(base_index).ffill()
        r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        r_cols[inst] = r.shift(-1).fillna(0.0)
    R = pd.DataFrame(r_cols, index=base_index).astype(np.float32)

    # Ensure output dirs
    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_returns), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_dates), exist_ok=True)

    X.to_csv(args.out_features, index=True)
    R.to_csv(args.out_returns, index=True)
    pd.Series(base_index.astype('datetime64[ns]')).dt.strftime('%Y-%m-%d').to_csv(args.out_dates, index=False, header=False)

    print({
        "status": "saved",
        "features": args.out_features,
        "returns": args.out_returns,
        "dates": args.out_dates,
        "rows": int(X.shape[0]),
        "cols": int(X.shape[1]),
    })


def parse_cli() -> Args:
    parser = argparse.ArgumentParser(description="Build grid indicators for FX + ETFs (daily)")
    parser.add_argument("--instruments", default=",".join(DEFAULT_OANDA_20))
    parser.add_argument("--environment", choices=["practice","live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end")
    parser.add_argument("--use-all-etfs", action="store_true")
    parser.add_argument("--etf-tickers", default="")
    parser.add_argument("--out-features", default="forex-rl/supervised-ae/data/multi_features.csv")
    parser.add_argument("--out-returns", default="forex-rl/supervised-ae/data/fx_returns.csv")
    parser.add_argument("--out-dates", default="forex-rl/supervised-ae/data/dates.csv")
    a = parser.parse_args()
    insts = [s.strip() for s in (a.instruments or "").split(",") if s.strip()]
    etfs = [s.strip().upper() for s in (a.etf_tickers or "").split(",") if s.strip()]
    return Args(
        instruments=insts if len(insts) > 0 else DEFAULT_OANDA_20,
        environment=a.environment,
        account_id=a.account_id,
        access_token=a.access_token,
        start=a.start,
        end=a.end,
        use_all_etfs=bool(a.use_all_etfs),
        etf_tickers=etfs if len(etfs) > 0 else None,
        out_features=a.out_features,
        out_returns=a.out_returns,
        out_dates=a.out_dates,
    )


if __name__ == "__main__":
    args = parse_cli()
    build_and_save(args)
