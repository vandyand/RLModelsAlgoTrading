from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import gc

import numpy as np
import pandas as pd

from indicators import (
    ema,
    rsi,
    macd,
    atr,
    roc,
    stoch_kd,
    cci,
    bollinger_width,
)

INSTRUMENTS = [
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "AUD_USD",
    "EUR_JPY",
    "EUR_AUD",
    "EUR_GBP",
]


@dataclass
class InstrumentFrame:
    instrument: str
    df: pd.DataFrame


def _load_csv(csv_path: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index(kind="mergesort")[
        ["open", "high", "low", "close", "volume"]
    ]
    # Downcast dtypes aggressively
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'float32' or df[col].dtype == 'float16':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64' or str(df[col].dtype).startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast='integer')
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df.astype({c: 'float32' for c in ["open","high","low","close","volume"]})


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.to_datetime(index, utc=True)
    hour = idx.tz_convert("UTC").hour
    weekday = idx.tz_convert("UTC").weekday
    # Sessions (approx): Asia 00-07, London 07-16, NY 13-21 UTC
    asia = ((hour >= 0) & (hour < 7)).astype(float)
    london = ((hour >= 7) & (hour < 16)).astype(float)
    ny = ((hour >= 13) & (hour < 21)).astype(float)
    return pd.DataFrame({
        "tod_hour": hour.astype(float),
        "dow": weekday.astype(float),
        "sess_asia": asia,
        "sess_london": london,
        "sess_ny": ny,
    }, index=index).astype(np.float32)


def _zscore_df(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    roll_mean = df.rolling(window, min_periods=1).mean()
    roll_std = df.rolling(window, min_periods=1).std().replace(0.0, np.nan).fillna(1e-6)
    z = (df - roll_mean) / roll_std
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z.columns = [f"{c}_z" for c in df.columns]
    return z


def build_features(
    data_dir: str,
    instruments: List[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    subsample: int = 1,
    max_rows: int | None = None,
    feature_set: str = "core",  # core|minimal|full
    parquet_cache_path: Optional[str] = None,
    cache_mode: Optional[str] = None,  # 'read'|'write'|None
) -> Dict[str, pd.DataFrame]:
    inst_to_df: Dict[str, pd.DataFrame] = {}
    insts = instruments if instruments is not None and len(instruments) > 0 else INSTRUMENTS
    start_ts = pd.to_datetime(start, utc=True) if start else None
    end_ts = pd.to_datetime(end, utc=True) if end else None
    missing: List[str] = []
    loaded: List[str] = []
    for inst in insts:
        p = os.path.join(data_dir, f"{inst}_M5.csv")
        if not os.path.exists(p):
            missing.append(inst)
            continue
        base = _load_csv(p, start=start_ts, end=end_ts)
        inst_to_df[inst] = base
        loaded.append(inst)
        # Free any temp objects proactively
        gc.collect()

    # Align on intersection of indices
    if not inst_to_df:
        raise RuntimeError("No instrument CSVs found in data dir")
    common_index = None
    for df in inst_to_df.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    # Downsample index early and limit rows to reduce memory
    if subsample and subsample > 1:
        common_index = common_index[::int(subsample)]
    if max_rows is not None and len(common_index) > max_rows:
        common_index = common_index[-int(max_rows):]
    for k in list(inst_to_df.keys()):
        inst_to_df[k] = inst_to_df[k].reindex(common_index).ffill().dropna()

    # Build per-instrument indicator features
    feats: Dict[str, pd.DataFrame] = {}
    for inst, df in inst_to_df.items():
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        macd_line, macd_sig, macd_hist = macd(close)
        k, d = stoch_kd(high, low, close, 14, 3)
        features = pd.DataFrame({
            f"{inst}_close": close,
            f"{inst}_ema_10": ema(close, 10),
            f"{inst}_ema_50": ema(close, 50),
            f"{inst}_rsi_14": rsi(close, 14),
            f"{inst}_roc_10": roc(close, 10),
            f"{inst}_stochk_14": k / 100.0,
            f"{inst}_stochd_14": d / 100.0,
            f"{inst}_cci_20": cci(high, low, close, 20),
            f"{inst}_atr_14": atr(high, low, close, 14),
            f"{inst}_bbwidth_20": bollinger_width(close, 20, 2.0),
            f"{inst}_macd": macd_line,
            f"{inst}_macds": macd_sig,
            f"{inst}_macdh": macd_hist,
            f"{inst}_vol": vol,
        }, index=df.index)
        # Downcast memory
        for c in features.columns:
            features[c] = pd.to_numeric(features[c], downcast='float')
        feats[inst] = features.astype(np.float32)
        del features
        gc.collect()

    # Merge all instruments and add time features
    combined = pd.concat(list(feats.values()) + [time_features(common_index)], axis=1)
    combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # Z-score all numeric features except session flags which are already 0/1
    sess_cols = ["sess_asia", "sess_london", "sess_ny"]
    numeric = combined.drop(columns=[c for c in sess_cols if c in combined.columns], errors="ignore").astype(float)
    z = _zscore_df(numeric, window=50)
    # Keep session flags as-is; also keep original close columns for PnL reference
    close_cols = [c for c in combined.columns if c.endswith("_close")]
    keep = combined[sess_cols + close_cols] if all(c in combined.columns for c in sess_cols) else combined[close_cols]
    out = pd.concat([z, keep], axis=1).astype(np.float32)

    # Feature selection
    if feature_set != "full":
        # Minimal subset
        minimal_names = [
            "ema_10_z", "ema_50_z", "rsi_14_z", "roc_10_z",
            "stochk_14_z", "cci_20_z", "atr_14_z", "bbwidth_20_z",
            "macd_z", "macds_z",
        ]
        core_names = minimal_names  # current core equals minimal for memory
        wanted_suffixes = minimal_names if feature_set == "minimal" else core_names
        keep_cols: List[str] = []
        for inst in loaded:
            for suf in wanted_suffixes:
                col = f"{inst}_{suf}"
                if col in out.columns:
                    keep_cols.append(col)
        # Always keep session flags and closes
        keep_cols.extend([c for c in out.columns if c in ["sess_asia", "sess_london", "sess_ny"]])
        keep_cols.extend([c for c in out.columns if c.endswith("_close")])
        out = out[keep_cols]

    # Optional parquet cache write
    if cache_mode == 'write' and parquet_cache_path:
        try:
            out.to_parquet(parquet_cache_path, compression='snappy')
        except Exception:
            pass

    return {"X": out, "loaded": loaded, "missing": missing}
