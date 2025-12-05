from __future__ import annotations
import os
from typing import Dict, List, Tuple
import pandas as pd

from config import Config
from instruments import underscore_to_slash
from feature_select import prefixed_feature_names


def _read_features_csv(path: str, usecols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", *usecols])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Ensure index uniqueness to allow reindex
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _read_close_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df["close"].astype(float)


def _features_path(root: str, gran: str, instrument: str) -> str:
    return os.path.join(root, "features", gran, f"{instrument}_{gran}_features.csv.gz")


def _price_path(root: str, gran: str, instrument: str) -> str:
    return os.path.join(root, f"{instrument}_{gran}.csv")


def load_feature_panel(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, closes) where:
    - X: MultiIndex columns (instrument, feature_name), M5-indexed
    - closes: columns by instrument (M5 close), aligned to X.index
    """
    instruments = list(cfg.instruments)
    # 1) Build M5 index range across instruments within lookback window
    m5_closes: Dict[str, pd.Series] = {}
    for inst in instruments:
        ppath = _price_path(cfg.data_root, "M5", inst)
        s = _read_close_csv(ppath)
        m5_closes[inst] = s
    # Intersect indices and clip to lookback
    idx = None
    for s in m5_closes.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) == 0:
        raise RuntimeError("No overlapping M5 timestamps across instruments")
    end_ts = idx.max()
    # If explicit dates provided, clamp range to [start_date, end_date]
    if cfg.start_date:
        start_ts = max(pd.Timestamp(cfg.start_date, tz="UTC"), idx.min())
    else:
        start_ts = end_ts - pd.Timedelta(days=int(cfg.lookback_days))
    if cfg.end_date:
        end_ts = min(pd.Timestamp(cfg.end_date, tz="UTC"), idx.max())
    idx = idx[(idx >= start_ts) & (idx <= end_ts)]
    # Ensure unique timeline
    if idx.has_duplicates:
        idx = pd.DatetimeIndex(sorted(set(idx)))
    # 2) Load features per instrument for M5/H1/D and align to M5 index
    X_blocks: List[pd.DataFrame] = []
    closes_df = pd.DataFrame({k: v.reindex(idx).ffill().astype(float) for k, v in m5_closes.items()}, index=idx)

    for inst in instruments:
        inst_slash = underscore_to_slash(inst)
        # M5
        m5_cols = prefixed_feature_names(inst_slash, "M5")
        m5_path = _features_path(cfg.data_root, "M5", inst)
        m5_df = _read_features_csv(m5_path, m5_cols).reindex(idx).ffill()
        m5_df.columns = [f"M5::{c.split(inst_slash + '_', 1)[1]}" for c in m5_df.columns]
        # H1
        h1_cols = prefixed_feature_names(inst_slash, "H1")
        h1_path = _features_path(cfg.data_root, "H1", inst)
        h1_df = _read_features_csv(h1_path, h1_cols)
        h1_df = h1_df.reindex(idx, method="ffill").ffill()
        h1_df.columns = [f"H1::{c.split(inst_slash + '_', 1)[1]}" for c in h1_df.columns]
        # D
        d_cols = prefixed_feature_names(inst_slash, "D")
        d_path = _features_path(cfg.data_root, "D", inst)
        d_df = _read_features_csv(d_path, d_cols)
        d_df = d_df.reindex(idx, method="ffill").ffill()
        d_df.columns = [f"D::{c.split(inst_slash + '_', 1)[1]}" for c in d_df.columns]
        inst_block = pd.concat([m5_df, h1_df, d_df], axis=1)
        # Add instrument level to columns
        inst_block.columns = pd.MultiIndex.from_product([[inst], inst_block.columns])
        X_blocks.append(inst_block)

    X = pd.concat(X_blocks, axis=1).sort_index(axis=1)
    # Final sanity: drop any rows with remaining NaNs
    X = X.dropna(axis=0, how="any")
    closes_df = closes_df.reindex(X.index)
    return X, closes_df
