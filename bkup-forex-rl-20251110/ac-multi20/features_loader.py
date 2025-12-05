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
    - X: MultiIndex columns (instrument, feature_name)
    - closes: columns by instrument (close), aligned to X.index

    Granularity selection:
    - By default uses M5,H1,D with M5 step index.
    - If cfg.include_grans is provided (e.g., ["H1","D"]), the step index is the lowest granularity included
      (priority: M5 < H1 < D), and only specified granularities are loaded and forward-filled to the step index.
    """
    instruments = list(cfg.instruments)
    include_grans = getattr(cfg, "include_grans", None)
    if include_grans is None:
        include = ["M5", "H1", "D"]
    else:
        include = [g.strip().upper() for g in include_grans if str(g).strip()]
        include = [g for g in include if g in {"M5","H1","D"}]
        if not include:
            include = ["M5", "H1", "D"]
    # Determine step/base granularity
    priority = ["M5", "H1", "D"]
    base_gran = next((g for g in priority if g in include), "M5")

    # 1) Build base index range across instruments within lookback window
    base_closes: Dict[str, pd.Series] = {}
    for inst in instruments:
        ppath = _price_path(cfg.data_root, base_gran, inst)
        s = _read_close_csv(ppath)
        base_closes[inst] = s
    idx = None
    for s in base_closes.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"No overlapping {base_gran} timestamps across instruments")
    end_ts = idx.max()
    # Clamp range
    if cfg.start_date:
        start_ts = max(pd.Timestamp(cfg.start_date, tz="UTC"), idx.min())
    else:
        start_ts = end_ts - pd.Timedelta(days=int(cfg.lookback_days))
    if cfg.end_date:
        end_ts = min(pd.Timestamp(cfg.end_date, tz="UTC"), idx.max())
    idx = idx[(idx >= start_ts) & (idx <= end_ts)]
    if idx.has_duplicates:
        idx = pd.DatetimeIndex(sorted(set(idx)))

    X_blocks: List[pd.DataFrame] = []
    closes_df = pd.DataFrame({k: v.reindex(idx).ffill().astype(float) for k, v in base_closes.items()}, index=idx)

    for inst in instruments:
        inst_slash = underscore_to_slash(inst)
        per_inst_blocks: List[pd.DataFrame] = []
        if "M5" in include:
            m5_cols = prefixed_feature_names(inst_slash, "M5")
            m5_path = _features_path(cfg.data_root, "M5", inst)
            m5_df = _read_features_csv(m5_path, m5_cols).reindex(idx, method="ffill").ffill()
            m5_df.columns = [f"M5::{c.split(inst_slash + '_', 1)[1]}" for c in m5_df.columns]
            per_inst_blocks.append(m5_df)
        if "H1" in include:
            h1_cols = prefixed_feature_names(inst_slash, "H1")
            h1_path = _features_path(cfg.data_root, "H1", inst)
            h1_df = _read_features_csv(h1_path, h1_cols).reindex(idx, method="ffill").ffill()
            h1_df.columns = [f"H1::{c.split(inst_slash + '_', 1)[1]}" for c in h1_df.columns]
            per_inst_blocks.append(h1_df)
        if "D" in include:
            d_cols = prefixed_feature_names(inst_slash, "D")
            d_path = _features_path(cfg.data_root, "D", inst)
            d_df = _read_features_csv(d_path, d_cols).reindex(idx, method="ffill").ffill()
            d_df.columns = [f"D::{c.split(inst_slash + '_', 1)[1]}" for c in d_df.columns]
            per_inst_blocks.append(d_df)

        if per_inst_blocks:
            inst_block = pd.concat(per_inst_blocks, axis=1)
            inst_block.columns = pd.MultiIndex.from_product([[inst], inst_block.columns])
            X_blocks.append(inst_block)

    X = pd.concat(X_blocks, axis=1).sort_index(axis=1)
    X = X.dropna(axis=0, how="any")
    closes_df = closes_df.reindex(X.index)
    return X, closes_df
