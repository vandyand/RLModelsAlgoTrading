#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Compute forex-rl directory and add unsupervised-ae to path for importing grid_features
FX_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
UNSUP_AE_DIR = os.path.join(FX_ROOT, "unsupervised-ae")
if UNSUP_AE_DIR not in sys.path:
    sys.path.append(UNSUP_AE_DIR)

# Reuse existing grid builder utilities
from grid_features import (  # type: ignore
    fetch_fx_ohlcv,
    fetch_etf_ohlcv,
    compute_indicator_grid,
    time_cyclical_features_from_index,
    get_etf_universe,
)


def load_oanda_instruments(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    col = df.columns[0]
    raw = [str(x).strip() for x in df[col].tolist() if isinstance(x, (str,))]
    out: List[str] = []
    for r in raw:
        if r.upper() == "INSTRUMENT" or not r:
            continue
        r2 = r.replace(" ", "")
        r2 = r2.replace("/", "_")
        out.append(r2)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for inst in out:
        if inst not in seen:
            uniq.append(inst)
            seen.add(inst)
    return uniq


def build_features(
    instruments: List[str],
    start: str,
    end: Optional[str],
    environment: str,
    access_token: Optional[str],
    use_all_etfs: bool,
    out_features: str,
    out_returns: str,
    out_dates: str,
    write_mode: str = "per-file",
    features_dir: Optional[str] = None,
) -> None:
    """Build features in either single-CSV or memory-efficient per-file mode.

    - single: original behavior, returns a single wide CSV (may be heavy)
    - per-file: compute base_index from FX closes, then write each instrument's
      features to disk individually to minimize memory usage
    """
    periods = [5, 15, 45, 135, 405]

    # Pass 1: determine base index via FX closes only (lightweight)
    fx_close_map: Dict[str, pd.Series] = {}
    base_index: Optional[pd.DatetimeIndex] = None
    for inst in instruments:
        print({"status": "fetch_fx", "instrument": inst})
        d_df = fetch_fx_ohlcv(inst, start, end, environment, access_token)
        fx_close_map[inst] = d_df["close"]
        if base_index is None:
            base_index = d_df.index
        else:
            base_index = base_index.intersection(d_df.index)
    assert base_index is not None

    # Save dates and returns early
    os.makedirs(os.path.dirname(out_returns), exist_ok=True)
    os.makedirs(os.path.dirname(out_dates), exist_ok=True)
    # Returns (next-day log returns per FX instrument)
    r_cols: Dict[str, pd.Series] = {}
    for inst, cls in fx_close_map.items():
        close = cls.reindex(base_index).ffill()
        r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        r_cols[inst] = r.shift(-1).fillna(0.0)
    R = pd.DataFrame(r_cols, index=base_index).astype(np.float32)
    R.to_csv(out_returns, index=True)
    pd.Series(pd.to_datetime(base_index)).dt.strftime('%Y-%m-%d').to_csv(out_dates, index=False, header=False)

    # Time features (once)
    t_feats = time_cyclical_features_from_index(base_index).astype(np.float32)

    if write_mode == "single":
        # Original behavior: materialize all blocks then concat
        fx_frames: List[pd.DataFrame] = []
        for inst in instruments:
            print({"status": "compute_fx_features", "instrument": inst})
            # Re-fetch instrument OHLCV to compute indicators (or cache earlier d_df if desired)
            d_df = fetch_fx_ohlcv(inst, start, end, environment, access_token)
            feats = compute_indicator_grid(d_df, prefix=f"FX_{inst}_", periods=periods)
            fx_frames.append(feats.reindex(base_index).fillna(0.0))

        etf_frames: List[pd.DataFrame] = []
        etf_list = get_etf_universe(use_all=use_all_etfs)
        if etf_list:
            print({"status": "fetch_etf_batch", "count": len(etf_list)})
            # Fetch in manageable batches to reduce memory footprint
            for tkr in etf_list:
                etf_map = fetch_etf_ohlcv([tkr], start, end)
                for tkr2, df in etf_map.items():
                    print({"status": "compute_etf_features", "ticker": tkr2})
                    feats = compute_indicator_grid(df, prefix=f"ETF_{tkr2}_", periods=periods)
                    etf_frames.append(feats.reindex(base_index).ffill().fillna(0.0))

        blocks: List[pd.DataFrame] = []
        blocks.extend(fx_frames)
        blocks.extend(etf_frames)
        blocks.append(t_feats)
        X = pd.concat(blocks, axis=1).astype(np.float32)
        os.makedirs(os.path.dirname(out_features), exist_ok=True)
        X.to_csv(out_features, index=True)
        print({
            "status": "saved",
            "features": out_features,
            "returns": out_returns,
            "dates": out_dates,
            "rows": int(X.shape[0]),
            "cols": int(X.shape[1]),
        })
        return

    # Memory-efficient per-file mode
    assert features_dir is not None, "features_dir required for per-file mode"
    fx_dir = os.path.join(features_dir, "FX")
    etf_dir = os.path.join(features_dir, "ETF")
    os.makedirs(fx_dir, exist_ok=True)
    os.makedirs(etf_dir, exist_ok=True)

    # FX features one-by-one
    for inst in instruments:
        print({"status": "compute_fx_features", "instrument": inst})
        d_df = fetch_fx_ohlcv(inst, start, end, environment, access_token)
        feats = compute_indicator_grid(d_df, prefix=f"FX_{inst}_", periods=periods)
        feats = feats.reindex(base_index).fillna(0.0).astype(np.float32)
        feats.to_csv(os.path.join(fx_dir, f"{inst}.csv"), index=True)

    # ETFs processed sequentially to limit memory
    etf_list = get_etf_universe(use_all=use_all_etfs)
    if etf_list:
        print({"status": "fetch_etf_batch", "count": len(etf_list)})
        for tkr in etf_list:
            try:
                etf_map = fetch_etf_ohlcv([tkr], start, end)
                for tkr2, df in etf_map.items():
                    print({"status": "compute_etf_features", "ticker": tkr2})
                    feats = compute_indicator_grid(df, prefix=f"ETF_{tkr2}_", periods=periods)
                    feats = feats.reindex(base_index).ffill().fillna(0.0).astype(np.float32)
                    feats.to_csv(os.path.join(etf_dir, f"{tkr2}.csv"), index=True)
            except Exception as exc:
                print({"status": "etf_error", "ticker": tkr, "error": str(exc)})

    # Time features
    t_feats.to_csv(os.path.join(features_dir, "time_features.csv"), index=True)

    print({
        "status": "saved_per_file",
        "features_dir": features_dir,
        "returns": out_returns,
        "dates": out_dates,
        "rows": int(len(base_index)),
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Build daily feature grid for all OANDA FX + all ETFs")
    p.add_argument("--oanda-csv", default="forex-rl/hybrid-agnostic/proj/docs/oanda-instruments.csv")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--out-features", default="forex-rl/hybrid-agnostic/data/all_features.csv")
    p.add_argument("--out-returns", default="forex-rl/hybrid-agnostic/data/fx_returns.csv")
    p.add_argument("--out-dates", default="forex-rl/hybrid-agnostic/data/dates.csv")
    p.add_argument("--write-mode", choices=["single", "per-file"], default="per-file")
    p.add_argument("--features-dir", default="forex-rl/hybrid-agnostic/data/parts")
    args = p.parse_args()

    instruments = load_oanda_instruments(args.oanda_csv)
    print({"status": "instruments", "count": len(instruments)})

    build_features(
        instruments=instruments,
        start=args.start,
        end=args.end,
        environment=args.environment,
        access_token=args.access_token,
        use_all_etfs=True,
        out_features=args.out_features,
        out_returns=args.out_returns,
        out_dates=args.out_dates,
        write_mode=args.write_mode,
        features_dir=args.features_dir,
    )


if __name__ == "__main__":
    main()
