#!/usr/bin/env python3
"""
Run inference for the unsupervised two-headed autoencoder.

End-to-end pipeline options:
- Load precomputed features CSV (fastest), or
- Build features on the fly from OANDA (FX) + yfinance (ETFs)
- Apply saved standardization -> PCA transform -> whitening
- Load trained AE checkpoint and output target units per instrument

Usage (from precomputed features):
  python forex-rl/unsupervised-ae/infer_unsupervised_ae.py \
    --features forex-rl/unsupervised-ae/data/multi_features.csv \
    --pca-meta forex-rl/unsupervised-ae/data/pca_meta.npz \
    --model-path forex-rl/unsupervised-ae/checkpoints/unsup_ae.pt \
    --instruments EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD

Usage (build features on the fly):
  python forex-rl/unsupervised-ae/infer_unsupervised_ae.py \
    --start 2019-01-01 --end 2025-08-31 \
    --instruments EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD \
    --pca-meta forex-rl/unsupervised-ae/data/pca_meta.npz \
    --model-path forex-rl/unsupervised-ae/checkpoints/unsup_ae.pt \
    --access-token $OANDA_DEMO_KEY

To align to OANDA positions, pipe the last-row JSON to align tool:
  python forex-rl/unsupervised-ae/infer_unsupervised_ae.py ... \
    --last-only --json \
    | python forex-rl/actor-critic/align_positions_to_targets.py --targets - --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Repo path setup (match training scripts)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

# Add current script directory (unsupervised-ae) to path so we can import sibling modules
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

# Import feature builders and model from local modules
from grid_features import (  # type: ignore
    fetch_fx_ohlcv,
    fetch_etf_ohlcv,
    get_etf_universe,
    compute_indicator_grid,
    time_cyclical_features_from_index,
)

from unsupervised_autoencoder import UnsupervisedAE  # type: ignore


def build_features_on_the_fly(
    instruments: List[str],
    start: str,
    end: Optional[str],
    environment: str,
    access_token: Optional[str],
    use_all_etfs: bool,
    etf_tickers: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    periods = [5, 15, 45, 135, 405]

    fx_frames: List[pd.DataFrame] = []
    fx_close_map: Dict[str, pd.Series] = {}

    for inst in instruments:
        print(json.dumps({"status": "fetch_fx", "instrument": inst}), file=sys.stderr)
        d_df = fetch_fx_ohlcv(inst, start, end, environment, access_token)
        fx_close_map[inst] = d_df["close"]
        print(json.dumps({"status": "compute_fx_features", "instrument": inst}), file=sys.stderr)
        feats = compute_indicator_grid(d_df, prefix=f"FX_{inst}_", periods=periods)
        fx_frames.append(feats)

    base_index = fx_frames[0].index
    for f in fx_frames[1:]:
        base_index = base_index.intersection(f.index)

    etf_frames: List[pd.DataFrame] = []
    etf_list = (etf_tickers if etf_tickers and len(etf_tickers) > 0 else get_etf_universe(use_all_etfs))
    if etf_list:
        print(json.dumps({"status": "fetch_etf", "count": len(etf_list)}), file=sys.stderr)
        etf_map = fetch_etf_ohlcv(etf_list, start, end)
        for tkr, df in etf_map.items():
            print(json.dumps({"status": "compute_etf_features", "ticker": tkr}), file=sys.stderr)
            feats = compute_indicator_grid(df, prefix=f"ETF_{tkr}_", periods=periods)
            feats = feats.reindex(base_index).ffill().fillna(0.0)
            etf_frames.append(feats)

    blocks: List[pd.DataFrame] = []
    for frm in fx_frames:
        blocks.append(frm.reindex(base_index).fillna(0.0))
    blocks.extend(etf_frames)
    blocks.append(time_cyclical_features_from_index(base_index))

    X = pd.concat(blocks, axis=1).astype(np.float32)
    return X, base_index


def transform_with_pca_and_whitening(
    X: pd.DataFrame,
    pca_meta_path: str,
) -> np.ndarray:
    meta = np.load(pca_meta_path)
    cols = [c for c in meta["columns"].tolist()]
    f_mean = meta["mean"].astype(np.float32)
    f_std = meta["std"].astype(np.float32)
    pca_components = meta["pca_components"].astype(np.float32)  # shape (n_comp, n_features)
    pca_mean = meta["pca_mean"].astype(np.float32)  # shape (n_features,)
    z_mean = meta["z_mean"].astype(np.float32)
    z_std = meta["z_std"].astype(np.float32)

    # Align columns; fill missing with 0.0
    X_aligned = X.reindex(columns=cols).fillna(0.0).astype(np.float32)

    # Standardize features
    Xn = (X_aligned.values - f_mean[None, :]) / (f_std[None, :] + 1e-8)

    # PCA transform: (Xn - pca_mean) @ components.T
    Z = (Xn - pca_mean[None, :]) @ pca_components.T

    # Whitening: use saved z_mean/z_std from train split
    Zw = (Z - z_mean[None, :]) / (z_std[None, :] + 1e-8)
    return Zw.astype(np.float32)


def run_inference(
    Z_whitened: np.ndarray,
    model_path: str,
    latent_dim: Optional[int],
    num_outputs: int,
    max_units: float,
    device: torch.device,
) -> np.ndarray:
    ckpt = torch.load(model_path, map_location=device)
    meta_num_outputs = int(ckpt.get("num_outputs", num_outputs))
    meta_input_dim = int(ckpt.get("input_dim", Z_whitened.shape[1]))
    cfg = ckpt.get("cfg", {})
    if latent_dim is None:
        latent_dim = int(cfg.get("latent", 24))

    model = UnsupervisedAE(input_dim=meta_input_dim, latent_dim=int(latent_dim), num_outputs=meta_num_outputs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        X = torch.tensor(Z_whitened, dtype=torch.float32, device=device)
        _, _, pos = model(X)
        units = (pos * float(max_units)).cpu().numpy().astype(np.float32)
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for unsupervised AE (features -> PCA+whiten -> AE -> units)")

    # Mode A: precomputed features
    parser.add_argument("--features", help="Path to features CSV (from grid_features)")

    # Mode B: build features on the fly
    parser.add_argument("--start", help="YYYY-MM-DD (build features)")
    parser.add_argument("--end", help="YYYY-MM-DD (inclusive) for feature build")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--use-all-etfs", action="store_true")
    parser.add_argument("--etf-tickers", default="")

    # Shared
    parser.add_argument("--instruments", required=True, help="Comma-separated OANDA instruments (20)")
    parser.add_argument("--pca-meta", required=True, help="Path to PCA meta .npz produced by pca_reduce.py")
    parser.add_argument("--model-path", default="forex-rl/unsupervised-ae/checkpoints/unsup_ae.pt")
    parser.add_argument("--max-units", type=float, default=100.0)
    parser.add_argument("--latent", type=int, help="Override latent dim if needed (defaults to model cfg)")
    parser.add_argument("--json", action="store_true", help="Print JSON mapping of last row targets")
    parser.add_argument("--last-only", action="store_true", help="Output only the last row units")
    parser.add_argument("--out-csv", help="Optional CSV path to write full units timeseries")

    args = parser.parse_args()

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()]

    # Build or load features
    if args.features:
        # Warn that build flags are ignored when --features is provided
        if args.start or args.end or args.access_token or args.use_all_etfs or (args.etf_tickers and len(args.etf_tickers) > 0):
            print(json.dumps({
                "status": "warning",
                "message": "--features provided; build-related flags (start/end/etfs/token) are ignored"
            }), file=sys.stderr)
        print(json.dumps({"status": "load_features", "path": args.features}), file=sys.stderr)
        X = pd.read_csv(args.features, index_col=0)
        index = pd.to_datetime(X.index)
    else:
        etfs = [s.strip().upper() for s in (args.etf_tickers or "").split(",") if s.strip()]
        print(json.dumps({"status": "build_features", "start": args.start, "end": args.end, "use_all_etfs": bool(args.use_all_etfs), "etf_override": bool(etfs)}), file=sys.stderr)
        X, index = build_features_on_the_fly(
            instruments=instruments,
            start=args.start or "2019-01-01",
            end=args.end,
            environment=args.environment,
            access_token=args.access_token,
            use_all_etfs=bool(args.use_all_etfs),
            etf_tickers=etfs if len(etfs) > 0 else None,
        )

    print(json.dumps({"status": "transform_pca_whiten"}), file=sys.stderr)
    # Log the last feature date for clarity
    try:
        last_date = pd.to_datetime(index[-1]).strftime('%Y-%m-%d')
        print(json.dumps({"status": "last_feature_date", "date": last_date}), file=sys.stderr)
    except Exception:
        pass
    Z = transform_with_pca_and_whitening(X, args.pca_meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"status": "load_model", "model": args.model_path}), file=sys.stderr)
    units = run_inference(Z, args.model_path, args.latent, num_outputs=len(instruments), max_units=float(args.max_units), device=device)

    # Build outputs
    df_units = pd.DataFrame(units, index=index, columns=instruments)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df_units.to_csv(args.out_csv, index=True)
        print(json.dumps({"status": "saved_csv", "path": args.out_csv}), file=sys.stderr)

    if args.last_only or args.json:
        last = df_units.iloc[-1]
        mapping = {inst: int(round(float(last[inst]))) for inst in instruments}
        if args.json:
            print(json.dumps(mapping))
        else:
            print(mapping)
    else:
        # Print brief stats
        print(json.dumps({"status": "inference_done", "rows": int(df_units.shape[0]), "cols": int(df_units.shape[1])}))


if __name__ == "__main__":
    main()
