#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Repo paths
import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from unsupervised_autoencoder import UnsupervisedAE  # type: ignore
from grid_features import (  # type: ignore
    fetch_fx_ohlcv,
    fetch_etf_ohlcv,
    get_etf_universe,
    compute_indicator_grid,
    time_cyclical_features_from_index,
)
from td3_agent import TD3Agent, TD3Config  # type: ignore

# Defaults that allow running without flags
INSTRUMENTS_DEFAULT = (
    "EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,"
    "EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,"
    "AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD"
)

DEFAULT_PCA_META = os.path.join(FX_ROOT, "unsupervised-ae", "data", "pca_meta.npz")
DEFAULT_PCA_FEATURES = os.path.join(FX_ROOT, "unsupervised-ae", "data", "pca_features.npy")
DEFAULT_DATES_CSV = os.path.join(FX_ROOT, "unsupervised-ae", "data", "dates.csv")
DEFAULT_AE_MODEL = os.path.join(FX_ROOT, "unsupervised-ae", "checkpoints", "unsup_ae.pt")
DEFAULT_TD3_MODEL = os.path.join(FX_ROOT, "unsupervised-ae", "checkpoints", "td3.pt")


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
    fx_close_map: dict[str, pd.Series] = {}

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
    pca_components = meta["pca_components"].astype(np.float32)
    pca_mean = meta["pca_mean"].astype(np.float32)
    z_mean = meta["z_mean"].astype(np.float32)
    z_std = meta["z_std"].astype(np.float32)

    X_aligned = X.reindex(columns=cols).fillna(0.0).astype(np.float32)
    Xn = (X_aligned.values - f_mean[None, :]) / (f_std[None, :] + 1e-8)
    Z = (Xn - pca_mean[None, :]) @ pca_components.T
    Zw = (Z - z_mean[None, :]) / (z_std[None, :] + 1e-8)
    return Zw.astype(np.float32)


def run_inference(
    Z_whitened: np.ndarray,
    ae_model_path: str,
    td3_path: str,
    latent_dim: Optional[int],
    max_units: float,
    device: torch.device,
) -> np.ndarray:
    td3_ckpt = torch.load(td3_path, map_location=device)
    td3_cfg = td3_ckpt.get("td3_cfg")
    action_dim = int(td3_ckpt.get("action_dim", 20))

    ae_ckpt = torch.load(ae_model_path, map_location=device)
    meta_input_dim = int(ae_ckpt.get("input_dim", Z_whitened.shape[1]))
    cfg = ae_ckpt.get("cfg", {})
    if latent_dim is None:
        latent_dim = int(cfg.get("latent", 24))

    ae = UnsupervisedAE(input_dim=meta_input_dim, latent_dim=int(latent_dim), num_outputs=action_dim).to(device)
    ae.load_state_dict(ae_ckpt["model_state"])  # type: ignore
    ae.eval()

    agent = TD3Agent(TD3Config(state_dim=int(latent_dim), action_dim=action_dim), device)
    agent.actor.load_state_dict(td3_ckpt["actor_state"])  # type: ignore
    agent.critic.load_state_dict(td3_ckpt["critic_state"])  # type: ignore
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())

    with torch.no_grad():
        X = torch.tensor(Z_whitened, dtype=torch.float32, device=device)
        z = ae.encoder(X)
        actions = agent.select_action(z, noise_sigma=0.0)
        units = (actions * float(max_units)).cpu().numpy().astype(np.float32)
    return units


def main() -> None:
    p = argparse.ArgumentParser(description="Inference: PCA+whiten -> AE latent -> TD3 policy -> units")

    # Mode A: precomputed features
    p.add_argument("--features", help="Path to features CSV (from grid_features)")

    # Mode B: build features on the fly
    p.add_argument("--start", help="YYYY-MM-DD (build features)")
    p.add_argument("--end", help="YYYY-MM-DD (inclusive) for feature build")
    p.add_argument("--environment", choices=["practice", "live"], default="practice")
    p.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--use-all-etfs", action="store_true")
    p.add_argument("--etf-tickers", default="")

    # Mode C: directly provide whitened PCA features (npy)
    p.add_argument("--pca-features", default=DEFAULT_PCA_FEATURES, help="Path to precomputed whitened PCA features (.npy)")
    p.add_argument("--dates-csv", default=DEFAULT_DATES_CSV, help="CSV of dates aligned to features (one date per row)")

    # Shared
    p.add_argument("--instruments", default=INSTRUMENTS_DEFAULT, help="Comma-separated OANDA instruments (20)")
    p.add_argument("--pca-meta", default=DEFAULT_PCA_META, help="Path to PCA meta .npz produced by pca_reduce.py")
    p.add_argument("--ae-model", default=DEFAULT_AE_MODEL)
    p.add_argument("--td3-model", default=DEFAULT_TD3_MODEL)
    p.add_argument("--max-units", type=float, default=100.0)
    p.add_argument("--latent", type=int, help="Override latent dim if needed (defaults to model cfg)")
    p.add_argument("--json", action="store_true", help="Print JSON mapping of last row targets")
    p.add_argument("--last-only", action="store_true", help="Output only the last row units")
    p.add_argument("--out-csv", help="Optional CSV path to write full units timeseries")

    args = p.parse_args()

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()]

    # Build or load features
    stale_fallback_attempted = False
    if args.pca_features:
        # Load whitened PCA features directly
        print(json.dumps({"status": "load_pca_features", "path": args.pca_features}), file=sys.stderr)
        Z = np.load(args.pca_features).astype(np.float32)
        # Load dates index
        try:
            dates = pd.read_csv(args.dates_csv, header=None)
            index = pd.to_datetime(dates.iloc[:, 0])
        except Exception:
            # Fallback: make a simple RangeIndex of correct length
            index = pd.RangeIndex(start=0, stop=Z.shape[0], step=1)

        # If using default artifacts and data looks stale, build recent features on-the-fly and transform
        try:
            if isinstance(index, pd.Series) and len(index) > 0:
                last_val = index.iloc[-1]
                last_date = pd.to_datetime(last_val).date()
            elif isinstance(index, pd.DatetimeIndex) and len(index) > 0:
                last_date = pd.to_datetime(index[-1]).date()
            else:
                last_date = None
                # Compute latest expected trading date (last weekday before today UTC)
            now_utc = pd.Timestamp.utcnow().date()
            expected = now_utc - pd.Timedelta(days=1)
            while expected.weekday() >= 5:  # 5=Sat,6=Sun
                expected = expected - pd.Timedelta(days=1)
            is_default_artifacts = (
                (os.path.abspath(args.pca_features) == os.path.abspath(DEFAULT_PCA_FEATURES)) and
                (os.path.abspath(args.pca_meta) == os.path.abspath(DEFAULT_PCA_META))
            )
            if is_default_artifacts and last_date is not None and last_date < expected:
                stale_fallback_attempted = True
                print(json.dumps({
                    "status": "stale_precomputed_fallback",
                    "last_available": last_date.isoformat(),
                    "expected_at_least": str(expected),
                }), file=sys.stderr)
                print(json.dumps({
                    "status": "stale_precomputed_info",
                    "pca_features": os.path.abspath(args.pca_features),
                    "dates_csv": os.path.abspath(args.dates_csv),
                }), file=sys.stderr)
                # Build a recent window (>= longest indicator window) and transform using existing PCA meta
                start_for_fallback = (pd.Timestamp.utcnow().date() - pd.Timedelta(days=540)).isoformat()
                etfs = [s.strip().upper() for s in (args.etf_tickers or "").split(",") if s.strip()]
                X, index = build_features_on_the_fly(
                    instruments=instruments,
                    start=start_for_fallback,
                    end=None,
                    environment=args.environment,
                    access_token=args.access_token,
                    use_all_etfs=bool(args.use_all_etfs),
                    etf_tickers=etfs if len(etfs) > 0 else None,
                )
                print(json.dumps({"status": "transform_pca_whiten"}), file=sys.stderr)
                Z = transform_with_pca_and_whitening(X, args.pca_meta)
        except Exception as _e:
            print(json.dumps({"status": "stale_check_error", "error": str(_e)}), file=sys.stderr)
    elif args.features:
        if args.start or args.end or args.access_token or args.use_all_etfs or (args.etf_tickers and len(args.etf_tickers) > 0):
            print(json.dumps({"status": "warning", "message": "--features provided; build-related flags (start/end/etfs/token) are ignored"}), file=sys.stderr)
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

    if args.pca_features and not stale_fallback_attempted:
        # Already have Z from precomputed file
        pass
    else:
        print(json.dumps({"status": "transform_pca_whiten"}), file=sys.stderr)
        try:
            last_date = pd.to_datetime(index[-1]).strftime('%Y-%m-%d')
            print(json.dumps({"status": "last_feature_date", "date": last_date}), file=sys.stderr)
        except Exception:
            pass
        Z = transform_with_pca_and_whitening(X, args.pca_meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"status": "load_models", "ae": args.ae_model, "td3": args.td3_model}), file=sys.stderr)
    # Optimize memory if last_only requested
    if args.last_only and isinstance(Z, np.ndarray) and Z.ndim == 2 and Z.shape[0] > 1:
        Z_infer = Z[-1:, :]
    else:
        Z_infer = Z
    units = run_inference(Z_infer, args.ae_model, args.td3_model, args.latent, max_units=float(args.max_units), device=device)

    # Build outputs
    # Align index length with output rows
    if units.shape[0] == 1 and hasattr(index, '__len__') and len(index) > 0:
        out_index = pd.DatetimeIndex([index[-1]]) if hasattr(index, '__getitem__') else pd.RangeIndex(start=len(index)-1, stop=len(index))
    else:
        out_index = index
    df_units = pd.DataFrame(units, index=out_index, columns=instruments)

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
        print(json.dumps({"status": "inference_done", "rows": int(df_units.shape[0]), "cols": int(df_units.shape[1])}))


if __name__ == "__main__":
    main()
