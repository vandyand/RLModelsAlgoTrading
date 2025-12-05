#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import contextlib
import io

import numpy as np
import pandas as pd
import torch

# Make src/ importable when running as a script
SCRIPT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from models.multihead_autoencoder import MultiHeadAEConfig, MultiHeadAutoencoder  # type: ignore
from models.lstm_ddpg import DDPGConfig, LSTMDDPG  # type: ignore

# Bring in feature builders
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
FX_ROOT = os.path.join(REPO_ROOT, "unsupervised-ae")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
from grid_features import (  # type: ignore
    fetch_fx_ohlcv,
    fetch_etf_ohlcv,
    compute_indicator_grid,
    time_cyclical_features_from_index,
    get_etf_universe,
)


def compute_feature_window(
    feature_instruments_fx: List[str],
    start: str,
    end: str,
    access_token: str,
    use_all_etfs: bool = True,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    periods = [5, 15, 45, 135, 405]
    # FX
    fx_frames: List[pd.DataFrame] = []
    for inst in feature_instruments_fx:
        df = fetch_fx_ohlcv(inst, start, end, environment="practice", access_token=access_token)
        feats = compute_indicator_grid(df, prefix=f"FX_{inst}_", periods=periods)
        fx_frames.append(feats)
    # Use UNION of FX dates and forward-fill per instrument so we don't stall
    # the state when one instrument posts late
    base_index = fx_frames[0].index
    for f in fx_frames[1:]:
        base_index = base_index.union(f.index)
    base_index = base_index.sort_values()
    # ETFs
    etf_frames: List[pd.DataFrame] = []
    etf_list = get_etf_universe(use_all=use_all_etfs)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                etf_map = fetch_etf_ohlcv(etf_list, start, end)
    else:
        etf_map = fetch_etf_ohlcv(etf_list, start, end)
    for tkr, df in etf_map.items():
        feats = compute_indicator_grid(df, prefix=f"ETF_{tkr}_", periods=periods)
        feats = feats.reindex(base_index).ffill().fillna(0.0)
        etf_frames.append(feats)
    blocks: List[pd.DataFrame] = [frm.reindex(base_index).ffill().fillna(0.0) for frm in fx_frames]
    blocks.extend(etf_frames)
    blocks.append(time_cyclical_features_from_index(base_index))
    X = pd.concat(blocks, axis=1).astype(np.float32)
    return X, base_index


def load_plan_and_stats(plan_path: str, stats_path: str) -> Tuple[Dict[str, int], Dict[str, Tuple[float, float]]]:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    feature_to_shard = {str(k): int(v) for k, v in plan.get("feature_to_shard", {}).items()}
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_obj = json.load(f)
    stats = {k: (float(v[0]), float(v[1])) for k, v in stats_obj.items()}
    return feature_to_shard, stats


def encode_shards_from_dataframe(
    X: pd.DataFrame,
    feature_to_shard: Dict[str, int],
    stats: Dict[str, Tuple[float, float]],
    shards_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.startswith("shard_") and f.endswith(".pt")])
    # Invert plan to shard -> cols
    shard_to_cols: Dict[int, List[str]] = {}
    for feat, sid in feature_to_shard.items():
        if feat in X.columns:
            shard_to_cols.setdefault(sid, []).append(feat)
    zs: List[pd.DataFrame] = []
    for sid, fname in enumerate(shard_files):
        ck = torch.load(os.path.join(shards_dir, fname), map_location=device)
        cfgd = ck.get("cfg", {})
        cols_expected: List[str] = ck.get("columns", [])
        if not cols_expected:
            # Fallback to plan-derived subset for this shard
            cols_expected = shard_to_cols.get(sid, [])
        if not cols_expected:
            continue
        input_dim = int(ck.get("input_dim", len(cols_expected)))
        latent_dim = int(cfgd.get("latent_dim", 64))
        head_count = int(cfgd.get("head_count", 6))
        hidden_dims = cfgd.get("hidden_dims", [min(2048, max(256, input_dim // 2)), 256])

        model = MultiHeadAutoencoder(MultiHeadAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            head_count=head_count,
            hidden_dims=hidden_dims,
            dropout=float(cfgd.get("dropout", 0.0)),
            noise_std=float(cfgd.get("noise_std", 0.0)),
            randomized_heads=bool(cfgd.get("randomized_heads", False)),
            head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
            head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
            head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
            head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
            head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
            head_random_seed=(cfgd.get("head_random_seed", None)),
        )).to(device)
        model.load_state_dict(ck["model_state"])  # type: ignore[index]
        model.eval()

        # Build standardized input matrix aligned to expected columns
        present = [c for c in cols_expected if c in X.columns]
        missing = [c for c in cols_expected if c not in X.columns]
        # Start with zeros: standardized zeros = feature at mean
        Xmat = np.zeros((X.shape[0], len(cols_expected)), dtype=np.float32)
        if present:
            Xm = X[present].values.astype(np.float32)
            m = np.array([stats.get(c, (0.0, 1.0))[0] for c in present], dtype=np.float32)
            s = np.array([max(stats.get(c, (0.0, 1.0))[1], 1e-8) for c in present], dtype=np.float32)
            Xstd = (Xm - m[None, :]) / s[None, :]
            # place into Xmat respecting expected order
            pos = [cols_expected.index(c) for c in present]
            Xmat[:, pos] = Xstd
        with torch.no_grad():
            xb = torch.tensor(Xmat, dtype=torch.float32, device=device)
            z, _ = model(xb)
            z_np = z.cpu().numpy().astype(np.float32)
        cols_out = [f"z_s{sid:02d}_{i:02d}" for i in range(z_np.shape[1])]
        zs.append(pd.DataFrame(z_np, index=X.index, columns=cols_out))
    Z = pd.concat(zs, axis=1)
    return Z


def encode_final_latent(Z_concat: pd.DataFrame, final_ae_path: str, device: torch.device) -> pd.DataFrame:
    ck = torch.load(final_ae_path, map_location=device)
    cfgd = ck.get("cfg", {})
    input_dim = int(ck.get("input_dim", Z_concat.shape[1]))
    latent_dim = int(cfgd.get("latent_dim", 64))
    head_count = int(cfgd.get("head_count", 6))
    hidden_dims = cfgd.get("hidden_dims", [512, 128])
    model = MultiHeadAutoencoder(MultiHeadAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        head_count=head_count,
        hidden_dims=hidden_dims,
        dropout=float(cfgd.get("dropout", 0.0)),
        noise_std=float(cfgd.get("noise_std", 0.0)),
        randomized_heads=bool(cfgd.get("randomized_heads", False)),
        head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
        head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
        head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
        head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
        head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
        head_random_seed=(cfgd.get("head_random_seed", None)),
    )).to(device)
    model.load_state_dict(ck["model_state"])  # type: ignore[index]
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(Z_concat.values, dtype=torch.float32, device=device)
        z, _ = model(xb)
        z_np = z.cpu().numpy().astype(np.float32)
    cols_out = [f"z_final_{i:02d}" for i in range(z_np.shape[1])]
    return pd.DataFrame(z_np, index=Z_concat.index, columns=cols_out)


def infer_targets(
    action_instruments_fx: List[str],
    feature_instruments_fx: List[str],
    end_date: str,
    seq_len: int,
    access_token: str,
    shards_dir: str,
    final_ae_path: str,
    plan_path: str,
    stats_path: str,
    ddpg_ckpt: str,
    max_units_override: float | None = None,
    quiet: bool = False,
    temperature: float = 1.0,
    ) -> Dict[str, int]:
    # Compute minimal window: seq_len + max feature period (405 days)
    end = pd.to_datetime(end_date).date()
    start = (end - timedelta(days=seq_len + 420)).isoformat()
    X, index = compute_feature_window(feature_instruments_fx, start, end.isoformat(), access_token, use_all_etfs=True, quiet=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_to_shard, stats = load_plan_and_stats(plan_path, stats_path)

    if not quiet:
        print(json.dumps({"status": "encode_shards"}))
    Z_concat = encode_shards_from_dataframe(X, feature_to_shard, stats, shards_dir, device)
    if not quiet:
        print(json.dumps({"status": "encode_final", "rows": int(Z_concat.shape[0])}))
    Z_final = encode_final_latent(Z_concat, final_ae_path, device)

    # Take last seq_len rows
    if len(Z_final) < seq_len:
        raise RuntimeError(f"Not enough rows for seq_len={seq_len}; got {len(Z_final)}")
    Z_seq = Z_final.iloc[-seq_len:]

    # Load DDPG and act deterministically
    ck = torch.load(ddpg_ckpt, map_location=device)
    cfgd = ck.get("cfg", {})
    cfg = DDPGConfig(
        state_dim=Z_seq.shape[1],
        action_dim=len(action_instruments_fx),
        lstm_hidden=int(cfgd.get("lstm_hidden", 64)),
        actor_hidden=int(cfgd.get("actor_hidden", 128)),
        critic_hidden=int(cfgd.get("critic_hidden", 128)),
        seq_len=int(cfgd.get("seq_len", seq_len)),
        gamma=float(cfgd.get("gamma", 0.99)),
        tau=float(cfgd.get("tau", 0.005)),
        actor_lr=float(cfgd.get("actor_lr", 1e-4)),
        critic_lr=float(cfgd.get("critic_lr", 1e-3)),
        noise_sigma=float(cfgd.get("noise_sigma", 0.0)),
        max_units=float(cfgd.get("max_units", 100.0)),
        use_gru=bool(cfgd.get("use_gru", True)),
        num_threads=1,
        q_clip=float(cfgd.get("q_clip", 10.0)),
        factorized_action=bool(cfgd.get("factorized_action", False)),
    )
    agent = LSTMDDPG(cfg, device=device)
    # Restore weights
    if isinstance(ck.get("actor"), dict):
        agent.actor.load_state_dict(ck["actor"])  # type: ignore[index]
    if isinstance(ck.get("critic"), dict):
        agent.critic.load_state_dict(ck["critic"])  # type: ignore[index]

    with torch.no_grad():
        x = torch.tensor(Z_seq.values[None, ...], dtype=torch.float32, device=device)
        a = agent.actor(x)[0].cpu().numpy()
    max_units = float(max_units_override) if max_units_override is not None else float(cfg.max_units)
    a_scaled = np.clip(a * float(temperature), -1.0, 1.0)
    units = (a_scaled * max_units).astype(np.int32)
    return {action_instruments_fx[i]: int(units[i]) for i in range(len(action_instruments_fx))}


def main() -> None:
    p = argparse.ArgumentParser(description="Infer today's target positions using shard AEs + final AE + GRU-DDPG")
    p.add_argument("--instruments", required=True, help="Comma-separated FX instruments to trade (outputs)")
    p.add_argument("--feature-instruments", help="Optional comma-separated FX instruments to use for features (superset)")
    p.add_argument("--end", help="YYYY-MM-DD (defaults to yesterday UTC)")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--shards-dir", default="forex-rl/hybrid-agnostic/checkpoints/shards")
    p.add_argument("--final-ae", default="forex-rl/hybrid-agnostic/checkpoints/final_ae.pt")
    p.add_argument("--plan-path", default="forex-rl/hybrid-agnostic/artifacts/shard_plan.json")
    p.add_argument("--stats-path", default="forex-rl/hybrid-agnostic/artifacts/feature_stats.json")
    p.add_argument("--ddpg-ckpt", default="forex-rl/hybrid-agnostic/checkpoints/lstm_ddpg.pt")
    p.add_argument("--max-units", type=float)
    p.add_argument("--json", action="store_true")
    p.add_argument("--last-only", action="store_true", help="Suppress all logs; print only final JSON map")
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 1.0)), help="Scale actor outputs before units (e.g., 0.9)")
    p.add_argument("--emit-components", action="store_true", help="Output direction [-1,1] and magnitude [0,1] per instrument instead of units")
    args = p.parse_args()

    if not args.access_token:
        raise RuntimeError("Missing OANDA access token (OANDA_DEMO_KEY)")

    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]
    if args.feature_instruments:
        feature_instruments = [s.strip() for s in args.feature_instruments.split(',') if s.strip()]
    else:
        feature_instruments = instruments
    # Default end = yesterday UTC
    if args.end:
        end = args.end
    else:
        end = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

    targets = infer_targets(
        action_instruments_fx=instruments,
        feature_instruments_fx=feature_instruments,
        end_date=end,
        seq_len=int(args.seq_len),
        access_token=args.access_token,
        shards_dir=args.shards_dir,
        final_ae_path=args.final_ae,
        plan_path=args.plan_path,
        stats_path=args.stats_path,
        ddpg_ckpt=args.ddpg_ckpt,
        max_units_override=args.max_units,
        quiet=bool(args.last_only or args.json),
        temperature=float(args.temperature),
    )
    if bool(args.emit_components):
        # Recompute last components for the same sequence window and model
        # Load same sequence features quickly
        end_dt = (pd.to_datetime(end).date())
        start_dt = (end_dt - timedelta(days=int(args.seq_len) + 420)).isoformat()
        X_tmp, _ = compute_feature_window(feature_instruments, start_dt, end_dt.isoformat(), args.access_token, use_all_etfs=True, quiet=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_to_shard, stats = load_plan_and_stats(args.plan_path, args.stats_path)
        Z_concat = encode_shards_from_dataframe(X_tmp, feature_to_shard, stats, args.shards_dir, device)
        Z_final = encode_final_latent(Z_concat, args.final_ae, device)
        Z_seq = Z_final.iloc[-int(args.seq_len):]
        # Load agent as above
        ck = torch.load(args.ddpg_ckpt, map_location=device)
        cfgd = ck.get("cfg", {})
        cfg = DDPGConfig(
            state_dim=Z_seq.shape[1],
            action_dim=len(instruments),
            lstm_hidden=int(cfgd.get("lstm_hidden", 64)),
            actor_hidden=int(cfgd.get("actor_hidden", 128)),
            critic_hidden=int(cfgd.get("critic_hidden", 128)),
            seq_len=int(cfgd.get("seq_len", int(args.seq_len))),
            gamma=float(cfgd.get("gamma", 0.99)),
            tau=float(cfgd.get("tau", 0.005)),
            actor_lr=float(cfgd.get("actor_lr", 1e-4)),
            critic_lr=float(cfgd.get("critic_lr", 1e-3)),
            noise_sigma=0.0,
            max_units=float(cfgd.get("max_units", 100.0)),
            use_gru=bool(cfgd.get("use_gru", True)),
            num_threads=1,
            q_clip=float(cfgd.get("q_clip", 10.0)),
            factorized_action=bool(cfgd.get("factorized_action", False)),
        )
        agent = LSTMDDPG(cfg, device=device)
        if isinstance(ck.get("actor"), dict):
            agent.actor.load_state_dict(ck["actor"])  # type: ignore[index]
        # Components for the last window
        with torch.no_grad():
            d, m = agent.act_components(Z_seq.values)
        out = {
            "direction": {instruments[i]: float(d[i]) for i in range(len(instruments))},
            "magnitude": {instruments[i]: float(m[i]) for i in range(len(instruments))},
        }
        print(json.dumps(out))
    else:
        # When piping to aligner, use --last-only to print the JSON only
        if args.last_only or args.json:
            print(json.dumps(targets))
        else:
            print(targets)


if __name__ == "__main__":
    main()
