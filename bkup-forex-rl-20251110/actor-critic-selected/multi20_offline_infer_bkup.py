#!/usr/bin/env python3
"""
Inference script: produce today's target position sizes for 20 FX instruments.
- Loads checkpoint from multi20_offline_actor_critic trainer
- Rebuilds features over a recent lookback window (daily FX + ETF + cyclical time)
- Standardizes using saved stats, runs encoder+policy, returns targets
- Output: JSON mapping {instrument: target_units}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Ensure we can import sibling training module and forex-rl utilities
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
AC_DIR = os.path.dirname(__file__)
if AC_DIR not in sys.path:
    sys.path.append(AC_DIR)

from multi20_offline_actor_critic import (  # type: ignore
    Config,
    DEFAULT_OANDA_20,
    get_etf_universe,
    build_dataset,
    standardize_apply,
    AutoEncoder,
    ActorCriticMulti,
)


def load_checkpoint(path: str) -> Dict[str, object]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid checkpoint format")
    return ckpt


def build_inference_config(saved_cfg: Dict[str, object], args: argparse.Namespace) -> Config:
    # Instruments: CLI override or saved
    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()]
    if not instruments:
        instruments = list(saved_cfg.get("instruments", DEFAULT_OANDA_20))  # type: ignore[arg-type]
    # ETF selection
    etf_override = [s.strip().upper() for s in (args.etf_tickers or "").split(",") if s.strip()]
    use_all_etfs = bool(args.use_all_etfs) if args.use_all_etfs is not None else bool(saved_cfg.get("use_all_etfs", False))

    cfg = Config(
        instruments=instruments,
        environment=args.environment,
        account_id=args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID"),
        access_token=args.access_token or os.environ.get("OANDA_DEMO_KEY"),
        start=args.start,
        end=args.end,
        include_weekly=False,
        include_hourly=False,
        use_all_etfs=use_all_etfs,
        etf_tickers=(etf_override if etf_override else (saved_cfg.get("etf_tickers") or None)),  # type: ignore[arg-type]
        ae_latent_dim=int(saved_cfg.get("ae_latent_dim", args.ae_latent)),
        ae_epochs=0,
        epochs=0,
        batch_size=max(1, int(args.batch_size)),
        gamma=float(saved_cfg.get("gamma", 0.99)),
        actor_sigma=float(saved_cfg.get("actor_sigma", 0.3)),
        entropy_coef=float(saved_cfg.get("entropy_coef", 0.001)),
        value_coef=float(saved_cfg.get("value_coef", 0.5)),
        max_grad_norm=float(saved_cfg.get("max_grad_norm", 1.0)),
        adv_clip=float(saved_cfg.get("adv_clip", 5.0)),
        reward_scale=float(saved_cfg.get("reward_scale", 1.0)),
        model_path=args.checkpoint,
        seed=int(saved_cfg.get("seed", 42)),
    )
    # Preserve hidden widths and max_units from training if present
    if isinstance(saved_cfg.get("ae_hidden"), list):
        cfg.ae_hidden = list(saved_cfg.get("ae_hidden"))  # type: ignore[assignment]
    cfg.max_units = int(saved_cfg.get("max_units", args.max_units))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer today's target position sizes for 20 FX instruments")
    parser.add_argument("--checkpoint", default="forex-rl/actor-critic/checkpoints/multi20_offline_ac.pt")
    parser.add_argument("--instruments", default=",")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--use-all-etfs", action="store_true")
    parser.add_argument("--etf-tickers", default="")
    parser.add_argument("--lookback-days", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ae-latent", type=int, default=64)
    parser.add_argument("--max-units", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise RuntimeError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = load_checkpoint(args.checkpoint)
    saved_cfg = ckpt.get("cfg") if isinstance(ckpt.get("cfg"), dict) else {}
    feature_stats: Dict[str, List[float]] = ckpt.get("feature_stats", {})  # type: ignore[assignment]
    meta = ckpt.get("meta", {}) if isinstance(ckpt.get("meta"), dict) else {}

    # Time window
    # Use 5pm New York (ET) as daily close boundary for FX; map to UTC.
    # If current time is before today's 5pm ET close, we should effectively use yesterday as the last complete day.
    now_utc = datetime.now(timezone.utc)
    # 5pm ET in UTC depends on DST; approximate with US/Eastern via pandas (no new deps)
    try:
        eastern = pd.Timestamp(now_utc).tz_convert('US/Eastern')
        is_after_close = (eastern.hour > 17) or (eastern.hour == 17 and eastern.minute >= 0)
    except Exception:
        # Fallback: assume close passed to avoid surprising behavior
        is_after_close = True
    last_complete_day = now_utc.date() if is_after_close else (now_utc.date() - timedelta(days=1))
    start = (last_complete_day - timedelta(days=max(200, int(args.lookback_days))))
    end = last_complete_day  # inclusive for FX; ETF loader uses end+1 day

    # Build inference config (daily-only)
    args.start = start.isoformat()  # type: ignore[attr-defined]
    args.end = end.isoformat()      # type: ignore[attr-defined]
    cfg = build_inference_config(saved_cfg, args)

    # Build dataset and standardize using saved stats
    X, R, dates = build_dataset(cfg)
    # Align columns strictly to training
    keys = list(feature_stats.keys())
    X = X.reindex(columns=keys).fillna(0.0)
    Xn = standardize_apply(X, feature_stats)
    if len(Xn) == 0:
        raise RuntimeError("No feature rows produced for inference window")

    x_last = torch.tensor(Xn.iloc[-1].values, dtype=torch.float32).unsqueeze(0)

    # Rebuild encoder + policy
    input_dim = int(meta.get("input_dim", Xn.shape[1]))
    latent_dim = int(meta.get("latent_dim", saved_cfg.get("ae_latent_dim", args.ae_latent)))
    num_inst = int(meta.get("num_instruments", len(cfg.instruments)))

    # AE hidden widths from saved cfg or default
    ae_hidden = saved_cfg.get("ae_hidden") if isinstance(saved_cfg.get("ae_hidden"), list) else [2048, 512, 128]

    from multi20_offline_actor_critic import AutoEncoder as AECls, ActorCriticMulti as ACMCls  # type: ignore
    ae = AECls(input_dim=input_dim, hidden_dims=ae_hidden, latent_dim=latent_dim)
    encoder_state = ckpt.get("ae_encoder_state")
    if isinstance(encoder_state, dict):
        ae.encoder.load_state_dict(encoder_state)
    model = ACMCls(encoder=ae.encoder, latent_dim=latent_dim, num_instruments=num_inst)
    # Load policy/value
    pol_state = ckpt.get("policy_state")
    if isinstance(pol_state, dict):
        model.policy.load_state_dict(pol_state)
    val_state = ckpt.get("value_state")
    if isinstance(val_state, dict):
        model.value.load_state_dict(val_state)

    model.eval()
    with torch.no_grad():
        z = model.encoder(x_last)
        mu = model.policy(z)
        a = torch.tanh(mu)[0].cpu().numpy()

    # Scale by max_units and format
    max_units = int(cfg.max_units)
    targets: Dict[str, int] = {}
    for i, inst in enumerate(cfg.instruments):
        if i >= len(a):
            break
        units = int(np.round(a[i] * max_units))
        targets[inst] = units

    print(json.dumps(targets))


if __name__ == "__main__":
    main()
