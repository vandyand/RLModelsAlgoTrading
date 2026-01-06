#!/usr/bin/env python3
"""
Inference script: produce today's raw actor outputs for the Alpaca crypto universe.

- Loads checkpoint from `crypto_offline_actor_critic` trainer.
- Rebuilds features over a recent lookback window (daily OHLCV + cyclical time).
- Optionally appends pre-trained TCN features (same as used during training).
- Standardizes using saved stats, runs the policy on the latest state.
- Outputs a JSON mapping {symbol: action} where:
    action_i ∈ (-1, 1)
  is exactly the `tanh(mu)` output of the policy head for that symbol
  (no additional clipping, scaling, or long-only mapping).

These raw actions can then be mapped to notionals/weights by a downstream
aligner script, preserving the model’s original semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure repo imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
AC_DIR = os.path.dirname(__file__)
if AC_DIR not in sys.path:
    sys.path.append(AC_DIR)

from crypto_offline_actor_critic import (  # type: ignore
    Config,
    build_crypto_dataset,
    standardize_apply,
    ActorCriticMulti,
    _build_tcn_features_from_checkpoint,
)


def load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid checkpoint format")
    return ckpt  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer today's target weights for Alpaca crypto universe."
    )
    parser.add_argument(
        "--checkpoint",
        default="forex-rl/actor-critic/checkpoints/crypto_offline_ac.pt",
        help="Path to crypto AC checkpoint (.pt file) produced by crypto_offline_actor_critic.",
    )
    parser.add_argument(
        "--environment",
        choices=["practice", "live"],
        default="practice",
        help="Environment tag for upstream (practice strongly recommended).",
    )
    parser.add_argument(
        "--candle-cache-base-url",
        default=os.environ.get("CANDLE_CACHE_BASE_URL", "http://127.0.0.1:9100"),
        help="Base URL for candle_cache_service (OANDA-compatible).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=200,
        help="Number of calendar days to look back when building features (default: 200).",
    )
    parser.add_argument(
        "--use-tcn",
        action="store_true",
        help=(
            "When set, load a pre-trained crypto TCN checkpoint and append its features "
            "before inference. This must match how the AC was trained."
        ),
    )
    parser.add_argument(
        "--tcn-ckpt",
        type=str,
        default="forex-rl/actor-critic/checkpoints/crypto_tcn.pt",
        help="Path to pre-trained TCN checkpoint used during training (if any).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="-",
        help="Where to write JSON targets (default: '-' = stdout).",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    saved_cfg: Dict[str, Any] = ckpt.get("config", {})  # type: ignore[assignment]
    feature_stats: Dict[str, Any] = ckpt.get("feature_stats", {})  # type: ignore[assignment]
    symbols: List[str] = list(saved_cfg.get("symbols", []))  # type: ignore[arg-type]
    input_dim = int(ckpt.get("input_dim", 0))
    if input_dim <= 0:
        input_dim = None  # will infer from X shape later

    if not symbols:
        raise SystemExit("No symbols found in checkpoint config; cannot build crypto universe.")

    # Determine last complete trading day in UTC.
    now_utc = datetime.now(timezone.utc)
    # For daily bars, treat yesterday as last complete day if we run intraday; for simplicity, always use yesterday.
    last_complete_day = (now_utc - timedelta(days=1)).date()
    start_date = last_complete_day - timedelta(days=max(60, int(args.lookback_days)))
    end_date = last_complete_day

    cfg = Config(
        symbols=symbols,
        environment=args.environment,
        candle_cache_base_url=args.candle_cache_base_url,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        epochs=0,
        batch_size=1,
        gamma=float(saved_cfg.get("gamma", 0.99)),
        actor_sigma=float(saved_cfg.get("actor_sigma", 0.3)),
        entropy_coef=float(saved_cfg.get("entropy_coef", 0.001)),
        value_coef=float(saved_cfg.get("value_coef", 0.5)),
        reward_scale=float(saved_cfg.get("reward_scale", 1.0)),
        max_notional=float(saved_cfg.get("max_notional", 1.0)),
        seed=int(saved_cfg.get("seed", 42)),
        policy_hidden=list(saved_cfg.get("policy_hidden", [256, 256])),  # type: ignore[list-item]
        value_hidden=list(saved_cfg.get("value_hidden", [256, 256])),    # type: ignore[list-item]
    )

    print(
        json.dumps(
            {
                "status": "crypto_infer_build_dataset",
                "symbols": cfg.symbols,
                "start": cfg.start,
                "end": cfg.end,
                "candle_cache_base_url": cfg.candle_cache_base_url,
                "use_tcn_flag": bool(args.use_tcn),
            }
        ),
        flush=True,
    )

    X, R, dates = build_crypto_dataset(cfg)
    if not feature_stats:
        raise RuntimeError("Checkpoint missing feature_stats; cannot standardize features.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Decide whether to use TCN features:
    has_tcn_in_stats = any(str(k).startswith("TCN_") for k in feature_stats.keys())
    use_tcn = bool(args.use_tcn or has_tcn_in_stats)

    if use_tcn:
        tcn_ckpt_path = args.tcn_ckpt
        if not os.path.isabs(tcn_ckpt_path):
            tcn_ckpt_path = os.path.join(REPO_ROOT, tcn_ckpt_path)
        if not os.path.exists(tcn_ckpt_path):
            raise SystemExit(
                f"TCN features required (stats contain TCN_ columns or --use-tcn set) "
                f"but TCN checkpoint not found at: {tcn_ckpt_path}"
            )
        tcn_feats = _build_tcn_features_from_checkpoint(
            X, cfg, symbols, ckpt_path=tcn_ckpt_path, device=device
        )
        X_aug = pd.concat([X, tcn_feats], axis=1)
    else:
        X_aug = X

    # Align columns to training feature set
    keys = list(feature_stats.keys())
    X_aug = X_aug.reindex(columns=keys).fillna(0.0)
    Xn = standardize_apply(X_aug, feature_stats)
    if len(Xn) == 0:
        raise RuntimeError("No feature rows produced for inference window.")

    x_last = torch.tensor(Xn.iloc[-1].values, dtype=torch.float32).unsqueeze(0)

    if input_dim is None or input_dim <= 0:
        input_dim = Xn.shape[1]
    num_inst = len(cfg.symbols)

    model = ActorCriticMulti(
        input_dim=input_dim,
        num_instruments=num_inst,
        policy_hidden=list(cfg.policy_hidden),
        value_hidden=list(cfg.value_hidden),
    )
    state_dict = ckpt.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint missing state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        _, mu, _ = model(x_last)
        a = torch.tanh(mu)[0].cpu().numpy()

    # Return actions exactly as produced by the model (tanh outputs in (-1,1)).
    targets: Dict[str, float] = {}
    for i, sym in enumerate(cfg.symbols):
        if i >= len(a):
            break
        targets[sym] = float(a[i])

    out_json = json.dumps(targets)
    if args.output_path == "-" or not args.output_path:
        print(out_json)
    else:
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()

