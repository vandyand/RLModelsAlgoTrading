#!/usr/bin/env python3
"""
Train a shared TCN on OANDA FX pairs to predict next-day returns.

This script builds the same OHLCV-derived daily feature set as
`forex_offline_actor_critic`, then trains a `TCNScalarHead` (from the WFO
TCN stack) across all symbols to predict the next-day log return per symbol.

The trained TCN is saved to a checkpoint that `forex_offline_actor_critic`
can later load and use as an auxiliary feature generator.

Usage (from forex-rl root):

  python -m forex-rl.actor-critic.forex_tcn_train \\
      --start 2021-01-01 --end 2025-12-31 \\
      --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from .forex_offline_actor_critic import (
    Config,
    DEFAULT_OANDA_20,
    build_forex_dataset,
)
from wfo.adapters.tcn_scalar_threshold import TCNScalarHead  # type: ignore


def _build_tcn_training_data(
    X: pd.DataFrame,
    R: pd.DataFrame,
    symbols: List[str],
    lookback: int,
    train_len: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build (X_seq, y, in_channels) for TCN training from features/returns."""
    idx = X.index
    n = len(idx)
    L = int(lookback)
    if n <= L + 2:
        raise RuntimeError("Window too short to train TCN")

    seqs: List[np.ndarray] = []
    targets: List[float] = []
    in_channels: Optional[int] = None

    for sym in symbols:
        prefix = f"FX_{sym}_D_"
        feat_cols = [c for c in X.columns if c.startswith(prefix)]
        if not feat_cols:
            continue
        F_sym = X[feat_cols].values.astype(np.float32)  # (n, C_sym)
        # Per-symbol feature standardization for TCN input
        mean = F_sym.mean(axis=0, keepdims=True)
        std = F_sym.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        F_norm = (F_sym - mean) / std

        if in_channels is None:
            in_channels = F_norm.shape[1]
        else:
            if in_channels != F_norm.shape[1]:
                raise RuntimeError(f"Inconsistent TCN input channels for symbol {sym}")

        r_sym = R[sym].values.astype(np.float32)
        max_t_train = min(train_len - 1, n - 2)
        for t in range(L, max_t_train):
            x_seq = F_norm[t - L : t, :].T  # (C, L)
            y = float(r_sym[t])
            seqs.append(x_seq)
            targets.append(y)

    if not seqs or in_channels is None:
        raise RuntimeError("No TCN training data constructed from features; check symbols/window.")

    X_seq = np.stack(seqs, axis=0)  # (B, C, L)
    y_arr = np.asarray(targets, dtype=np.float32)  # (B,)
    return X_seq, y_arr, int(in_channels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TCN for FX next-day return prediction.")
    ap.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_OANDA_20),
        help="Comma-separated list of OANDA FX instruments (default: DEFAULT_OANDA_20).",
    )
    ap.add_argument("--environment", type=str, default="practice", choices=["practice", "live"])
    ap.add_argument(
        "--account-id",
        type=str,
        default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"),
        help="OANDA account id (for metadata only).",
    )
    ap.add_argument(
        "--access-token",
        type=str,
        default=os.environ.get("OANDA_DEMO_KEY"),
        help="OANDA API access token (if omitted, falls back to OANDA_DEMO_KEY / OANDA_LIVE_KEY env).",
    )
    ap.add_argument("--start", type=str, default="2021-01-01")
    ap.add_argument("--end", type=str, default="2025-12-31")
    ap.add_argument("--epochs", type=int, default=8, help="TCN training epochs (default: 8).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lookback",
        type=int,
        default=64,
        help="Number of past days per TCN input sequence (default: 64).",
    )
    ap.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channels of TCNScalarHead (default: 32).",
    )
    ap.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Kernel size for TCN convolutions (default: 5).",
    )
    ap.add_argument(
        "--num-blocks",
        type=int,
        default=3,
        help="Number of residual TCN blocks (default: 3).",
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout in TCN blocks (default: 0.1).",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for TCN optimizer (default: 1e-3).",
    )
    ap.add_argument(
        "--ckpt-path",
        type=str,
        default="forex-rl/actor-critic/checkpoints/forex_tcn.pt",
        help="Where to save the trained TCN checkpoint.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided for TCN training.")

    cfg = Config(
        symbols=symbols,
        environment=args.environment,
        account_id=args.account_id,
        access_token=args.access_token,
        start=args.start,
        end=args.end,
    )

    print(
        json.dumps(
            {
                "status": "tcn_build_dataset",
                "symbols": symbols,
                "start": args.start,
                "end": args.end,
                "environment": args.environment,
            }
        ),
        flush=True,
    )

    X, R, dates = build_forex_dataset(cfg)
    n = len(X)
    start_eff = dates[0] if len(dates) > 0 else None
    end_eff = dates[-1] if len(dates) > 0 else None
    print(
        json.dumps(
            {
                "status": "tcn_effective_window",
                "aligned_days": int(n),
                "aligned_start": start_eff.isoformat() if start_eff is not None else None,
                "aligned_end": end_eff.isoformat() if end_eff is not None else None,
            }
        ),
        flush=True,
    )

    split = int(n * 0.8)
    if split < 1:
        split = n

    X_seq, y_arr, in_channels = _build_tcn_training_data(
        X, R, symbols, lookback=args.lookback, train_len=split
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32, device=device)

    model = TCNScalarHead(
        in_channels=in_channels,
        hidden_channels=int(args.hidden_channels),
        kernel_size=int(args.kernel_size),
        num_blocks=int(args.num_blocks),
        dropout=float(args.dropout),
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=float(args.lr))
    B = X_tensor.size(0)
    batch_size = min(B, int(args.batch_size))

    for epoch in range(int(args.epochs)):
        perm = torch.randperm(B, device=device)
        epoch_loss = 0.0
        steps = 0
        for i in range(0, B, batch_size):
            idx_b = perm[i : i + batch_size]
            xb = X_tensor[idx_b]
            yb = y_tensor[idx_b]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            steps += 1
        print(
            json.dumps(
                {
                    "phase": "tcn_train",
                    "epoch": epoch + 1,
                    "loss": epoch_loss / max(1, steps),
                }
            ),
            flush=True,
        )

    ckpt: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "in_channels": int(in_channels),
        "tcn_lookback": int(args.lookback),
        "hidden_channels": int(args.hidden_channels),
        "kernel_size": int(args.kernel_size),
        "num_blocks": int(args.num_blocks),
        "dropout": float(args.dropout),
        "symbols": symbols,
        "start": args.start,
        "end": args.end,
    }

    ckpt_path = args.ckpt_path
    if not os.path.isabs(ckpt_path):
        here = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
        ckpt_path = os.path.join(project_root, ckpt_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp = ckpt_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)
    print(json.dumps({"status": "tcn_saved", "ckpt_path": ckpt_path}), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()

