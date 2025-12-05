#!/usr/bin/env python3
"""
Unsupervised two-headed autoencoder (joint training):
- Reconstruction head: reconstructs PCA features (identity task)
- Position head: predicts 20 position sizes in [-1, 1] optimized by a Sharpe-like reward

Training data:
- Inputs X: PCA outputs from pca_reduce.py (numpy .npy)
- Returns R: next-day FX returns matrix (DataFrame->CSV) aligned to X rows

Loss:
- Recon loss: MSE(X_recon, X)
- Reward loss: - mean(contrib) / (std(contrib)+eps), where contrib = (tanh(head) * max_units) * returns
- Total loss = weight_recon * ReconLoss + weight_pos * RewardLoss

Outputs:
- Trained weights
- Numpy file of predicted positions over full series for inspection

Usage (aligned defaults):
  # 1) Build features
  python forex-rl/unsupervised-ae/grid_features.py \
    --start 2015-01-01 --end 2025-08-31 \
    --instruments EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD \
    --out-features forex-rl/unsupervised-ae/data/multi_features.csv \
    --out-returns forex-rl/unsupervised-ae/data/fx_returns.csv \
    --out-dates forex-rl/unsupervised-ae/data/dates.csv

  # 2) PCA reduce to 64 dims
  python forex-rl/unsupervised-ae/pca_reduce.py \
    --features forex-rl/unsupervised-ae/data/multi_features.csv \
    --components 64 \
    --out-features forex-rl/unsupervised-ae/data/pca_features.npy \
    --out-meta forex-rl/unsupervised-ae/data/pca_meta.npz

  # 3) Train AE with 24-dim bottleneck
  python forex-rl/unsupervised-ae/unsupervised_autoencoder.py \
    --pca-features forex-rl/unsupervised-ae/data/pca_features.npy \
    --returns-csv forex-rl/unsupervised-ae/data/fx_returns.csv \
    --instruments EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD \
    --latent 24 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math


class RunningNormalizer:
    """Exponential moving normalization for scalar loss terms.

    Maintains running mean/variance from detached values and returns a tanh-clipped
    normalized tensor preserving gradients.
    """
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8) -> None:
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.initialized: bool = False

    def update(self, value: torch.Tensor) -> None:
        v_mean = float(value.detach().mean().item())
        try:
            v_var = float(value.detach().var(unbiased=False).item())
        except Exception:
            v_var = 0.0
        if not self.initialized:
            self.running_mean = v_mean
            self.running_var = v_var if v_var > 0.0 else 1.0
            self.initialized = True
        else:
            m = self.momentum
            self.running_mean = m * self.running_mean + (1.0 - m) * v_mean
            self.running_var = m * self.running_var + (1.0 - m) * (v_var if v_var > 0.0 else self.running_var)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.update(value)
        else:
            self.update(value)
        device = value.device
        dtype = value.dtype
        mean_t = torch.tensor(self.running_mean, device=device, dtype=dtype)
        std_t = torch.tensor(math.sqrt(max(self.running_var, 0.0)) + self.eps, device=device, dtype=dtype)
        norm = (value - mean_t) / std_t
        return torch.tanh(norm)


def load_returns(returns_csv: str, instruments: List[str]) -> np.ndarray:
    R = pd.read_csv(returns_csv, index_col=0)
    missing = [i for i in instruments if i not in R.columns]
    if missing:
        raise RuntimeError(f"Returns CSV missing instruments: {missing}")
    R = R[instruments]
    return R.values.astype(np.float32)


class UnsupervisedAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_outputs: int) -> None:
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        # Decoder (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        # Supervised head: outputs raw actions in [-1,1]
        self.pos_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        pos_raw = self.pos_head(z)
        pos = torch.tanh(pos_raw)
        return z, recon, pos


@dataclass
class TrainConfig:
    latent: int = 24
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    weight_recon: float = 0.5
    weight_pos: float = 0.5
    max_units: float = 100.0
    train_ratio: float = 0.8
    seed: int = 42
    normalizer_momentum: float = 0.95


def train(model: UnsupervisedAE, X: np.ndarray, R: np.ndarray, cfg: TrainConfig, device: torch.device, checkpoint_path: Optional[str] = None, checkpoint_every: int = 5) -> dict:
    n, d = X.shape
    n_inst = R.shape[1]
    split = int(n * cfg.train_ratio)
    Xtr = torch.tensor(X[:split], dtype=torch.float32, device=device)
    Rtr = torch.tensor(R[:split], dtype=torch.float32, device=device)
    Xva = torch.tensor(X[split:], dtype=torch.float32, device=device)
    Rva = torch.tensor(R[split:], dtype=torch.float32, device=device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    recon_norm = RunningNormalizer(momentum=0.99)
    reward_norm = RunningNormalizer(momentum=cfg.normalizer_momentum if hasattr(cfg, 'normalizer_momentum') else 0.95)

    def step_epoch(Xb: torch.Tensor, Rb: torch.Tensor, train: bool) -> dict:
        model.train(train)
        T = Xb.size(0)
        total = {"loss": 0.0, "raw_recon": 0.0, "raw_reward": 0.0, "norm_recon": 0.0, "norm_reward": 0.0, "reward": 0.0}
        steps = 0
        for t in range(0, T - 1):
            xb = Xb[t:t+1]
            rb = Rb[t+1]  # next-day returns
            _, recon, pos = model(xb)
            # Reconstruction
            loss_recon = mse(recon, xb)
            # Position Sharpe-like loss
            units = pos[0] * cfg.max_units
            contrib = units * rb
            mean_c = torch.mean(contrib)
            std_c = torch.std(contrib)
            sharpe_like = mean_c / (std_c + 1e-8)
            loss_pos = -sharpe_like
            # Normalize and combine
            norm_recon = recon_norm.normalize(loss_recon)
            norm_reward = reward_norm.normalize(loss_pos)
            loss = cfg.weight_recon * norm_recon + cfg.weight_pos * norm_reward
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total["loss"] += float(loss.item())
            total["raw_recon"] += float(loss_recon.item())
            total["raw_reward"] += float(loss_pos.item())
            total["norm_recon"] += float(norm_recon.item())
            total["norm_reward"] += float(norm_reward.item())
            total["reward"] += float(sharpe_like.item())
            steps += 1
        for k in total:
            total[k] /= max(1, steps)
        return total

    for epoch in range(cfg.epochs):
        tr = step_epoch(Xtr, Rtr, train=True)
        va = step_epoch(Xva, Rva, train=False)
        print(json.dumps({
            "epoch": epoch + 1,
            **{f"tr_{k}": v for k, v in tr.items()},
            **{f"va_{k}": v for k, v in va.items()}
        }), flush=True)

        # Periodic checkpointing
        if checkpoint_path and checkpoint_every > 0 and ((epoch + 1) % int(checkpoint_every) == 0):
            try:
                base_dir = os.path.dirname(checkpoint_path)
                base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
                ext = os.path.splitext(checkpoint_path)[1] or ".pt"
                os.makedirs(base_dir, exist_ok=True)
                ck_path = os.path.join(base_dir, f"{base_name}_ep{epoch + 1:03d}{ext}")
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "input_dim": int(X.shape[1]),
                    "num_outputs": int(R.shape[1]),
                }, ck_path)
                print(json.dumps({"checkpoint_saved": ck_path}), flush=True)
            except Exception as _e:
                print(json.dumps({"checkpoint_error": str(_e)}), flush=True)

    # Final full-sequence inference
    model.eval()
    with torch.no_grad():
        Xall = torch.tensor(X, dtype=torch.float32, device=device)
        _, _, pos_all = model(Xall)
        pos_all = torch.tanh(pos_all).cpu().numpy().astype(np.float32)
    return {"positions": pos_all}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train unsupervised two-headed autoencoder (joint recon + reward) with running normalization")
    parser.add_argument("--pca-features", required=True)
    parser.add_argument("--returns-csv", required=True)
    parser.add_argument("--instruments", required=True, help="Comma-separated 20 FX instruments in column order of returns CSV")
    parser.add_argument("--latent", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-recon", type=float, default=0.5)
    parser.add_argument("--weight-pos", type=float, default=0.5)
    parser.add_argument("--max-units", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out-model", default="forex-rl/unsupervised-ae/checkpoints/unsup_ae.pt")
    parser.add_argument("--out-positions", default="forex-rl/unsupervised-ae/data/pred_positions.npy")
    parser.add_argument("--normalizer-momentum", type=float, default=0.95)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    print(json.dumps({
        "status": "start_unsup_ae",
        "latent": int(args.latent),
        "weights": {"recon": float(args.weight_recon), "pos": float(args.weight_pos)},
        "max_units": float(args.max_units)
    }))

    X = np.load(args.pca_features).astype(np.float32)
    print(json.dumps({"status": "loaded_pca", "rows": int(X.shape[0]), "dims": int(X.shape[1])}))
    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]
    R = load_returns(args.returns_csv, instruments)
    print(json.dumps({"status": "loaded_returns", "rows": int(R.shape[0]), "instruments": len(instruments)}))
    n, d = X.shape
    if R.shape[0] != n:
        raise RuntimeError(f"Length mismatch: PCA rows {n} vs returns rows {R.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnsupervisedAE(input_dim=d, latent_dim=int(args.latent), num_outputs=R.shape[1]).to(device)

    cfg = TrainConfig(
        latent=int(args.latent),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_recon=float(args.weight_recon),
        weight_pos=float(args.weight_pos),
        max_units=float(args.max_units),
        train_ratio=float(args.train_ratio),
        normalizer_momentum=float(args["normalizer_momentum"]) if hasattr(args, "__getitem__") else float(getattr(args, "normalizer_momentum", 0.95)),
    )

    out = train(model, X, R, cfg, device, checkpoint_path=args.out_model, checkpoint_every=5)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_positions), exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "cfg": asdict(cfg),
        "input_dim": int(d),
        "num_outputs": int(R.shape[1]),
    }, args.out_model)
    np.save(args.out_positions, out["positions"])

    print(json.dumps({
        "saved_model": args.out_model,
        "saved_positions": args.out_positions,
        "rows": int(n),
        "latent": int(args.latent),
    }))


if __name__ == "__main__":
    main()
