#!/usr/bin/env python3
"""
Supervised Autoencoder with dual heads:
- Reconstruction head: reconstructs PCA features (identity task)
- Supervised head: predicts 20-target position sizes in [-1, 1]

Training data:
- Inputs X: PCA outputs from pca_reduce.py (numpy .npy)
- Targets Y_pos: next-day FX portfolio signal derived from returns and a baseline simple rule
  or load externally provided supervision file of target positions.

Reward/Loss:
- Recon loss: MSE(X_recon, X)
- Position head loss: Sharpe-like surrogate built on predicted positions vs actual next-day returns
  L_pos = - mean(contrib) / (std(contrib)+eps), where contrib = units * returns
- Total loss = 0.5 * ReconLoss + 0.5 * PosLoss (scale to similar magnitudes)

Outputs:
- Trained weights and scaler metadata
- Numpy file of predicted positions over full series for inspection

Example:
  python forex-rl/supervised-ae/supervised_autoencoder.py \
    --pca-features forex-rl/supervised-ae/data/pca_features.npy \
    --returns-csv forex-rl/supervised-ae/data/fx_returns.csv \
    --instruments EUR_USD,... (20) \
    --latent 25 --epochs 20 --batch-size 128
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


def load_returns(returns_csv: str, instruments: List[str]) -> np.ndarray:
    R = pd.read_csv(returns_csv, index_col=0)
    missing = [i for i in instruments if i not in R.columns]
    if missing:
        raise RuntimeError(f"Returns CSV missing instruments: {missing}")
    R = R[instruments]
    return R.values.astype(np.float32)


class SupervisedAE(nn.Module):
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
    latent: int = 25
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_recon: float = 0.5
    weight_pos: float = 0.5
    max_units: float = 100.0
    train_ratio: float = 0.8
    seed: int = 42


def train(model: SupervisedAE, X: np.ndarray, R: np.ndarray, cfg: TrainConfig, device: torch.device) -> dict:
    n, d = X.shape
    n_inst = R.shape[1]
    split = int(n * cfg.train_ratio)
    Xtr = torch.tensor(X[:split], dtype=torch.float32, device=device)
    Rtr = torch.tensor(R[:split], dtype=torch.float32, device=device)
    Xva = torch.tensor(X[split:], dtype=torch.float32, device=device)
    Rva = torch.tensor(R[split:], dtype=torch.float32, device=device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    def step_epoch(Xb: torch.Tensor, Rb: torch.Tensor, train: bool) -> dict:
        model.train(train)
        T = Xb.size(0)
        total = {"loss": 0.0, "recon": 0.0, "pos": 0.0, "reward": 0.0}
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
            # Total
            loss = cfg.weight_recon * loss_recon + cfg.weight_pos * loss_pos
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total["loss"] += float(loss.item())
            total["recon"] += float(loss_recon.item())
            total["pos"] += float(loss_pos.item())
            total["reward"] += float(sharpe_like.item())
            steps += 1
        for k in total:
            total[k] /= max(1, steps)
        return total

    for epoch in range(cfg.epochs):
        tr = step_epoch(Xtr, Rtr, train=True)
        va = step_epoch(Xva, Rva, train=False)
        print(json.dumps({"epoch": epoch+1, **{f"tr_{k}": v for k,v in tr.items()}, **{f"va_{k}": v for k,v in va.items()}}), flush=True)

    # Final full-sequence inference
    model.eval()
    with torch.no_grad():
        Xall = torch.tensor(X, dtype=torch.float32, device=device)
        _, _, pos_all = model(Xall)
        pos_all = torch.tanh(pos_all).cpu().numpy().astype(np.float32)
    return {"positions": pos_all}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised autoencoder (dual-head)")
    parser.add_argument("--pca-features", required=True)
    parser.add_argument("--returns-csv", required=True)
    parser.add_argument("--instruments", required=True, help="Comma-separated 20 FX instruments in column order of returns CSV")
    parser.add_argument("--latent", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-recon", type=float, default=0.5)
    parser.add_argument("--weight-pos", type=float, default=0.5)
    parser.add_argument("--max-units", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out-model", default="forex-rl/supervised-ae/checkpoints/sup_ae.pt")
    parser.add_argument("--out-positions", default="forex-rl/supervised-ae/data/pred_positions.npy")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    X = np.load(args.pca_features).astype(np.float32)
    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]
    R = load_returns(args.returns_csv, instruments)
    n, d = X.shape
    if R.shape[0] != n:
        raise RuntimeError(f"Length mismatch: PCA rows {n} vs returns rows {R.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupervisedAE(input_dim=d, latent_dim=int(args.latent), num_outputs=R.shape[1]).to(device)

    cfg = TrainConfig(
        latent=int(args.latent),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_recon=float(args.weight_recon),
        weight_pos=float(args.weight_pos),
        max_units=float(args.max_units),
        train_ratio=float(args.train_ratio),
    )

    out = train(model, X, R, cfg, device)

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
