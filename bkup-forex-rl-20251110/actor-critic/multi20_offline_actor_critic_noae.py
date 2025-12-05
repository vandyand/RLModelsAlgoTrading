#!/usr/bin/env python3
"""
Multi-Asset Offline Actor-Critic (no autoencoder)
- Uses the SAME engineered features as multi20_offline_actor_critic.py
  (FX Daily [+Weekly optional], ETF Daily, Time(10)), but skips AE pretrain
- A small learnable MLP trunk replaces the AE encoder and is trained end-to-end

Run example:
  python forex-rl/actor-critic/multi20_offline_actor_critic_noae.py \
      --start 2019-01-01 --end 2025-08-31 --epochs 8 --use-all-etfs \
      --latent 128 --trunk-hidden 2048,512 \
      --model-path forex-rl/actor-critic/checkpoints/multi20_offline_ac_noae.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repo paths are importable (match original script behavior)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
AC_DIR = os.path.dirname(__file__)
if AC_DIR not in sys.path:
    sys.path.append(AC_DIR)

# Reuse dataset/feature pipeline and AC head from the original script
# Note: build_dataset reads OANDA/yfinance and constructs feature and return matrices
from multi20_offline_actor_critic import (  # type: ignore
    build_dataset,
    standardize_fit,
    DEFAULT_OANDA_20,
    ActorCriticMulti,
)


@dataclass
class Config:
    # Data/source
    instruments: List[str]
    environment: str = "practice"
    account_id: Optional[str] = None
    access_token: Optional[str] = None
    start: str = "2019-01-01"
    end: Optional[str] = None
    include_weekly: bool = False
    include_hourly: bool = False
    use_all_etfs: bool = False
    etf_tickers: Optional[List[str]] = None

    # Model/training
    trunk_hidden: List[int] = None  # e.g., [2048, 512]
    latent_dim: int = 128
    policy_hidden: int = 256
    value_hidden: int = 256
    batch_size: int = 128
    epochs: int = 6
    gamma: float = 0.99
    actor_sigma: float = 0.3
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    adv_clip: float = 5.0
    reward_scale: float = 1.0
    max_units: int = 100
    action_mapping: str = "threshold"
    seed: int = 42
    autosave_secs: float = 120.0
    model_path: str = "forex-rl/actor-critic/checkpoints/multi20_offline_ac_noae.pt"

    def __post_init__(self) -> None:
        if self.trunk_hidden is None:
            self.trunk_hidden = [2048, 512]


class DirectTrunk(nn.Module):
    """Simple MLP trunk: input_dim -> hidden... -> latent_dim -> ReLU"""

    def __init__(self, input_dim: int, hidden_sizes: List[int], latent_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = int(input_dim)
        for h in (hidden_sizes or []):
            layers += [nn.Linear(last, int(h)), nn.ReLU()]
            last = int(h)
        layers += [nn.Linear(last, int(latent_dim)), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def tensorize(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)


def train_actor_critic_direct(
    X_train: pd.DataFrame,
    R_train: pd.DataFrame,
    X_val: pd.DataFrame,
    R_val: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> Dict[str, Any]:
    num_inst = R_train.shape[1]
    input_dim = int(X_train.shape[1])

    trunk = DirectTrunk(input_dim=input_dim, hidden_sizes=cfg.trunk_hidden, latent_dim=cfg.latent_dim).to(device)
    model = ActorCriticMulti(
        encoder=trunk,
        latent_dim=int(cfg.latent_dim),
        num_instruments=num_inst,
        policy_hidden=int(cfg.policy_hidden),
        value_hidden=int(cfg.value_hidden),
    ).to(device)

    # Train the entire model (including trunk)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    X_tr = tensorize(X_train).to(device)
    R_tr = tensorize(R_train).to(device)
    X_va = tensorize(X_val).to(device)
    R_va = tensorize(R_val).to(device)

    sigma = float(cfg.actor_sigma)
    log_sigma_const = float(np.log(sigma + 1e-8))

    def step_epoch(Xb: torch.Tensor, Rb: torch.Tensor, train: bool) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()
        T = Xb.size(0)
        pos_vec = torch.zeros(Rb.size(1), dtype=torch.float32, device=Xb.device)
        total_loss = total_actor = total_value = total_entropy = total_reward = 0.0
        steps = 0
        for t in range(0, T - 1):
            s_t = Xb[t:t + 1]
            s_tp1 = Xb[t + 1:t + 2]
            r_tp1 = Rb[t + 1]

            _, mu_t, v_t = model(s_t)
            with torch.no_grad():
                _, _, v_tp1 = model(s_tp1)

            # Sample pre-squash action from Normal(mu, sigma)
            eps = torch.randn_like(mu_t)
            pre = mu_t + sigma * eps
            a_t = torch.tanh(pre)

            if (cfg.action_mapping or "threshold") == "continuous":
                pos = a_t[0] * float(cfg.max_units)
            else:
                a = a_t[0]
                maxu = float(cfg.max_units)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxu), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxu), pos_vec)
                pos = pos_vec
            contrib = pos * r_tp1
            mean_c = torch.mean(contrib)
            std_c = torch.std(contrib)
            sharpe_like = mean_c / (std_c + 1e-8)
            r_scalar = cfg.reward_scale * sharpe_like

            adv = (r_scalar + cfg.gamma * v_tp1[0] - v_t[0]).detach()
            if cfg.adv_clip > 0:
                adv = torch.clamp(adv, -cfg.adv_clip, cfg.adv_clip)

            logprob = -0.5 * torch.sum(((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const, dim=1)
            entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))

            actor_loss = -adv * logprob.mean()
            value_target = (r_scalar + cfg.gamma * v_tp1[0]).detach()
            value_loss = cfg.value_coef * 0.5 * (value_target - v_t[0]) ** 2
            loss = actor_loss + value_loss - cfg.entropy_coef * entropy

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

            total_loss += float(loss.item())
            total_actor += float(actor_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)
            total_reward += float(r_scalar.item())
            steps += 1
        return {
            "loss": total_loss / max(1, steps),
            "actor": total_actor / max(1, steps),
            "value": total_value / max(1, steps),
            "entropy": total_entropy / max(1, steps),
            "reward": total_reward / max(1, steps),
        }

    for epoch in range(cfg.epochs):
        tr = step_epoch(X_tr, R_tr, train=True)
        va = step_epoch(X_va, R_va, train=False)
        print(json.dumps({
            "phase": "train",
            "epoch": epoch + 1,
            **{f"tr_{k}": v for k, v in tr.items()},
            **{f"va_{k}": v for k, v in va.items()},
        }), flush=True)

    # Final greedy validation (no sampling)
    model.eval()
    with torch.no_grad():
        T = X_va.size(0)
        cum = 0.0
        pos_vec = torch.zeros(R_va.size(1), dtype=torch.float32, device=X_va.device)
        for t in range(0, T - 1):
            s_t = X_va[t:t + 1]
            _, mu_t, _ = model(s_t)
            a_t = torch.tanh(mu_t)
            r_tp1 = R_va[t + 1]
            if (cfg.action_mapping or "threshold") == "continuous":
                pos = a_t[0] * float(cfg.max_units)
            else:
                a = a_t[0]
                maxu = float(cfg.max_units)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxu), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxu), pos_vec)
                pos = pos_vec
            contrib = pos * r_tp1
            sharpe_like = torch.mean(contrib) / (torch.std(contrib) + 1e-8)
            cum += float(sharpe_like.item())
        avg_daily = cum / max(1, T - 1)
        print(json.dumps({"phase": "eval", "val_avg_daily_reward": avg_daily}), flush=True)

    return {"model": model}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Actor-Critic (no autoencoder)")
    parser.add_argument("--instruments", default=",".join(DEFAULT_OANDA_20), help="Comma-separated OANDA instruments (20 recommended)")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end")
    parser.add_argument("--include-weekly", action="store_true")
    parser.add_argument("--no-include-weekly", dest="include_weekly", action="store_false")
    parser.set_defaults(include_weekly=False)
    parser.add_argument("--include-hourly", action="store_true")  # reserved
    parser.add_argument("--use-all-etfs", action="store_true")
    parser.add_argument("--etf-tickers", default="", help="Comma-separated custom ETF tickers (overrides --use-all-etfs if provided)")

    # Trunk/heads
    parser.add_argument("--trunk-hidden", default="2048,512", help="Comma-separated hidden sizes for trunk (e.g., 2048,512)")
    parser.add_argument("--latent", type=int, default=128, help="Latent dim output of trunk")
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--value-hidden", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-sigma", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--max-units", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mapping", choices=["threshold", "continuous"], default="threshold")
    parser.add_argument("--model-path", default=None, help="Optional path; default saves to checkpoints with timestamp")

    args = parser.parse_args()

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()] or DEFAULT_OANDA_20
    etf_list = [s.strip().upper() for s in (args.etf_tickers or "").split(",") if s.strip()]
    try:
        trunk_hidden = [int(x.strip()) for x in (args.trunk_hidden or "").split(",") if x.strip()]
    except Exception:
        trunk_hidden = [2048, 512]

    # Default model path with timestamp if not provided
    default_model_path = args.model_path or (
        f"forex-rl/actor-critic/checkpoints/multi20_offline_ac_noae_{datetime.now().strftime('%Y-%m-%d_%H%M')}.pt"
    )

    cfg = Config(
        instruments=instruments,
        environment=args.environment,
        account_id=args.account_id,
        access_token=args.access_token,
        start=args.start,
        end=args.end,
        include_weekly=bool(args.include_weekly),
        include_hourly=bool(args.include_hourly),
        use_all_etfs=bool(args.use_all_etfs),
        etf_tickers=(etf_list if len(etf_list) > 0 else None),
        trunk_hidden=trunk_hidden,
        latent_dim=int(args.latent),
        policy_hidden=int(args.policy_hidden),
        value_hidden=int(args.value_hidden),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        gamma=float(args.gamma),
        actor_sigma=float(args.actor_sigma),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        adv_clip=float(args.adv_clip),
        reward_scale=float(args.reward_scale),
        model_path=default_model_path,
        seed=int(args.seed),
        max_units=int(args.max_units),
        action_mapping=str(args.action_mapping),
    )

    # Seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataset + standardize
    print(json.dumps({
        "status": "fetch_data",
        "instruments": cfg.instruments,
        "use_all_etfs": cfg.use_all_etfs,
        "etf_override": bool(cfg.etf_tickers),
        "start": cfg.start,
        "end": cfg.end,
    }), flush=True)
    X, R, _ = build_dataset(cfg)  # uses OANDA/yfinance via original helpers
    Xn, stats = standardize_fit(X)

    # Time split
    n = len(Xn)
    if n < 200:
        raise RuntimeError("Not enough data to train. Increase date range.")
    split = int(n * 0.8)
    X_train = Xn.iloc[:split]
    R_train = R.iloc[:split]
    X_val = Xn.iloc[split:]
    R_val = R.iloc[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train actor-critic directly on features
    out = train_actor_critic_direct(X_train, R_train, X_val, R_val, cfg, device)
    model: ActorCriticMulti = out["model"]

    # Save checkpoint
    try:
        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
        torch.save({
            "cfg": cfg.__dict__,
            "feature_stats": stats,
            "trunk_state": model.encoder.state_dict(),
            "policy_state": model.policy.state_dict(),
            "value_state": model.value.state_dict(),
            "meta": {
                "input_dim": int(X_train.shape[1]),
                "latent_dim": int(cfg.latent_dim),
                "num_instruments": int(R.shape[1]),
            },
        }, cfg.model_path)
        print(json.dumps({"saved": cfg.model_path}), flush=True)
    except Exception as exc:
        print(json.dumps({"save_error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
