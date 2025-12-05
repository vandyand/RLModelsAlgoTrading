from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math

from features_loader import load_feature_panel


# -------------------- Config --------------------


@dataclass
class ACConfig:
    instruments: List[str]
    lookback_days: int = 20
    start_date: str | None = None
    end_date: str | None = None
    include_grans: List[str] = None  # set in main; e.g., ["M5","H1","D"]
    # RL
    epochs: int = 5
    gamma: float = 0.99
    actor_sigma: float = 0.3
    entropy_coef: float = 1e-3
    lr: float = 1e-3
    hidden: int = 256
    reward_scale: float = 1.0
    max_grad_norm: float = 1.0
    # Threshold mapping (absolute mode only for now)
    enter_long: float = 0.80
    exit_long: float = 0.60
    enter_short: float = 0.20
    exit_short: float = 0.40
    # Logging / checkpoint
    model_path: str = "ac-multi20/checkpoints/ac_multi20.pt"


# -------------------- Model --------------------


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.policy(h)
        v = self.value(h).squeeze(-1)
        return h, mu, v


# -------------------- Utils --------------------


def standardize_fit(X: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Tuple[float, float]]]:
    stats: dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean()); s = float(X[c].std())
        if s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def tensorize(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)


# -------------------- Training --------------------


def train_actor_critic(X: pd.DataFrame, closes: pd.DataFrame, cfg: ACConfig) -> None:
    # Flatten features deterministically: by instrument then feature
    flat_cols: list[tuple[str, str]] = []
    for inst in cfg.instruments:
        for col in X[inst].columns:
            flat_cols.append((inst, col))
    X_flat = X.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
    X_flat.columns = [f"{i}::{c}" for i, c in X_flat.columns]

    # Compute per-instrument log returns aligned to next step (r_t = log(c_t/c_{t-1}))
    px = closes.astype(float)
    r = np.log(px / px.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Align: action at t-1 earns reward at t. We'll drop the first row.
    X_flat = X_flat.iloc[1:]
    r = r.iloc[1:]

    # Standardize features and remember stats/column order for inference
    col_order = [str(c) for c in X_flat.columns]
    Xn, stats = standardize_fit(X_flat)

    T = len(Xn)
    num_inst = r.shape[1]
    input_dim = Xn.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(input_dim=input_dim, num_instruments=num_inst, hidden=int(cfg.hidden)).to(device)
    opt = optim.Adam(model.parameters(), lr=float(cfg.lr))

    X_t = tensorize(Xn).to(device)
    R_t = tensorize(r).to(device)

    sigma = float(cfg.actor_sigma)
    log_sigma_const = float(np.log(max(1e-8, sigma)))

    for epoch in range(int(cfg.epochs)):
        model.train()
        total_loss = 0.0
        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        steps = 0
        # Position state per instrument: -1,0,1
        pos_state = torch.zeros((num_inst,), dtype=torch.float32)
        for t in range(0, T - 1):
            s_t = X_t[t:t+1]
            s_tp1 = X_t[t+1:t+2]
            r_tp1 = R_t[t+1]  # next-step returns vector

            _, mu_t, v_t = model(s_t)
            with torch.no_grad():
                _, _, v_tp1 = model(s_tp1)

            # Diagonal Gaussian policy; sigmoid-squashed actions in [0,1)
            eps = torch.randn_like(mu_t)
            pre = mu_t + sigma * eps
            a_t = torch.sigmoid(pre)[0]  # shape: [num_inst], in (0,1)

            # Map to discrete state using absolute thresholds (no bands)
            # prev 0 -> long if a>enter_long; short if a<enter_short
            # prev +1 -> exit to 0 if a<exit_long
            # prev -1 -> exit to 0 if a>exit_short
            enter_long = float(cfg.enter_long)
            exit_long = float(cfg.exit_long)
            enter_short = float(cfg.enter_short)
            exit_short = float(cfg.exit_short)
            new_state = pos_state.clone()
            flat_mask = (pos_state == 0.0)
            long_mask = (pos_state > 0.0)
            short_mask = (pos_state < 0.0)
            new_state = torch.where(flat_mask & (a_t > enter_long), torch.ones_like(new_state), new_state)
            new_state = torch.where(flat_mask & (a_t < enter_short), -torch.ones_like(new_state), new_state)
            new_state = torch.where(long_mask & (a_t < exit_long), torch.zeros_like(new_state), new_state)
            new_state = torch.where(short_mask & (a_t > exit_short), torch.zeros_like(new_state), new_state)

            # Reward: mean(position * returns)
            reward = cfg.reward_scale * torch.mean(new_state * r_tp1)
            # Ensure it's a scalar float for logging accumulation
            reward = reward + 0.0 * v_t.mean()

            # Advantage TD(0)
            adv = (reward + cfg.gamma * v_tp1[0] - v_t[0]).detach()

            # Log-prob on pre-squash (ignore sigmoid correction for simplicity)
            logprob = -0.5 * torch.sum(((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const, dim=1)
            entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))

            actor_loss = -adv * logprob.mean()
            value_target = (reward + cfg.gamma * v_tp1[0]).detach()
            value_loss = 0.5 * (value_target - v_t[0]) ** 2
            loss = actor_loss + value_loss - cfg.entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            opt.step()

            total_loss += float(loss.item())
            total_actor += float(actor_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy))
            total_reward += float(reward.item())
            steps += 1
            pos_state = new_state.detach()  # advance state

        # Greedy eval on the same sequence (deterministic actions a = tanh(mu))
        model.eval()
        with torch.no_grad():
            greedy_reward_sum = 0.0
            greedy_steps = 0
            greedy_action_abs_sum = 0.0
            greedy_action_count = 0
            equity = 1.0  # unscaled cumulative return as equity curve
            pos_state = torch.zeros((num_inst,), dtype=torch.float32)
            for t in range(0, T - 1):
                s_t = X_t[t:t+1]
                r_tp1 = R_t[t+1]
                _, mu_t, _ = model(s_t)
                a_t = torch.sigmoid(mu_t)[0]
                # Map to new state (absolute thresholds)
                enter_long = float(cfg.enter_long)
                exit_long = float(cfg.exit_long)
                enter_short = float(cfg.enter_short)
                exit_short = float(cfg.exit_short)
                new_state = pos_state.clone()
                flat_mask = (pos_state == 0.0)
                long_mask = (pos_state > 0.0)
                short_mask = (pos_state < 0.0)
                new_state = torch.where(flat_mask & (a_t > enter_long), torch.ones_like(new_state), new_state)
                new_state = torch.where(flat_mask & (a_t < enter_short), -torch.ones_like(new_state), new_state)
                new_state = torch.where(long_mask & (a_t < exit_long), torch.zeros_like(new_state), new_state)
                new_state = torch.where(short_mask & (a_t > exit_short), torch.zeros_like(new_state), new_state)

                # Scaled reward for comparability with training logs
                r_scaled = float(cfg.reward_scale) * float(torch.mean(new_state * r_tp1).item())
                greedy_reward_sum += r_scaled
                greedy_steps += 1
                greedy_action_abs_sum += float(torch.mean(torch.abs(a_t)).item())
                greedy_action_count += 1
                # Unscaled daily step return contributes to equity curve
                step_ret_unscaled = float(torch.mean(new_state * r_tp1).item())
                equity *= (1.0 + step_ret_unscaled)
                pos_state = new_state
            greedy_reward_mean = (greedy_reward_sum / max(1, greedy_steps)) if greedy_steps > 0 else 0.0
            greedy_action_mean_abs = (greedy_action_abs_sum / max(1, greedy_action_count)) if greedy_action_count > 0 else 0.0
            greedy_cum_return = (equity - 1.0)

        def fmt(x: float) -> str:
            # Pretty, human-readable fixed-or-sci format with 3 sig figs
            val = float(x)
            if val == 0.0:
                return "0"
            mag = abs(val)
            if mag >= 1e-2 and mag < 1e3:
                return f"{val:.6f}".rstrip('0').rstrip('.')
            return f"{val:.3e}"

        print(json.dumps({
            "ph": "train",
            "ep": int(epoch + 1),
            "loss": fmt(total_loss / max(1, steps)),
            "act": fmt(total_actor / max(1, steps)),
            "val": fmt(total_value / max(1, steps)),
            "ent": fmt(total_entropy / max(1, steps)),
            "rew": fmt(total_reward / max(1, steps) + 0.0),
            "g_rew": fmt(greedy_reward_mean + 0.0),
            "g_cum": fmt(greedy_cum_return + 0.0),
            "g_aabs": fmt(greedy_action_mean_abs + 0.0),
        }), flush=True)

    # Save checkpoint
    try:
        # Build timestamped checkpoint path
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = cfg.model_path
        if base.lower().endswith(".pt"):
            base_no_ext, _ = os.path.splitext(base)
            save_path = f"{base_no_ext}-{ts}.pt"
        else:
            os.makedirs(base, exist_ok=True)
            save_path = os.path.join(base, f"ac_multi20-{ts}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        meta = {
            "input_dim": int(input_dim),
            "num_instruments": int(num_inst),
            "instruments": list(cfg.instruments),
            "hidden": int(cfg.hidden),
            "grans": list(cfg.include_grans or ["M5","H1","D"]),
            "thresholds": {
                "enter_long": float(cfg.enter_long),
                "exit_long": float(cfg.exit_long),
                "enter_short": float(cfg.enter_short),
                "exit_short": float(cfg.exit_short),
            },
        }
        torch.save({
            "model_state": model.state_dict(),
            "meta": meta,
            "feature_stats": stats,
            "col_order": col_order,
        }, save_path)
        # Sidecar meta JSON for downstream tools
        try:
            meta_path = os.path.splitext(save_path)[0] + ".meta.json"
            with open(meta_path, "w") as f:
                json.dump({"meta": meta, "col_order": col_order}, f)
        except Exception:
            pass
        print(json.dumps({"saved": save_path, "meta": meta_path}), flush=True)
    except Exception as exc:
        print(json.dumps({"save_error": str(exc)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Actor-Critic training on M5 features for multi-20 FX")
    p.add_argument("--instruments", default=",")
    p.add_argument("--lookback-days", type=int, default=20)
    p.add_argument("--start", default="", help="YYYY-MM-DD start date (UTC)")
    p.add_argument("--end", default="", help="YYYY-MM-DD end date (UTC)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-sigma", type=float, default=0.3)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--model-path", default="ac-multi20/checkpoints/ac_multi20.pt")
    # Absolute threshold mapping (no bands mode in AC)
    p.add_argument("--enter-long", type=float, default=0.80)
    p.add_argument("--exit-long", type=float, default=0.60)
    p.add_argument("--enter-short", type=float, default=0.20)
    p.add_argument("--exit-short", type=float, default=0.40)
    # Granularity selection: comma-separated subset of M5,H1,D; default all
    p.add_argument("--grans", default="M5,H1,D", help="Granularities to include (e.g., 'H1,D')")
    args = p.parse_args()

    # Instruments
    if args.instruments.strip() and args.instruments.strip() != ",":
        instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    else:
        # If not specified, try to mirror ga-multi20 defaults via local instruments module
        try:
            from instruments import DEFAULT_OANDA_20 as _DEF
            instruments = list(_DEF)
        except Exception:
            instruments = []

    cfg = ACConfig(
        instruments=instruments,
        lookback_days=int(args.lookback_days),
        start_date=(args.start.strip() or None),
        end_date=(args.end.strip() or None),
        epochs=int(args.epochs),
        gamma=float(args.gamma),
        actor_sigma=float(args.actor_sigma),
        entropy_coef=float(args.entropy_coef),
        lr=float(args.lr),
        hidden=int(args.hidden),
        reward_scale=float(args.reward_scale),
        model_path=str(args.model_path),
        enter_long=float(args.enter_long),
        exit_long=float(args.exit_long),
        enter_short=float(args.enter_short),
        exit_short=float(args.exit_short),
    )

    # Load features and closes at M5 resolution
    # Reuse existing loader which aligns M5/H1/D features to M5 index
    print(json.dumps({"st": "load", "n_inst": len(cfg.instruments)}), flush=True)
    grans = [g.strip().upper() for g in (args.grans or "M5,H1,D").split(',') if g.strip()]
    cfg.include_grans = list(grans)
    X_panel, closes = load_feature_panel(
        type("_Tmp", (), {
            "instruments": cfg.instruments,
            "lookback_days": cfg.lookback_days,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "data_root": "continuous-trader/data",
            "features_root": "continuous-trader/data/features",
            "include_grans": grans,
        })()
    )

    train_actor_critic(X_panel, closes, cfg)


if __name__ == "__main__":
    main()
