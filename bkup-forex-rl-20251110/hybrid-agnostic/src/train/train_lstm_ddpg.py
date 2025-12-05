#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Make src/ importable when running as a script
import os, sys
SCRIPT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from models.lstm_ddpg import DDPGConfig, LSTMDDPG, ReplayBuffer  # type: ignore


def sortino_reward(positions: np.ndarray, returns: np.ndarray, eps: float = 1e-8) -> float:
    """Compute portfolio Sortino-like reward for a single day.
    positions: shape (N,) in [-1,1] scaled later by max_units
    returns: shape (N,) daily instrument returns
    """
    contrib = positions * returns
    mean_r = float(np.mean(contrib))
    downside = contrib[contrib < 0.0]
    dd = float(np.sqrt(np.mean(downside ** 2)) if downside.size > 0 else 0.0)
    # Robustify: add epsilon floor and clamp reward to reasonable range
    sortino = mean_r / (dd + 1e-4)
    return float(np.clip(sortino, -10.0, 10.0))


def build_state_sequences(final_latents_csv: str, seq_len: int) -> Tuple[np.ndarray, List[str]]:
    Z = pd.read_csv(final_latents_csv, index_col=0)
    idx = Z.index
    X = Z.values.astype(np.float32)
    T, D = X.shape
    # Build strides view for efficient rolling windows
    import numpy.lib.stride_tricks as st
    num = max(0, T - seq_len)
    if num <= 0:
        return np.zeros((0, seq_len, D), dtype=np.float32), []
    s0, s1 = X.strides
    windows = st.as_strided(X, shape=(num, seq_len, D), strides=(s0, s0, s1)).copy()
    return windows, list(idx[seq_len - 1:-1])


def load_fx_returns(returns_csv: str, instruments: List[str]) -> np.ndarray:
    R = pd.read_csv(returns_csv, index_col=0)
    R = R[instruments]
    return R.values.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM-DDPG on final latent states")
    p.add_argument("--final-latents-csv", required=True, help="CSV of final AE latent per day (index dates)")
    p.add_argument("--returns-csv", required=True, help="CSV of daily FX returns per instrument (next-day aligned)")
    p.add_argument("--instruments", required=True, help="Comma-separated 20 FX instruments matching returns columns")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=5000)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--noise-sigma", type=float, default=0.2)
    p.add_argument("--max-units", type=float, default=100.0)
    p.add_argument("--out-model", default="forex-rl/hybrid-agnostic/checkpoints/lstm_ddpg.pt")
    p.add_argument("--factorized-action", action="store_true", default=True, help="Use direction [-1,1] and magnitude [0,1] action heads")
    args = p.parse_args()

    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]

    # Build state sequences
    states, state_dates = build_state_sequences(args.final_latents_csv, seq_len=int(args.seq_len))
    # Align returns
    R = load_fx_returns(args.returns_csv, instruments)
    # returns for t+1 already aligned to state at t in your builders; so for sequences ending at t, reward uses R[t+1]
    if len(R) != (len(states) + 1):
        # Fallback: trim to min length - alignment guard
        L = min(len(R) - 1, len(states))
        states = states[-L:]
        R = R[:L+1]
    state_dim = states.shape[-1]

    cfg = DDPGConfig(
        state_dim=state_dim,
        action_dim=len(instruments),
        lstm_hidden=64,
        actor_hidden=128,
        critic_hidden=128,
        seq_len=int(args.seq_len),
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        noise_sigma=float(args.noise_sigma),
        max_units=float(args.max_units),
        use_gru=True,
        num_threads=1,
        factorized_action=bool(args.factorized_action),
    )

    agent = LSTMDDPG(cfg)
    buffer = ReplayBuffer(capacity=int(args.buffer_size), state_dim=state_dim, action_dim=len(instruments), seq_len=int(args.seq_len))

    # Pre-fill buffer from historical sequences (off-policy) and train online over epochs
    for epoch in range(int(args.epochs)):
        total = {"actor": 0.0, "critic": 0.0, "reward": 0.0}
        steps = 0
        for t in range(len(states)):
            s_seq = states[t]
            # Action
            a = agent.act(s_seq, noise=True)
            # Reward uses next-day returns
            r_vec = R[t + 1]
            pos_units = a * cfg.max_units
            r = sortino_reward(pos_units, r_vec)
            # Next state
            if t + 1 < len(states):
                s2_seq = states[t + 1]
            else:
                s2_seq = s_seq  # terminal-like
            buffer.add(s_seq, a, r, s2_seq)

            # Train step when enough samples
            if buffer.ptr >= int(args.batch_size):
                out = agent.train_step(buffer, batch_size=int(args.batch_size))
                total["actor"] += out["actor_loss"]
                total["critic"] += out["critic_loss"]
            total["reward"] += float(r)
            steps += 1
            if (t + 1) % 200 == 0:
                print(json.dumps({"epoch": epoch + 1, "t": t + 1, "avg_actor": total["actor"]/max(1, steps), "avg_critic": total["critic"]/max(1, steps), "avg_reward": total["reward"]/max(1, steps)}), flush=True)
        print(json.dumps({"epoch": epoch + 1, "done": True, "avg_actor": total["actor"]/max(1, steps), "avg_critic": total["critic"]/max(1, steps), "avg_reward": total["reward"]/max(1, steps)}), flush=True)

        # Deterministic evaluation (no noise): how the current policy performs
        eval_sum = 0.0
        eval_steps = 0
        for t in range(len(states)):
            a = agent.act(states[t], noise=False)
            r_vec = R[t + 1]
            pos_units = a * cfg.max_units
            eval_sum += sortino_reward(pos_units, r_vec)
            eval_steps += 1
        print(json.dumps({"epoch": epoch + 1, "eval_avg_reward": eval_sum / max(1, eval_steps)}), flush=True)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({
        "cfg": cfg.__dict__,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, args.out_model)
    print(json.dumps({"saved": args.out_model}))


if __name__ == "__main__":
    main()
