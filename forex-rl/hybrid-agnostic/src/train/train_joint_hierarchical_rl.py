#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Make src/ importable when running as a script
import os, sys
SCRIPT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from models.joint_hierarchical_rl import (
    HierarchicalAEConfig,
    JointRLConfig,
    JointHierarchicalRL,
)
from models.sharding import plan_shards  # type: ignore


def build_shard_slices(columns: Sequence[str], plan_path: str) -> Tuple[List[Optional[List[int]]], Dict[int, int]]:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    feature_to_shard: Dict[str, int] = {str(k): int(v) for k, v in plan.get("feature_to_shard", {}).items()}
    col_index: Dict[str, int] = {c: i for i, c in enumerate(columns)}
    num_shards = int(plan.get("num_shards", max(feature_to_shard.values(), default=-1) + 1))
    slices: List[Optional[List[int]]]= [None for _ in range(num_shards)]
    input_dims: Dict[int, int] = {sid: 0 for sid in range(num_shards)}
    for feat, sid in feature_to_shard.items():
        idx = col_index.get(feat)
        if idx is None:
            continue
        if slices[sid] is None:
            slices[sid] = []
        slices[sid].append(idx)
    for sid in range(num_shards):
        if slices[sid] is not None:
            slices[sid].sort()
            input_dims[sid] = len(slices[sid])
    return slices, input_dims


def ensure_plan_and_stats(features_csv: str, stats_path: str, plan_path: str, num_shards_hint: Optional[int] = None) -> None:
    need_stats = not os.path.exists(stats_path)
    need_plan = not os.path.exists(plan_path)
    if not (need_stats or need_plan):
        return
    X = pd.read_csv(features_csv, index_col=0)
    # Stats
    if need_stats:
        stats: Dict[str, Tuple[float, float]] = {}
        for c in X.columns:
            col = X[c].values.astype(np.float32)
            m = float(col.mean())
            s = float(col.std())
            if s < 1e-8:
                s = 1.0
            stats[c] = (m, s)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({k: [float(v[0]), float(v[1])] for k, v in stats.items()}, f)
    # Plan
    if need_plan:
        nshards = int(num_shards_hint) if num_shards_hint else 10
        plan = plan_shards(list(X.columns), num_shards=nshards, group_related=True)
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({"num_shards": int(plan.num_shards), "feature_to_shard": plan.feature_to_shard}, f)


def build_state_sequences(features_csv: str, seq_len: int, plan_path: str, stats_path: str) -> Tuple[np.ndarray, List[str], List[Optional[List[int]]], Dict[int, int]]:
    X = pd.read_csv(features_csv, index_col=0)
    # load stats and standardize
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    for c in X.columns:
        m, s = stats.get(c, [0.0, 1.0])
        s = 1.0 if s == 0.0 else float(s)
        X[c] = (X[c] - float(m)) / s
    X = X.astype(np.float32)
    slices, input_dims = build_shard_slices(list(X.columns), plan_path)

    idx = X.index
    M, D = X.shape
    import numpy.lib.stride_tricks as st
    num = max(0, M - seq_len)
    if num <= 0:
        return np.zeros((0, seq_len, D), dtype=np.float32), [], slices, input_dims
    Xv = X.values.astype(np.float32)
    s0, s1 = Xv.strides
    windows = st.as_strided(Xv, shape=(num, seq_len, D), strides=(s0, s0, s1)).copy()
    dates = list(idx[seq_len - 1:-1])
    return windows, dates, slices, input_dims


def load_fx_returns(returns_csv: str, instruments: List[str]) -> np.ndarray:
    R = pd.read_csv(returns_csv, index_col=0)
    R = R[instruments]
    return R.values.astype(np.float32)


def sortino_reward(actions: np.ndarray, returns: np.ndarray, max_units: float, eps: float = 1e-8) -> float:
    positions = actions * float(max_units)
    contrib = positions * returns
    mean_r = float(np.mean(contrib))
    downside = contrib[contrib < 0.0]
    dd = float(np.sqrt(np.mean(downside ** 2)) if downside.size > 0 else 0.0)
    return float(np.clip(mean_r / (dd + 1e-4), -10.0, 10.0))


def main() -> None:
    p = argparse.ArgumentParser(description="Joint training of hierarchical AEs and GRU-DDPG on final latents")
    p.add_argument("--features-csv", required=True)
    p.add_argument("--returns-csv", required=True)
    p.add_argument("--plan-path", default="forex-rl/hybrid-agnostic/artifacts/shard_plan.json")
    p.add_argument("--stats-path", default="forex-rl/hybrid-agnostic/artifacts/feature_stats.json")
    p.add_argument("--instruments", required=True, help="Comma-separated FX instruments matching returns columns")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    # AE
    p.add_argument("--num-shards", type=int)
    p.add_argument("--shard-latent", type=int, default=64)
    p.add_argument("--final-latent", type=int, default=64)
    p.add_argument("--head-count", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    # AE randomized head options
    p.add_argument("--randomized-heads", action="store_true", default=True, help="Enable randomized decoder heads for shard/final AEs")
    p.add_argument("--head-arch-min-layers", type=int, default=1)
    p.add_argument("--head-arch-max-layers", type=int, default=3)
    p.add_argument("--head-hidden-min", type=int, default=64)
    p.add_argument("--head-hidden-max", type=int, default=512)
    p.add_argument("--head-activations", default="relu,gelu,silu,elu", help="Comma-separated activation set for random heads")
    p.add_argument("--head-random-seed", type=int)
    # RL
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--noise-sigma", type=float, default=0.2)
    p.add_argument("--max-units", type=float, default=100.0)
    p.add_argument("--use-gru", action="store_true", default=True)
    p.add_argument("--q-clip", type=float, default=10.0)
    p.add_argument("--detach-state-for-rl", action="store_true", default=True)
    p.add_argument("--out-model", default="forex-rl/hybrid-agnostic/checkpoints/joint_hier_rl.pt")
    args = p.parse_args()

    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]

    # Ensure plan and stats exist (self-contained run)
    ensure_plan_and_stats(args.features_csv, args.stats_path, args.plan_path, args.num_shards)
    # Build sequences and shard mapping
    states_raw, state_dates, shard_slices, input_dims = build_state_sequences(args.features_csv, int(args.seq_len), args.plan_path, args.stats_path)
    R = load_fx_returns(args.returns_csv, instruments)
    # Align: sequences end at t; reward uses R[t+1]
    if len(R) != (len(states_raw) + 1):
        L = min(len(R) - 1, len(states_raw))
        states_raw = states_raw[-L:]
        R = R[:L+1]

    num_shards = int(args.num_shards) if args.num_shards else len(input_dims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_cfg = HierarchicalAEConfig(
        num_shards=num_shards,
        shard_input_dims=input_dims,
        shard_latent_dim=int(args.shard_latent),
        final_latent_dim=int(args.final_latent),
        head_count=int(args.head_count),
        dropout=float(args.dropout),
        noise_std=float(args.noise_std),
    )
    # Attach random head options to ae_cfg (consumed within model when creating MultiHeadAEConfig)
    setattr(ae_cfg, "randomized_heads", bool(args.randomized_heads))
    setattr(ae_cfg, "head_arch_min_layers", int(args.head_arch_min_layers))
    setattr(ae_cfg, "head_arch_max_layers", int(args.head_arch_max_layers))
    setattr(ae_cfg, "head_hidden_min", int(args.head_hidden_min))
    setattr(ae_cfg, "head_hidden_max", int(args.head_hidden_max))
    setattr(ae_cfg, "head_activation_set", [s.strip() for s in str(args.head_activations).split(',') if s.strip()])
    setattr(ae_cfg, "head_random_seed", (int(args.head_random_seed) if args.head_random_seed is not None else None))
    rl_cfg = JointRLConfig(
        ae=ae_cfg,
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
        use_gru=bool(args.use_gru),
        q_clip=float(args.q_clip),
        detach_state_for_rl=bool(args.detach_state_for_rl),
        factorized_action=True,
    )

    model = JointHierarchicalRL(rl_cfg).to(device)

    # Optimizers: separate for AE and RL so we can scale losses differently
    opt_ae = optim.AdamW(model.hier.parameters(), lr=1e-3)
    opt_actor = optim.AdamW(model.actor.parameters(), lr=float(args.actor_lr))
    opt_critic = optim.AdamW(model.critic.parameters(), lr=float(args.critic_lr))

    # Training loop: off-policy over pre-built sequences (like earlier trainer)
    B = int(args.batch_size)
    seq_len = int(args.seq_len)

    def sample_batch() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample indices such that a valid next state exists
        max_idx = len(states_raw) - 1
        idx = np.random.randint(0, max_idx, size=B)
        s = states_raw[idx]  # [B, T, D]
        s2 = states_raw[idx + 1]
        # Placeholder rewards; actual rewards computed on-the-fly from actions and returns
        r = np.zeros((B,), dtype=np.float32)
        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(s2, dtype=torch.float32, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.tensor([i for i in idx], dtype=torch.long, device=device),
        )

    for epoch in range(int(args.epochs)):
        totals = {"ae": 0.0, "actor": 0.0, "critic": 0.0}
        steps = 0
        for it in range(max(1, len(states_raw) // B)):
            s, s2, r_placeholder, idx_t = sample_batch()
            # AE forward on both s and s2 to compute recon loss; treat frames independently
            Bx, T, D = s.shape
            z_s, ae_loss_s = model.encode_final_sequence(s, shard_slices)
            z_s2, ae_loss_s2 = model.encode_final_sequence(s2, shard_slices)
            ae_loss = model.hier.recon_mse_total({"dummy": torch.zeros((), device=device)})  # init
            # recompute AE loss weights on both s and s2
            ae_loss = ae_loss_s + ae_loss_s2

            # RL losses use rewards computed from next-day returns aligned to batch indices
            # Build rewards from returns using actor actions on s
            with torch.no_grad():
                a_det = model.act(z_s, noise_sigma=0.0)
                # map each idx to its returns at t+1
                r_vec = torch.tensor(R[idx_t.cpu().numpy() + 1], dtype=torch.float32, device=device)
                pos_units = torch.clamp(a_det, -1.0, 1.0) * float(args.max_units)
                # Sortino-like per-sample reward across instruments
                contrib = pos_units * r_vec  # [B, N]
                mean_r = contrib.mean(dim=-1)  # [B]
                neg_mask = (contrib < 0.0).to(contrib.dtype)
                neg_count = neg_mask.sum(dim=-1).clamp(min=1.0)
                downside = (contrib.pow(2) * neg_mask).sum(dim=-1) / neg_count
                dd = torch.sqrt(downside + 1e-4)
                rewards = torch.clamp(mean_r / dd, -10.0, 10.0)

            actor_loss, critic_loss = model.rl_losses(z_s, a_det, rewards, z_s2)

            # Optimize: RL first (critic then actor), then AE
            # Critic update
            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            opt_critic.step()

            # Actor update (recompute to avoid graph reuse)
            a_det2 = model.act(z_s, noise_sigma=0.0)
            actor_loss2, _ = model.rl_losses(z_s, a_det2, rewards, z_s2)
            opt_actor.zero_grad()
            actor_loss2.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            opt_actor.step()

            opt_ae.zero_grad()
            ae_loss.backward()
            nn.utils.clip_grad_norm_(model.hier.parameters(), 1.0)
            opt_ae.step()

            # Soft update targets
            model._soft_update(model.actor, model.target_actor, float(args.tau))
            model._soft_update(model.critic, model.target_critic, float(args.tau))

            totals["ae"] += float(ae_loss.detach().item())
            totals["actor"] += float(actor_loss2.detach().item())
            totals["critic"] += float(critic_loss.detach().item())
            steps += 1
            if (it + 1) % 100 == 0:
                print(json.dumps({"epoch": epoch + 1, "iter": it + 1, "avg_ae": totals["ae"]/max(1, steps), "avg_actor": totals["actor"]/max(1, steps), "avg_critic": totals["critic"]/max(1, steps)}), flush=True)
        print(json.dumps({"epoch": epoch + 1, "done": True, "avg_ae": totals["ae"]/max(1, steps), "avg_actor": totals["actor"]/max(1, steps), "avg_critic": totals["critic"]/max(1, steps)}), flush=True)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    ckpt = model.export_checkpoint()
    torch.save(ckpt, args.out_model)
    print(json.dumps({"saved": args.out_model, "components": list(ckpt.keys())}))


if __name__ == "__main__":
    main()
