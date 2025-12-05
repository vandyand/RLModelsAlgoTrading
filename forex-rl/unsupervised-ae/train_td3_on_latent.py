#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from unsupervised_autoencoder import UnsupervisedAE  # type: ignore
from td3_agent import TD3Agent, TD3Config, ReplayBuffer, sharpe_like_reward  # type: ignore


INSTRUMENTS_DEFAULT = (
    "EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,"
    "EUR_JPY,GBP_JPY,EUR_GBP,EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,"
    "AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD"
)


def load_returns(returns_csv: str, instruments: List[str]) -> np.ndarray:
    R = pd.read_csv(returns_csv, index_col=0)
    missing = [i for i in instruments if i not in R.columns]
    if missing:
        raise RuntimeError(f"Returns CSV missing instruments: {missing}")
    R = R[instruments]
    return R.values.astype(np.float32)


def compute_latents(
    ae_ckpt_path: str,
    X_whitened: np.ndarray,
    latent_dim: int | None,
    device: torch.device,
) -> Tuple[np.ndarray, int, int]:
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    meta_input_dim = int(ckpt.get("input_dim", X_whitened.shape[1]))
    cfg = ckpt.get("cfg", {})
    if latent_dim is None:
        latent_dim = int(cfg.get("latent", 24))
    model = UnsupervisedAE(input_dim=meta_input_dim, latent_dim=int(latent_dim), num_outputs=int(ckpt.get("num_outputs", 20))).to(device)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_whitened, dtype=torch.float32, device=device)
        z = model.encoder(X)  # [T, latent]
        z_np = z.cpu().numpy().astype(np.float32)
    return z_np, int(latent_dim), int(ckpt.get("num_outputs", 20))


def build_transitions(
    latents: np.ndarray,
    returns_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align s_t, s_{t+1}, returns_{t+1} for t=0..T-2."""
    T = latents.shape[0]
    s = latents[:-1]
    s2 = latents[1:]
    r = returns_mat[1:]
    assert s.shape[0] == r.shape[0] == s2.shape[0], "Alignment mismatch"
    return s, s2, r


def main() -> None:
    p = argparse.ArgumentParser(description="Train TD3 on frozen AE latent states with Sharpe-like per-step reward")
    p.add_argument("--pca-features", default="forex-rl/unsupervised-ae/data/pca_features.npy")
    p.add_argument("--returns-csv", default="forex-rl/unsupervised-ae/data/fx_returns.csv")
    p.add_argument("--instruments", default=INSTRUMENTS_DEFAULT, help="Comma-separated 20 FX instruments matching returns columns")
    p.add_argument("--ae-model", default="forex-rl/unsupervised-ae/checkpoints/unsup_ae.pt")
    p.add_argument("--latent", type=int)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--explore-noise", type=float, default=0.1)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--max-units", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default="forex-rl/unsupervised-ae/checkpoints/td3.pt")
    p.add_argument("--out-positions", default="forex-rl/unsupervised-ae/data/td3_positions.npy")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action="store_true", help="Resume from --save checkpoint if it exists")
    p.add_argument("--resume-from", default="", help="Explicit checkpoint path to resume from (overrides --resume)")
    args = p.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    instruments = [s.strip() for s in args.instruments.split(',') if s.strip()]

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    print(json.dumps({"status": "load_data", "pca": args.pca_features, "returns": args.returns_csv, "epochs": int(args.epochs), "resume": bool(args.resume) or bool(args.resume_from)}))
    Xw = np.load(args.pca_features).astype(np.float32)
    R = load_returns(args.returns_csv, instruments)
    if Xw.shape[0] != R.shape[0]:
        raise RuntimeError(f"Length mismatch: PCA rows {Xw.shape[0]} vs returns rows {R.shape[0]}")

    print(json.dumps({"status": "encode_latent", "ae": args.ae_model}))
    latents, latent_dim, action_dim = compute_latents(args.ae_model, Xw, args.latent, device)

    s, s2, r_next = build_transitions(latents, R)

    cfg = TD3Config(
        state_dim=int(latent_dim),
        action_dim=int(action_dim),
        actor_lr=1e-3,
        critic_lr=1e-3,
        tau=0.005,
        gamma=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        max_action=1.0,
    )
    agent = TD3Agent(cfg, device)

    # Optional resume
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif bool(args.resume):
        resume_path = args.save
    if resume_path and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            agent.actor.load_state_dict(ck["actor_state"])  # type: ignore
            agent.critic.load_state_dict(ck["critic_state"])  # type: ignore
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            print(json.dumps({"status": "resumed", "path": resume_path}), flush=True)
        except Exception as _e:
            print(json.dumps({"status": "resume_failed", "path": resume_path, "error": str(_e)}), flush=True)

    replay = ReplayBuffer(state_dim=cfg.state_dim, action_dim=cfg.action_dim, capacity=int(args.buffer_size), device=device)

    # Warmup with random policy
    print(json.dumps({"status": "warmup", "steps": int(args.warmup_steps)}))
    with torch.no_grad():
        T = s.shape[0]
        idx = np.arange(T)
        np.random.shuffle(idx)
        idx = idx[: int(min(args.warmup_steps, T))]
        st = torch.tensor(s[idx], dtype=torch.float32, device=device)
        st2 = torch.tensor(s2[idx], dtype=torch.float32, device=device)
        rvec = torch.tensor(r_next[idx], dtype=torch.float32, device=device)
        a = torch.empty((st.shape[0], cfg.action_dim), dtype=torch.float32, device=device).uniform_(-1.0, 1.0)
        rew = sharpe_like_reward(a, rvec, max_units=float(args.max_units))
        done = torch.zeros((st.shape[0], 1), dtype=torch.float32, device=device)
        replay.add(st, a, rew, st2, done)

    # Training epochs: roll through time in order, add transitions with exploration noise, train TD3
    total_updates = 0
    for epoch in range(int(args.epochs)):
        ep_info = {"epoch": epoch + 1, "actor": 0.0, "critic": 0.0, "reward": 0.0, "steps": 0}
        for t in range(s.shape[0]):
            st = torch.tensor(s[t:t+1], dtype=torch.float32, device=device)
            st2 = torch.tensor(s2[t:t+1], dtype=torch.float32, device=device)
            rvec = torch.tensor(r_next[t:t+1], dtype=torch.float32, device=device)
            # Policy action and reward computation do not need gradients
            with torch.no_grad():
                a = agent.select_action(st, noise_sigma=float(args.explore_noise))
                rew = sharpe_like_reward(a, rvec, max_units=float(args.max_units))
                done = torch.tensor([[0.0]], dtype=torch.float32, device=device)
                replay.add(st, a, rew, st2, done)

            # Train updates per step (with gradients enabled)
            for _ in range(int(args.updates_per_step)):
                if replay.size >= int(args.batch_size):
                    out = agent.train_step(replay, int(args.batch_size))
                    ep_info["actor"] += float(out.get("actor_loss", 0.0))
                    ep_info["critic"] += float(out.get("critic_loss", 0.0))
                    total_updates += 1
            ep_info["reward"] += float(rew.mean().item())
            ep_info["steps"] += 1
        # Log epoch averages
        denom = max(1, ep_info["steps"])
        print(json.dumps({
            "epoch": epoch + 1,
            "avg_actor": ep_info["actor"] / max(1, total_updates),
            "avg_critic": ep_info["critic"] / max(1, total_updates),
            "avg_reward": ep_info["reward"] / denom,
            "buffer_size": int(replay.size),
        }), flush=True)

        # Periodic checkpointing
        if int(args.checkpoint_every) > 0 and ((epoch + 1) % int(args.checkpoint_every) == 0):
            try:
                os.makedirs(os.path.dirname(args.save), exist_ok=True)
                base_dir = os.path.dirname(args.save)
                base_name = os.path.splitext(os.path.basename(args.save))[0]
                ext = os.path.splitext(args.save)[1] or ".pt"
                ckpt_path = os.path.join(base_dir, f"{base_name}_ep{epoch + 1:03d}{ext}")
                payload = {
                    "actor_state": agent.actor.state_dict(),
                    "critic_state": agent.critic.state_dict(),
                    "td3_cfg": asdict(cfg),
                    "ae_model": args.ae_model,
                    "latent_dim": int(latent_dim),
                    "action_dim": int(action_dim),
                    "max_units": float(args.max_units),
                    "instruments": instruments,
                }
                torch.save(payload, ckpt_path)
                torch.save(payload, args.save)  # also refresh latest
                print(json.dumps({"checkpoint_saved": ckpt_path, "latest": args.save}), flush=True)
            except Exception as _e:
                print(json.dumps({"checkpoint_error": str(_e)}), flush=True)

    # Save checkpoint
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        "actor_state": agent.actor.state_dict(),
        "critic_state": agent.critic.state_dict(),
        "td3_cfg": asdict(cfg),
        "ae_model": args.ae_model,
        "latent_dim": int(latent_dim),
        "action_dim": int(action_dim),
        "max_units": float(args.max_units),
        "instruments": instruments,
    }, args.save)
    print(json.dumps({"saved_td3": args.save}))

    # Full deterministic pass for positions
    with torch.no_grad():
        Zt = torch.tensor(s, dtype=torch.float32, device=device)
        actions = agent.select_action(Zt, noise_sigma=0.0).cpu().numpy().astype(np.float32)
        pos_units = actions * float(args.max_units)
    os.makedirs(os.path.dirname(args.out_positions), exist_ok=True)
    np.save(args.out_positions, pos_units)
    print(json.dumps({"saved_positions": args.out_positions, "rows": int(pos_units.shape[0])}))


if __name__ == "__main__":
    main()
