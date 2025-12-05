#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DDPGConfig:
    state_dim: int
    action_dim: int = 20
    lstm_hidden: int = 64
    actor_hidden: int = 128
    critic_hidden: int = 128
    seq_len: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    noise_sigma: float = 0.2
    max_units: float = 100.0
    use_gru: bool = True
    num_threads: int = 1
    q_clip: float = 10.0
    # New: factorize action into direction [-1,1] and magnitude [0,1]
    factorized_action: bool = True


class ActorRecurrent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, actor_hidden: int, use_gru: bool, factorized_action: bool = False) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.factorized_action = bool(factorized_action)
        self.rnn = nn.GRU(state_dim, hidden_size, batch_first=True) if use_gru else nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.base = nn.Sequential(
            nn.Linear(hidden_size, actor_hidden), nn.ReLU(),
        )
        if self.factorized_action:
            self.dir_head = nn.Linear(actor_hidden, action_dim)
            self.mag_head = nn.Linear(actor_hidden, action_dim)
        else:
            self.out_head = nn.Sequential(
                nn.Linear(actor_hidden, action_dim), nn.Tanh(),
            )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, state_dim]
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        h = self.base(last)
        if self.factorized_action:
            direction = torch.tanh(self.dir_head(h))
            magnitude = torch.sigmoid(self.mag_head(h))
            a = direction * magnitude
            return a
        else:
            a = self.out_head(h)
            return a

    def forward_components(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (direction, magnitude) components; if not factorized, emulate with (a, |a|)."""
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        h = self.base(last)
        if self.factorized_action:
            direction = torch.tanh(self.dir_head(h))
            magnitude = torch.sigmoid(self.mag_head(h))
        else:
            a = self.out_head(h)
            direction = a
            magnitude = torch.clamp(torch.abs(a), 0.0, 1.0)
        return direction, magnitude


class CriticRecurrent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, critic_hidden: int, use_gru: bool) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.rnn = nn.GRU(state_dim, hidden_size, batch_first=True) if use_gru else nn.LSTM(state_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + action_dim, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(self, x_seq: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        q = self.head(torch.cat([last, a], dim=-1))
        return q.squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seq_len: int) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.state = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, seq_len, state_dim), dtype=np.float32)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray) -> None:
        i = self.ptr
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = s2
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or (self.ptr == 0)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_idx = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)
        return self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx]


class LSTMDDPG:
    def __init__(self, cfg: DDPGConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(max(1, int(cfg.num_threads)))
        self.actor = ActorRecurrent(cfg.state_dim, cfg.action_dim, cfg.lstm_hidden, cfg.actor_hidden, cfg.use_gru, cfg.factorized_action).to(self.device)
        self.critic = CriticRecurrent(cfg.state_dim, cfg.action_dim, cfg.lstm_hidden, cfg.critic_hidden, cfg.use_gru).to(self.device)
        self.target_actor = ActorRecurrent(cfg.state_dim, cfg.action_dim, cfg.lstm_hidden, cfg.actor_hidden, cfg.use_gru).to(self.device)
        self.target_critic = CriticRecurrent(cfg.state_dim, cfg.action_dim, cfg.lstm_hidden, cfg.critic_hidden, cfg.use_gru).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, state_seq: np.ndarray, noise: bool = True) -> np.ndarray:
        x = torch.tensor(state_seq[None, ...], dtype=torch.float32, device=self.device)
        a = self.actor(x)[0].cpu().numpy()
        if noise and self.cfg.noise_sigma > 0.0:
            a = np.clip(a + np.random.normal(0.0, self.cfg.noise_sigma, size=a.shape), -1.0, 1.0)
        return a

    @torch.no_grad()
    def act_components(self, state_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.tensor(state_seq[None, ...], dtype=torch.float32, device=self.device)
        d, m = self.actor.forward_components(x)
        return d[0].cpu().numpy(), m[0].cpu().numpy()

    def _soft_update(self, src: nn.Module, dst: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p in zip(dst.parameters(), src.parameters()):
                p_t.data.mul_(1 - tau).add_(tau * p.data)

    def train_step(self, buffer: ReplayBuffer, batch_size: int) -> dict:
        s, a, r, s2 = buffer.sample(batch_size)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2 = self.target_actor(s2_t)
            q2 = self.target_critic(s2_t, a2)
            y = r_t + self.cfg.gamma * q2
            # Clip TD targets to stabilize training
            if self.cfg.q_clip is not None and self.cfg.q_clip > 0:
                y = torch.clamp(y, -self.cfg.q_clip, self.cfg.q_clip)

        # Critic update
        q = self.critic(s_t, a_t)
        critic_loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        # Actor update (deterministic policy gradient)
        a_pred = self.actor(s_t)
        actor_loss = -self.critic(s_t, a_pred).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # Targets
        self._soft_update(self.actor, self.target_actor, self.cfg.tau)
        self._soft_update(self.critic, self.target_critic, self.cfg.tau)

        return {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
        }
