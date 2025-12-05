#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_mlp(sizes: Tuple[int, ...], activation: nn.Module = nn.ReLU(), output_activation: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else (output_activation if output_activation is not None else nn.Identity())
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.net = make_mlp((state_dim, hidden[0], hidden[1], action_dim), activation=nn.ReLU(), output_activation=nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        # Q1
        self.q1 = make_mlp((state_dim + action_dim, hidden[0], hidden[1], 1), activation=nn.ReLU())
        # Q2
        self.q2 = make_mlp((state_dim + action_dim, hidden[0], hidden[1], 1), activation=nn.ReLU())

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu), self.q2(xu)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([state, action], dim=-1)
        return self.q1(xu)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.not_done = torch.ones((capacity, 1), dtype=torch.float32, device=device)

    @torch.no_grad()
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> None:
        n = state.shape[0]
        idx = torch.arange(n, device=self.device) + self.ptr
        idx = idx % self.capacity
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward.view(-1, 1)
        self.next_state[idx] = next_state
        self.not_done[idx] = 1.0 - done.view(-1, 1)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.not_done[idx]


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    actor_hidden: Tuple[int, int] = (256, 256)
    critic_hidden: Tuple[int, int] = (256, 256)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    tau: float = 0.005
    gamma: float = 0.99
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_action: float = 1.0


class TD3Agent:
    def __init__(self, cfg: TD3Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.actor_hidden).to(device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim, cfg.actor_hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.critic_hidden).to(device)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim, cfg.critic_hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, noise_sigma: float = 0.0) -> torch.Tensor:
        action = self.actor(state)
        if noise_sigma > 0.0:
            action = action + noise_sigma * torch.randn_like(action)
        return torch.clamp(action, -self.cfg.max_action, self.cfg.max_action)

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> Dict[str, Any]:
        self.total_it += 1
        state, action, reward, next_state, not_done = replay.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.cfg.max_action, self.cfg.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        actor_loss_val = torch.tensor(0.0, device=self.device)
        if self.total_it % self.cfg.policy_delay == 0:
            # Actor loss: maximize Q, i.e., minimize -Q
            actor_action = self.actor(state)
            actor_loss = -self.critic.q1_forward(state, actor_action).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.detach()

            # Update targets
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - self.cfg.tau)
                    target_param.data.add_(self.cfg.tau * param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1 - self.cfg.tau)
                    target_param.data.add_(self.cfg.tau * param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss_val.item()),
            "step": int(self.total_it),
        }


@torch.no_grad()
def sharpe_like_reward(actions: torch.Tensor, returns_vec: torch.Tensor, max_units: float, eps: float = 1e-8) -> torch.Tensor:
    """Compute Sharpe-like reward across instruments for a batch.

    actions: [B, N] in [-1,1]
    returns_vec: [B, N] daily returns for next day
    """
    pos_units = actions * float(max_units)
    contrib = pos_units * returns_vec
    mean_r = torch.mean(contrib, dim=1)
    std_r = torch.std(contrib, dim=1)
    sharpe = mean_r / (std_r + eps)
    # Clamp to avoid exploding targets
    return torch.clamp(sharpe, -10.0, 10.0).view(-1, 1)
