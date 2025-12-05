#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .multihead_autoencoder import MultiHeadAEConfig, MultiHeadAutoencoder
from .lstm_ddpg import ActorRecurrent, CriticRecurrent


@dataclass
class HierarchicalAEConfig:
    num_shards: int
    shard_input_dims: Dict[int, int]
    shard_latent_dim: int = 64
    final_latent_dim: int = 64
    head_count: int = 6
    shard_hidden: Optional[List[int]] = None
    final_hidden: Optional[List[int]] = None
    dropout: float = 0.0
    noise_std: float = 0.0


class HierarchicalAutoencoders(nn.Module):
    """Sharded multi-head AEs + final multi-head AE.

    - Each shard AE reconstructs its own feature subset using multiple random decoder heads.
    - The final AE reconstructs the concatenated shard latents using multiple random decoder heads.
    - Reconstruction losses are pure MSE averaged across heads, as requested.
    """

    def __init__(self, cfg: HierarchicalAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_shards = int(cfg.num_shards)
        self.shard_input_dims = dict(cfg.shard_input_dims)
        # Build shard AEs
        self.shard_models = nn.ModuleList()
        for sid in range(self.num_shards):
            in_dim = int(self.shard_input_dims.get(sid, 0))
            if in_dim <= 0:
                # Placeholder for empty shards to keep indexing stable
                self.shard_models.append(nn.Identity())
                continue
            mcfg = MultiHeadAEConfig(
                input_dim=in_dim,
                latent_dim=int(cfg.shard_latent_dim),
                head_count=int(cfg.head_count),
                hidden_dims=cfg.shard_hidden or [min(2048, max(256, in_dim // 2)), 256],
                dropout=float(cfg.dropout),
                noise_std=float(cfg.noise_std),
                randomized_heads=bool(getattr(cfg, "randomized_heads", False)),
                head_arch_min_layers=int(getattr(cfg, "head_arch_min_layers", 1)),
                head_arch_max_layers=int(getattr(cfg, "head_arch_max_layers", 3)),
                head_hidden_min=int(getattr(cfg, "head_hidden_min", 64)),
                head_hidden_max=int(getattr(cfg, "head_hidden_max", 512)),
                head_activation_set=list(getattr(cfg, "head_activation_set", ["relu","gelu","silu","elu"])),
                head_random_seed=(getattr(cfg, "head_random_seed", None)),
            )
            self.shard_models.append(MultiHeadAutoencoder(mcfg))
        # Final AE over concatenated shard latents
        final_in = self.num_shards * int(cfg.shard_latent_dim)
        fcfg = MultiHeadAEConfig(
            input_dim=final_in,
            latent_dim=int(cfg.final_latent_dim),
            head_count=int(cfg.head_count),
            hidden_dims=cfg.final_hidden or [512, 128],
            dropout=float(cfg.dropout),
            noise_std=float(cfg.noise_std),
            randomized_heads=bool(getattr(cfg, "randomized_heads", False)),
            head_arch_min_layers=int(getattr(cfg, "head_arch_min_layers", 1)),
            head_arch_max_layers=int(getattr(cfg, "head_arch_max_layers", 3)),
            head_hidden_min=int(getattr(cfg, "head_hidden_min", 64)),
            head_hidden_max=int(getattr(cfg, "head_hidden_max", 512)),
            head_activation_set=list(getattr(cfg, "head_activation_set", ["relu","gelu","silu","elu"])),
            head_random_seed=(getattr(cfg, "head_random_seed", None)),
        )
        self.final_model = MultiHeadAutoencoder(fcfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x_flat: torch.Tensor,
        shard_slices: List[Optional[Sequence[int]]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode features through shards then final AE.

        Args:
            x_flat: [N, D] standardized features for one or more time steps
            shard_slices: list of column index sequences per shard mapping into x_flat
        Returns:
            z_final: [N, final_latent_dim]
            aux: dict with per-stage reconstruction MSE terms for logging
        """
        z_blocks: List[torch.Tensor] = []
        losses: Dict[str, torch.Tensor] = {}
        for sid, (model, cols) in enumerate(zip(self.shard_models, shard_slices)):
            if isinstance(model, nn.Identity) or cols is None or len(cols) == 0:
                # No features in this shard; feed zeros latent
                z_blocks.append(torch.zeros((x_flat.size(0), int(self.cfg.shard_latent_dim)), dtype=x_flat.dtype, device=x_flat.device))
                continue
            x_s = x_flat[:, cols]
            z_s, recons = model(x_s)
            # Reconstruction MSE across heads
            mse_terms = [torch.nn.functional.mse_loss(r, x_s) for r in recons]
            mse = sum(mse_terms) / max(1, len(mse_terms))
            # Mild regularizers (optional): latent decorrelation and sparsity
            with torch.no_grad():
                pass
            # Re-encode shard input to get latent for regularization
            # (model.forward already produced z_s)
            zc = z_s - z_s.mean(dim=0, keepdim=True)
            cov = (zc.T @ zc) / max(1, zc.size(0) - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            decor = (off_diag ** 2).mean()
            diag = torch.diag(cov)
            var_id = ((diag - 1.0) ** 2).mean()
            sparse = torch.mean(torch.abs(z_s))
            losses[f"shard_{sid:02d}_mse"] = mse
            losses[f"shard_{sid:02d}_decor"] = decor
            losses[f"shard_{sid:02d}_var"] = var_id
            losses[f"shard_{sid:02d}_sparse"] = sparse
            z_blocks.append(z_s)
        z_concat = torch.cat(z_blocks, dim=-1)
        z_final, recons_final = self.final_model(z_concat)
        # Final AE reconstruction in latent space
        mse_final_terms = [torch.nn.functional.mse_loss(r, z_concat) for r in recons_final]
        final_mse = sum(mse_final_terms) / max(1, len(mse_final_terms))
        # Regularizers on final latent
        zf = z_final
        zfc = zf - zf.mean(dim=0, keepdim=True)
        covf = (zfc.T @ zfc) / max(1, zfc.size(0) - 1)
        off_diag_f = covf - torch.diag(torch.diag(covf))
        decor_f = (off_diag_f ** 2).mean()
        diagf = torch.diag(covf)
        var_id_f = ((diagf - 1.0) ** 2).mean()
        sparse_f = torch.mean(torch.abs(zf))
        losses["final_mse"] = final_mse
        losses["final_decor"] = decor_f
        losses["final_var"] = var_id_f
        losses["final_sparse"] = sparse_f
        return z_final, losses

    def recon_mse_total(
        self,
        losses: Dict[str, torch.Tensor],
        weight_shard: float = 1.0,
        weight_final: float = 1.0,
        weight_decor: float = 0.1,
        weight_var: float = 0.05,
        weight_sparse: float = 0.02,
    ) -> torch.Tensor:
        total = torch.zeros((), dtype=next(self.parameters()).dtype, device=self.device)
        for k, v in losses.items():
            if k.startswith("shard_") and (k.endswith("_mse") or k.endswith("_decor") or k.endswith("_var") or k.endswith("_sparse")):
                if k.endswith("_mse"):
                    total = total + float(weight_shard) * v
                elif k.endswith("_decor"):
                    total = total + float(weight_decor) * v
                elif k.endswith("_var"):
                    total = total + float(weight_var) * v
                elif k.endswith("_sparse"):
                    total = total + float(weight_sparse) * v
            elif k.startswith("final_"):
                if k == "final_mse":
                    total = total + float(weight_final) * v
                elif k == "final_decor":
                    total = total + float(weight_decor) * v
                elif k == "final_var":
                    total = total + float(weight_var) * v
                elif k == "final_sparse":
                    total = total + float(weight_sparse) * v
        return total

    def export_config(self) -> dict:
        return {
            "ae": {
                "num_shards": int(self.cfg.num_shards),
                "shard_latent_dim": int(self.cfg.shard_latent_dim),
                "final_latent_dim": int(self.cfg.final_latent_dim),
                "head_count": int(self.cfg.head_count),
                "shard_hidden": list(self.cfg.shard_hidden or []),
                "final_hidden": list(self.cfg.final_hidden or []),
                "dropout": float(self.cfg.dropout),
                "noise_std": float(self.cfg.noise_std),
                "shard_input_dims": {int(k): int(v) for k, v in self.shard_input_dims.items()},
            }
        }


@dataclass
class JointRLConfig:
    # AE config
    ae: HierarchicalAEConfig
    # RL config
    action_dim: int
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
    q_clip: float = 10.0
    detach_state_for_rl: bool = True
    factorized_action: bool = True


class JointHierarchicalRL(nn.Module):
    """One big model that includes hierarchical AEs and GRU-DDPG actor/critic.

    - AE reconstruction uses pure MSE averaged across random decoder heads.
    - RL is trained concurrently; by default, AE latents are detached when fed into RL
      so AE gradients are driven by reconstruction only, as requested.
    """

    def __init__(self, cfg: JointRLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hier = HierarchicalAutoencoders(cfg.ae)
        state_dim = int(cfg.ae.final_latent_dim)
        self.actor = ActorRecurrent(state_dim, int(cfg.action_dim), int(cfg.lstm_hidden), int(cfg.actor_hidden), bool(cfg.use_gru), bool(cfg.factorized_action))
        self.critic = CriticRecurrent(state_dim, int(cfg.action_dim), int(cfg.lstm_hidden), int(cfg.critic_hidden), bool(cfg.use_gru))
        self.target_actor = ActorRecurrent(state_dim, int(cfg.action_dim), int(cfg.lstm_hidden), int(cfg.actor_hidden), bool(cfg.use_gru), bool(cfg.factorized_action))
        self.target_critic = CriticRecurrent(state_dim, int(cfg.action_dim), int(cfg.lstm_hidden), int(cfg.critic_hidden), bool(cfg.use_gru))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_final_sequence(
        self,
        x_seq: torch.Tensor,
        shard_slices: List[Optional[Sequence[int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of sequences of standardized features into final latents.

        Args:
            x_seq: [B, T, D]
            shard_slices: mapping into feature columns per shard
        Returns:
            z_seq: [B, T, final_latent_dim]
            recon_loss: scalar total AE MSE loss across shards + final on all frames
        """
        B, T, D = x_seq.shape
        x_flat = x_seq.reshape(B * T, D)
        z_flat, losses = self.hier(x_flat, shard_slices)
        recon_loss = self.hier.recon_mse_total(losses)
        z_seq = z_flat.reshape(B, T, -1)
        return z_seq, recon_loss

    @torch.no_grad()
    def act(self, z_seq: torch.Tensor, noise_sigma: Optional[float] = None) -> torch.Tensor:
        a = self.actor(z_seq)
        if noise_sigma is None:
            noise_sigma = float(self.cfg.noise_sigma)
        if noise_sigma > 0.0 and self.training:
            a = torch.clamp(a + noise_sigma * torch.randn_like(a), -1.0, 1.0)
        return a

    def _soft_update(self, src: nn.Module, dst: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p_t, p in zip(dst.parameters(), src.parameters()):
                p_t.data.mul_(1 - tau).add_(tau * p.data)

    def critic_target(self, z_next: torch.Tensor, rewards: torch.Tensor, gamma: float, q_clip: Optional[float]) -> torch.Tensor:
        with torch.no_grad():
            a2 = self.target_actor(z_next)
            q2 = self.target_critic(z_next, a2)
            y = rewards + gamma * q2
            if q_clip is not None and q_clip > 0:
                y = torch.clamp(y, -q_clip, q_clip)
        return y

    def rl_losses(
        self,
        z_seq: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        z_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(self.cfg.detach_state_for_rl):
            z_seq = z_seq.detach()
            z_next = z_next.detach()
        y = self.critic_target(z_next, rewards, float(self.cfg.gamma), float(self.cfg.q_clip))
        q = self.critic(z_seq, actions)
        critic_loss = torch.nn.functional.smooth_l1_loss(q, y)
        a_pred = self.actor(z_seq)
        actor_loss = -self.critic(z_seq, a_pred).mean()
        return actor_loss, critic_loss

    def export_checkpoint(self) -> dict:
        return {
            "cfg": {
                "rl": {
                    "action_dim": int(self.cfg.action_dim),
                    "lstm_hidden": int(self.cfg.lstm_hidden),
                    "actor_hidden": int(self.cfg.actor_hidden),
                    "critic_hidden": int(self.cfg.critic_hidden),
                    "seq_len": int(self.cfg.seq_len),
                    "gamma": float(self.cfg.gamma),
                    "tau": float(self.cfg.tau),
                    "actor_lr": float(self.cfg.actor_lr),
                    "critic_lr": float(self.cfg.critic_lr),
                    "noise_sigma": float(self.cfg.noise_sigma),
                    "max_units": float(self.cfg.max_units),
                    "use_gru": bool(self.cfg.use_gru),
                    "q_clip": float(self.cfg.q_clip),
                    "detach_state_for_rl": bool(self.cfg.detach_state_for_rl),
                },
                **self.hier.export_config(),
            },
            "hier": {
                "state": self.hier.state_dict(),
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
