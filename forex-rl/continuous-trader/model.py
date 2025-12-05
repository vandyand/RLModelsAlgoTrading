from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Dict as TDict

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    ae_hidden: Tuple[int, ...] = (2048, 512, 128)
    ae_latent: int = 64
    policy_hidden: int = 256
    value_hidden: int = 256


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        enc_layers = []
        last = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        enc_layers += [nn.Linear(last, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers += [nn.Linear(last, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class ActorCriticSingle(nn.Module):
    """Single-output actor-critic: actor outputs a scalar in [0,1).

    We'll use tanh+affine to map to (0,1): a = sigmoid(logit).
    """
    def __init__(self, encoder: nn.Module, latent_dim: int, policy_hidden: int, value_hidden: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, 1),  # scalar logit
        )
        self.value = nn.Sequential(
            nn.Linear(latent_dim, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.training):
            z = self.encoder(x)
        logit = self.policy(z)
        v = self.value(z).squeeze(-1)
        # actor output in [0,1)
        a = torch.sigmoid(logit).squeeze(-1)
        return a, logit.squeeze(-1), v


class PerInstrumentTrunk(nn.Module):
    """Small MLP trunk applied Siamese-style to each instrument's feature vector for a given granularity."""

    def __init__(self, input_dim: int, embed_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_instruments, D_inst)
        return self.net(x)  # (N_instruments, embed_dim)


class SiameseMultiGranActorCritic(nn.Module):
    """Siamese per-instrument trunks per granularity + gran-branch aggregation, then actor/value heads.

    Input: flat feature vector (batch, D_total). We use a mapping to slice per (gran, instrument).
    For each granularity g:
      - Slice features per instrument using indices mapping
      - Apply shared per-instrument trunk_g to each instrument -> (N_inst, E)
      - Aggregate across instruments (mean) -> (E)
    Concatenate gran latents -> (E_total)
    Actor/Value MLPs operate on this concatenated latent.
    """

    def __init__(
        self,
        flat_input_dim: int,
        indices_by_gran_inst: TDict[str, TDict[str, List[int]]],
        embed_dim: int = 64,
        hidden_per_inst: int = 256,
        policy_hidden: int = 512,
        value_hidden: int = 512,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.indices_by_gran_inst = indices_by_gran_inst
        # Build per-gran Siamese trunks (shared across instruments within a gran)
        self.trunks: nn.ModuleDict = nn.ModuleDict()
        self.attn: nn.ModuleDict = nn.ModuleDict()
        self.use_attention = use_attention
        # Infer per-instrument input dims per gran from the first instrument mapping
        for gran, by_inst in indices_by_gran_inst.items():
            # Find first non-empty instrument
            inst0 = next((k for k, v in by_inst.items() if len(v) > 0), None)
            if inst0 is None:
                continue
            d_inst = len(by_inst[inst0])
            self.trunks[gran] = PerInstrumentTrunk(input_dim=d_inst, embed_dim=embed_dim, hidden=hidden_per_inst)
            if use_attention:
                self.attn[gran] = nn.Linear(embed_dim, 1)

        total_embed = embed_dim * max(1, len(self.trunks))
        self.policy = nn.Sequential(
            nn.LayerNorm(total_embed),
            nn.Linear(total_embed, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, 1),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(total_embed),
            nn.Linear(total_embed, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_flat: (batch, D_total). We operate per-sample due to different slicing per gran/inst.
        batch = x_flat.size(0)
        latents: List[torch.Tensor] = []
        for gran, trunk in self.trunks.items():
            by_inst = self.indices_by_gran_inst.get(gran, {})
            # collect per-instrument tensors
            inst_vecs: List[torch.Tensor] = []
            for inst, idxs in by_inst.items():
                if not idxs:
                    continue
                xi = x_flat[:, idxs]  # (batch, D_inst)
                # Siamese expects (N_inst, D) but we have batch. We process per batch then mean across instruments
                inst_vecs.append(xi)
            if not inst_vecs:
                # No features for this gran, use zeros
                latents.append(torch.zeros((batch, trunk.net[-2].out_features), device=x_flat.device))
                continue
            # Stack instruments along dim=1: (batch, N_inst, D_inst)
            Xg = torch.stack(inst_vecs, dim=1)
            # Flatten instruments for trunk: (batch*N_inst, D_inst)
            B, N, D = Xg.shape
            Xg2 = Xg.reshape(B * N, D)
            Z = trunk(Xg2)  # (B*N, E)
            Z = Z.reshape(B, N, -1)
            if self.use_attention and gran in self.attn:
                # Attention weights over instruments
                scores = self.attn[gran](Z)  # (B, N, 1)
                weights = torch.softmax(scores, dim=1)
                Zg = (weights * Z).sum(dim=1)  # (B, E)
            else:
                # Aggregate across instruments (mean)
                Zg = Z.mean(dim=1)  # (batch, E)
            latents.append(Zg)

        if len(latents) == 0:
            # Fallback empty
            H = torch.zeros((batch, 1), device=x_flat.device)
        else:
            H = torch.cat(latents, dim=1)

        logit = self.policy(H)
        v = self.value(H).squeeze(-1)
        a = torch.sigmoid(logit).squeeze(-1)
        return a, logit.squeeze(-1), v
