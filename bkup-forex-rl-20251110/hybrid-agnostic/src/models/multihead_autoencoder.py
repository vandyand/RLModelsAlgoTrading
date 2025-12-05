#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNormalizer:
    def __init__(self, momentum: float = 0.99, eps: float = 1e-8) -> None:
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.mean: float = 0.0
        self.var: float = 1.0
        self.initialized = False

    def update(self, x: torch.Tensor) -> None:
        m = float(x.detach().mean().item())
        try:
            v = float(x.detach().var(unbiased=False).item())
        except Exception:
            v = 0.0
        if not self.initialized:
            self.mean = m
            self.var = v if v > 0.0 else 1.0
            self.initialized = True
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * m
            self.var = self.momentum * self.var + (1.0 - self.momentum) * (v if v > 0.0 else self.var)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.update(x)
        mean_t = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std_t = torch.tensor(math.sqrt(max(self.var, 0.0)) + self.eps, dtype=x.dtype, device=x.device)
        return torch.tanh((x - mean_t) / std_t)


@dataclass
class MultiHeadAEConfig:
    input_dim: int
    latent_dim: int = 64
    head_count: int = 6
    hidden_dims: Optional[List[int]] = None  # encoder/decoder base widths
    dropout: float = 0.0
    noise_std: float = 0.0  # denoising AE input noise
    # Randomized head decoder options (backward-compatible: off by default)
    randomized_heads: bool = True
    head_arch_min_layers: int = 1
    head_arch_max_layers: int = 3
    head_hidden_min: int = 64
    head_hidden_max: int = 512
    head_activation_set: Optional[List[str]] = None  # e.g., ["relu","gelu","silu","elu","tanh"]
    head_random_seed: Optional[int] = None


class RandomHeadDecoder(nn.Module):
    """A decoder head with randomized per-head architecture.

    Backward-compatible: if randomized mode is disabled, falls back to the
    original 4-variant templates keyed by variant_id.
    """
    def __init__(
        self,
        cfg: MultiHeadAEConfig,
        output_dim: int,
        variant_id: int,
    ) -> None:
        super().__init__()

        if not bool(getattr(cfg, "randomized_heads", False)):
            # Legacy 4-template behavior for backward compatibility with old checkpoints
            v = variant_id % 4
            if v == 0:
                self.net = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 256), nn.ReLU(),
                    nn.Linear(256, output_dim),
                )
            elif v == 1:
                self.net = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 128), nn.ReLU(), nn.Dropout(cfg.dropout),
                    nn.Linear(128, 128), nn.ReLU(),
                    nn.Linear(128, output_dim),
                )
            elif v == 2:
                self.net = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 512), nn.ReLU(),
                    nn.Linear(512, 128), nn.ReLU(),
                    nn.Linear(128, output_dim),
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(cfg.latent_dim, 128), nn.GELU(), nn.Dropout(cfg.dropout),
                    nn.Linear(128, output_dim),
                )
            return

        # Randomized architecture path
        seed_val: int = int(cfg.head_random_seed) if cfg.head_random_seed is not None else 1337
        rnd = random.Random(seed_val + int(variant_id))

        min_layers = max(1, int(getattr(cfg, "head_arch_min_layers", 1)))
        max_layers = max(min_layers, int(getattr(cfg, "head_arch_max_layers", 3)))
        num_hidden_layers = rnd.randint(min_layers, max_layers)

        hmin = max(8, int(getattr(cfg, "head_hidden_min", 64)))
        hmax = max(hmin, int(getattr(cfg, "head_hidden_max", 512)))

        act_set = getattr(cfg, "head_activation_set", None) or ["relu", "gelu", "silu", "elu"]

        layers: List[nn.Module] = []
        in_dim = int(cfg.latent_dim)
        for li in range(num_hidden_layers):
            width = int(rnd.randint(hmin, hmax))
            layers.append(nn.Linear(in_dim, width))
            act_name = rnd.choice(act_set)
            if act_name == "relu":
                layers.append(nn.ReLU())
            elif act_name == "gelu":
                layers.append(nn.GELU())
            elif act_name == "silu":
                layers.append(nn.SiLU())
            elif act_name == "elu":
                layers.append(nn.ELU())
            elif act_name == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            # Optional dropout per layer with 50% chance
            if float(cfg.dropout) > 0.0 and rnd.random() < 0.5:
                layers.append(nn.Dropout(float(cfg.dropout)))
            in_dim = width

        # Final linear projection to output_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MultiHeadAutoencoder(nn.Module):
    def __init__(self, cfg: MultiHeadAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hd = cfg.hidden_dims or [1024, 256]
        enc: List[nn.Module] = []
        last = cfg.input_dim
        for h in hd:
            enc += [nn.Linear(last, h), nn.ReLU()]
            if cfg.dropout > 0.0:
                enc += [nn.Dropout(cfg.dropout)]
            last = h
        enc += [nn.Linear(last, cfg.latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        # A pool of reconstruction heads with per-head architecture variants
        self.decoders = nn.ModuleList([
            RandomHeadDecoder(cfg=cfg, output_dim=cfg.input_dim, variant_id=h)
            for h in range(cfg.head_count)
        ])

        # Auxiliary heads: enforce structure in latent
        self.use_decorrelation = True
        self.use_sparse = True
        self.use_contractive = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        xin = x
        if self.cfg.noise_std > 0.0 and self.training:
            xin = xin + self.cfg.noise_std * torch.randn_like(xin)
        z = self.encoder(xin)
        recons = [dec(z) for dec in self.decoders]
        return z, recons

    @staticmethod
    def loss(
        x: torch.Tensor,
        z: torch.Tensor,
        recons: List[torch.Tensor],
        weights: Dict[str, float],
        normalizers: Dict[str, RunningNormalizer],
        use_decorrelation: bool = True,
        use_sparse: bool = True,
        use_contractive: bool = False,
        encoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse_terms = [F.mse_loss(r, x) for r in recons]
        recon = sum(mse_terms) / max(1, len(mse_terms))

        # Latent decorrelation: off-diagonal covariance penalty
        decor = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        var_id = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if use_decorrelation:
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.T @ zc) / max(1, zc.size(0) - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            decor = (off_diag ** 2).mean()
            # Variance-to-identity penalty to prevent collapse (encourage diag ~ 1)
            diag = torch.diag(cov)
            var_id = ((diag - 1.0) ** 2).mean()

        # Sparsity: L1 on latent
        sparse = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if use_sparse:
            sparse = torch.mean(torch.abs(z))

        # Contractive penalty: ||Jacobian||^2 (approx via gradients); optional due to cost
        contr = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if use_contractive and encoder is not None:
            z_sum = z.sum()
            grads = torch.autograd.grad(z_sum, [p for p in encoder.parameters() if p.requires_grad], create_graph=True, retain_graph=True, allow_unused=True)
            sq = [g.pow(2).sum() for g in grads if g is not None]
            if len(sq) > 0:
                contr = sum(sq) / float(len(sq))

        # Normalize terms to similar scales then weight
        nr = normalizers["recon"].normalize(recon)
        nd = normalizers["decor"].normalize(decor)
        nv = normalizers["var"].normalize(var_id)
        ns = normalizers["sparse"].normalize(sparse)
        nc = normalizers["contr"].normalize(contr)

        loss = (
            weights.get("recon", 1.0) * nr
            + weights.get("decor", 0.1) * nd
            + weights.get("var", 0.0) * nv
            + weights.get("sparse", 0.05) * ns
            + weights.get("contr", 0.0) * nc
        )

        metrics = {
            "recon": float(recon.detach().item()),
            "decor": float(decor.detach().item()),
            "var": float(var_id.detach().item()),
            "sparse": float(sparse.detach().item()),
            "contr": float(contr.detach().item()),
            "norm_recon": float(nr.detach().item()),
            "norm_decor": float(nd.detach().item()),
            "norm_var": float(nv.detach().item()),
            "norm_sparse": float(ns.detach().item()),
            "norm_contr": float(nc.detach().item()),
            "total": float(loss.detach().item()),
        }
        return loss, metrics
