#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List

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

from models.multihead_autoencoder import MultiHeadAEConfig, MultiHeadAutoencoder, RunningNormalizer  # type: ignore


def _assemble_shard_matrix_from_parts(features_dir: str, dates_csv: str, cols: List[str]) -> pd.DataFrame:
    """Build shard matrix by loading only needed columns from their source files.
    Avoids scanning every part file and avoids reindex/astype to reduce CPU/memory.
    """
    dates = pd.read_csv(dates_csv, header=None)[0].astype(str).tolist()
    base_index = pd.to_datetime(pd.Series(dates))
    Xs = pd.DataFrame(index=base_index)

    # Group requested columns by their source file
    groups: Dict[str, List[str]] = {}
    for c in cols:
        if c.startswith("FX_"):
            parts = c.split("_")
            if len(parts) >= 4:  # FX, AAA, BBB, ...
                inst = parts[1] + "_" + parts[2]
                key = os.path.join(features_dir, "FX", f"{inst}.csv")
            else:
                # Fallback: scan FX dir (should be rare)
                key = os.path.join(features_dir, "FX", "__ALL__")
        elif c.startswith("ETF_"):
            parts = c.split("_")
            if len(parts) >= 2:
                tkr = parts[1]
                key = os.path.join(features_dir, "ETF", f"{tkr}.csv")
            else:
                key = os.path.join(features_dir, "ETF", "__ALL__")
        else:
            key = os.path.join(features_dir, "time_features.csv")
        groups.setdefault(key, []).append(c)

    for path, want_cols in groups.items():
        if os.path.basename(path) == "__ALL__":
            # Fallback slow path: scan dir
            parent = os.path.dirname(path)
            if os.path.isdir(parent):
                for fname in os.listdir(parent):
                    if not fname.endswith('.csv'):
                        continue
                    p = os.path.join(parent, fname)
                    # Check header quickly
                    hdr = pd.read_csv(p, nrows=0)
                    take = [c for c in want_cols if c in hdr.columns]
                    if not take:
                        continue
                    dfp = pd.read_csv(p, usecols=[hdr.columns[0], *take], index_col=0)
                    # Parts were saved already aligned to base_index, so skip reindex
                    Xs.loc[:, take] = dfp[take]
            continue

        if not os.path.exists(path):
            # Missing part (e.g., time features not requested)
            continue
        try:
            # Read only required columns (plus index column)
            hdr = pd.read_csv(path, nrows=0)
            take = [c for c in want_cols if c in hdr.columns]
            if not take:
                continue
            dfp = pd.read_csv(path, usecols=[hdr.columns[0], *take], index_col=0)
            # Parts are already aligned to base_index; avoid reindex for speed
            Xs.loc[:, take] = dfp[take]
        except Exception:
            # Fallback: full read
            dfp = pd.read_csv(path, index_col=0)
            take = [c for c in want_cols if c in dfp.columns]
            if not take:
                continue
            Xs.loc[:, take] = dfp[take]

    # Ensure float32 later when converting to tensor
    Xs = Xs.fillna(0.0)
    return Xs


def load_shard_latents_from_csv(features_csv: str, shards_dir: str, plan_path: str, stats_path: str) -> pd.DataFrame:
    """Encode features through each shard AE and concatenate z's per day.
    Returns a DataFrame with index aligned to features rows and columns [z_s00_*, z_s01_*, ...].
    """
    # Load standardized features using saved stats
    X = pd.read_csv(features_csv, index_col=0)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_obj = json.load(f)
    stats = {k: (float(v[0]), float(v[1])) for k, v in stats_obj.items()}
    # Apply
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        X[c] = (X[c] - m) / s
    X = X.astype(np.float32)

    # Load each shard, encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zs: List[pd.DataFrame] = []

    # Enumerate shard files by order
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.startswith("shard_") and f.endswith(".pt")])
    for sid, fname in enumerate(shard_files):
        ck = torch.load(os.path.join(shards_dir, fname), map_location=device)
        cols: List[str] = ck.get("columns", [])
        input_dim: int = int(ck.get("input_dim", len(cols)))
        cfgd = ck.get("cfg", {})
        latent_dim = int(cfgd.get("latent_dim", 64))
        head_count = int(cfgd.get("head_count", 6))
        dropout = float(cfgd.get("dropout", 0.0))
        noise_std = float(cfgd.get("noise_std", 0.0))
        hidden_dims = cfgd.get("hidden_dims", [min(2048, max(256, input_dim // 2)), 256])

        model = MultiHeadAutoencoder(MultiHeadAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            head_count=head_count,
            dropout=dropout,
            noise_std=noise_std,
            hidden_dims=hidden_dims,
        )).to(device)
        model.load_state_dict(ck["model_state"])  # type: ignore[index]
        model.eval()

        Xs = X[cols].values
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32, device=device)
            z, _ = model(xb)
            z_np = z.cpu().numpy().astype(np.float32)
        cols_out = [f"z_s{sid:02d}_{i:02d}" for i in range(z_np.shape[1])]
        zs.append(pd.DataFrame(z_np, index=X.index, columns=cols_out))

    return pd.concat(zs, axis=1)


def load_shard_latents_from_parts(features_dir: str, dates_csv: str, shards_dir: str, plan_path: str, stats_path: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    feature_to_shard = {str(k): int(v) for k, v in plan.get("feature_to_shard", {}).items()}
    # Invert to shard -> columns
    shard_to_cols: Dict[int, List[str]] = {}
    for feat, sid in feature_to_shard.items():
        shard_to_cols.setdefault(int(sid), []).append(feat)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_all = json.load(f)

    zs: List[pd.DataFrame] = []
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.startswith("shard_") and f.endswith(".pt")])
    for sid, fname in enumerate(shard_files):
        cols = shard_to_cols.get(sid, [])
        if not cols:
            continue
        try:
            print(json.dumps({"status": "assemble_shard", "sid": sid, "cols": len(cols)}), flush=True)
        except Exception:
            pass
        # Assemble shard matrix
        Xs = _assemble_shard_matrix_from_parts(features_dir, dates_csv, cols)
        # Standardize shard columns
        for c in cols:
            m, s = stats_all.get(c, [0.0, 1.0])
            if s == 0.0:
                s = 1.0
            Xs[c] = (Xs[c] - float(m)) / float(s)
        Xs = Xs.astype(np.float32)

        ck = torch.load(os.path.join(shards_dir, fname), map_location=device)
        input_dim: int = int(ck.get("input_dim", len(cols)))
        cfgd = ck.get("cfg", {})
        latent_dim = int(cfgd.get("latent_dim", 64))
        head_count = int(cfgd.get("head_count", 6))
        dropout = float(cfgd.get("dropout", 0.0))
        noise_std = float(cfgd.get("noise_std", 0.0))
        hidden_dims = cfgd.get("hidden_dims", [min(2048, max(256, input_dim // 2)), 256])

        model = MultiHeadAutoencoder(MultiHeadAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            head_count=head_count,
            dropout=dropout,
            noise_std=noise_std,
            hidden_dims=hidden_dims,
        )).to(device)
        model.load_state_dict(ck["model_state"])  # type: ignore[index]
        model.eval()

        try:
            print(json.dumps({"status": "encode_shard", "sid": sid, "rows": int(Xs.shape[0]), "input_dim": int(input_dim)}), flush=True)
        except Exception:
            pass
        with torch.no_grad():
            xb = torch.tensor(Xs.values, dtype=torch.float32, device=device)
            z, _ = model(xb)
            z_np = z.cpu().numpy().astype(np.float32)
        cols_out = [f"z_s{sid:02d}_{i:02d}" for i in range(z_np.shape[1])]
        zs.append(pd.DataFrame(z_np, index=Xs.index, columns=cols_out))
    return pd.concat(zs, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train final multi-head AE on concatenated shard latents")
    p.add_argument("--features")
    p.add_argument("--features-dir", help="Directory of per-file features (FX/ ETF/ time_features.csv)")
    p.add_argument("--dates-csv", default="forex-rl/hybrid-agnostic/data/dates.csv")
    p.add_argument("--shards-dir", default="forex-rl/hybrid-agnostic/checkpoints/shards")
    p.add_argument("--plan-path", default="forex-rl/hybrid-agnostic/artifacts/shard_plan.json")
    p.add_argument("--stats-path", default="forex-rl/hybrid-agnostic/artifacts/feature_stats.json")
    p.add_argument("--out-model", default="forex-rl/hybrid-agnostic/checkpoints/final_ae.pt")
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head-count", type=int, default=6)
    # Randomized head architecture options
    p.add_argument("--randomized-heads", action="store_true", default=True, help="Enable per-head randomized architectures")
    p.add_argument("--head-arch-min-layers", type=int, default=1)
    p.add_argument("--head-arch-max-layers", type=int, default=3)
    p.add_argument("--head-hidden-min", type=int, default=64)
    p.add_argument("--head-hidden-max", type=int, default=512)
    p.add_argument("--head-activations", default="relu,gelu,silu,elu", help="Comma-separated activation set for random heads")
    p.add_argument("--head-random-seed", type=int)
    args = p.parse_args()

    if args.features:
        Z = load_shard_latents_from_csv(args.features, args.shards_dir, args.plan_path, args.stats_path)
    else:
        if not args.features_dir:
            raise SystemExit("Provide --features (CSV) or --features-dir (per-file)")
        Z = load_shard_latents_from_parts(args.features_dir, args.dates_csv, args.shards_dir, args.plan_path, args.stats_path)

    # Final AE on Z
    input_dim = Z.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MultiHeadAEConfig(
        input_dim=input_dim,
        latent_dim=int(args.latent),
        head_count=int(args.head_count),
        hidden_dims=[512, 128],
        dropout=0.0,
        noise_std=0.0,
        randomized_heads=bool(args.randomized_heads),
        head_arch_min_layers=int(args.head_arch_min_layers),
        head_arch_max_layers=int(args.head_arch_max_layers),
        head_hidden_min=int(args.head_hidden_min),
        head_hidden_max=int(args.head_hidden_max),
        head_activation_set=[s.strip() for s in str(args.head_activations).split(',') if s.strip()],
        head_random_seed=(int(args.head_random_seed) if args.head_random_seed is not None else None),
    )
    model = MultiHeadAutoencoder(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr))

    dataset = torch.utils.data.TensorDataset(torch.tensor(Z.values, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)

    norms = {k: RunningNormalizer(0.99) for k in ["recon", "decor", "var", "sparse", "contr"]}
    weights = {"recon": 1.0, "decor": 0.1, "var": 0.05, "sparse": 0.02, "contr": 0.0}

    for epoch in range(int(args.epochs)):
        model.train()
        total = {"loss": 0.0, "recon": 0.0, "decor": 0.0, "var": 0.0, "sparse": 0.0, "contr": 0.0}
        n = 0
        for (xb,) in loader:
            xb = xb.to(device)
            z, recons = model(xb)
            loss, metrics = MultiHeadAutoencoder.loss(
                xb, z, recons, weights=weights, normalizers=norms,
                use_decorrelation=True, use_sparse=True, use_contractive=False, encoder=model.encoder
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = xb.size(0)
            total["loss"] += float(loss.item()) * bs
            total["recon"] += metrics["recon"] * bs
            total["decor"] += metrics["decor"] * bs
            total["sparse"] += metrics["sparse"] * bs
            total["var"] += metrics.get("var", 0.0) * bs
            total["contr"] += metrics["contr"] * bs
            n += bs
        log = {k: (v / max(1, n)) for k, v in total.items()}
        print(json.dumps({"epoch": epoch + 1, **log}), flush=True)

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "cfg": asdict(cfg),
        "input_dim": int(input_dim),
        "cols": list(Z.columns),
    }, args.out_model)
    print(json.dumps({"saved": args.out_model, "input_dim": int(input_dim)}))


if __name__ == "__main__":
    main()
