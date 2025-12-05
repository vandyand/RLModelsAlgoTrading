#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

# Make src/ importable when running as a script
import os, sys
SCRIPT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from models.multihead_autoencoder import MultiHeadAEConfig, MultiHeadAutoencoder  # type: ignore


def load_and_standardize(features_csv: str, stats_path: str) -> pd.DataFrame:
    X = pd.read_csv(features_csv, index_col=0)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_obj = json.load(f)
    stats = {k: (float(v[0]), float(v[1])) for k, v in stats_obj.items()}
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        X[c] = (X[c] - m) / s
    return X.astype(np.float32)


def encode_shard_latents(X: pd.DataFrame, shards_dir: str, device: torch.device) -> pd.DataFrame:
    zs: List[pd.DataFrame] = []
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
            randomized_heads=bool(cfgd.get("randomized_heads", False)),
            head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
            head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
            head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
            head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
            head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
            head_random_seed=(cfgd.get("head_random_seed", None)),
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


def _assemble_shard_array_from_parts(features_dir: str, dates_csv: str, cols: List[str]) -> np.ndarray:
    dates = pd.read_csv(dates_csv, header=None)[0].astype(str).tolist()
    num_rows = len(dates)
    arr = np.zeros((num_rows, len(cols)), dtype=np.float32)
    # Column position map
    col_pos: Dict[str, int] = {c: i for i, c in enumerate(cols)}
    # Group requested columns by source file
    groups: Dict[str, List[str]] = {}
    for c in cols:
        if c.startswith("FX_"):
            parts = c.split("_")
            key = os.path.join(features_dir, "FX", f"{parts[1]}_{parts[2]}.csv") if len(parts) >= 3 else os.path.join(features_dir, "FX", "__ALL__")
        elif c.startswith("ETF_"):
            parts = c.split("_")
            key = os.path.join(features_dir, "ETF", f"{parts[1]}.csv") if len(parts) >= 2 else os.path.join(features_dir, "ETF", "__ALL__")
        else:
            key = os.path.join(features_dir, "time_features.csv")
        groups.setdefault(key, []).append(c)

    for path, want_cols in groups.items():
        if os.path.basename(path) == "__ALL__":
            parent = os.path.dirname(path)
            if os.path.isdir(parent):
                for fname in os.listdir(parent):
                    if not fname.endswith('.csv'):
                        continue
                    p = os.path.join(parent, fname)
                    hdr = pd.read_csv(p, nrows=0)
                    take = [c for c in want_cols if c in hdr.columns]
                    if not take:
                        continue
                    dfp = pd.read_csv(p, usecols=[hdr.columns[0], *take], index_col=0)
                    vals = dfp[take].values.astype(np.float32, copy=False)
                    for j, c in enumerate(take):
                        arr[:, col_pos[c]] = vals[:, j]
            continue
        if not os.path.exists(path):
            continue
        hdr = pd.read_csv(path, nrows=0)
        take = [c for c in want_cols if c in hdr.columns]
        if not take:
            continue
        dfp = pd.read_csv(path, usecols=[hdr.columns[0], *take], index_col=0)
        vals = dfp[take].values.astype(np.float32, copy=False)
        for j, c in enumerate(take):
            arr[:, col_pos[c]] = vals[:, j]
    return arr


def encode_final_from_parts(features_dir: str, dates_csv: str, shards_dir: str, final_ae_path: str, plan_path: str, stats_path: str, device: torch.device) -> pd.DataFrame:
    # Load shard plan and stats
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    feature_to_shard = {str(k): int(v) for k, v in plan.get("feature_to_shard", {}).items()}
    shard_to_cols: Dict[int, List[str]] = {}
    for feat, sid in feature_to_shard.items():
        shard_to_cols.setdefault(int(sid), []).append(feat)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_all = json.load(f)

    # Encode each shard then concatenate
    # Precompute rows and allocate final Z_concat array (rows x total_latent)
    num_rows = len(pd.read_csv(dates_csv, header=None))
    # Determine total latent dim
    total_latent = 0
    shard_latent_dims: Dict[int, int] = {}
    shard_files = sorted([f for f in os.listdir(shards_dir) if f.startswith("shard_") and f.endswith(".pt")])
    for sid, fname in enumerate(shard_files):
        ck = torch.load(os.path.join(shards_dir, fname), map_location=device)
        cfgd = ck.get("cfg", {})
        latent_dim = int(cfgd.get("latent_dim", 64))
        shard_latent_dims[sid] = latent_dim
        total_latent += latent_dim
    Z_concat = np.zeros((num_rows, total_latent), dtype=np.float32)
    offset = 0
    for sid, fname in enumerate(shard_files):
        cols = shard_to_cols.get(sid, [])
        if not cols:
            continue
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
            randomized_heads=bool(cfgd.get("randomized_heads", False)),
            head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
            head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
            head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
            head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
            head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
            head_random_seed=(cfgd.get("head_random_seed", None)),
        )).to(device)
        model.load_state_dict(ck["model_state"])  # type: ignore[index]
        model.eval()

        Xarr = _assemble_shard_array_from_parts(features_dir, dates_csv, cols)
        # Vectorized standardization
        m = np.array([stats_all.get(c, [0.0, 1.0])[0] for c in cols], dtype=np.float32)
        s = np.array([max(stats_all.get(c, [0.0, 1.0])[1], 1e-8) for c in cols], dtype=np.float32)
        Xarr = (Xarr - m[None, :]) / s[None, :]
        with torch.no_grad():
            xb = torch.tensor(Xarr, dtype=torch.float32, device=device)
            z, _ = model(xb)
            z_np = z.cpu().numpy().astype(np.float32)
        Z_concat[:, offset:offset + latent_dim] = z_np
        offset += latent_dim

    Z = Z_concat

    # Load final AE and encode to final latent
    ck_final = torch.load(final_ae_path, map_location=device)
    cfgd = ck_final.get("cfg", {})
    input_dim = int(ck_final.get("input_dim", Z.shape[1]))
    latent_dim = int(cfgd.get("latent_dim", 64))
    head_count = int(cfgd.get("head_count", 6))
    hidden_dims = cfgd.get("hidden_dims", [512, 128])
    dropout = float(cfgd.get("dropout", 0.0))
    noise_std = float(cfgd.get("noise_std", 0.0))

    final_ae = MultiHeadAutoencoder(MultiHeadAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        head_count=head_count,
        hidden_dims=hidden_dims,
        dropout=dropout,
        noise_std=noise_std,
        randomized_heads=bool(cfgd.get("randomized_heads", False)),
        head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
        head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
        head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
        head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
        head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
        head_random_seed=(cfgd.get("head_random_seed", None)),
    )).to(device)
    final_ae.load_state_dict(ck_final["model_state"])  # type: ignore[index]
    final_ae.eval()
    with torch.no_grad():
        xb = torch.tensor(Z, dtype=torch.float32, device=device)
        z_final, _ = final_ae(xb)
        zf_np = z_final.cpu().numpy().astype(np.float32)
    # Build index from dates
    idx = pd.read_csv(dates_csv, header=None)[0].astype(str).tolist()
    cols_out = [f"z_final_{i:02d}" for i in range(zf_np.shape[1])]
    return pd.DataFrame(zf_np, index=idx, columns=cols_out)


essential_keys = ["model_state", "cfg", "input_dim", "cols"]


def encode_final_latents(Z_concat: pd.DataFrame, final_ckpt: str, device: torch.device) -> pd.DataFrame:
    ck = torch.load(final_ckpt, map_location=device)
    cfgd = ck.get("cfg", {})
    input_dim = int(ck.get("input_dim", Z_concat.shape[1]))
    latent_dim = int(cfgd.get("latent_dim", 64))
    head_count = int(cfgd.get("head_count", 6))
    hidden_dims = cfgd.get("hidden_dims", [512, 128])
    dropout = float(cfgd.get("dropout", 0.0))
    noise_std = float(cfgd.get("noise_std", 0.0))

    model = MultiHeadAutoencoder(MultiHeadAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        head_count=head_count,
        hidden_dims=hidden_dims,
        dropout=dropout,
        noise_std=noise_std,
        randomized_heads=bool(cfgd.get("randomized_heads", False)),
        head_arch_min_layers=int(cfgd.get("head_arch_min_layers", 1)),
        head_arch_max_layers=int(cfgd.get("head_arch_max_layers", 3)),
        head_hidden_min=int(cfgd.get("head_hidden_min", 64)),
        head_hidden_max=int(cfgd.get("head_hidden_max", 512)),
        head_activation_set=list(cfgd.get("head_activation_set", ["relu","gelu","silu","elu"])),
        head_random_seed=(cfgd.get("head_random_seed", None)),
    )).to(device)
    model.load_state_dict(ck["model_state"])  # type: ignore[index]
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(Z_concat.values, dtype=torch.float32, device=device)
        z, _ = model(xb)
        z_np = z.cpu().numpy().astype(np.float32)
    cols_out = [f"z_final_{i:02d}" for i in range(z_np.shape[1])]
    return pd.DataFrame(z_np, index=Z_concat.index, columns=cols_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Export final latent sequences from features using trained shard + final AEs")
    p.add_argument("--features")
    p.add_argument("--features-dir", help="Directory of per-file features (FX/ ETF/ time_features.csv)")
    p.add_argument("--dates-csv", default="forex-rl/hybrid-agnostic/data/dates.csv")
    p.add_argument("--stats-path", default="forex-rl/hybrid-agnostic/artifacts/feature_stats.json")
    p.add_argument("--shards-dir", default="forex-rl/hybrid-agnostic/checkpoints/shards")
    p.add_argument("--final-ae", default="forex-rl/hybrid-agnostic/checkpoints/final_ae.pt")
    p.add_argument("--plan-path", default="forex-rl/hybrid-agnostic/artifacts/shard_plan.json")
    p.add_argument("--out-csv", default="forex-rl/hybrid-agnostic/data/final_latents.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.features:
        X = load_and_standardize(args.features, args.stats_path)
        Z_concat = encode_shard_latents(X, args.shards_dir, device)
        Z_final = encode_final_latents(Z_concat, args.final_ae, device)
    else:
        if not args.features_dir:
            raise SystemExit("Provide --features (CSV) or --features-dir (per-file)")
        Z_final = encode_final_from_parts(
            features_dir=args.features_dir,
            dates_csv=args.dates_csv,
            shards_dir=args.shards_dir,
            final_ae_path=args.final_ae,
            plan_path=args.plan_path,
            stats_path=args.stats_path,
            device=device,
        )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    Z_final.to_csv(args.out_csv, index=True)
    print(json.dumps({"saved": args.out_csv, "rows": int(Z_final.shape[0]), "cols": int(Z_final.shape[1])}))


if __name__ == "__main__":
    main()
