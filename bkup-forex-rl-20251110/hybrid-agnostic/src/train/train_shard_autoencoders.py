#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

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

from models.sharding import ShardPlan, plan_shards, shard_dataframe_columns  # type: ignore
from models.multihead_autoencoder import MultiHeadAEConfig, MultiHeadAutoencoder, RunningNormalizer  # type: ignore


@dataclass
class TrainConfig:
    num_shards: int = 10
    latent_dim: int = 64
    head_count: int = 6
    dropout: float = 0.0
    noise_std: float = 0.0
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_recon: float = 1.0
    weight_decor: float = 0.1
    weight_sparse: float = 0.05
    weight_contr: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def standardize_fit(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean())
        s = float(X[c].std())
        if s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Train sharded multi-head autoencoders on feature grid (wide CSV or per-file parts)")
    p.add_argument("--features", help="CSV path of wide features (from grid generator)")
    p.add_argument("--features-dir", help="Directory of per-file features (FX/*.csv, ETF/*.csv, time_features.csv)")
    p.add_argument("--dates-csv", default="forex-rl/hybrid-agnostic/data/dates.csv", help="Dates CSV from builder (for index alignment in per-file mode)")
    p.add_argument("--out-dir", default="forex-rl/hybrid-agnostic/checkpoints/shards")
    p.add_argument("--plan-path", default="forex-rl/hybrid-agnostic/artifacts/shard_plan.json")
    p.add_argument("--stats-path", default="forex-rl/hybrid-agnostic/artifacts/feature_stats.json")
    p.add_argument("--num-shards", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
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
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--noise-std", type=float, default=0.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.plan_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_path), exist_ok=True)

    if not args.features and not args.features_dir:
        raise SystemExit("Provide either --features or --features-dir")

    # Mode A: wide CSV (original behavior)
    if args.features:
        X = pd.read_csv(args.features, index_col=0)
        Xn, stats = standardize_fit(X)
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump({k: [float(v[0]), float(v[1])] for k, v in stats.items()}, f)
        plan = plan_shards(list(Xn.columns), num_shards=int(args.num_shards), group_related=True)
        plan.save(args.plan_path)
        col_map = shard_dataframe_columns(list(Xn.columns), plan)
    else:
        # Mode B: per-file parts. Build column list and stats without materializing all features.
        features_dir = args.features_dir
        assert features_dir is not None
        fx_dir = os.path.join(features_dir, "FX")
        etf_dir = os.path.join(features_dir, "ETF")
        time_csv = os.path.join(features_dir, "time_features.csv")
        parts: List[str] = []
        if os.path.isdir(fx_dir):
            parts += [os.path.join(fx_dir, f) for f in os.listdir(fx_dir) if f.endswith('.csv')]
        if os.path.isdir(etf_dir):
            parts += [os.path.join(etf_dir, f) for f in os.listdir(etf_dir) if f.endswith('.csv')]
        if os.path.exists(time_csv):
            parts.append(time_csv)
        if len(parts) == 0:
            raise SystemExit(f"No feature part CSVs found in {features_dir}")

        # Base index from dates.csv
        dates = pd.read_csv(args.dates_csv, header=None)[0].astype(str).tolist()
        base_index = pd.to_datetime(pd.Series(dates))
        # Prepare column list and per-column stats
        all_columns: List[str] = []
        stats_map: Dict[str, Tuple[float, float]] = {}

        for pth in sorted(parts):
            df = pd.read_csv(pth, index_col=0)
            # Align and fill
            df = df.reindex(base_index).fillna(0.0).astype(np.float32)
            # Accumulate columns and stats
            for c in df.columns:
                all_columns.append(c)
                col = df[c].values.astype(np.float32)
                m = float(col.mean())
                s = float(col.std())
                if s < 1e-8:
                    s = 1.0
                stats_map[c] = (m, s)

        # Plan shards and persist
        plan = plan_shards(all_columns, num_shards=int(args.num_shards), group_related=True)
        plan.save(args.plan_path)
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump({k: [float(v[0]), float(v[1])] for k, v in stats_map.items()}, f)

        # Build column index map per shard for later assembly
        # In per-file mode we won't use DataFrame-wide indices; collect names per shard
        col_map = {sid: [] for sid in range(plan.num_shards)}
        for feat, sid in plan.feature_to_shard.items():
            col_map[sid].append(feat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sid in range(plan.num_shards):
        # Assemble shard matrix either from Xn or from parts
        if args.features:
            cols_idx = col_map[sid]
            if len(cols_idx) == 0:
                continue
            cols = [Xn.columns[i] for i in cols_idx]
            Xs = Xn.iloc[:, cols_idx]
            input_dim = Xs.shape[1]
        else:
            cols = col_map[sid]
            if len(cols) == 0:
                continue
            # Assemble shard features from per-file parts
            features_dir = args.features_dir
            assert features_dir is not None
            dates = pd.read_csv(args.dates_csv, header=None)[0].astype(str).tolist()
            base_index = pd.to_datetime(pd.Series(dates))
            Xs = pd.DataFrame(index=base_index)
            fx_dir = os.path.join(features_dir, "FX")
            etf_dir = os.path.join(features_dir, "ETF")
            time_csv = os.path.join(features_dir, "time_features.csv")
            # Helper to merge subset columns from a part file
            def merge_subset(path: str, wanted: set) -> None:
                dfp = pd.read_csv(path, index_col=0)
                cols_here = [c for c in dfp.columns if c in wanted]
                if not cols_here:
                    return
                dfp = dfp.reindex(base_index).fillna(0.0).astype(np.float32)
                Xs.loc[:, cols_here] = dfp[cols_here]

            wanted = set(cols)
            # Iterate FX and ETF parts
            for d in [fx_dir, etf_dir]:
                if os.path.isdir(d):
                    for fname in os.listdir(d):
                        if not fname.endswith('.csv'):
                            continue
                        merge_subset(os.path.join(d, fname), wanted)
            if os.path.exists(time_csv):
                merge_subset(time_csv, wanted)

            # Standardize using saved stats map
            with open(args.stats_path, "r", encoding="utf-8") as f:
                stats_all = json.load(f)
            for c in cols:
                m, s = stats_all.get(c, [0.0, 1.0])
                if s == 0.0:
                    s = 1.0
                Xs[c] = (Xs[c] - float(m)) / float(s)
            Xs = Xs.astype(np.float32)
            input_dim = Xs.shape[1]

        cfg = MultiHeadAEConfig(
            input_dim=input_dim,
            latent_dim=int(args.latent),
            head_count=int(args.head_count),
            dropout=float(args.dropout),
            noise_std=float(args.noise_std),
            hidden_dims=[min(2048, max(256, input_dim // 2)), 256],
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

        dataset = torch.utils.data.TensorDataset(torch.tensor(Xs.values, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)

        norms = {
            "recon": RunningNormalizer(0.99),
            "decor": RunningNormalizer(0.99),
            "var": RunningNormalizer(0.99),
            "sparse": RunningNormalizer(0.99),
            "contr": RunningNormalizer(0.99),
        }
        weights = {
            "recon": 1.0,
            "decor": 0.1,
            "var": 0.05,
            "sparse": 0.02,
            "contr": 0.0,
        }

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
            print(json.dumps({"shard": sid, "epoch": epoch + 1, **log}), flush=True)

        # Save checkpoint and shard columns
        ck = {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "input_dim": int(input_dim),
            "columns": cols,
        }
        outp = os.path.join(args.out_dir, f"shard_{sid:02d}.pt")
        torch.save(ck, outp)
        print(json.dumps({"saved": outp, "cols": len(cols)}))


if __name__ == "__main__":
    main()
