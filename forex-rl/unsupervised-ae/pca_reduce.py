#!/usr/bin/env python3
"""
PCA reducer for grid features (with whitening of PCA scores by default).

- Loads wide features CSV produced by grid_features.py
- Standardizes each column using train split means/stds
- Fits PCA to retain N components (default 64) that explain most variance
- Whitened output: PCA scores are standardized (train-mean/variance) to unit variance
- Saves transformed features and PCA artifacts (component matrix, mean/std, explained variance)

Usage (aligned with grid_features defaults):
  python forex-rl/unsupervised-ae/pca_reduce.py \
    --features forex-rl/unsupervised-ae/data/multi_features.csv \
    --components 64 \
    --train-ratio 0.8 \
    --out-features forex-rl/unsupervised-ae/data/pca_features.npy \
    --out-meta forex-rl/unsupervised-ae/data/pca_meta.npz
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
    parser = argparse.ArgumentParser(description="Reduce feature dimensionality with PCA")
    parser.add_argument("--features", required=True)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out-features", default="forex-rl/unsupervised-ae/data/pca_features.npy")
    parser.add_argument("--out-meta", default="forex-rl/unsupervised-ae/data/pca_meta.npz")
    args = parser.parse_args()

    print(json.dumps({"status": "start_pca", "features": args.features, "components": int(args.components)}))
    X = pd.read_csv(args.features, index_col=0)

    # Train/val split by time
    n = len(X)
    split = int(n * args.train_ratio)
    X_train = X.iloc[:split]
    X_val = X.iloc[split:]

    print(json.dumps({"status": "standardize", "train_rows": int(X_train.shape[0]), "val_rows": int(X_val.shape[0])}))
    Xn_train, stats = standardize_fit(X_train)
    Xn_val = standardize_apply(X_val, stats)

    pca = PCA(n_components=int(args.components), svd_solver="auto", random_state=42)
    Z_train = pca.fit_transform(Xn_train.values)
    Z_val = pca.transform(Xn_val.values)

    # Whiten PCA scores by standardizing on train split
    z_mean = Z_train.mean(axis=0)
    z_std = Z_train.std(axis=0) + 1e-8
    Z_train_w = (Z_train - z_mean) / z_std
    Z_val_w = (Z_val - z_mean) / z_std
    Z = np.vstack([Z_train_w, Z_val_w]).astype(np.float32)

    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_meta), exist_ok=True)

    np.save(args.out_features, Z)
    np.savez_compressed(
        args.out_meta,
        columns=np.array(list(X.columns)),
        mean=np.array([stats[c][0] for c in X.columns], dtype=np.float32),
        std=np.array([stats[c][1] for c in X.columns], dtype=np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        z_mean=z_mean.astype(np.float32),
        z_std=z_std.astype(np.float32),
    )

    print(json.dumps({
        "status": "pca_saved",
        "in": args.features,
        "out_features": args.out_features,
        "out_meta": args.out_meta,
        "n_components": int(args.components),
        "explained_var_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "whitened": True,
    }))


if __name__ == "__main__":
    main()
