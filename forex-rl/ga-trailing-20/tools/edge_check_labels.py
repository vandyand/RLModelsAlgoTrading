#!/usr/bin/env python3
"""Pre-NN edge check on trailing-stop episode labels.

This script:
  1) Loads aligned multi-instrument features via DatasetLoader.
  2) Uses the MultiHeadGatedNNStrategy's label builder to generate
     per-instrument trailing-stop "episode returns" (Y).
  3) Optionally restricts to a feature subset (e.g. top-50 from ranking).
  4) Optionally expands features with simple temporal lags.
  5) On a capped number of bars, fits very simple baselines:
       - Ridge regression on raw episode returns (regression edge).
       - Logistic regression on |ret| > gate_ret_threshold (gate edge).
  6) Reports per-instrument and aggregate metrics (mean/std of labels,
     fraction of "interesting" episodes, R^2, ROC-AUC, etc.).

The goal is to sanity-check whether there is any detectable edge in the
features *before* committing to heavier neural training runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

# Make ga-trailing-20 importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig, SplitConfig  # type: ignore[import]
from strategies.multihead_gated_nn import (  # type: ignore[import]
    MultiHeadGatedNNConfig,
    MultiHeadGatedNNStrategy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-NN edge check on trailing-stop episode labels")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    p.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    p.add_argument(
        "--instruments",
        default="EUR_USD,USD_JPY,GBP_USD,AUD_USD,USD_CHF,USD_CAD,NZD_USD,EUR_JPY,GBP_JPY,EUR_GBP,"
        "EUR_CHF,EUR_AUD,EUR_CAD,GBP_CHF,AUD_JPY,AUD_CHF,CAD_JPY,NZD_JPY,GBP_AUD,AUD_NZD",
        help="Comma-separated list of instruments (default = OANDA 20 universe)",
    )
    p.add_argument("--train", default="2023-01-01:2025-06-30", help="Train range for label construction")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D", help="Aux granularities, e.g. 'D' or 'M5,D' (empty string for none)")
    p.add_argument("--max-bars", type=int, default=800, help="Max number of bars used for the edge check")
    p.add_argument("--max-rows", type=int, default=20000, help="Optional cap on rows loaded per instrument")
    p.add_argument(
        "--gate-ret-threshold",
        type=float,
        default=0.0075,
        help="Absolute episode return threshold used to define positive gate labels",
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-3,
        help="L2 strength for Ridge regression on raw episode returns",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum number of samples required to run regressions/classifiers",
    )
    # Feature subset controls (e.g. MULTI20 top-K list)
    p.add_argument(
        "--feature-subset-file",
        default="",
        help="Optional path to a text file with one feature name per line; if set, restricts loader to this subset.",
    )
    p.add_argument(
        "--feature-top-k",
        type=int,
        default=50,
        help="If >0, use only the first K features from --feature-subset-file.",
    )
    # Simple temporal lags applied *after* label construction.
    p.add_argument(
        "--feature-lags",
        default="0",
        help="Comma-separated non-negative lags in bars, e.g. '0,7,49'. 0 = current bar only.",
    )
    return p.parse_args()


def _parse_range(s: str) -> Tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def _materialize_split(split, max_bars: int) -> List[Dict]:
    records: List[Dict] = []
    if split is None:
        return records
    for idx, rec in enumerate(split):
        if idx >= max_bars:
            break
        records.append(rec)
    return records


def _maybe_load_feature_subset(path: str, top_k: int) -> List[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    names: List[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if not name:
                continue
            names.append(name)
            if top_k and len(names) >= int(top_k):
                break
    return names or None


def _apply_lags_to_xy(
    X: np.ndarray,
    Y: np.ndarray,
    lags: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply simple time lags to already-built (X, Y).

    X is assumed to be ordered in time with shape [T, F]. For each time index
    t >= max(lags) we build a new feature vector by concatenating X[t - lag]
    across all lags. Y is shifted accordingly to keep alignment.
    """
    if not lags or lags == [0]:
        return X, Y

    lags = sorted({int(l) for l in lags if int(l) >= 0})
    if 0 not in lags:
        lags.insert(0, 0)
    max_lag = max(lags)

    T, F = X.shape
    if T <= max_lag + 1:
        return X, Y

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    for t in range(max_lag, T):
        feats_per_lag: List[np.ndarray] = []
        for lag in lags:
            feats_per_lag.append(X[t - lag])
        X_list.append(np.concatenate(feats_per_lag, axis=-1))
        Y_list.append(Y[t])

    X_new = np.stack(X_list, axis=0)
    Y_new = np.stack(Y_list, axis=0)
    return X_new, Y_new


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]

    # Optional feature subset: load names from text file (e.g. MULTI20 ranking)
    feature_subset = _maybe_load_feature_subset(args.feature_subset_file, int(args.feature_top_k))

    # Loader configuration
    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity=args.base_gran.upper(),
        aux_granularities=aux,
        normalize=True,
        feature_subset=feature_subset,
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
    )
    loader = DatasetLoader(lcfg)

    train_range = _parse_range(args.train)
    split_cfg = SplitConfig(train=train_range)
    train_split, _, _ = loader.split_by_dates(split_cfg)

    records = _materialize_split(train_split, int(args.max_bars))
    if len(records) < args.min_samples:
        print(
            {
                "type": "edge_check_summary",
                "status": "insufficient_samples",
                "num_records": len(records),
                "min_samples": args.min_samples,
                "train_range": args.train,
            }
        )
        return

    # Use the strategy's internal label builder to generate trailing-stop
    # episode returns per instrument, but we do NOT train the NN here.
    cfg = MultiHeadGatedNNConfig(gate_ret_threshold=float(args.gate_ret_threshold))
    strat = MultiHeadGatedNNStrategy(instruments=instruments, config=cfg)
    X, Y = strat._build_xy(records)  # noqa: SLF001 - intentional internal reuse

    # Optional lag expansion of X (and aligned trimming of Y).
    try:
        lags = [int(tok.strip()) for tok in args.feature_lags.split(",") if tok.strip()]
    except Exception:
        lags = [0]
    X, Y = _apply_lags_to_xy(X, Y, lags)

    n_samples, n_features = X.shape
    _, n_inst = Y.shape
    if n_samples < args.min_samples:
        print(
            {
                "type": "edge_check_summary",
                "status": "insufficient_samples_after_xy",
                "num_samples": n_samples,
                "min_samples": args.min_samples,
            }
        )
        return

    # Simple train/validation split that preserves time order.
    split = int(0.7 * n_samples)
    if split <= 10 or n_samples - split <= 10:
        print(
            {
                "type": "edge_check_summary",
                "status": "insufficient_split",
                "num_samples": n_samples,
                "split_index": split,
            }
        )
        return

    X_train, X_val = X[:split], X[split:]
    results: Dict[str, Dict[str, float]] = {}

    # Lazy imports of sklearn so that this script only requires it when used.
    from sklearn.linear_model import Ridge, LogisticRegression  # type: ignore[import]
    from sklearn.metrics import r2_score, roc_auc_score  # type: ignore[import]

    gate_thr = float(args.gate_ret_threshold)

    for j, inst in enumerate(instruments):
        y_raw = Y[:, j]
        y_tr = y_raw[:split]
        y_va = y_raw[split:]

        # Basic label distribution diagnostics.
        mean_ret = float(np.mean(y_raw))
        std_ret = float(np.std(y_raw))
        frac_big = float(np.mean(np.abs(y_raw) > gate_thr))

        # Ridge regression on raw returns.
        ridge_edge = float("nan")
        try:
            ridge = Ridge(alpha=float(args.ridge_alpha))
            ridge.fit(X_train, y_tr)
            y_pred = ridge.predict(X_val)
            ridge_edge = float(r2_score(y_va, y_pred))
        except Exception:
            pass

        # Logistic regression on "gate-worthy" episodes.
        y_gate = (np.abs(y_raw) > gate_thr).astype(np.int32)
        y_gate_tr = y_gate[:split]
        y_gate_va = y_gate[split:]

        auc = float("nan")
        try:
            # Require at least two classes in the training segment.
            if y_gate_tr.min() != y_gate_tr.max():
                clf = LogisticRegression(max_iter=200)
                clf.fit(X_train, y_gate_tr)
                prob = clf.predict_proba(X_val)[:, 1]
                # Guard against degenerate validation labels.
                if y_gate_va.min() != y_gate_va.max():
                    auc = float(roc_auc_score(y_gate_va, prob))
        except Exception:
            pass

        results[inst] = {
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "frac_big": frac_big,
            "ridge_r2": ridge_edge,
            "gate_auc": auc,
        }

    # Aggregate metrics across instruments.
    ridge_vals = [v["ridge_r2"] for v in results.values() if np.isfinite(v["ridge_r2"])]
    auc_vals = [v["gate_auc"] for v in results.values() if np.isfinite(v["gate_auc"])]
    frac_big_vals = [v["frac_big"] for v in results.values()]

    summary = {
        "type": "edge_check_summary",
        "status": "ok",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_instruments": n_inst,
        "gate_ret_threshold": gate_thr,
        "mean_frac_big": float(np.mean(frac_big_vals)) if frac_big_vals else float("nan"),
        "mean_ridge_r2": float(np.mean(ridge_vals)) if ridge_vals else float("nan"),
        "mean_gate_auc": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "per_instrument": results,
    }
    print(summary)


if __name__ == "__main__":  # pragma: no cover
    main()
