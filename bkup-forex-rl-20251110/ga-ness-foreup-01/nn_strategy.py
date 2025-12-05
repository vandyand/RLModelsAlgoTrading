from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from gp_optimize import metrics_from_positions


@dataclass
class NNConfig:
    hidden1: int = 64
    hidden2: int = 32
    epochs: int = 400
    lr: float = 0.05
    reg: float = 1e-4
    # Threshold mapping in [0,1]
    enter_long: float = 0.65
    exit_long: float = 0.55
    exit_short: float = 0.45
    enter_short: float = 0.35
    # Costs and extras
    cost_bps: float = 1.0


class _NumpyMLPBin:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mu) / (self.sigma + 1e-12)
        Z1 = Xn @ self.W1 + self.b1
        H1 = np.maximum(0.0, Z1)
        Z2 = H1 @ self.W2 + self.b2
        H2 = np.maximum(0.0, Z2)
        Z3 = H2 @ self.W3 + self.b3
        P = 1.0 / (1.0 + np.exp(-Z3))
        return P.reshape(-1)


def _train_mlp_bin(X: np.ndarray, y: np.ndarray, hidden1: int, hidden2: int, epochs: int, lr: float, reg: float, seed: int = 42) -> _NumpyMLPBin:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    Xn = (X - mu) / sigma
    rng = np.random.default_rng(int(seed))
    n_in = Xn.shape[1]
    H1, H2 = int(hidden1), int(hidden2)
    W1 = 0.1 * rng.standard_normal((n_in, H1))
    b1 = np.zeros((H1,), dtype=float)
    W2 = 0.1 * rng.standard_normal((H1, H2))
    b2 = np.zeros((H2,), dtype=float)
    W3 = 0.1 * rng.standard_normal((H2, 1))
    b3 = np.zeros((1,), dtype=float)
    yv = y.reshape(-1, 1).astype(float)
    for _ in range(int(epochs)):
        # forward
        Z1 = Xn @ W1 + b1
        H1a = np.maximum(0.0, Z1)
        Z2 = H1a @ W2 + b2
        H2a = np.maximum(0.0, Z2)
        Z3 = H2a @ W3 + b3
        P = 1.0 / (1.0 + np.exp(-Z3))
        # gradients (binary cross-entropy derivative)
        dZ3 = (P - yv)
        dW3 = (H2a.T @ dZ3) / Xn.shape[0] + reg * W3
        db3 = dZ3.mean(axis=0)
        dH2 = dZ3 @ W3.T
        dZ2 = dH2 * (Z2 > 0)
        dW2 = (H1a.T @ dZ2) / Xn.shape[0] + reg * W2
        db2 = dZ2.mean(axis=0)
        dH1 = dZ2 @ W2.T
        dZ1 = dH1 * (Z1 > 0)
        dW1 = (Xn.T @ dZ1) / Xn.shape[0] + reg * W1
        db1 = dZ1.mean(axis=0)
        # update
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    return _NumpyMLPBin(mu, sigma, W1, b1, W2, b2, W3, b3)


def _positions_from_thresholds(score: pd.Series, cfg: NNConfig, min_hold_candles: int, cooldown_candles: int) -> pd.Series:
    s = score.values.astype(float)
    gl = s > float(cfg.enter_long)
    xl = s < float(cfg.exit_long)
    gs = s < float(cfg.enter_short)
    xs = s > float(cfg.exit_short)
    state = 0
    hold = 0
    cooldown = 0
    out = np.zeros_like(s, dtype=float)
    for i in range(len(s)):
        if cooldown > 0:
            cooldown -= 1
        if state != 0:
            hold += 1
        else:
            hold = 0
        if state == 0:
            if cooldown == 0:
                if gl[i]:
                    state = 1
                    hold = 0
                elif gs[i]:
                    state = -1
                    hold = 0
        elif state == 1:
            if hold >= int(min_hold_candles) and (xl[i] or gs[i]):
                state = 0
                cooldown = int(cooldown_candles)
        elif state == -1:
            if hold >= int(min_hold_candles) and (xs[i] or gl[i]):
                state = 0
                cooldown = int(cooldown_candles)
        out[i] = float(state)
    pos = pd.Series(out, index=score.index)
    return pos.shift(1).fillna(0.0)


def run_nn_once(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    close_col: str,
    cfg: NNConfig,
    min_hold: int,
    cooldown: int,
    seed: int,
) -> Dict[str, Any]:
    # Prepare supervised labels: next-bar return > 0
    feature_cols = [c for c in X_train.columns if c != close_col and not c.endswith("_close")]
    ret_train = X_train[close_col].pct_change().shift(-1).fillna(0.0)
    y_train = (ret_train > 0.0).astype(int).values
    Xtr = X_train[feature_cols].astype(float).values
    model = _train_mlp_bin(Xtr, y_train, cfg.hidden1, cfg.hidden2, cfg.epochs, cfg.lr, cfg.reg, seed)

    # Train predictions and positions
    s_train = pd.Series(model.predict_proba(Xtr), index=X_train.index)
    pos_train = _positions_from_thresholds(s_train, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
    tm = metrics_from_positions(X_train[close_col], pos_train, cost_bps=float(cfg.cost_bps))

    # Validation
    val_metrics: Optional[Dict[str, float]] = None
    pos_val: Optional[pd.Series] = None
    if X_val is not None and len(X_val) > 0:
        Xv = X_val[feature_cols].astype(float).values
        s_val = pd.Series(model.predict_proba(Xv), index=X_val.index)
        pos_val = _positions_from_thresholds(s_val, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
        val_metrics = metrics_from_positions(X_val[close_col], pos_val, cost_bps=float(cfg.cost_bps))

    # Wrap as final_population entry for downstream tools
    trades_tr = int(pos_train.ne(pos_train.shift(1)).sum())
    bars_tr = int(len(X_train))
    tf_train = (float(trades_tr) * 4000.0) / float(max(1, bars_tr))
    fp = [{
        "rank": 1,
        "fitness_train": float(tm.get("sharpe", 0.0)),
        "metrics_train": tm,
        "trades_train": trades_tr,
        "metrics_val": val_metrics or {},
        "trades_val": int(pos_val.ne(pos_val.shift(1)).sum()) if pos_val is not None else 0,
        "tree_size": 0,
        "tree_depth": 0,
        "tf_train": float(tf_train),
        "tf_val": float((float((pos_val.ne(pos_val.shift(1)).sum()) if pos_val is not None else 0) * 4000.0) / float(max(1, len(X_val) if X_val is not None else 0))) if X_val is not None and len(X_val) > 0 else float("nan"),
        "extras_train": {},
        "extras_val": {},
        "tree": {"entry_long": None, "entry_short": None, "exit_long": None, "exit_short": None},
    }]

    return {
        "train_metrics": tm,
        "val_metrics": val_metrics,
        "final_population": fp,
        "model": {
            "hidden": [int(cfg.hidden1), int(cfg.hidden2)],
            "thresholds": {
                "enter_long": float(cfg.enter_long),
                "exit_long": float(cfg.exit_long),
                "exit_short": float(cfg.exit_short),
                "enter_short": float(cfg.enter_short),
            },
        },
    }
