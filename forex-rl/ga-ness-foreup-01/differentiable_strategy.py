from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Reuse existing metrics for final (discrete) evaluation
from gp_optimize import metrics_from_positions

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class DiffConfig:
    hidden1: int = 64
    hidden2: int = 32
    epochs: int = 80
    lr: float = 1e-3
    l2: float = 1e-5
    tv_penalty: float = 1e-3  # penalty on position changes (smoothness)
    beta: float = 5.0         # steepness for tanh position mapping
    mid: float = 0.5          # center for tanh position mapping
    # Differentiable objective weights (train-only; no lookahead)
    w_sharpe: float = 1.0
    w_cum: float = 0.5
    w_pf: float = 0.1
    w_tf_band: float = 0.1
    w_neg_cum_pen: float = 0.5
    w_val_sharpe: float = 1.0
    w_gap: float = 0.3
    w_ratio: float = 0.3
    w_exposure: float = 0.1
    exposure_target: float = 0.2
    # Target trade frequency band (per-1000 bars approx via smooth proxy)
    tf_low: float = 10.0
    tf_high: float = 25.0
    tf_maxok: float = 40.0
    # Cum return shaping
    cum_tanh_k: float = 4.0
    # Discrete thresholds used only for reporting/evaluation
    enter_long: float = 0.65
    exit_long: float = 0.55
    exit_short: float = 0.45
    enter_short: float = 0.35
    cost_bps: float = 1.0


def _discrete_positions_from_thresholds(score: pd.Series, cfg: DiffConfig, min_hold_candles: int, cooldown_candles: int) -> pd.Series:
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


def _numpy_fallback(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    close_col: str,
    cfg: DiffConfig,
    min_hold: int,
    cooldown: int,
    seed: int,
) -> Dict[str, Any]:
    # If torch is unavailable, return a no-op baseline using zero scores
    feature_cols = [c for c in X_train.columns if c != close_col and not c.endswith("_close")]
    s_train = pd.Series(np.zeros(len(X_train), dtype=float) + 0.5, index=X_train.index)
    pos_train = _discrete_positions_from_thresholds(s_train, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
    tm = metrics_from_positions(X_train[close_col], pos_train, cost_bps=float(cfg.cost_bps))
    val_metrics: Optional[Dict[str, float]] = None
    if X_val is not None and len(X_val) > 0:
        s_val = pd.Series(np.zeros(len(X_val), dtype=float) + 0.5, index=X_val.index)
        pos_val = _discrete_positions_from_thresholds(s_val, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
        val_metrics = metrics_from_positions(X_val[close_col], pos_val, cost_bps=float(cfg.cost_bps))
    fp = [{
        "rank": 1,
        "fitness_train": float(tm.get("sharpe", 0.0)),
        "metrics_train": tm,
        "trades_train": int(pos_train.ne(pos_train.shift(1)).sum()),
        "metrics_val": val_metrics or {},
        "trades_val": 0,
        "tree_size": 0,
        "tree_depth": 0,
        "tf_train": 0.0,
        "tf_val": float("nan"),
        "extras_train": {},
        "extras_val": {},
        "tree": {"entry_long": None, "entry_short": None, "exit_long": None, "exit_short": None},
    }]
    return {"train_metrics": tm, "val_metrics": val_metrics, "final_population": fp}


class _StrategyNet(nn.Module):
    def __init__(self, n_features: int, h1: int, h2: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        xn = self.scale * x + self.bias
        s = self.net(xn).squeeze(-1)
        return s


def _diff_backtest(
    score: torch.Tensor,
    close: torch.Tensor,
    cfg: DiffConfig,
) -> torch.Tensor:
    # Map to continuous position in [-1,1]
    pos = torch.tanh(torch.tensor(cfg.beta, dtype=score.dtype, device=score.device) * (score - float(cfg.mid)))
    # Use previous position to apply to current return
    pos_shift = torch.cat([torch.zeros(1, dtype=pos.dtype, device=pos.device), pos[:-1]])
    c = close.float()
    # Replace non-finite values with zeros for safety
    c = torch.where(torch.isfinite(c), c, torch.zeros_like(c))
    # Compute simple returns r[t] = c[t]/c[t-1] - 1 with r[0]=0
    ret = torch.zeros_like(c)
    if c.numel() > 1:
        ret[1:] = c[1:] / (c[:-1] + 1e-12) - 1.0
    strat_ret = pos_shift * ret
    # Transaction cost on position changes
    if float(cfg.cost_bps) > 0.0:
        dpos = torch.abs(pos_shift - torch.cat([torch.zeros(1, dtype=pos.dtype, device=pos.device), pos_shift[:-1]]))
        cost = float(cfg.cost_bps) * 1e-4
        strat_ret = strat_ret - cost * dpos
    return strat_ret


def run_diff_once(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    close_col: str,
    cfg: DiffConfig,
    min_hold: int,
    cooldown: int,
    seed: int,
) -> Dict[str, Any]:
    if torch is None or nn is None:
        return _numpy_fallback(X_train, X_val, close_col, cfg, min_hold, cooldown, seed)

    torch.manual_seed(int(seed))
    feature_cols = [c for c in X_train.columns if c != close_col and not c.endswith("_close")]
    xtr = torch.tensor(X_train[feature_cols].astype(float).values, dtype=torch.float32)
    xva = torch.tensor(X_val[feature_cols].astype(float).values, dtype=torch.float32) if X_val is not None and len(X_val) > 0 else None
    ctr = torch.tensor(X_train[close_col].astype(float).values, dtype=torch.float32)
    cva = torch.tensor(X_val[close_col].astype(float).values, dtype=torch.float32) if X_val is not None and len(X_val) > 0 else None

    model = _StrategyNet(n_features=xtr.shape[1], h1=int(cfg.hidden1), h2=int(cfg.hidden2))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.l2))

    def sharpe_like(r: torch.Tensor) -> torch.Tensor:
        mu = torch.mean(r)
        sd = torch.std(r, unbiased=True) if r.numel() > 1 else torch.tensor(0.0, dtype=r.dtype, device=r.device)
        return mu / (sd + 1e-8)

    def cumret_like(r: torch.Tensor) -> torch.Tensor:
        # Stable cumulative return proxy via log-sum; returns approx (prod(1+r)-1)
        return torch.tanh(cfg.cum_tanh_k * torch.sum(torch.log1p(torch.clamp(r, min=-0.99)) / max(1, r.numel())))

    def trade_freq_band_penalty(pos_shift: torch.Tensor) -> torch.Tensor:
        # Smooth proxy for number of trades via |delta pos|; scale to per-1000 bars
        if pos_shift.numel() <= 1:
            return torch.tensor(0.0, dtype=pos_shift.dtype, device=pos_shift.device)
        dpos = torch.abs(pos_shift[1:] - pos_shift[:-1])
        # Effective trade rate ~ average |dpos| per step
        per_k = 1000.0 * torch.mean(dpos)
        low = torch.tensor(cfg.tf_low, dtype=pos_shift.dtype)
        high = torch.tensor(cfg.tf_high, dtype=pos_shift.dtype)
        maxok = torch.tensor(cfg.tf_maxok, dtype=pos_shift.dtype)
        below = torch.relu(low - per_k) / torch.clamp(low, min=1e-6)
        within = torch.zeros_like(per_k)
        above = torch.relu(per_k - high) / torch.clamp(maxok - high, min=1e-6)
        return torch.where(per_k < low, below, torch.where(per_k <= high, within, above))

    for _ in range(int(cfg.epochs)):
        model.train()
        s_tr = model(xtr)
        r_tr = _diff_backtest(s_tr, ctr, cfg)
        sh_tr = sharpe_like(r_tr)
        cr_tr = cumret_like(r_tr)
        # Validation stream (used for gap penalty)
        if xva is not None:
            s_va = model(xva)
            r_va = _diff_backtest(s_va, cva, cfg)
            sh_va = sharpe_like(r_va)
            # Ratio alignment: expect val cumret ~ train_cum * (val_len/train_len)
            ratio_target = float((len(X_val) if X_val is not None else 0) / max(1, len(X_train)))
            cr_va = cumret_like(r_va)
            ratio_err = (cr_va - ratio_target * cr_tr).abs()
        else:
            sh_va = torch.tensor(0.0, dtype=sh_tr.dtype)
            ratio_err = torch.tensor(0.0, dtype=sh_tr.dtype)
        # Smoothness penalty (TV on positions)
        pos_tr = torch.tanh(torch.tensor(cfg.beta, dtype=s_tr.dtype) * (s_tr - float(cfg.mid)))
        dpos = torch.abs(pos_tr[1:] - pos_tr[:-1]).mean() if pos_tr.numel() > 1 else torch.tensor(0.0, dtype=s_tr.dtype)
        # Trade frequency band penalty
        tf_pen = trade_freq_band_penalty(pos_tr)
        # Exposure encouragement (avoid trivial zero positions)
        exposure = torch.mean(torch.abs(pos_tr)) if pos_tr.numel() > 0 else torch.tensor(0.0, dtype=s_tr.dtype)
        exposure_shortfall = torch.relu(torch.tensor(cfg.exposure_target, dtype=s_tr.dtype) - exposure)
        # Loss: maximize train objectives + val Sharpe; penalize train>val gap and activity
        gap = torch.relu(sh_tr - sh_va)
        # Multi-term objective (negative for loss)
        neg_cum_pen = cfg.w_neg_cum_pen * torch.relu(-cr_tr)
        obj = cfg.w_sharpe * sh_tr + cfg.w_cum * cr_tr - cfg.w_tf_band * tf_pen - neg_cum_pen
        loss = (
            -(obj + cfg.w_val_sharpe * sh_va)
            + cfg.w_gap * gap
            + cfg.w_ratio * ratio_err
            + float(cfg.tv_penalty) * dpos
            + cfg.w_exposure * exposure_shortfall
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Final scores for reporting
    model.eval()
    with torch.no_grad():
        s_tr = model(xtr).cpu().numpy()
        if xva is not None:
            s_va = model(xva).cpu().numpy()
        else:
            s_va = None

    s_train = pd.Series(s_tr, index=X_train.index)
    pos_train = _discrete_positions_from_thresholds(s_train, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
    tm = metrics_from_positions(X_train[close_col], pos_train, cost_bps=float(cfg.cost_bps))

    val_metrics: Optional[Dict[str, float]] = None
    if X_val is not None and len(X_val) > 0 and s_va is not None:
        s_val = pd.Series(s_va, index=X_val.index)
        pos_val = _discrete_positions_from_thresholds(s_val, cfg, min_hold_candles=int(min_hold), cooldown_candles=int(cooldown))
        val_metrics = metrics_from_positions(X_val[close_col], pos_val, cost_bps=float(cfg.cost_bps))

    trades_tr = int(pos_train.ne(pos_train.shift(1)).sum())
    bars_tr = int(len(X_train))
    tf_train = (float(trades_tr) * 4000.0) / float(max(1, bars_tr))

    fp = [{
        "rank": 1,
        "fitness_train": float(tm.get("sharpe", 0.0)),
        "metrics_train": tm,
        "trades_train": trades_tr,
        "metrics_val": val_metrics or {},
        "trades_val": 0,
        "tree_size": 0,
        "tree_depth": 0,
        "tf_train": float(tf_train),
        "tf_val": float("nan"),
        "extras_train": {},
        "extras_val": {},
        "tree": {"entry_long": None, "entry_short": None, "exit_long": None, "exit_short": None},
    }]

    return {
        "train_metrics": tm,
        "val_metrics": val_metrics,
        "final_population": fp,
    }
