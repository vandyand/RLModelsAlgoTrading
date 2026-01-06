from __future__ import annotations

"""
Lean strategy adapters for faster WFO experimentation.

Families:
- MLPActionAdapter   : small MLP over per-bar features -> {-1,0,+1} logits
- RuleActionAdapter  : simple hand-crafted momentum rule with thresholds

Both adapters:
- Reuse the scalar TCN feature construction for consistency
- Emit GA-style metrics (cum_return, sharpe, dd, trades, win/loss breakdown)
  so that existing compute_objective functions can be reused.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse GA backtester metrics
_ROOT = Path(__file__).resolve().parents[2]
_GA_MULTI20_DIR = _ROOT / "ga-multi20"
import sys  # noqa: E402

if str(_GA_MULTI20_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_MULTI20_DIR))

from fitness import compute_portfolio_metrics  # type: ignore  # noqa: E402
from backtester import _extract_trades as _bt_extract_trades  # type: ignore  # noqa: E402

from .tcn_scalar_threshold import (  # type: ignore
    TCNScalarConfig,
    TCNScalarThresholdAdapter,
    _standardize_fit,
    _standardize_apply,
)

# Optional flag to skip some heavier per-window diagnostics (e.g. equity R^2).
_LIGHT_METRICS = os.environ.get("WFO_LIGHT_METRICS", "0") == "1"


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _build_action_samples_and_labels(
    Xn: pd.DataFrame,
    closes: pd.DataFrame,
    instrument: str,
    lookback_bars: int,
    target_horizon: int,
    label_long_thresh: float,
    label_short_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-bar samples and discrete labels {-1,0,+1} encoded as {0,1,2}.

    Unlike the TCN adapters, this uses *single-bar* feature vectors (no history stack)
    to keep the model extremely small and fast.
    """
    values = Xn.values  # (T, F)
    px = closes[instrument].astype(float).values  # (T,)
    T, F = values.shape
    L = lookback_bars
    H = target_horizon
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for t in range(L, T - H):
        x_t = values[t, :]  # (F,)
        ret = math.log(max(px[t + H], 1e-12) / max(px[t], 1e-12))
        if ret >= label_long_thresh:
            y = 2  # long
        elif ret <= label_short_thresh:
            y = 0  # short
        else:
            y = 1  # flat
        xs.append(x_t)
        ys.append(y)
    if not xs:
        return np.zeros((0, F), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X_arr = np.stack(xs).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.int64)
    return X_arr, y_arr


def _compute_metrics_from_positions(
    closes: pd.DataFrame, positions: pd.DataFrame, trade_cost_bps: float
) -> Dict[str, float]:
    """Compute portfolio + trade-level metrics given positions and closes."""
    trade_cost = float(trade_cost_bps) / 1e4 if trade_cost_bps else 0.0
    m = compute_portfolio_metrics(closes, positions, trade_cost)
    trades = _bt_extract_trades(positions, closes, trade_cost)
    closed_trades = len(trades)

    pf = 0.0
    wl = 0.0
    wr = 0.0
    r2 = 0.0
    trade_sharpe = 0.0
    if trades:
        profits = np.array([t.net_pnl for t in trades], dtype=float)
        gains = profits[profits > 0]
        losses = -profits[profits < 0]
        if losses.size > 0:
            pf = float(gains.sum() / max(1e-12, losses.sum()))
        elif gains.size > 0:
            pf = float("inf")
        n_win = int((profits > 0).sum())
        n_loss = int((profits < 0).sum())
        if n_loss > 0:
            wl = float(n_win / max(1, n_loss))
        elif n_win > 0:
            wl = float("inf")
        wr = float(n_win / max(1, len(profits)))

        # Trade-based Sharpe: mean(net_pnl) / std(net_pnl) over trades
        mu = float(profits.mean()) if profits.size > 0 else 0.0
        sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
        if sigma > 1e-12:
            trade_sharpe = mu / sigma
        else:
            trade_sharpe = 0.0

        # Equity curve R^2 (can be skipped when light metrics are enabled)
        if not _LIGHT_METRICS:
            px = closes.to_numpy(dtype=np.float32, copy=False)
            pos_np = positions.to_numpy(dtype=np.float32, copy=False)
            ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
            strat_ret = (pos_np[1:, :] * ret).mean(axis=1)
            if strat_ret.size > 0:
                eq = np.cumprod(1.0 + strat_ret)
                x = np.arange(eq.size, dtype=np.float64)
                y = eq.astype(np.float64)
                x_mean = x.mean()
                y_mean = y.mean()
                cov = float(((x - x_mean) * (y - y_mean)).sum())
                varx = float(((x - x_mean) ** 2).sum())
                if varx > 1e-12:
                    beta = cov / varx
                    alpha = y_mean - beta * x_mean
                    yhat = alpha + beta * x
                    ss_res = float(((y - yhat) ** 2).sum())
                    ss_tot = float(((y - y_mean) ** 2).sum())
                    r2 = float(1.0 - ss_res / max(1e-12, ss_tot)) if ss_tot > 1e-12 else 0.0

        longs = [t for t in trades if t.side > 0]
        shorts = [t for t in trades if t.side < 0]
        long_trades = len(longs)
        short_trades = len(shorts)
        long_win = len([t for t in longs if t.net_pnl > 0.0])
        long_loss = len([t for t in longs if t.net_pnl < 0.0])
        short_win = len([t for t in shorts if t.net_pnl > 0.0])
        short_loss = len([t for t in shorts if t.net_pnl < 0.0])
    else:
        long_trades = short_trades = long_win = long_loss = short_win = short_loss = 0

    return {
        "cum_return": float(m.cum_return),
        # Sharpe is now trade-based; keep bar-based Sharpe separately if needed.
        "sharpe": float(trade_sharpe),
        "bar_sharpe": float(m.sharpe),
        "sortino": float(m.sortino),
        "max_dd": float(m.max_drawdown),
        "trades": float(closed_trades),
        "time_in_mkt": float(m.time_in_market),
        "win_rate": float(wr),
        "win_loss": float(wl),
        "profit_factor": float(pf),
        "equity_r2": float(r2),
        "long_trades": float(long_trades),
        "short_trades": float(short_trades),
        "long_win_trades": float(long_win),
        "long_loss_trades": float(long_loss),
        "short_win_trades": float(short_win),
        "short_loss_trades": float(short_loss),
    }


# -----------------------------------------------------------------------------
# MLP Action Adapter
# -----------------------------------------------------------------------------


@dataclass
class MLPActionConfig:
    instrument: str = "EUR_USD"
    grans: str = "M5,H1,D"
    base_gran: str = "M5"
    candle_cache_base: str = "http://127.0.0.1:9100"
    lookback_bars: int = 32
    target_horizon: int = 4
    epochs: int = 3
    lr: float = 1e-3
    hidden_dim: int = 32  # if <= 0, becomes linear baseline
    dropout: float = 0.1
    label_long_thresh: float = 0.0005
    label_short_thresh: float = -0.0005
    trade_cost_bps: float = 0.5


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_classes: int = 3) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # Pure linear baseline
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        return self.net(x)


class MLPActionAdapter:
    """Small MLP or linear action classifier over per-bar features.

    Labels are derived from forward returns as in TCNActionAdapter, but inputs are
    single-bar feature vectors instead of TCN sequences, making training very fast.
    """

    name = "mlp-action-classifier"

    def __init__(
        self,
        *,
        instrument: str = "EUR_USD",
        grans: str = "M5,H1,D",
        base_gran: str = "M5",
        candle_cache_base: str = "",
        lookback_bars: int = 32,
        target_horizon: int = 4,
        epochs: int = 3,
        lr: float = 1e-3,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        label_long_thresh: float = 0.0005,
        label_short_thresh: float = -0.0005,
        trade_cost_bps: float = 0.5,
    ) -> None:
        self.cfg = MLPActionConfig(
            instrument=instrument.upper(),
            grans=grans,
            base_gran=base_gran.upper(),
            candle_cache_base=candle_cache_base
            or os.environ.get("CANDLE_CACHE_BASE", "http://127.0.0.1:9100"),
            lookback_bars=int(lookback_bars),
            target_horizon=int(target_horizon),
            epochs=int(epochs),
            lr=float(lr),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            label_long_thresh=float(label_long_thresh),
            label_short_thresh=float(label_short_thresh),
            trade_cost_bps=float(trade_cost_bps),
        )
        self.grans: List[str] = [g.strip().upper() for g in self.cfg.grans.split(",") if g.strip()]
        if self.cfg.base_gran not in self.grans:
            self.grans = [self.cfg.base_gran] + self.grans
        self._train_stats: Dict[str, Tuple[float, float]] | None = None
        self._model: _MLPHead | None = None
        # For global trade logging
        self._last_positions: pd.DataFrame | None = None
        self._last_trades: List[Any] | None = None

    # ------------------------------------------------------------------
    # Adapter API
    # ------------------------------------------------------------------

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Reuse scalar adapter's feature construction
        scalar_cfg = TCNScalarConfig(
            instrument=self.cfg.instrument,
            grans=self.cfg.grans,
            base_gran=self.cfg.base_gran,
            candle_cache_base=self.cfg.candle_cache_base,
            lookback_bars=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
        )
        tmp = TCNScalarThresholdAdapter(
            instrument=scalar_cfg.instrument,
            grans=scalar_cfg.grans,
            base_gran=scalar_cfg.base_gran,
            candle_cache_base=scalar_cfg.candle_cache_base,
            lookback_bars=scalar_cfg.lookback_bars,
            target_horizon=scalar_cfg.target_horizon,
        )
        return tmp.load_window(start, end)

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        if X.empty or closes.empty:
            self._model = None
            return None
        Xn, stats = _standardize_fit(X)
        self._train_stats = stats

        X_arr, y_arr = _build_action_samples_and_labels(
            Xn,
            closes,
            instrument=self.cfg.instrument,
            lookback_bars=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
            label_long_thresh=self.cfg.label_long_thresh,
            label_short_thresh=self.cfg.label_short_thresh,
        )
        if X_arr.shape[0] == 0:
            self._model = None
            return None
        B, F = X_arr.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _MLPHead(
            in_dim=F,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_classes=3,
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_arr, dtype=torch.long, device=device)
        model.train()
        for ep in range(self.cfg.epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(X_t)
            loss = nn.functional.cross_entropy(logits, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        self._model = model
        return model

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        if model is None or X.empty or closes.empty or self._train_stats is None:
            return {
                "cum_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_dd": 0.0,
                "trades": 0.0,
                "time_in_mkt": 0.0,
                "win_rate": 0.0,
                "win_loss": 0.0,
                "profit_factor": 0.0,
                "equity_r2": 0.0,
                "long_trades": 0.0,
                "short_trades": 0.0,
                "long_win_trades": 0.0,
                "long_loss_trades": 0.0,
                "short_win_trades": 0.0,
                "short_loss_trades": 0.0,
            }
        Xn = _standardize_apply(X, self._train_stats)
        X_arr, y_arr = _build_action_samples_and_labels(
            Xn,
            closes,
            instrument=self.cfg.instrument,
            lookback_bars=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
            label_long_thresh=self.cfg.label_long_thresh,
            label_short_thresh=self.cfg.label_short_thresh,
        )
        idx = closes.index
        T = len(idx)
        positions = pd.Series(0.0, index=idx, name=self.cfg.instrument)
        if X_arr.shape[0] == 0:
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            self._last_positions = pos_df
            self._last_trades = []
            return _compute_metrics_from_positions(closes, pos_df, self.cfg.trade_cost_bps)

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
            logits = model(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # 0,1,2

        # Map from sample index to bar index: samples start at lookback bar
        L = self.cfg.lookback_bars
        for i, cls in enumerate(preds):
            t = L + i
            if t >= T:
                break
            if cls == 0:
                new_pos = -1.0
            elif cls == 1:
                new_pos = 0.0
            else:
                new_pos = 1.0
            positions.iloc[t] = new_pos

        pos_df = pd.DataFrame({self.cfg.instrument: positions})
        self._last_positions = pos_df
        # recompute trades for logging (metrics already computed trade stats internally)
        self._last_trades = _bt_extract_trades(pos_df, closes, float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0)
        return _compute_metrics_from_positions(closes, pos_df, self.cfg.trade_cost_bps)


# -----------------------------------------------------------------------------
# Rule-based adapter
# -----------------------------------------------------------------------------


@dataclass
class RuleActionConfig:
    instrument: str = "EUR_USD"
    grans: str = "M5,H1,D"
    base_gran: str = "M5"
    candle_cache_base: str = "http://127.0.0.1:9100"
    momentum_lookback: int = 48  # bars
    long_ret_bp: float = 5.0  # entry threshold in basis points
    short_ret_bp: float = 5.0
    trade_cost_bps: float = 0.5


class RuleActionAdapter:
    """Simple momentum-based rule adapter over single instrument.

    Rule (on base-gran closes):
      r_t = log(P_t / P_{t-L})
      if r_t > +long_ret_threshold  -> long
      elif r_t < -short_ret_threshold -> short
      else -> flat
    """

    name = "rule-action"

    def __init__(
        self,
        *,
        instrument: str = "EUR_USD",
        grans: str = "M5,H1,D",
        base_gran: str = "M5",
        candle_cache_base: str = "",
        momentum_lookback: int = 48,
        long_ret_bp: float = 5.0,
        short_ret_bp: float = 5.0,
        trade_cost_bps: float = 0.5,
    ) -> None:
        self.cfg = RuleActionConfig(
            instrument=instrument.upper(),
            grans=grans,
            base_gran=base_gran.upper(),
            candle_cache_base=candle_cache_base
            or os.environ.get("CANDLE_CACHE_BASE", "http://127.0.0.1:9100"),
            momentum_lookback=int(momentum_lookback),
            long_ret_bp=float(long_ret_bp),
            short_ret_bp=float(short_ret_bp),
            trade_cost_bps=float(trade_cost_bps),
        )
        self.grans: List[str] = [g.strip().upper() for g in self.cfg.grans.split(",") if g.strip()]
        if self.cfg.base_gran not in self.grans:
            self.grans = [self.cfg.base_gran] + self.grans
        # For global trade logging
        self._last_positions: pd.DataFrame | None = None
        self._last_trades: List[Any] | None = None

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Reuse scalar adapter's candle loader for closes
        scalar_cfg = TCNScalarConfig(
            instrument=self.cfg.instrument,
            grans=self.cfg.grans,
            base_gran=self.cfg.base_gran,
            candle_cache_base=self.cfg.candle_cache_base,
            lookback_bars=self.cfg.momentum_lookback,
            target_horizon=1,
        )
        tmp = TCNScalarThresholdAdapter(
            instrument=scalar_cfg.instrument,
            grans=scalar_cfg.grans,
            base_gran=scalar_cfg.base_gran,
            candle_cache_base=scalar_cfg.candle_cache_base,
            lookback_bars=scalar_cfg.lookback_bars,
            target_horizon=scalar_cfg.target_horizon,
        )
        X, closes = tmp.load_window(start, end)
        # We only need closes for rule; carry X through to satisfy interface.
        return X, closes

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        # No training; purely rule-based.
        return None

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        idx = closes.index
        T = len(idx)
        positions = pd.Series(0.0, index=idx, name=self.cfg.instrument)
        if T == 0:
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            self._last_positions = pos_df
            self._last_trades = []
            return _compute_metrics_from_positions(closes, pos_df, self.cfg.trade_cost_bps)

        px = closes[self.cfg.instrument].astype(float).values  # (T,)
        L = self.cfg.momentum_lookback
        long_thr = self.cfg.long_ret_bp / 1e4
        short_thr = self.cfg.short_ret_bp / 1e4

        for t in range(L, T):
            p_now = px[t]
            p_past = px[t - L]
            if p_past <= 0.0 or p_now <= 0.0:
                continue
            r = math.log(p_now / p_past)
            if r > long_thr:
                positions.iloc[t] = 1.0
            elif r < -short_thr:
                positions.iloc[t] = -1.0
            else:
                positions.iloc[t] = 0.0

        pos_df = pd.DataFrame({self.cfg.instrument: positions})
        self._last_positions = pos_df
        self._last_trades = _bt_extract_trades(pos_df, closes, float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0)
        return _compute_metrics_from_positions(closes, pos_df, self.cfg.trade_cost_bps)


