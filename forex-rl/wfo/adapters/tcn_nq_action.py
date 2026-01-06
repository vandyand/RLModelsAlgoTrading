from __future__ import annotations

"""NQ-specific TCN + MLP policy adapter.

This restores a *model-driven* NQ path while staying strictly causal:

- A TCN backbone consumes causal feature windows (same features as FX).
- A small MLP policy head outputs a continuous action a_t in (-1, 1).
- Hysteresis thresholds on a_t (config fields ``nq_enter_long`` / ``nq_exit_long`` /
  ``nq_enter_short`` / ``nq_exit_short``) are used to enter/exit long/short.
- Supervision is multi-task:
    * z-scored horizon log-return (regression target)
    * semantic action label in {-1, 0, +1} (regression target)

There is **no look-ahead in validation**: positions at time t depend only on
features up through t and model parameters learned on the training window.

Usage in JSON base config (unchanged):

    "adapter_spec": "wfo.adapters.tcn_nq_action:TCNActionNQAdapter"

Constructor signature and adapter_kwargs are identical to ``TCNActionAdapter``,
so existing GA / WFO wiring continues to work. FX paths are unaffected.
"""

from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from . import tcn_scalar_threshold as _base


# -----------------------------------------------------------------------------
# NQ-specific TCN backbone + policy model
# -----------------------------------------------------------------------------


class _NQBackboneTCN(nn.Module):
    """Lightweight causal TCN that produces a fixed-size feature vector.

    Architecture mirrors the FX TCN (causal conv blocks + pooling) but exposes
    the pooled hidden state instead of a scalar head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Reuse the same causal blocks as the scalar/head models.
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            dilation = 2**i
            blocks.append(_base._TCNBlock(hidden_channels, kernel_size, dilation, dropout))  # type: ignore[attr-defined]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        return h


class _NQPolicyHead(nn.Module):
    """MLP policy head mapping backbone features -> continuous action.

    Output domain is constrained to (-1, 1) via ``tanh`` so that thresholds in
    config are stable and interpretable.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        mlp_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        # Build a flexible MLP: hidden_dim -> ...mlp_layers... -> 1
        layer_sizes: List[int] = []
        if mlp_layers:
            layer_sizes = [int(x) for x in mlp_layers if int(x) > 0]
        modules: List[nn.Module] = []
        if layer_sizes:
            in_dim = hidden_dim
            for out_dim in layer_sizes:
                modules.append(nn.Linear(in_dim, out_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
                in_dim = out_dim
            modules.append(nn.Linear(in_dim, 1))
        else:
            # Fallback: original single-hidden-layer behavior
            modules.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                ]
            )
        self.net = nn.Sequential(*modules)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, hidden)
        a = self.net(feats).squeeze(-1)
        return torch.tanh(a)  # (B,)


class _NQPolicyModel(nn.Module):
    """Joint backbone + dual heads for NQ.

    - ``ret_head`` regresses z-scored horizon returns.
    - ``policy_head`` regresses semantic actions in {-1,0,+1}.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float,
        mlp_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.backbone = _NQBackboneTCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout=dropout,
        )
        self.ret_head = nn.Linear(hidden_channels, 1)
        self.policy_head = _NQPolicyHead(
            hidden_channels,
            dropout=dropout,
            mlp_layers=mlp_layers,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)  # (B, hidden)
        ret_pred = self.ret_head(feats).squeeze(-1)  # (B,)
        act_pred = self.policy_head(feats)  # (B,), in (-1,1)
        return feats, ret_pred, act_pred


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------


class TCNActionNQAdapter(_base.TCNActionAdapter):
    """NQ-only adapter using TCN backbone + MLP policy head.

    Training:
      - Uses the same causal feature construction and WFO geometry as FX.
      - Builds sequences of length ``lookback_bars`` and horizon ``target_horizon``.
      - Supervises on both:
          * z-scored horizon log-return (regression)
          * semantic action in {-1,0,+1} (regression)

    Validation:
      - Ignores the FX classifier head entirely.
      - Runs the NQ policy model to get actions a_t in (-1,1).
      - Applies NQ-specific hysteresis thresholds to produce positions
        {-1,0,+1} with no look-ahead.
    """

    # Distinct adapter name so runs are clearly separated on disk.
    name = "tcn-action-nq"

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        # Training-normalization state for NQ return regression.
        self._train_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._ret_mean: float = 0.0
        self._ret_std: float = 1.0
        # Mean action on the training window, used to de-bias policy outputs
        # so that actions on validation are roughly centered around zero.
        self._act_bias: float = 0.0
        # Last positions/trades for core.py to persist.
        self._last_positions: Optional[pd.DataFrame] = None
        self._last_trades: Optional[List[Any]] = None

    # ------------------------------------------------------------------
    # Sequence/target construction (shared between fit/validate)
    # ------------------------------------------------------------------

    def _build_sequences_targets(
        self,
        Xn: pd.DataFrame,
        closes: pd.DataFrame,
        train: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y_ret_z, y_act) for NQ.

        - X: (N, F, L) feature windows
        - y_ret_z: z-scored horizon log-returns (N,)
        - y_act: semantic actions in {-1,0,+1} (N,), as floats for regression
        """

        values = Xn.values  # (T, F)
        if self.cfg.instrument not in closes.columns:
            return (
                np.zeros((0, 0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        px = closes[self.cfg.instrument].astype(float).values  # (T,)
        T, F = values.shape
        L = int(self.cfg.lookback_bars)
        H = int(self.cfg.target_horizon)
        if T <= 0 or F <= 0 or L <= 0 or H <= 0 or T <= L + H:
            return (
                np.zeros((0, F, max(L, 1)), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        windows: List[np.ndarray] = []
        rets: List[float] = []
        for t in range(L, T - H):
            window = values[t - L : t, :]  # (L, F)
            r = math.log(max(px[t + H], 1e-12) / max(px[t], 1e-12))
            windows.append(window.T)
            rets.append(r)

        if not windows:
            return (
                np.zeros((0, F, L), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        rets_arr = np.asarray(rets, dtype=float)

        # Build semantic actions {-1,0,+1} using the same logic as the FX
        # classifier (bps or quantile modes).
        ys: List[float] = []
        if self.cfg.label_mode == "quantile":
            try:
                long_q = float(np.quantile(rets_arr, self.cfg.label_long_quantile))
                short_q = float(np.quantile(rets_arr, self.cfg.label_short_quantile))
            except Exception:
                long_q = float(np.quantile(rets_arr, 0.7))
                short_q = float(np.quantile(rets_arr, 0.3))
            # Fallback if quantiles are degenerate or inverted.
            if not math.isfinite(long_q) or not math.isfinite(short_q) or short_q >= long_q:
                long_q = float(np.percentile(rets_arr, 66))
                short_q = float(np.percentile(rets_arr, 33))

            for r in rets_arr:
                if r >= long_q:
                    ys.append(1.0)
                elif r <= short_q:
                    ys.append(-1.0)
                else:
                    ys.append(0.0)
        else:
            long_t = float(self.cfg.label_long_thresh)
            short_t = float(self.cfg.label_short_thresh)
            for r in rets_arr:
                if r >= long_t:
                    ys.append(1.0)
                elif r <= short_t:
                    ys.append(-1.0)
                else:
                    ys.append(0.0)

        X_arr = np.stack(windows).astype(np.float32)
        y_act = np.asarray(ys, dtype=np.float32)

        # Z-score horizon returns for regression stability.
        if train:
            mu = float(rets_arr.mean())
            sigma = float(rets_arr.std())
            if not math.isfinite(mu):
                mu = 0.0
            if not math.isfinite(sigma) or sigma < 1e-6:
                sigma = 1.0
            self._ret_mean = mu
            self._ret_std = sigma

        mu = self._ret_mean
        sigma = self._ret_std if self._ret_std > 0.0 else 1.0
        y_ret_z = ((rets_arr - mu) / sigma).astype(np.float32)
        return X_arr, y_ret_z, y_act

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:  # type: ignore[override]
        """Train the NQ TCN+MLP policy model.

        This entirely replaces the FX classifier's ``fit``; the base model is
        *not* used for NQ.
        """

        if X.empty or closes.empty:
            self._model = None
            self._train_stats = None
            return None

        # Standardize features using the shared helpers.
        Xn, stats = _base._standardize_fit(X)
        self._train_stats = stats

        X_arr, y_ret_z, y_act = self._build_sequences_targets(Xn, closes, train=True)
        if X_arr.shape[0] == 0:
            self._model = None
            return None

        B, C, L = X_arr.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _NQPolicyModel(
            in_channels=C,
            hidden_channels=self.cfg.hidden_channels,
            kernel_size=self.cfg.kernel_size,
            num_blocks=self.cfg.num_blocks,
            dropout=self.cfg.dropout,
            mlp_layers=getattr(self.cfg, "mlp_layers", None),
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        y_ret_t = torch.tensor(y_ret_z, dtype=torch.float32, device=device)
        y_act_t = torch.tensor(y_act, dtype=torch.float32, device=device)

        model.train()
        for _ep in range(int(self.cfg.epochs)):
            opt.zero_grad(set_to_none=True)
            _, pred_ret, pred_act = model(X_t)
            loss_ret = nn.functional.mse_loss(pred_ret, y_ret_t)
            loss_act = nn.functional.mse_loss(pred_act, y_act_t)
            loss = loss_ret + loss_act
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Estimate and store the average action over the training window so we
        # can de-bias actions at validation time. This keeps the policy from
        # drifting to a persistent long-only or short-only bias.
        model.eval()
        with torch.no_grad():
            feats_tr, _pred_ret_tr, pred_act_tr = model(X_t)
            act_bias = float(pred_act_tr.mean().item()) if pred_act_tr.numel() > 0 else 0.0
        if not math.isfinite(act_bias):
            act_bias = 0.0
        self._act_bias = act_bias

        self._model = model
        return model

    # ------------------------------------------------------------------
    # Validation / trading
    # ------------------------------------------------------------------

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:  # type: ignore[override]
        """Run NQ validation.

        - When ``debug_force_all_long`` is True, forces +1 positions on all bars
          to verify plumbing.
        - Otherwise, uses the NQ TCN+MLP policy with hysteresis thresholds.
        """

        # Debug path: force always-long positions to validate pipeline.
        if getattr(self.cfg, "debug_force_all_long", False):
            if X.empty or closes.empty:
                return _base.TCNActionAdapter.validate(self, None, X, closes)

            idx = closes.index
            positions = pd.Series(1.0, index=idx, name=self.cfg.instrument)
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            self._last_positions = pos_df

            trade_cost = float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0
            m = _base.compute_portfolio_metrics(closes, pos_df, trade_cost)
            trades = _base._bt_extract_trades(pos_df, closes, trade_cost)
            self._last_trades = trades

            long_trades = len([t for t in trades if t.side > 0])
            short_trades = len([t for t in trades if t.side < 0])
            long_win = len([t for t in trades if t.side > 0 and t.net_pnl > 0.0])
            long_loss = len([t for t in trades if t.side > 0 and t.net_pnl < 0.0])
            short_win = len([t for t in trades if t.side < 0 and t.net_pnl > 0.0])
            short_loss = len([t for t in trades if t.side < 0 and t.net_pnl < 0.0])

            if trades:
                profits = np.array([t.net_pnl for t in trades], dtype=float)
                gains = profits[profits > 0]
                losses = -profits[profits < 0]
                if losses.size > 0:
                    pf = float(gains.sum() / max(1e-12, losses.sum()))
                elif gains.size > 0:
                    pf = float("inf")
                else:
                    pf = 0.0
                n_win = int((profits > 0).sum())
                n_loss = int((profits < 0).sum())
                if n_loss > 0:
                    wl = float(n_win / max(1, n_loss))
                elif n_win > 0:
                    wl = float("inf")
                else:
                    wl = 0.0
                wr = float(n_win / max(1, len(profits)))
                mu = float(profits.mean()) if profits.size > 0 else 0.0
                sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
                trade_sharpe = mu / sigma if sigma > 1e-12 else 0.0
            else:
                pf = wl = wr = trade_sharpe = 0.0

            return {
                "cum_return": float(m.cum_return),
                "sharpe": float(trade_sharpe),
                "bar_sharpe": float(m.sharpe),
                "sortino": float(m.sortino),
                "max_dd": float(m.max_drawdown),
                "trades": float(len(trades)),
                "time_in_mkt": float(m.time_in_market),
                "win_rate": float(wr),
                "win_loss": float(wl),
                "profit_factor": float(pf),
                "equity_r2": float(m.equity_r2),
                "long_trades": float(long_trades),
                "short_trades": float(short_trades),
                "long_win_trades": float(long_win),
                "long_loss_trades": float(long_loss),
                "short_win_trades": float(short_win),
                "short_loss_trades": float(short_loss),
            }

        # Normal NQ policy path.
        if model is None or X.empty or closes.empty or self._train_stats is None:
            # Degenerate window: fall back to FX adapter's empty-metrics behavior.
            return _base.TCNActionAdapter.validate(self, None, X, closes)

        if self.cfg.instrument not in closes.columns:
            return _base.TCNActionAdapter.validate(self, None, X, closes)

        Xn = _base._standardize_apply(X, self._train_stats)
        X_arr, _y_ret_z, _y_act = self._build_sequences_targets(Xn, closes, train=False)

        idx = closes.index
        T = len(idx)
        positions = pd.Series(0.0, index=idx, name=self.cfg.instrument)

        if X_arr.shape[0] == 0:
            # Flat; metrics will be effectively zero; also zero trades => Sharpe = 0.
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            self._last_positions = pos_df
            trade_cost = float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0
            m = _base.compute_portfolio_metrics(closes, pos_df, trade_cost)
            trades = _base._bt_extract_trades(pos_df, closes, trade_cost)
            self._last_trades = trades

            # Diagnostics for degenerate windows
            acts_mean = acts_std = acts_min = acts_max = 0.0
            frac_gt_enter_long = frac_lt_enter_short = 0.0
            pos_frac_long = pos_frac_short = 0.0
            pos_frac_flat = 1.0

            pf = wl = wr = trade_sharpe = 0.0
            long_trades = short_trades = long_win = long_loss = short_win = short_loss = 0
            return {
                "cum_return": float(m.cum_return),
                "sharpe": float(trade_sharpe),
                "bar_sharpe": float(m.sharpe),
                "sortino": float(m.sortino),
                "max_dd": float(m.max_drawdown),
                "trades": float(len(trades)),
                "time_in_mkt": float(m.time_in_market),
                "win_rate": float(wr),
                "win_loss": float(wl),
                "profit_factor": float(pf),
                "equity_r2": float(m.equity_r2),
                "long_trades": float(long_trades),
                "short_trades": float(short_trades),
                "long_win_trades": float(long_win),
                "long_loss_trades": float(long_loss),
                "short_win_trades": float(short_win),
                "short_loss_trades": float(short_loss),
                # Action / position diagnostics
                "acts_mean": float(acts_mean),
                "acts_std": float(acts_std),
                "acts_min": float(acts_min),
                "acts_max": float(acts_max),
                "acts_frac_gt_enter_long": float(frac_gt_enter_long),
                "acts_frac_lt_enter_short": float(frac_lt_enter_short),
                "pos_frac_long": float(pos_frac_long),
                "pos_frac_short": float(pos_frac_short),
                "pos_frac_flat": float(pos_frac_flat),
                # No sequences => treat raw action as neutral.
                "last_action": 0.0,
            }

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
            _feats, _pred_ret, pred_act = model(X_t)
            # De-bias actions using the mean action observed on the training
            # window so that validation actions are approximately centered.
            act_bias = float(getattr(self, "_act_bias", 0.0) or 0.0)
            acts = (pred_act - act_bias).cpu().numpy().astype(float)  # (N_seq,)

        L = int(self.cfg.lookback_bars)
        enter_long = float(self.cfg.nq_enter_long)
        exit_long = float(self.cfg.nq_exit_long)
        enter_short = float(self.cfg.nq_enter_short)
        exit_short = float(self.cfg.nq_exit_short)

        for i, a in enumerate(acts):
            t = L + i
            if t >= T:
                break
            prev_pos = positions.iloc[t - 1] if t > 0 else 0.0
            new_pos = prev_pos

            if prev_pos == 0.0:
                if a >= enter_long:
                    new_pos = 1.0
                elif a <= enter_short:
                    new_pos = -1.0
            elif prev_pos > 0.0:
                if a <= exit_long:
                    new_pos = 0.0
            else:  # prev_pos < 0
                if a >= exit_short:
                    new_pos = 0.0

            positions.iloc[t] = new_pos

        # Carry the most recent position forward to the end of the window so
        # that the "current" position at the last bar reflects the latest
        # policy decision rather than reverting to flat purely due to the
        # target horizon geometry.
        positions = positions.ffill().fillna(0.0)

        # Diagnostics: distribution of actions and resulting positions
        try:
            acts_mean = float(np.mean(acts)) if acts.size > 0 else 0.0
            acts_std = float(np.std(acts)) if acts.size > 0 else 0.0
            acts_min = float(np.min(acts)) if acts.size > 0 else 0.0
            acts_max = float(np.max(acts)) if acts.size > 0 else 0.0
            frac_gt_enter_long = float(np.mean(acts >= enter_long)) if acts.size > 0 else 0.0
            frac_lt_enter_short = float(np.mean(acts <= enter_short)) if acts.size > 0 else 0.0
        except Exception:
            acts_mean = acts_std = acts_min = acts_max = 0.0
            frac_gt_enter_long = frac_lt_enter_short = 0.0

        pos_df = pd.DataFrame({self.cfg.instrument: positions})
        self._last_positions = pos_df

        trade_cost = float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0
        m = _base.compute_portfolio_metrics(closes, pos_df, trade_cost)
        trades = _base._bt_extract_trades(pos_df, closes, trade_cost)
        self._last_trades = trades

        long_trades = len([t for t in trades if t.side > 0])
        short_trades = len([t for t in trades if t.side < 0])
        long_win = len([t for t in trades if t.side > 0 and t.net_pnl > 0.0])
        long_loss = len([t for t in trades if t.side > 0 and t.net_pnl < 0.0])
        short_win = len([t for t in trades if t.side < 0 and t.net_pnl > 0.0])
        short_loss = len([t for t in trades if t.side < 0 and t.net_pnl < 0.0])

        pos_arr = pos_df.to_numpy(dtype=np.float32, copy=False)
        if pos_arr.size > 0:
            pos_frac_long = float((pos_arr > 0).mean())
            pos_frac_short = float((pos_arr < 0).mean())
            pos_frac_flat = float((pos_arr == 0).mean())
        else:
            pos_frac_long = pos_frac_short = 0.0
            pos_frac_flat = 1.0

        if trades:
            profits = np.array([t.net_pnl for t in trades], dtype=float)
            gains = profits[profits > 0]
            losses = -profits[profits < 0]
            if losses.size > 0:
                pf = float(gains.sum() / max(1e-12, losses.sum()))
            elif gains.size > 0:
                pf = float("inf")
            else:
                pf = 0.0
            n_win = int((profits > 0).sum())
            n_loss = int((profits < 0).sum())
            if n_loss > 0:
                wl = float(n_win / max(1, n_loss))
            elif n_win > 0:
                wl = float("inf")
            else:
                wl = 0.0
            wr = float(n_win / max(1, len(profits)))
            mu = float(profits.mean()) if profits.size > 0 else 0.0
            sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
            trade_sharpe = mu / sigma if sigma > 1e-12 else 0.0
        else:
            pf = wl = wr = trade_sharpe = 0.0

        last_action = float(acts[-1]) if acts.size > 0 else 0.0

        return {
            "cum_return": float(m.cum_return),
            "sharpe": float(trade_sharpe),
            "bar_sharpe": float(m.sharpe),
            "sortino": float(m.sortino),
            "max_dd": float(m.max_drawdown),
            "trades": float(len(trades)),
            "time_in_mkt": float(m.time_in_market),
            "win_rate": float(wr),
            "win_loss": float(wl),
            "profit_factor": float(pf),
            "equity_r2": float(m.equity_r2),
            "long_trades": float(long_trades),
            "short_trades": float(short_trades),
            "long_win_trades": float(long_win),
            "long_loss_trades": float(long_loss),
            "short_win_trades": float(short_win),
            "short_loss_trades": float(short_loss),
            # Action / position diagnostics
            "acts_mean": float(acts_mean),
            "acts_std": float(acts_std),
            "acts_min": float(acts_min),
            "acts_max": float(acts_max),
            "acts_frac_gt_enter_long": float(frac_gt_enter_long),
            "acts_frac_lt_enter_short": float(frac_lt_enter_short),
            "pos_frac_long": float(pos_frac_long),
            "pos_frac_short": float(pos_frac_short),
            "pos_frac_flat": float(pos_frac_flat),
            # Raw policy output on the most recent bar used for trading.
            "last_action": last_action,
        }
