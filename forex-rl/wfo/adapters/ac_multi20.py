from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse existing AC utilities from ac-multi20
import os
import sys
from pathlib import Path

# Ensure ac-multi20 package path is importable regardless of CWD
_ROOT = Path(__file__).resolve().parents[2]
_AC_DIR = _ROOT / "ac-multi20"
if str(_AC_DIR) not in sys.path:
    sys.path.insert(0, str(_AC_DIR))

from features_loader import load_feature_panel  # type: ignore
from model import hysteresis_map  # type: ignore
from fitness import compute_portfolio_metrics  # type: ignore
from backtester import _extract_trades as _bt_extract_trades  # type: ignore


class _ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.policy(h)
        v = self.value(h).squeeze(-1)
        return h, mu, v


def _standardize_fit(X: pd.DataFrame):
    stats: Dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean()); s = float(X[c].std())
        if s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def _standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]):
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


def _flatten_panel(X_panel: pd.DataFrame, instruments: List[str]):
    flat_cols: List[Tuple[str, str]] = []
    for inst in instruments:
        for col in X_panel[inst].columns:
            flat_cols.append((inst, col))
    X_flat = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
    X_flat.columns = [f"{i}::{c}" for i, c in X_flat.columns]
    col_order = [str(c) for c in X_flat.columns]
    return X_flat, col_order


class ACMulti20Adapter:
    name = "ac-multi20"

    def __init__(
        self,
        *,
        instruments: str = ",",
        grans: str = "M5,H1,D",
        epochs: int = 5,
        gamma: float = 0.99,
        actor_sigma: float = 0.3,
        entropy_coef: float = 1e-3,
        lr: float = 1e-3,
        hidden: int = 256,
        reward_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        enter_long: float = 0.80,
        exit_long: float = 0.60,
        enter_short: float = 0.20,
        exit_short: float = 0.40,
        no_save: bool = True,
        carry_forward: bool = False,
        init_model: str = "",
    ) -> None:
        self.grans = [g.strip().upper() for g in grans.split(',') if g.strip()]
        self.instruments = self._load_instruments(instruments)
        self.epochs = int(epochs)
        self.gamma = float(gamma)
        self.actor_sigma = float(actor_sigma)
        self.entropy_coef = float(entropy_coef)
        self.lr = float(lr)
        self.hidden = int(hidden)
        self.reward_scale = float(reward_scale)
        self.max_grad_norm = float(max_grad_norm)
        self.enter_long = float(enter_long)
        self.exit_long = float(exit_long)
        self.enter_short = float(enter_short)
        self.exit_short = float(exit_short)
        self.carry_forward = bool(carry_forward)
        self.prev_state: Optional[Dict[str, Any]] = None
        self.init_state: Optional[Dict[str, Any]] = None
        if init_model:
            try:
                _ck = torch.load(init_model, map_location="cpu")
                self.init_state = _ck.get("model_state", None)
            except Exception:
                self.init_state = None

        self._train_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._col_order: Optional[List[str]] = None
        self._model: Optional[_ActorCritic] = None

    def _load_instruments(self, arg: str) -> List[str]:
        if arg.strip() and arg.strip() != ",":
            return [s.strip().upper() for s in arg.split(',') if s.strip()]
        # fallbacks
        try:
            from instruments import DEFAULT_OANDA_20  # type: ignore
            return list(DEFAULT_OANDA_20)
        except Exception:
            pass
        return [
            "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF",
            "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
            "EUR_CHF", "EUR_AUD", "EUR_CAD", "GBP_CHF", "AUD_JPY",
            "AUD_CHF", "CAD_JPY", "NZD_JPY", "GBP_AUD", "AUD_NZD",
        ]

    # Adapter API
    def load_window(self, start: pd.Timestamp, end: pd.Timestamp):
        X, closes = load_feature_panel(type("_Tmp", (), {
            "instruments": self.instruments,
            "lookback_days": 99999,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "data_root": "continuous-trader/data",
            "features_root": "continuous-trader/data/features",
            "include_grans": self.grans,
        })())
        return X, closes

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame):
        print(json.dumps({"event": "ac_fit_start", "rows": int(len(X)), "cols": int(X.shape[1])}), flush=True)
        Xf, col_order = _flatten_panel(X, self.instruments)
        px = closes.astype(float)
        r = np.log(px / px.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        Xf = Xf.iloc[1:]
        r = r.iloc[1:]
        Xn, stats = _standardize_fit(Xf)
        self._train_stats = stats
        self._col_order = col_order
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _ActorCritic(input_dim=Xn.shape[1], num_instruments=r.shape[1], hidden=self.hidden).to(device)
        if self.carry_forward and self.prev_state is not None:
            try:
                model.load_state_dict(self.prev_state, strict=False)
            except Exception:
                pass
        elif self.init_state is not None:
            try:
                model.load_state_dict(self.init_state, strict=False)
            except Exception:
                pass
        opt = optim.Adam(model.parameters(), lr=self.lr)
        sigma = float(self.actor_sigma)
        log_sigma_const = float(np.log(max(1e-8, sigma)))
        T = len(Xn)
        X_t = torch.tensor(Xn.values, dtype=torch.float32, device=device)
        R_t = torch.tensor(r.values, dtype=torch.float32, device=device)
        num_inst = r.shape[1]
        for ep in range(self.epochs):
            # Light progress beacon per epoch
            if ep == 0 or (ep + 1 == self.epochs):
                print(json.dumps({"event": "ac_epoch", "ep": int(ep + 1), "of": int(self.epochs)}), flush=True)
            model.train()
            pos_state = torch.zeros((num_inst,), dtype=torch.float32, device=device)
            for t in range(0, T - 1):
                s_t = X_t[t:t+1]
                s_tp1 = X_t[t+1:t+2]
                r_tp1 = R_t[t+1]
                _, mu_t, v_t = model(s_t)
                with torch.no_grad():
                    _, _, v_tp1 = model(s_tp1)
                eps = torch.randn_like(mu_t)
                pre = mu_t + sigma * eps
                a_t = torch.sigmoid(pre)[0]
                new_state = pos_state.clone()
                flat_mask = (pos_state == 0.0)
                long_mask = (pos_state > 0.0)
                short_mask = (pos_state < 0.0)
                new_state = torch.where(flat_mask & (a_t > self.enter_long), torch.ones_like(new_state), new_state)
                new_state = torch.where(flat_mask & (a_t < self.enter_short), -torch.ones_like(new_state), new_state)
                new_state = torch.where(long_mask & (a_t < self.exit_long), torch.zeros_like(new_state), new_state)
                new_state = torch.where(short_mask & (a_t > self.exit_short), torch.zeros_like(new_state), new_state)
                reward = float(self.reward_scale) * torch.mean(new_state * r_tp1)
                adv = (reward + float(self.gamma) * v_tp1[0] - v_t[0]).detach()
                logprob = -0.5 * torch.sum(((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const, dim=1)
                entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))
                actor_loss = -adv * logprob.mean()
                value_target = (reward + float(self.gamma) * v_tp1[0]).detach()
                value_loss = 0.5 * (value_target - v_t[0]) ** 2
                loss = actor_loss + value_loss - float(self.entropy_coef) * entropy
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), float(self.max_grad_norm)); opt.step()
                pos_state = new_state.detach()
        self._model = model
        print(json.dumps({"event": "ac_fit_done"}), flush=True)
        if self.carry_forward:
            try:
                self.prev_state = model.state_dict()
            except Exception:
                self.prev_state = None
        return model

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame):
        assert self._train_stats is not None and self._col_order is not None, "fit must be called first"
        Xf, _ = _flatten_panel(X, self.instruments)
        px = closes.astype(float)
        r = np.log(px / px.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        Xf = Xf.iloc[1:]
        r = r.iloc[1:]
        # align columns to train order
        for c in self._col_order:
            if c not in Xf.columns:
                Xf[c] = 0.0
        Xf = Xf[self._col_order]
        Xn = _standardize_apply(Xf, self._train_stats)
        device = next(model.parameters()).device
        with torch.no_grad():
            _, mu, _ = model(torch.tensor(Xn.values, dtype=torch.float32, device=device))
            a = torch.sigmoid(mu).cpu().numpy()
        pos_mat = hysteresis_map(
            a,
            enter_long=self.enter_long,
            exit_long=self.exit_long,
            enter_short=self.enter_short,
            exit_short=self.exit_short,
            mode="absolute",
            band_enter=0.05,
            band_exit=0.02,
        )
        positions = pd.DataFrame(pos_mat, index=Xn.index, columns=closes.columns)
        metrics = compute_portfolio_metrics(closes.loc[Xn.index], positions, 0.0)
        trades = _bt_extract_trades(positions, closes.loc[Xn.index], 0.0)
        pf = float("nan")
        n_win = n_loss = 0
        wl = wr = 0.0
        if trades:
            profits = np.array([t.net_pnl for t in trades], dtype=float)
            gains = profits[profits > 0]
            losses = -profits[profits < 0]
            pf = float(gains.sum() / max(1e-12, losses.sum())) if losses.size > 0 else (float('inf') if gains.size > 0 else 0.0)
            n_win = int((profits > 0).sum())
            n_loss = int((profits < 0).sum())
            wl = float(n_win / max(1, n_loss)) if n_loss > 0 else (float('inf') if n_win > 0 else 0.0)
            wr = float(n_win / max(1, len(profits)))
        return {
            "cum_return": float(metrics.cum_return),
            "sharpe": float(metrics.sharpe),
            "sortino": float(metrics.sortino),
            "max_dd": float(metrics.max_drawdown),
            "trades": float(metrics.trades),
            "time_in_mkt": float(metrics.time_in_market),
            "num_days": int(metrics.num_trade_days),
            "profit_factor": float(pf),
            "win_rate": float(wr),
            "win_loss": float(wl),
            "n_win": int(n_win),
            "n_loss": int(n_loss),
        }
