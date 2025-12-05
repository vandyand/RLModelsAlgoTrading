#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from features_loader import load_feature_panel
from model import hysteresis_map
from fitness import compute_portfolio_metrics
from backtester import _extract_trades as _bt_extract_trades
import importlib
import importlib.util
import sys
from pathlib import Path


# ---------- Minimal in-process reuse of AC trainer components ----------

@dataclass
class ACConfig:
    instruments: List[str]
    start_date: str | None
    end_date: str | None
    include_grans: List[str]
    epochs: int
    gamma: float
    actor_sigma: float
    entropy_coef: float
    lr: float
    hidden: int
    reward_scale: float
    max_grad_norm: float
    enter_long: float
    exit_long: float
    enter_short: float
    exit_short: float


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.policy(h)
        v = self.value(h).squeeze(-1)
        return h, mu, v


def _standardize_fit(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean()); s = float(X[c].std())
        if s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def _standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


def _flatten_panel(X_panel: pd.DataFrame, instruments: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    flat_cols: List[Tuple[str, str]] = []
    for inst in instruments:
        for col in X_panel[inst].columns:
            flat_cols.append((inst, col))
    X_flat = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
    X_flat.columns = [f"{i}::{c}" for i, c in X_flat.columns]
    col_order = [str(c) for c in X_flat.columns]
    return X_flat, col_order


def _greedy_eval(model: ActorCritic, Xn: pd.DataFrame, returns: pd.DataFrame, cfg: ACConfig) -> Dict[str, float]:
    device = next(model.parameters()).device
    X_t = torch.tensor(Xn.values, dtype=torch.float32, device=device)
    R_t = torch.tensor(returns.values, dtype=torch.float32, device=device)
    T = len(Xn)
    num_inst = returns.shape[1]
    pos_state = torch.zeros((num_inst,), dtype=torch.float32, device=device)
    greedy_reward_sum = 0.0
    greedy_steps = 0
    greedy_action_abs_sum = 0.0
    greedy_action_count = 0
    equity = 1.0
    for t in range(0, T - 1):
        s_t = X_t[t:t+1]
        r_tp1 = R_t[t+1]
        _, mu_t, _ = model(s_t)
        a_t = torch.sigmoid(mu_t)[0]
        enter_long = float(cfg.enter_long)
        exit_long = float(cfg.exit_long)
        enter_short = float(cfg.enter_short)
        exit_short = float(cfg.exit_short)
        new_state = pos_state.clone()
        flat_mask = (pos_state == 0.0)
        long_mask = (pos_state > 0.0)
        short_mask = (pos_state < 0.0)
        new_state = torch.where(flat_mask & (a_t > enter_long), torch.ones_like(new_state), new_state)
        new_state = torch.where(flat_mask & (a_t < enter_short), -torch.ones_like(new_state), new_state)
        new_state = torch.where(long_mask & (a_t < exit_long), torch.zeros_like(new_state), new_state)
        new_state = torch.where(short_mask & (a_t > exit_short), torch.zeros_like(new_state), new_state)
        r_scaled = float(cfg.reward_scale) * float(torch.mean(new_state * r_tp1).item())
        greedy_reward_sum += r_scaled
        greedy_steps += 1
        greedy_action_abs_sum += float(torch.mean(torch.abs(a_t)).item())
        greedy_action_count += 1
        step_ret_unscaled = float(torch.mean(new_state * r_tp1).item())
        equity *= (1.0 + step_ret_unscaled)
        pos_state = new_state
    greedy_reward_mean = (greedy_reward_sum / max(1, greedy_steps)) if greedy_steps > 0 else 0.0
    greedy_action_mean_abs = (greedy_action_abs_sum / max(1, greedy_action_count)) if greedy_action_count > 0 else 0.0
    greedy_cum_return = (equity - 1.0)
    return {
        "g_rew": float(greedy_reward_mean),
        "g_cum": float(greedy_cum_return),
        "g_aabs": float(greedy_action_mean_abs),
    }


def _portfolio_eval(
    model: ActorCritic,
    Xn: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    enter_long: float,
    exit_long: float,
    enter_short: float,
    exit_short: float,
    trade_cost: float = 0.0,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    X_full = torch.tensor(Xn.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, mu, _ = model(X_full)
        a = torch.sigmoid(mu).cpu().numpy()  # shape [T, N]
    pos_mat = hysteresis_map(
        a,
        enter_long=enter_long,
        exit_long=exit_long,
        enter_short=enter_short,
        exit_short=exit_short,
        mode="absolute",
        band_enter=0.05,
        band_exit=0.02,
    )
    positions = pd.DataFrame(pos_mat, index=Xn.index, columns=closes.columns)
    metrics = compute_portfolio_metrics(closes, positions, trade_cost)
    # Derive trade-level stats
    trades = _bt_extract_trades(positions, closes, trade_cost)
    pf = float("nan")
    n_win = n_loss = 0
    wl = wr = 0.0
    eq_r2 = 0.0
    if trades:
        profits = np.array([t.net_pnl for t in trades], dtype=float)
        gains = profits[profits > 0]
        losses = -profits[profits < 0]
        pf = float(gains.sum() / max(1e-12, losses.sum())) if losses.size > 0 else (float('inf') if gains.size > 0 else 0.0)
        n_win = int((profits > 0).sum())
        n_loss = int((profits < 0).sum())
        wl = float(n_win / max(1, n_loss)) if n_loss > 0 else (float('inf') if n_win > 0 else 0.0)
        wr = float(n_win / max(1, len(profits)))
        # Equity R^2
        px = closes.to_numpy(dtype=np.float32, copy=False)
        pos = positions.to_numpy(dtype=np.float32, copy=False)
        ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
        strat_ret = (pos[1:, :] * ret).mean(axis=1)
        if strat_ret.size > 1:
            eq = np.cumprod(1.0 + strat_ret)
            x = np.arange(eq.size, dtype=np.float64)
            y = eq.astype(np.float64)
            x_mean = x.mean(); y_mean = y.mean()
            cov = float(((x - x_mean) * (y - y_mean)).sum())
            varx = float(((x - x_mean) ** 2).sum())
            if varx > 1e-12:
                beta = cov / varx
                alpha = y_mean - beta * x_mean
                yhat = alpha + beta * x
                ss_res = float(((y - yhat) ** 2).sum())
                ss_tot = float(((y - y_mean) ** 2).sum())
                eq_r2 = float(1.0 - ss_res / max(1e-12, ss_tot)) if ss_tot > 1e-12 else 0.0
            else:
                eq_r2 = 0.0
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
        "equity_r2": float(eq_r2),
    }


def _fmt(x: float) -> str:
    val = float(x)
    if val == 0.0:
        return "0"
    mag = abs(val)
    if mag >= 1e-2 and mag < 1e3:
        return f"{val:.6f}".rstrip('0').rstrip('.')
    return f"{val:.3e}"


# ---------- Windowing ----------

def _base_from_grans(grans: List[str]) -> str:
    for g in ["M5", "H1", "D"]:
        if g in grans:
            return g
    return "M5"


def _add_offset(ts: pd.Timestamp, n: float, unit: str, base_gran: str = "M5") -> pd.Timestamp:
    """Add an offset of size n in given unit to timestamp ts.

    Supports fractional n for weeks/months via day approximation, and a 'steps'
    unit where one step equals the lowest granularity among selected features.
    """
    if unit == "steps":
        # Map steps to base granularity duration
        steps = max(1, int(round(float(n))))
        if base_gran == "H1":
            delta = pd.Timedelta(hours=steps)
        elif base_gran == "D":
            delta = pd.Timedelta(days=steps)
        else:  # M5 default
            delta = pd.Timedelta(minutes=5 * steps)
        return (ts + delta).normalize()
    if unit == "days":
        delta = pd.Timedelta(days=float(n))
        return (ts + delta).normalize()
    if unit == "weeks":
        # Support fractional weeks by converting to days
        delta = pd.Timedelta(days=7.0 * float(n))
        return (ts + delta).normalize()
    # months (default): approximate fractional months as 30 days each
    delta = pd.Timedelta(days=30.0 * float(n))
    return (ts + delta).normalize()


def generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_n: float,
    val_n: float,
    step_n: float,
    unit: str = "months",
    base_gran: str = "M5",
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur_train_start = start
    while True:
        cur_train_end = _add_offset(cur_train_start, float(train_n), unit, base_gran) - pd.Timedelta(seconds=1)
        cur_val_start = cur_train_end + pd.Timedelta(seconds=1)
        cur_val_end = _add_offset(cur_val_start, float(val_n), unit, base_gran) - pd.Timedelta(seconds=1)
        if cur_val_end > end:
            break
        windows.append((cur_train_start, cur_train_end, cur_val_start, cur_val_end))
        cur_train_start = _add_offset(cur_train_start, float(step_n), unit, base_gran)
    return windows


# ---------- Walk Forward Driver ----------

def _load_default_instruments() -> List[str]:
    # Try local package import first
    try:
        from instruments import DEFAULT_OANDA_20 as _DEF  # type: ignore
        return list(_DEF)
    except Exception:
        pass
    # Try absolute package name
    try:
        mod = importlib.import_module("ac-multi20.instruments".replace('-', '_'))
        return list(getattr(mod, "DEFAULT_OANDA_20"))
    except Exception:
        pass
    # Try loading by file path
    try:
        here = Path(__file__).resolve().parent
        inst_path = here / "instruments.py"
        if inst_path.exists():
            spec = importlib.util.spec_from_file_location("_wfo_inst_mod", str(inst_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["_wfo_inst_mod"] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                return list(getattr(mod, "DEFAULT_OANDA_20"))
    except Exception:
        pass
    # Hardcoded fallback (same as ac-multi20/instruments.py)
    return [
        "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF",
        "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
        "EUR_CHF", "EUR_AUD", "EUR_CAD", "GBP_CHF", "AUD_JPY",
        "AUD_CHF", "CAD_JPY", "NZD_JPY", "GBP_AUD", "AUD_NZD",
    ]


def run_wfo(args: argparse.Namespace) -> None:
    # Prepare run directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_dir = args.base_dir or os.path.join("ac-multi20", "runs")
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, f"wfo-ac-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Instruments
    if args.instruments.strip() and args.instruments.strip() != ",":
        instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    else:
        instruments = _load_default_instruments()
    if not instruments:
        raise RuntimeError("No instruments provided")

    # Granularities
    grans = [g.strip().upper() for g in (args.grans or "M5,H1,D").split(',') if g.strip()]
    base_gran = _base_from_grans(grans)

    # Windows
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    # Resolve unit precedence: --unit overrides flags; otherwise --weeks/--days select unit; default months
    unit = (args.unit.strip().lower() if args.unit else None)
    if not unit:
        if args.weeks:
            unit = "weeks"
        elif args.days:
            unit = "days"
        else:
            unit = "months"
    if unit not in {"months", "weeks", "days", "steps"}:
        unit = "months"
    windows = generate_windows(start_ts, end_ts, float(args.train_n), float(args.val_n), float(args.step_n), unit=unit, base_gran=base_gran)
    if int(args.windows_limit) > 0:
        windows = windows[: int(args.windows_limit)]

    meta = {"args": vars(args), "windows": len(windows)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    out_path = os.path.join(run_dir, "windows.jsonl")
    outf = open(out_path, "w")

    prev_model_state: Optional[Dict[str, Any]] = None
    init_model_state: Optional[Dict[str, Any]] = None
    if args.init_model:
        try:
            _ck = torch.load(args.init_model, map_location="cpu")
            init_model_state = _ck.get("model_state", None)
        except Exception:
            init_model_state = None

    for wi, (ts_tr_s, ts_tr_e, ts_v_s, ts_v_e) in enumerate(windows, start=1):
        # Load TRAIN panel
        X_train, closes_train = load_feature_panel(type("_Tmp", (), {
            "instruments": instruments,
            "lookback_days": 99999,  # ignored when start/end provided
            "start_date": ts_tr_s.isoformat(),
            "end_date": ts_tr_e.isoformat(),
            "data_root": "continuous-trader/data",
            "features_root": "continuous-trader/data/features",
            "include_grans": grans,
        })())
        if X_train.empty or len(X_train) < 3:
            outf.write(json.dumps({"event": "skip", "win": wi, "reason": "no_train_data"})+"\n"); outf.flush();
            continue
        # Flatten and returns
        Xf_train, col_order = _flatten_panel(X_train, instruments)
        px_train = closes_train.astype(float)
        r_train = np.log(px_train / px_train.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Align drop first
        Xf_train = Xf_train.iloc[1:]
        r_train = r_train.iloc[1:]
        Xn_train, stats = _standardize_fit(Xf_train)

        # Init model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = Xn_train.shape[1]
        num_inst = r_train.shape[1]
        model = ActorCritic(input_dim=input_dim, num_instruments=num_inst, hidden=int(args.hidden)).to(device)
        # Optionally initialize from previous window or provided init model
        init_mode = "fresh"
        if args.carry_forward and prev_model_state is not None:
            try:
                model.load_state_dict(prev_model_state, strict=False)
                init_mode = "carry"
            except Exception:
                init_mode = "fresh"
        elif init_model_state is not None:
            try:
                model.load_state_dict(init_model_state, strict=False)
                init_mode = "init_model"
            except Exception:
                init_mode = "fresh"
        print(json.dumps({"ph": "w_init", "win": wi, "mode": init_mode}), flush=True)
        opt = optim.Adam(model.parameters(), lr=float(args.lr))
        sigma = float(args.actor_sigma)
        log_sigma_const = float(np.log(max(1e-8, sigma)))

        # Train epochs
        T = len(Xn_train)
        X_t = torch.tensor(Xn_train.values, dtype=torch.float32, device=device)
        R_t = torch.tensor(r_train.values, dtype=torch.float32, device=device)
        for ep in range(int(args.epochs)):
            model.train()
            total_loss = total_actor = total_value = total_entropy = total_reward = 0.0
            steps = 0
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
                enter_long = float(args.enter_long)
                exit_long = float(args.exit_long)
                enter_short = float(args.enter_short)
                exit_short = float(args.exit_short)
                new_state = pos_state.clone()
                flat_mask = (pos_state == 0.0)
                long_mask = (pos_state > 0.0)
                short_mask = (pos_state < 0.0)
                new_state = torch.where(flat_mask & (a_t > enter_long), torch.ones_like(new_state), new_state)
                new_state = torch.where(flat_mask & (a_t < enter_short), -torch.ones_like(new_state), new_state)
                new_state = torch.where(long_mask & (a_t < exit_long), torch.zeros_like(new_state), new_state)
                new_state = torch.where(short_mask & (a_t > exit_short), torch.zeros_like(new_state), new_state)
                reward = float(args.reward_scale) * torch.mean(new_state * r_tp1)
                reward = reward + 0.0 * v_t.mean()
                adv = (reward + float(args.gamma) * v_tp1[0] - v_t[0]).detach()
                logprob = -0.5 * torch.sum(((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const, dim=1)
                entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))
                actor_loss = -adv * logprob.mean()
                value_target = (reward + float(args.gamma) * v_tp1[0]).detach()
                value_loss = 0.5 * (value_target - v_t[0]) ** 2
                loss = actor_loss + value_loss - float(args.entropy_coef) * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                opt.step()
                total_loss += float(loss.item()); total_actor += float(actor_loss.item()); total_value += float(value_loss.item())
                total_entropy += float(entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy)); total_reward += float(reward.item())
                steps += 1
                pos_state = new_state.detach()
            # Optional: print compact epoch summary for this window
            print(json.dumps({"ph":"w_train","win":wi,"ep":int(ep+1),"loss":_fmt(total_loss/max(1,steps)),"rew":_fmt(total_reward/max(1,steps))}), flush=True)

        # Train greedy metrics
        tr_g = _greedy_eval(model, Xn_train, r_train, ACConfig(
            instruments=instruments, start_date=None, end_date=None, include_grans=grans,
            epochs=args.epochs, gamma=args.gamma, actor_sigma=args.actor_sigma, entropy_coef=args.entropy_coef,
            lr=args.lr, hidden=args.hidden, reward_scale=args.reward_scale, max_grad_norm=args.max_grad_norm,
            enter_long=args.enter_long, exit_long=args.exit_long, enter_short=args.enter_short, exit_short=args.exit_short
        ))
        print(json.dumps({
            "ph": "w_train_g",
            "win": wi,
            "g_rew": _fmt(tr_g["g_rew"] + 0.0),
            "g_cum": _fmt(tr_g["g_cum"] + 0.0),
            "g_aabs": _fmt(tr_g["g_aabs"] + 0.0),
        }), flush=True)

        # Load VAL panel
        X_val, closes_val = load_feature_panel(type("_Tmp", (), {
            "instruments": instruments,
            "lookback_days": 99999,
            "start_date": ts_v_s.isoformat(),
            "end_date": ts_v_e.isoformat(),
            "data_root": "continuous-trader/data",
            "features_root": "continuous-trader/data/features",
            "include_grans": grans,
        })())
        if X_val.empty or len(X_val) < 3:
            outf.write(json.dumps({"event": "skip", "win": wi, "reason": "no_val_data"})+"\n"); outf.flush();
            continue
        # Flatten to train order and standardize with train stats
        Xf_val, _ = _flatten_panel(X_val, instruments)
        # Align and compute returns
        px_val = closes_val.astype(float)
        r_val = np.log(px_val / px_val.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        Xf_val = Xf_val.iloc[1:]
        r_val = r_val.iloc[1:]
        # Align cols to col_order
        for c in col_order:
            if c not in Xf_val.columns:
                Xf_val[c] = 0.0
        Xf_val = Xf_val[col_order]
        Xn_val = _standardize_apply(Xf_val, stats)

        # Greedy eval on validation
        val_g = _greedy_eval(model, Xn_val, r_val, ACConfig(
            instruments=instruments, start_date=None, end_date=None, include_grans=grans,
            epochs=args.epochs, gamma=args.gamma, actor_sigma=args.actor_sigma, entropy_coef=args.entropy_coef,
            lr=args.lr, hidden=args.hidden, reward_scale=args.reward_scale, max_grad_norm=args.max_grad_norm,
            enter_long=args.enter_long, exit_long=args.exit_long, enter_short=args.enter_short, exit_short=args.exit_short
        ))
        # Portfolio metrics on validation (trade-level)
        val_pf = _portfolio_eval(
            model,
            Xn_val,
            closes_val.loc[Xn_val.index],
            enter_long=float(args.enter_long),
            exit_long=float(args.exit_long),
            enter_short=float(args.enter_short),
            exit_short=float(args.exit_short),
            trade_cost=0.0,
        )
        print(json.dumps({
            "ph": "w_val",
            "win": wi,
            "g_rew": _fmt(val_g["g_rew"] + 0.0),
            "g_cum": _fmt(val_g["g_cum"] + 0.0),
            "g_aabs": _fmt(val_g["g_aabs"] + 0.0),
            "v_cum": _fmt(val_pf["cum_return"] + 0.0),
            "v_sh": _fmt(val_pf["sharpe"] + 0.0),
            "v_so": _fmt(val_pf["sortino"] + 0.0),
            "v_dd": _fmt(val_pf["max_dd"] + 0.0),
            "v_pf": _fmt(val_pf["profit_factor"] if np.isfinite(val_pf["profit_factor"]) else 0.0),
            "v_wr": _fmt(val_pf["win_rate"] + 0.0),
            "v_wl": _fmt(val_pf["win_loss"] if np.isfinite(val_pf["win_loss"]) else 0.0),
            "v_tr": int(val_pf["n_win"] + val_pf["n_loss"]),
            "v_r2": _fmt(val_pf["equity_r2"] + 0.0),
        }), flush=True)

        # Save per-window checkpoint (optional)
        if not args.no_save:
            ck_dir = os.path.join(run_dir, "checkpoints"); os.makedirs(ck_dir, exist_ok=True)
            ck_path = os.path.join(ck_dir, f"win{wi:03d}.pt")
            meta = {
                "input_dim": int(Xn_train.shape[1]),
                "num_instruments": int(len(instruments)),
                "instruments": list(instruments),
                "hidden": int(args.hidden),
                "grans": list(grans),
                "thresholds": {
                    "enter_long": float(args.enter_long),
                    "exit_long": float(args.exit_long),
                    "enter_short": float(args.enter_short),
                    "exit_short": float(args.exit_short),
                },
                "train_window": [ts_tr_s.isoformat(), ts_tr_e.isoformat()],
                "val_window": [ts_v_s.isoformat(), ts_v_e.isoformat()],
            }
            payload = {
                "model_state": model.state_dict(),
                "meta": meta,
                "feature_stats": stats,
                "col_order": col_order,
            }
            torch.save(payload, ck_path)
        # Save state dict for carry-forward if enabled
        if args.carry_forward:
            try:
                prev_model_state = model.state_dict()
            except Exception:
                prev_model_state = None

        rec = {
            "event": "window",
            "win": wi,
            "train": [ts_tr_s.isoformat(), ts_tr_e.isoformat()],
            "val": [ts_v_s.isoformat(), ts_v_e.isoformat()],
            "tr_g_rew": tr_g["g_rew"],
            "tr_g_cum": tr_g["g_cum"],
            "tr_g_aabs": tr_g["g_aabs"],
            "val_g_rew": val_g["g_rew"],
            "val_g_cum": val_g["g_cum"],
            "val_g_aabs": val_g["g_aabs"],
            "val_cum": val_pf["cum_return"],
            "val_sharpe": val_pf["sharpe"],
            "val_sortino": val_pf["sortino"],
            "val_max_dd": val_pf["max_dd"],
            "val_profit_factor": val_pf["profit_factor"],
            "val_win_rate": val_pf["win_rate"],
            "val_win_loss": val_pf["win_loss"],
            "val_trades": int(val_pf["n_win"] + val_pf["n_loss"]),
            "val_equity_r2": val_pf["equity_r2"],
        }
        outf.write(json.dumps(rec) + "\n"); outf.flush()

    outf.close()
    print(json.dumps({"run_dir": run_dir, "windows": len(windows), "out": out_path}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-Forward Optimization for AC Multi-20")
    p.add_argument("--start", required=True, help="Global start YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="Global end YYYY-MM-DD (UTC)")
    # Define window sizes either in months (default) or weeks (toggle)
    p.add_argument("--train-n", type=float, default=3.0, help="Training window size (in selected unit; supports decimals)")
    p.add_argument("--val-n", type=float, default=1.0, help="Validation window size (in selected unit; supports decimals)")
    p.add_argument("--step-n", type=float, default=1.0, help="Step size between windows (in selected unit; supports decimals)")
    p.add_argument("--weeks", action="store_true", help="Interpret sizes in weeks (deprecated; use --unit)")
    p.add_argument("--days", action="store_true", help="Interpret sizes in days (deprecated; use --unit)")
    p.add_argument("--unit", default="months", choices=["months", "weeks", "days", "steps"], help="Unit for window sizes: months (default), weeks, days, or steps (base gran)")
    p.add_argument("--windows-limit", type=int, default=0, help="Limit number of windows for quick runs")

    # Data/model config
    p.add_argument("--instruments", default=",")
    p.add_argument("--grans", default="M5,H1,D")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-sigma", type=float, default=0.3)
    p.add_argument("--entropy-coef", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # Threshold mapping
    p.add_argument("--enter-long", type=float, default=0.80)
    p.add_argument("--exit-long", type=float, default=0.60)
    p.add_argument("--enter-short", type=float, default=0.20)
    p.add_argument("--exit-short", type=float, default=0.40)

    # Output
    p.add_argument("--base-dir", default="ac-multi20/runs")
    p.add_argument("--no-save", action="store_true", help="Do not save per-window checkpoints")
    # Initialization and carry-forward
    p.add_argument("--carry-forward", action="store_true", help="Initialize each window from previous window's model")
    p.add_argument("--init-model", default="", help="Optional: initialize first window from this checkpoint path")

    args = p.parse_args()
    run_wfo(args)


if __name__ == "__main__":
    main()
