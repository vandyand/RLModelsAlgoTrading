#!/usr/bin/env python3
"""
Scalar Off+Online Actor-Critic (bi-directional) - Single Instrument

- Single scalar actor head in [0,1): maps to enter/exit decisions via thresholds
  - If y > 0.8 => enter long (+units) when flat
  - If y < 0.6 => exit long (flatten)
  - If y < 0.2 => enter short (-units) when flat
  - If y > 0.4 => exit short (flatten)
- Adds optional OFFLINE pretraining from REST candles using simple expert labels
  (MA/RSI) to avoid degeneracy (no-trade collapse), then ONLINE fine-tuning with
  micro-updates and optional DAgger-style expert supervision when uncertain.
- Fixed position size (?units), no TP/SL, no time-based closes.
- Episodic reward at close credited to entry (A2C update). Negative rewards are
  down-weighted by neg_reward_coef (default 0.1).
- Includes multi-horizon candle features: M1, M5, H1, D, W.

Environment:
  Requires OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY in environment.
"""
import argparse
import copy
from pathlib import Path
import json
import os
import sys
import time
import atexit
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.instruments as instruments_ep

# Reuse order helper
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from streamer.orders import place_market_order  # type: ignore
try:
    # Prefer local import from same directory
    from expert_baselines import combined_label  # type: ignore
except Exception:
    try:
        from forex_rl.actor_critic.expert_baselines import combined_label  # type: ignore
    except Exception:
        def combined_label(*args, **kwargs):
            return None


@dataclass
class Config:
    instrument: str
    environment: str = "practice"
    units: int = 100
    feature_ticks: int = 240
    reward_scale: float = 10000.0
    neg_reward_coef: float = 0.1
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    # Refresh cadence for instrument-level unrealized PnL sampling (online only)
    pnl_refresh_secs: float = 60.0
    lr: float = 1e-3
    entropy_coef: float = 0.001
    autosave_secs: float = 120.0
    model_path: str = ""
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_clip: float = 1.0
    adv_clip: float = 5.0
    gamma: float = 0.99
    # Thresholds for scalar decision
    enter_long_thresh: float = 0.8
    exit_long_thresh: float = 0.6
    enter_short_thresh: float = 0.2
    exit_short_thresh: float = 0.4
    flatten_on_start: bool = True
    flatten_on_exit: bool = True
    # Candle bars and refresh windows
    m1_bars: int = 300
    m5_bars: int = 300
    h1_bars: int = 60
    d_bars: int = 60
    w_bars: int = 60
    m1_refresh_secs: float = 5.0
    m5_refresh_secs: float = 30.0
    h1_refresh_secs: float = 60.0
    d_refresh_secs: float = 300.0
    w_refresh_secs: float = 3600.0


# ---------- Feature engineering (copied/trimmed from binary_enter_trader.py) ----------

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(arr, dtype=float)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14, eps: float = 1e-12) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:]) + eps
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class CandleCache:
    def __init__(self, api: API, instrument: str, h1_len: int, d_len: int, w_len: int) -> None:
        self.api = api
        self.instrument = instrument
        self.m1: Deque[Dict[str, Any]] = deque(maxlen=300)
        self.m5: Deque[Dict[str, Any]] = deque(maxlen=300)
        self.h1: Deque[Dict[str, Any]] = deque(maxlen=h1_len)
        self.d1: Deque[Dict[str, Any]] = deque(maxlen=d_len)
        self.w1: Deque[Dict[str, Any]] = deque(maxlen=w_len)
        self.last_m1_fetch: float = 0.0
        self.last_m5_fetch: float = 0.0
        self.last_h1_fetch: float = 0.0
        self.last_d_fetch: float = 0.0
        self.last_w_fetch: float = 0.0

    def _fetch(self, granularity: str, count: int) -> List[Dict[str, Any]]:
        try:
            req = instruments_ep.InstrumentsCandles(
                instrument=self.instrument,
                params={"granularity": granularity, "count": count, "price": "M"},
            )
            resp = self.api.request(req)
            out: List[Dict[str, Any]] = []
            for c in resp.get("candles", []):
                mid = c.get("mid") or {}
                out.append({
                    "time": c.get("time"),
                    "complete": bool(c.get("complete", False)),
                    "o": float(mid.get("o")) if mid.get("o") is not None else None,
                    "h": float(mid.get("h")) if mid.get("h") is not None else None,
                    "l": float(mid.get("l")) if mid.get("l") is not None else None,
                    "c": float(mid.get("c")) if mid.get("c") is not None else None,
                    "v": float(c.get("volume", 0.0)),
                })
            return out
        except Exception:
            return []

    def backfill(self, m1_count: int, m5_count: int, h1_count: int, d_count: int, w_count: int) -> None:
        if m1_count > 0:
            bars = self._fetch("M1", m1_count)
            self.m1.clear()
            for b in bars:
                self.m1.append(b)
            self.last_m1_fetch = time.time()
        if m5_count > 0:
            bars = self._fetch("M5", m5_count)
            self.m5.clear()
            for b in bars:
                self.m5.append(b)
            self.last_m5_fetch = time.time()
        if h1_count > 0:
            bars = self._fetch("H1", h1_count)
            self.h1.clear()
            for b in bars:
                self.h1.append(b)
            self.last_h1_fetch = time.time()
        if d_count > 0:
            bars = self._fetch("D", d_count)
            self.d1.clear()
            for b in bars:
                self.d1.append(b)
            self.last_d_fetch = time.time()
        if w_count > 0:
            bars = self._fetch("W", w_count)
            self.w1.clear()
            for b in bars:
                self.w1.append(b)
            self.last_w_fetch = time.time()

    def maybe_refresh(self, m1_refresh_secs: float, m5_refresh_secs: float, h1_refresh_secs: float, d_refresh_secs: float, w_refresh_secs: float) -> None:
        now = time.time()
        if (now - self.last_m1_fetch) >= m1_refresh_secs:
            bars = self._fetch("M1", 10)
            for b in bars[-10:]:
                if self.m1 and self.m1[-1]["time"] == b["time"]:
                    self.m1[-1] = b
                else:
                    self.m1.append(b)
            self.last_m1_fetch = now
        if (now - self.last_m5_fetch) >= m5_refresh_secs:
            bars = self._fetch("M5", 10)
            for b in bars[-10:]:
                if self.m5 and self.m5[-1]["time"] == b["time"]:
                    self.m5[-1] = b
                else:
                    self.m5.append(b)
            self.last_m5_fetch = now
        if (now - self.last_h1_fetch) >= h1_refresh_secs:
            bars = self._fetch("H1", 3)
            for b in bars[-3:]:
                if self.h1 and self.h1[-1]["time"] == b["time"]:
                    self.h1[-1] = b
                else:
                    self.h1.append(b)
            self.last_h1_fetch = now
        if (now - self.last_d_fetch) >= d_refresh_secs:
            bars = self._fetch("D", 3)
            for b in bars[-3:]:
                if self.d1 and self.d1[-1]["time"] == b["time"]:
                    self.d1[-1] = b
                else:
                    self.d1.append(b)
            self.last_d_fetch = now
        if (now - self.last_w_fetch) >= w_refresh_secs:
            bars = self._fetch("W", 3)
            for b in bars[-3:]:
                if self.w1 and self.w1[-1]["time"] == b["time"]:
                    self.w1[-1] = b
                else:
                    self.w1.append(b)
            self.last_w_fetch = now


class FeatureBuilder:
    def __init__(self, feature_ticks: int) -> None:
        self.feature_ticks = feature_ticks
        self.dom_depth = 5
        self.mid_window: Deque[float] = deque(maxlen=max(60, feature_ticks))
        self.spread_window: Deque[float] = deque(maxlen=200)
        self.eps = 1e-12
        self.last_microprice: Optional[float] = None

    def update_tick(self, bid: Optional[float], ask: Optional[float]) -> float:
        if bid is None and ask is None:
            return self.mid_window[-1] if self.mid_window else 0.0
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else (bid or ask or 0.0)
        self.mid_window.append(mid)
        if bid is not None and ask is not None:
            self.spread_window.append(max(self.eps, ask - bid))
        return mid

    def compute_tick_features(self) -> Optional[np.ndarray]:
        if len(self.mid_window) < 60:
            return None
        prices = np.array(self.mid_window, dtype=float)
        logp = np.log(prices + self.eps)
        r = np.diff(logp)

        def last_ratio(k: int) -> float:
            if len(prices) <= k:
                return 0.0
            return float(np.log((prices[-1] + self.eps) / (prices[-1 - k] + self.eps)))

        def roll_mean_std(window: int) -> Tuple[float, float]:
            if len(r) < window:
                return 0.0, 0.0
            seg = r[-window:]
            return float(np.mean(seg)), float(np.std(seg) + self.eps)

        r1 = last_ratio(1); r5 = last_ratio(5); r20 = last_ratio(20)
        m5, s5 = roll_mean_std(5)
        m20, s20 = roll_mean_std(20)
        m60, s60 = roll_mean_std(60)
        if len(prices) < 20:
            sma20 = prices[-1]; std20 = 1e-6
        else:
            sma20 = float(np.mean(prices[-20:])); std20 = float(np.std(prices[-20:]) + self.eps)
        z_sma20 = float((prices[-1] - sma20) / std20)
        if len(self.spread_window) >= 30:
            sp_arr = np.array(self.spread_window)
            sp = (sp_arr[-1] - sp_arr.mean()) / (sp_arr.std() + self.eps)
        else:
            sp = 0.0
        ema12 = ema(prices, 12); ema26 = ema(prices, 26)
        macd = ema12 - ema26; signal = ema(macd, 9); hist = macd - signal
        macd_last = float(macd[-1]); signal_last = float(signal[-1]); hist_last = float(hist[-1])
        rsi14 = rsi(prices, 14, self.eps)
        vol_of_vol = float(np.std(np.abs(r[-20:])) if len(r) >= 20 else 0.0)
        obv_tick = float(np.sum(np.sign(r[-60:]))) / 60.0 if len(r) > 0 else 0.0
        feats = np.array([
            1.0,
            r1, r5, r20,
            m5, s5,
            m20, s20,
            m60, s60,
            z_sma20,
            s60,
            sp,
            macd_last, signal_last, hist_last,
            rsi14 / 100.0,
            vol_of_vol,
            obv_tick,
        ], dtype=float)
        return np.clip(feats, -5.0, 5.0)

    def compute_dom_features(self, bids: List[Dict[str, Any]], asks: List[Dict[str, Any]]) -> np.ndarray:
        k = self.dom_depth
        eps = self.eps
        bid_px: List[float] = []
        bid_liq: List[float] = []
        ask_px: List[float] = []
        ask_liq: List[float] = []
        for i in range(min(k, len(bids))):
            try:
                bid_px.append(float(bids[i]["price"]))
            except Exception:
                bid_px.append(0.0)
            v = bids[i].get("liquidity") if i < len(bids) else None
            bid_liq.append(float(v) if v is not None else 0.0)
        for i in range(min(k, len(asks))):
            try:
                ask_px.append(float(asks[i]["price"]))
            except Exception:
                ask_px.append(0.0)
            v = asks[i].get("liquidity") if i < len(asks) else None
            ask_liq.append(float(v) if v is not None else 0.0)
        while len(bid_px) < k: bid_px.append(0.0)
        while len(ask_px) < k: ask_px.append(0.0)
        while len(bid_liq) < k: bid_liq.append(0.0)
        while len(ask_liq) < k: ask_liq.append(0.0)

        sb = float(np.sum(bid_liq)); sa = float(np.sum(ask_liq)); total = sb + sa + eps
        imb = (sb - sa) / total
        top_bid = bid_px[0] if bid_px else 0.0
        top_ask = ask_px[0] if ask_px else 0.0
        mid = (top_bid + top_ask) / 2.0 if (top_bid and top_ask) else (top_bid or top_ask)
        micro = 0.0
        if total > eps and top_bid and top_ask:
            micro = (top_ask * sb + top_bid * sa) / (sb + sa + eps)
        micro_delta = (micro - (self.last_microprice or micro))
        self.last_microprice = micro
        avg_bid = float(np.average(bid_px, weights=bid_liq)) if sb > 0 else 0.0
        avg_ask = float(np.average(ask_px, weights=ask_liq)) if sa > 0 else 0.0
        tilt = avg_bid - avg_ask
        liq_ratio = sb / (sb + sa + eps)
        dist_bid = [max(0.0, mid - px) for px in bid_px]
        dist_ask = [max(0.0, px - mid) for px in ask_px]
        pressure = (float(np.sum(np.array(dist_bid) * np.array(bid_liq))) - float(np.sum(np.array(dist_ask) * np.array(ask_liq)))) / (total)
        bid_share = (np.array(bid_liq) / (sb + eps)).tolist()
        ask_share = (np.array(ask_liq) / (sa + eps)).tolist()
        if mid != 0.0:
            bid_dnorm = ((mid - np.array(bid_px)) / abs(mid)).tolist()
            ask_dnorm = ((np.array(ask_px) - mid) / abs(mid)).tolist()
        else:
            bid_dnorm = [0.0] * k
            ask_dnorm = [0.0] * k
        vec = [imb, tilt, micro, micro_delta, sb, sa, liq_ratio, pressure]
        vec += bid_share + ask_share + bid_dnorm + ask_dnorm
        return np.array(vec, dtype=float)


def candle_features(bars: Deque[Dict[str, Any]], eps: float = 1e-12) -> np.ndarray:
    if len(bars) < 2:
        return np.zeros(16, dtype=float)
    close = np.array([b.get("c") or 0.0 for b in bars], dtype=float)
    high = np.array([b.get("h") or 0.0 for b in bars], dtype=float)
    low = np.array([b.get("l") or 0.0 for b in bars], dtype=float)
    vol = np.array([b.get("v") or 0.0 for b in bars], dtype=float)
    logc = np.log(np.maximum(close, eps))
    r = np.diff(logc)

    def last_ret(k: int) -> float:
        if len(close) <= k:
            return 0.0
        return float(np.log((close[-1] + eps)/(close[-1 - k] + eps)))

    def std_ret(k: int) -> float:
        if len(r) < k:
            return 0.0
        return float(np.std(r[-k:]) + eps)

    tr_list: List[float] = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_list.append(tr)
    atr20 = float(np.mean(tr_list[-20:])) if len(tr_list) >= 1 else 0.0

    def dist_max(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.max(close[-k:]))/(np.std(close[-k:]) + eps))

    def dist_min(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.min(close[-k:]))/(np.std(close[-k:]) + eps))

    if len(vol) >= 20:
        v20 = (vol[-1] - np.mean(vol[-20:]))/(np.std(vol[-20:]) + eps)
    else:
        v20 = 0.0

    ema12c = ema(close, 12); ema26c = ema(close, 26)
    macd = ema12c - ema26c; signal = ema(macd, 9); hist = macd - signal
    macd_last = float(macd[-1]); signal_last = float(signal[-1]); hist_last = float(hist[-1])
    rsi14 = rsi(close, 14, eps)
    vol_of_vol = float(np.std(np.abs(r[-20:])) if len(r) >= 20 else 0.0)

    obv = [0.0]
    for i in range(1, len(close)):
        delta = vol[i] if close[i] > close[i-1] else (-vol[i] if close[i] < close[i-1] else 0.0)
        obv.append(obv[-1] + delta)
    obv_arr = np.array(obv)
    if len(obv_arr) >= 20:
        obv_z = (obv_arr[-1] - np.mean(obv_arr[-20:])) / (np.std(obv_arr[-20:]) + eps)
    else:
        obv_z = 0.0

    feats = np.array([
        last_ret(1), last_ret(5), last_ret(20),
        std_ret(5), std_ret(20), std_ret(60),
        atr20,
        dist_max(20), dist_min(20),
        v20,
        macd_last, signal_last, hist_last,
        rsi14 / 100.0,
        vol_of_vol,
        obv_z,
    ], dtype=float)
    return np.clip(feats, -5.0, 5.0)


# ---------- Network ----------

class ActorCriticScalarNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, value_maxabs: float = 5.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.actor_raw = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.value_maxabs = float(value_maxabs)
        # Init: Kaiming for encoder; zero heads for neutral start (y?0.5, v?0)
        def _init_linear(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for layer in self.encoder:
            _init_linear(layer)
        nn.init.zeros_(self.actor_raw.weight); nn.init.zeros_(self.actor_raw.bias)
        nn.init.zeros_(self.critic.weight); nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.norm(h)
        a_raw = self.actor_raw(h).squeeze(-1)
        a_raw = torch.clamp(a_raw, -5.0, 5.0)  # keep logits in a sane range
        v_raw = self.critic(h).squeeze(-1)
        v = torch.tanh(v_raw) * self.value_maxabs
        return h, a_raw, v


# ---------- Helpers ----------

def fetch_nav(api: API, account_id: str) -> Optional[float]:
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        return float(resp.get("account", {}).get("NAV"))
    except Exception:
        return None


def refresh_units(api: API, account_id: str, instrument: str) -> int:
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for p in resp.get("positions", []):
            if p.get("instrument") == instrument:
                long_u = float((p.get("long") or {}).get("units") or 0.0)
                short_u = float((p.get("short") or {}).get("units") or 0.0)
                return int(round(long_u + short_u))
    except Exception:
        pass
    return 0


def flatten_position(api: API, account_id: str, instrument: str, client_tag: str) -> Optional[Dict[str, Any]]:
    units = refresh_units(api, account_id, instrument)
    if units != 0:
        try:
            order = place_market_order(
                api=api, account_id=account_id, instrument=instrument,
                units=-units, tp_pips=None, sl_pips=None,
                anchor=None, client_tag=client_tag, client_comment="auto flatten",
                fifo_safe=False, fifo_adjust=False,
            )
            return _summarize_order(order)
        except Exception as exc:
            print(json.dumps({"type": "FLATTEN_ERROR", "error": str(exc)}), flush=True)
    return None


def _summarize_order(order_obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = (order_obj or {}).get("response", {})
        create = resp.get("orderCreateTransaction", {})
        fill = resp.get("orderFillTransaction", {})
        instrument = create.get("instrument") or fill.get("instrument")
        out = {
            "instrument": instrument,
            "order_id": create.get("id"),
            "fill_id": fill.get("id"),
            "time": fill.get("time") or create.get("time"),
            "units": fill.get("units") or create.get("units"),
            "price": fill.get("price"),
            "pl": fill.get("pl"),
            "balance": fill.get("accountBalance"),
            "reason": fill.get("reason") or create.get("reason"),
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception:
        return {"raw": str(order_obj)[:300]}


def fetch_instrument_unrealized_pl(api: API, account_id: str, instrument: str) -> float:
    """Fetch current unrealized P/L for the given instrument from OANDA OpenPositions.
    Returns 0.0 on error or if no open position.
    """
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for p in resp.get("positions", []):
            if p.get("instrument") == instrument:
                lp = float((p.get("long") or {}).get("unrealizedPL") or 0.0)
                sp = float((p.get("short") or {}).get("unrealizedPL") or 0.0)
                return float(lp + sp)
    except Exception:
        pass
    return 0.0


def _compute_grad_norm(parameters: Any) -> float:
    total_sq = 0.0
    try:
        for p in parameters:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float(g.norm(2).item() ** 2)
    except Exception:
        return 0.0
    return float(np.sqrt(total_sq))

@dataclass
class TradeState:
    open: bool = False
    entry_features: Optional[np.ndarray] = None
    entry_action_long: Optional[int] = None  # 1 for long entry, 0 for short entry
    entry_logit: Optional[float] = None
    entry_v: Optional[float] = None
    entry_nav: Optional[float] = None


# ---------- Offline pretraining (candlestick only) ----------

def _fetch_ohlcv(api: API, instrument: str, granularity: str, count: int) -> List[Dict[str, Any]]:
    try:
        req = instruments_ep.InstrumentsCandles(
            instrument=instrument,
            params={"granularity": granularity, "count": int(count), "price": "M"},
        )
        resp = api.request(req)
        out: List[Dict[str, Any]] = []
        for c in resp.get("candles", []):
            mid = c.get("mid") or {}
            out.append({
                "time": c.get("time"),
                "complete": bool(c.get("complete", False)),
                "o": float(mid.get("o")) if mid.get("o") is not None else None,
                "h": float(mid.get("h")) if mid.get("h") is not None else None,
                "l": float(mid.get("l")) if mid.get("l") is not None else None,
                "c": float(mid.get("c")) if mid.get("c") is not None else None,
                "v": float(c.get("volume", 0.0)),
            })
        return out
    except Exception as _e:
        try:
            print(json.dumps({
                "type": "PREFETCH_ERROR",
                "instrument": instrument,
                "granularity": granularity,
                "count": int(count),
                "error": str(_e)
            }), flush=True)
        except Exception:
            pass
        return []


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def build_offline_examples_from_ohlcv(ohlcv: List[Dict[str, Any]], gran: str, expert: str, expert_kwargs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y_labels) for offline pretraining.
    - X: features matching live model input (including pos_pnl placeholder as last dim), so length = 138.
      The final element (pos_pnl) is set to 0.0 for offline data.
    - Y: labels in [0,1] from expert; rows with None labels are dropped.
    """
    # Prepare close array for labels
    close = [float(b.get("c") or 0.0) for b in ohlcv]
    # Build candle features per bar using same function as live for consistency
    # Reuse candle_features() from this module; it accepts a deque of dict bars
    bars: Deque[Dict[str, Any]] = deque(maxlen=max(2, len(ohlcv)))
    X_rows: List[np.ndarray] = []
    Y_rows: List[float] = []
    for i, b in enumerate(ohlcv):
        bars.append(b)
        if len(bars) < 2:
            continue
        # Decide which block to fill
        m1 = _zeros(16); m5 = _zeros(16); h1 = _zeros(16); d1 = _zeros(16); w1 = _zeros(16)
        feats = candle_features(bars)
        if gran == "M1":
            m1 = feats
        elif gran == "M5":
            m5 = feats
        elif gran == "H1":
            h1 = feats
        elif gran == "D":
            d1 = feats
        elif gran == "W":
            w1 = feats
        # Other blocks: tick/dom zero; pos_pnl placeholder 0.0 at the end
        x = np.concatenate([
            _zeros(19),  # tick
            _zeros(28),  # dom
            m1, m5, h1, d1, w1,
            np.zeros(10, dtype=np.float32),  # time cyclical placeholder
            np.array([0.0], dtype=np.float32),  # pos_pnl placeholder for offline
        ]).astype(np.float32)
        # Expert label uses close up to i
        y = combined_label(close[: i + 1], method=expert, **expert_kwargs)
        if y is None:
            continue
        X_rows.append(x)
        Y_rows.append(float(y))
    if len(X_rows) == 0:
        return np.zeros((0, 138), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(X_rows).astype(np.float32)
    Y = np.array(Y_rows, dtype=np.float32)
    return X, Y


def offline_pretrain(
    api: API,
    net: ActorCriticScalarNet,
    instrument: str,
    granularity: str,
    count: int,
    expert: str,
    expert_kwargs: Dict[str, Any],
    epochs: int,
    lr: float = 1e-3,
    validation_split: float = 0.2,
    min_output_std: float = 0.10,
    early_stop_patience: int = 3,
) -> int:
    """Lightweight offline pretraining on single-instrument OHLCV with expert labels.
    Returns number of supervised samples used.
    """
    data = _fetch_ohlcv(api, instrument, granularity, count)
    if not data:
        return 0
    X, Y = build_offline_examples_from_ohlcv(data, granularity, expert, expert_kwargs)
    if X.shape[0] == 0:
        return 0
    net.train()
    opt = optim.Adam(net.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    bs = 256
    N = X.shape[0]
    # Train/val split
    split = int(max(1, min(N - 1, round(N * (1.0 - float(validation_split))))))
    X_tr = X[:split]; Y_tr = Y[:split]
    X_va = X[split:]; Y_va = Y[split:]
    best_state = copy.deepcopy(net.state_dict())
    best_val = float('inf')
    no_improve = 0
    for ep in range(max(1, int(epochs))):
        idx = np.random.permutation(X_tr.shape[0])
        for s in range(0, X_tr.shape[0], bs):
            sel = idx[s : s + bs]
            xb = torch.from_numpy(X_tr[sel])
            yb = torch.from_numpy(Y_tr[sel])
            _, a_raw, _ = net(xb)
            loss = bce(a_raw, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        # Validation metrics
        if X_va.shape[0] > 0:
            with torch.no_grad():
                _, a_val, _ = net(torch.from_numpy(X_va))
                yhat = torch.sigmoid(a_val).cpu().numpy()
                val_loss = float(bce(a_val, torch.from_numpy(Y_va)).item())
                avg_out = float(np.mean(yhat))
                std_out = float(np.std(yhat))
            print(json.dumps({"type": "PRETRAIN_VAL", "epoch": ep + 1, "val_bce": val_loss, "avg_output": round(avg_out, 4), "output_std": round(std_out, 4)}), flush=True)
            # Hard stop on collapse
            if std_out < float(min_output_std):
                print(json.dumps({"type": "PRETRAIN_COLLAPSE", "reason": "low_output_variance", "output_std": round(std_out,4)}), flush=True)
                # restore best weights before exiting
                try:
                    net.load_state_dict(best_data_state := best_state)  # noqa: F841
                except Exception:
                    pass
                break
            # Early stopping on validation degradation
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(early_stop_patience):
                    print(json.dumps({"type": "PRETRAIN_EARLY_STOP", "epoch": ep + 1, "best_val": best_val}), flush=True)
                    try:
                        net.load_state_dict(best_state)
                    except Exception:
                        pass
                    break
    return int(N)


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scalar Off+Online Actor-Critic (bi-directional)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--account-suffix", type=int, default=None, help="Override last 3 digits of account id (e.g., 4 -> '004')")
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--feature-ticks", type=int, default=240)
    # Candle bar counts and refresh (overrides defaults in Config if provided)
    parser.add_argument("--m1-bars", type=int, default=300)
    parser.add_argument("--m5-bars", type=int, default=300)
    parser.add_argument("--h1-bars", type=int, default=60)
    parser.add_argument("--d-bars", type=int, default=60)
    parser.add_argument("--w-bars", type=int, default=60)
    parser.add_argument("--m1-refresh-secs", type=float, default=5.0)
    parser.add_argument("--m5-refresh-secs", type=float, default=30.0)
    parser.add_argument("--h1-refresh-secs", type=float, default=60.0)
    parser.add_argument("--d-refresh-secs", type=float, default=300.0)
    parser.add_argument("--w-refresh-secs", type=float, default=3600.0)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--neg-reward-coef", type=float, default=0.1)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    parser.add_argument("--pnl-refresh-secs", type=float, default=60.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/scalar_threshold_v001.pt")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Load on start only when explicitly requested
    parser.add_argument("--load-on-start", dest="load_on_start", action="store_true")
    parser.add_argument("--no-load-on-start", dest="load_on_start", action="store_false")
    parser.set_defaults(load_on_start=False)
    # Exploration knobs for scalar head (adds Gaussian noise to logit with prob eps)
    parser.add_argument("--explore-eps", type=float, default=0.05)
    parser.add_argument("--explore-sigma", type=float, default=0.75)
    parser.add_argument("--explore-sigma-open", type=float, default=0.25, help="Exploration sigma while a position is open")
    # Tick-level shaping reward (micro updates)
    parser.add_argument("--micro-update", action="store_true")
    parser.add_argument("--no-micro-update", dest="micro_update", action="store_false")
    parser.set_defaults(micro_update=True)
    parser.add_argument("--micro-reward-scale", type=float, default=100.0)
    parser.add_argument("--micro-actor-coef", type=float, default=0.02)
    parser.add_argument("--micro-value-coef", type=float, default=0.05)
    parser.add_argument("--micro-flat-bias", type=float, default=0.2, help="Scale factor for flat-state micro reward: bias * mid_delta")
    # Anti-churn gating and smoothing
    parser.add_argument("--y-ema-alpha", type=float, default=0.3, help="EMA smoothing for y used in decisions")
    parser.add_argument("--min-hold-ticks", type=int, default=6, help="Minimum PRICE ticks to hold before allowing exit")
    parser.add_argument("--consec-enter", type=int, default=2, help="Require N consecutive enter signals to open")
    parser.add_argument("--consec-exit", type=int, default=2, help="Require N consecutive exit signals to close")
    # OU noise (temporally correlated exploration)
    parser.add_argument("--ou-noise", action="store_true")
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-sigma", type=float, default=0.2)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    # Actor init bias (logit) to avoid neutral deadlock
    parser.add_argument("--init-logit-bias", type=float, default=0.0)
    # Curriculum warmup thresholds and window
    parser.add_argument("--warmup-seconds", type=float, default=120.0)
    parser.add_argument("--warmup-enter-long", type=float, default=0.6)
    parser.add_argument("--warmup-exit-long", type=float, default=0.5)
    parser.add_argument("--warmup-enter-short", type=float, default=0.4)
    parser.add_argument("--warmup-exit-short", type=float, default=0.5)
    # Tick-level commitment bonus toward non-neutral outputs
    parser.add_argument("--commit-reward-coef", type=float, default=0.001)
    parser.add_argument("--enter-long-thresh", type=float, default=0.8)
    parser.add_argument("--exit-long-thresh", type=float, default=0.6)
    parser.add_argument("--enter-short-thresh", type=float, default=0.2)
    parser.add_argument("--exit-short-thresh", type=float, default=0.4)
    parser.add_argument("--flatten-on-start", action="store_true")
    parser.add_argument("--no-flatten-on-start", dest="flatten_on_start", action="store_false")
    parser.set_defaults(flatten_on_start=True)
    parser.add_argument("--flatten-on-exit", action="store_true")
    parser.add_argument("--no-flatten-on-exit", dest="flatten_on_exit", action="store_false")
    parser.set_defaults(flatten_on_exit=True)
    # Offline pretraining options (candlestick only)
    parser.add_argument("--pretrain", action="store_true", help="Run a brief offline pretraining before streaming")
    parser.add_argument("--pretrain-granularity", choices=["D", "H1", "W", "M5", "M1"], default="D")
    parser.add_argument("--pretrain-count", type=int, default=1000)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--pretrain-expert", choices=["ma", "rsi", "ma_filt"], default="ma_filt")
    parser.add_argument("--pretrain-ma-fast", type=int, default=12)
    parser.add_argument("--pretrain-ma-slow", type=int, default=26)
    parser.add_argument("--pretrain-neutral-band", type=float, default=0.0002, help="Relative neutral band for expert (e.g., |ema diff|/price)")
    parser.add_argument("--pretrain-validation-split", type=float, default=0.2)
    parser.add_argument("--pretrain-min-output-std", type=float, default=0.10)
    parser.add_argument("--pretrain-early-stop-patience", type=int, default=3)
    parser.add_argument("--pretrain-only", action="store_true", help="Run offline pretraining then exit without starting live stream or flattening")
    parser.add_argument("--no-pretrain-only", dest="pretrain_only", action="store_false", help="After pretraining, continue into live streaming")
    parser.set_defaults(pretrain_only=True)
    parser.add_argument("--save-after-pretrain", action="store_true")
    parser.add_argument("--save-with-timestamp", action="store_true", help="Append UTC timestamp to model_path when saving pretrain snapshot")
    parser.add_argument("--no-save-with-timestamp", dest="save_with_timestamp", action="store_false")
    parser.set_defaults(save_with_timestamp=True)
    # DAgger-style supervision during live (expert guidance when uncertain)
    parser.add_argument("--uncertain-band", type=float, default=0.05, help="If |y_ema-0.5|<=band, consult expert baseline")
    parser.add_argument("--expert-baseline", choices=["none", "ma", "rsi", "ma_filt"], default="ma_filt")
    parser.add_argument("--expert-fast", type=int, default=12)
    parser.add_argument("--expert-slow", type=int, default=26)
    parser.add_argument("--expert-neutral-band", type=float, default=0.0002)
    parser.add_argument("--expert-confidence-threshold", type=float, default=0.15)
    parser.add_argument("--dagger-supervise", action="store_true", help="Apply supervised BCE step toward expert when consulted")
    parser.add_argument("--dagger-weight", type=float, default=0.1)
    parser.add_argument("--dagger-weight-decay", type=float, default=0.999)
    parser.add_argument("--dagger-weight-min", type=float, default=0.01)
    # KL regularization against pretrained policy to prevent forgetting
    parser.add_argument("--kl-coef", type=float, default=0.005)
    # Exploration decay
    parser.add_argument("--explore-sigma-decay", type=float, default=0.9999)
    parser.add_argument("--explore-sigma-min", type=float, default=0.02)
    # Training stats heartbeat
    parser.add_argument("--train-stats-secs", type=float, default=30.0)
    # Optional more aggressive thresholds to increase trading frequency
    parser.add_argument("--aggressive-thresholds", action="store_true", help="Use 0.7/0.55 long and 0.3/0.45 short thresholds")
    args = parser.parse_args()

    cfg = Config(
        instrument=args.instrument,
        environment=args.environment,
        units=args.units,
        feature_ticks=args.feature_ticks,
        reward_scale=args.reward_scale,
        neg_reward_coef=args.neg_reward_coef,
        nav_poll_secs=args.nav_poll_secs,
        pos_refresh_secs=args.pos_refresh_secs,
        pnl_refresh_secs=args.pnl_refresh_secs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        autosave_secs=args.autosave_secs,
        model_path=args.model_path,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        adv_clip=args.adv_clip,
        gamma=args.gamma,
        enter_long_thresh=args.enter_long_thresh,
        exit_long_thresh=args.exit_long_thresh,
        enter_short_thresh=args.enter_short_thresh,
        exit_short_thresh=args.exit_short_thresh,
        flatten_on_start=args.flatten_on_start,
        flatten_on_exit=args.flatten_on_exit,
        m1_bars=args.m1_bars,
        m5_bars=args.m5_bars,
        h1_bars=args.h1_bars,
        d_bars=args.d_bars,
        w_bars=args.w_bars,
        m1_refresh_secs=args.m1_refresh_secs,
        m5_refresh_secs=args.m5_refresh_secs,
        h1_refresh_secs=args.h1_refresh_secs,
        d_refresh_secs=args.d_refresh_secs,
        w_refresh_secs=args.w_refresh_secs,
    )

    # Optional thresholds override for more activity
    if bool(getattr(args, "aggressive_thresholds", False)):
        cfg.enter_long_thresh = 0.7
        cfg.exit_long_thresh = 0.55
        cfg.enter_short_thresh = 0.3
        cfg.exit_short_thresh = 0.45

    base_account = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not base_account or not access_token:
        raise RuntimeError("Missing OANDA credentials in env vars.")
    # Resolve account id with optional 3-digit suffix override
    if args.account_suffix is not None:
        if len(str(base_account)) < 3:
            raise RuntimeError("Account id must be at least 3 chars to use --account-suffix")
        account_id = str(base_account)[:-3] + f"{int(args.account_suffix):03d}"
    else:
        account_id = str(base_account)
    api = API(access_token=access_token, environment=args.environment)

    fb = FeatureBuilder(cfg.feature_ticks)
    ccache = CandleCache(api, args.instrument, h1_len=cfg.h1_bars, d_len=cfg.d_bars, w_len=cfg.w_bars)
    ccache.backfill(cfg.m1_bars, cfg.m5_bars, cfg.h1_bars, cfg.d_bars, cfg.w_bars)

    # Input dim: tick(19) + dom(28) + M1(16) + M5(16) + H1(16) + D(16) + W(16) + time(10) + pos_pnl(1) = 138
    input_dim = 138

    net = ActorCriticScalarNet(input_dim=input_dim, hidden_dim=128, value_maxabs=5.0)
    # Apply actor logit bias if requested
    try:
        if float(args.init_logit_bias) != 0.0:
            with torch.no_grad():
                net.actor_raw.bias.fill_(float(args.init_logit_bias))
    except Exception:
        pass
    net.train()
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    pretrained_net: Optional[ActorCriticScalarNet] = None

    # Optionally load checkpoint on start (explicit opt-in only)
    if bool(args.load_on_start) and args.model_path:
        try:
            if os.path.exists(args.model_path):
                ck = torch.load(args.model_path, map_location="cpu")
                state = ck.get("model") if isinstance(ck, dict) else None
                if state:
                    # Guard against input dimension mismatch (skip load if incompatible)
                    try:
                        w = state.get("encoder.0.weight") if hasattr(state, "get") else None
                        if w is not None and int(getattr(w, "shape")[1]) != int(input_dim):
                            print(json.dumps({
                                "type": "LOAD_SKIPPED",
                                "reason": "input_dim_mismatch",
                                "ckpt_dim": int(getattr(w, "shape")[1]),
                                "model_dim": int(input_dim),
                                "path": args.model_path,
                            }), flush=True)
                        else:
                            net.load_state_dict(state, strict=False)
                            print(json.dumps({"type": "LOADED", "path": args.model_path}), flush=True)
                    except Exception:
                        # Fallback to best-effort load
                        net.load_state_dict(state, strict=False)
                        print(json.dumps({"type": "LOADED", "path": args.model_path}), flush=True)
                # Optional: load optimizer
                if isinstance(ck, dict) and ck.get("opt"):
                    try:
                        opt.load_state_dict(ck["opt"])  # type: ignore[index]
                    except Exception:
                        pass
        except Exception as exc:
            print(json.dumps({"type": "LOAD_ERROR", "error": str(exc)}), flush=True)

    # Optional: offline pretraining prior to streaming
    if bool(args.pretrain):
        try:
            # Offline pretraining with validation
            used = offline_pretrain(
                api=api,
                net=net,
                instrument=args.instrument,
                granularity=str(args.pretrain_granularity),
                count=int(args.pretrain_count),
                expert=str(args.pretrain_expert),
                expert_kwargs={
                    "fast": int(args.pretrain_ma_fast),
                    "slow": int(args.pretrain_ma_slow),
                    "neutral_band": float(args.pretrain_neutral_band),
                },
                epochs=int(args.pretrain_epochs),
                lr=float(cfg.lr),
                validation_split=float(args.pretrain_validation_split),
                min_output_std=float(args.pretrain_min_output_std),
                early_stop_patience=int(args.pretrain_early_stop_patience),
            )
            print(json.dumps({"type": "PRETRAIN", "samples": used, "gran": args.pretrain_granularity}), flush=True)
            # Freeze a reference copy for KL regularization
            pretrained_net = copy.deepcopy(net)
            for p in pretrained_net.parameters():
                p.requires_grad = False
            pretrained_net.eval()
            if bool(args.save_after_pretrain) and cfg.model_path:
                try:
                    if used and used > 0:
                        base_path = Path(cfg.model_path)
                        if bool(args.save_with_timestamp):
                            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            suffix = ''.join(base_path.suffixes)
                            name = f"{base_path.stem}_{ts}{suffix if suffix else ''}"
                            save_path = base_path.with_name(name)
                        else:
                            save_path = base_path
                        os.makedirs(save_path.parent, exist_ok=True)
                        torch.save({"model": net.state_dict()}, str(save_path))
                        print(json.dumps({"type": "SAVED", "path": str(save_path)}), flush=True)
                    else:
                        print(json.dumps({"type": "PRETRAIN_SKIP_SAVE", "reason": "no_samples"}), flush=True)
                except Exception as e:
                    print(json.dumps({"type": "SAVE_ERROR", "error": str(e)}), flush=True)
        except Exception as exc:
            print(json.dumps({"type": "PRETRAIN_ERROR", "error": str(exc)}), flush=True)
        # Exit early if only offline pretraining was requested
        if bool(args.pretrain_only):
            print(json.dumps({"type": "PRETRAIN_DONE", "samples": used, "exiting": True}), flush=True)
            return

    # Optional: flatten any residual position on start
    if cfg.flatten_on_start:
        summ = flatten_position(api, account_id, args.instrument, client_tag="scal-thr")
        if summ is not None:
            print(json.dumps({"type": "AUTO_FLATTEN_START", "order": summ}), flush=True)

    # State
    st = TradeState(open=False)
    last_step_y: Optional[float] = None
    last_step_v: Optional[float] = None
    last_step_y_ema: Optional[float] = None
    last_nav = fetch_nav(api, account_id) or 1.0
    last_nav_poll = time.time()
    last_pos_refresh = time.time()
    last_pnl_refresh = time.time()
    current_pos_pnl: float = 0.0
    current_units = refresh_units(api, account_id, args.instrument)
    # Training stats trackers
    stats_y: Deque[float] = deque(maxlen=600)
    last_stats_emit: float = time.time()
    last_grad_norm: Optional[float] = None
    last_kl_pen: Optional[float] = None
    last_dagger_w: Optional[float] = None
    # Decision hysteresis counters and tick tracking
    consec_enter_long = 0
    consec_enter_short = 0
    consec_exit_long = 0
    consec_exit_short = 0
    ticks_open = 0

    # Ensure flatten on exit if requested
    if cfg.flatten_on_exit:
        def _on_exit() -> None:
            try:
                summ2 = flatten_position(api, account_id, args.instrument, client_tag="scal-thr")
                if summ2 is not None:
                    print(json.dumps({"type": "AUTO_FLATTEN_EXIT", "order": summ2}), flush=True)
            except Exception:
                pass
        atexit.register(_on_exit)

    # Resilient streaming loop with auto-reconnect
    # OU noise state
    ou_noise_val: float = 0.0
    start_ts = time.time()
    # If no offline pretrain occurred but a checkpoint loaded, still create a frozen copy for KL
    if pretrained_net is None:
        try:
            pretrained_net = copy.deepcopy(net)
            for p in pretrained_net.parameters():
                p.requires_grad = False
            pretrained_net.eval()
        except Exception:
            pretrained_net = None
    # Exploration decay bookkeeping
    step_idx: int = 0
    base_ou_sigma = float(args.ou_sigma)
    base_gauss_sigma = float(args.explore_sigma)
    base_gauss_sigma_open = float(args.explore_sigma_open)
    while True:
        try:
            stream = pricing.PricingStream(accountID=account_id, params={"instruments": args.instrument})
            for msg in api.request(stream):
                tnow = time.time()
                mtype = msg.get("type")
                if mtype == "HEARTBEAT":
                    if (tnow - last_nav_poll) >= cfg.nav_poll_secs:
                        nv = fetch_nav(api, account_id)
                        if nv is not None:
                            last_nav = nv
                        last_nav_poll = tnow
                    if (tnow - last_pos_refresh) >= cfg.pos_refresh_secs:
                        current_units = refresh_units(api, account_id, args.instrument)
                        last_pos_refresh = tnow
                    # refresh instrument-level unrealized P/L every pnl_refresh interval
                    if (tnow - last_pnl_refresh) >= getattr(cfg, 'pnl_refresh_secs', 60.0):
                        current_pos_pnl = fetch_instrument_unrealized_pl(
                            api, account_id, args.instrument
                        ) if current_units != 0 else 0.0
                        last_pnl_refresh = tnow
                    # Periodic refresh of candles
                    ccache.maybe_refresh(cfg.m1_refresh_secs, cfg.m5_refresh_secs, cfg.h1_refresh_secs, cfg.d_refresh_secs, cfg.w_refresh_secs)
                    hb: Dict[str, Any] = {
                        "type": "HB",
                        "nav": round(float(last_nav), 6) if isinstance(last_nav, (int, float)) else last_nav,
                        "units": int(current_units),
                        "open": bool(st.open),
                    }
                    if last_step_y is not None:
                        hb["y"] = round(float(last_step_y), 4)
                    if last_step_v is not None:
                        hb["v"] = round(float(last_step_v), 4)
                    hb["pos_pnl"] = round(float(current_pos_pnl), 6)
                    # Periodic training stats
                    if (tnow - last_stats_emit) >= float(getattr(args, 'train_stats_secs', 30.0)):
                        try:
                            y_mean = float(np.mean(stats_y)) if len(stats_y) > 0 else None
                            y_std = float(np.std(stats_y)) if len(stats_y) > 0 else None
                            print(json.dumps({
                                "type": "TRAIN_STATS",
                                "y_mean": round(y_mean, 4) if y_mean is not None else None,
                                "y_std": round(y_std, 4) if y_std is not None else None,
                                "grad_norm": round(float(last_grad_norm), 4) if last_grad_norm is not None else None,
                                "kl_pen": round(float(last_kl_pen), 6) if last_kl_pen is not None else None,
                                "dagger_w": round(float(last_dagger_w), 6) if last_dagger_w is not None else None,
                            }), flush=True)
                        except Exception:
                            pass
                        last_stats_emit = tnow
                    print(json.dumps(hb), flush=True)
                    continue
                if mtype != "PRICE":
                    continue

                bids = msg.get("bids") or []
                asks = msg.get("asks") or []
                try:
                    bid = float(bids[0]["price"]) if bids else None
                except Exception:
                    bid = None
                try:
                    ask = float(asks[0]["price"]) if asks else None
                except Exception:
                    ask = None

                mid = fb.update_tick(bid, ask)
                x_tick = fb.compute_tick_features()
                x_dom = fb.compute_dom_features(bids, asks)
                m1_feats = candle_features(ccache.m1)
                m5_feats = candle_features(ccache.m5)
                h1_feats = candle_features(ccache.h1)
                d1_feats = candle_features(ccache.d1)
                w1_feats = candle_features(ccache.w1)
                if x_tick is None:
                    continue
                # Time features (UTC) from message time if present
                try:
                    ts_str = msg.get("time") or msg.get("timestamp")
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                t_feats = time_cyclical_features(dt)
                x = np.concatenate([
                    x_tick, x_dom, m1_feats, m5_feats, h1_feats, d1_feats, w1_feats, t_feats,
                    np.array([current_pos_pnl], dtype=np.float32)
                ]).astype(np.float32)
                # Feature sanity
                if not np.all(np.isfinite(x)):
                    print(json.dumps({"type": "WARN", "msg": "nonfinite features"}), flush=True)
                    continue
                xt = torch.from_numpy(x)[None, :]

                # Inference
                net.train(False)
                with torch.no_grad():
                    _, a_raw_step, v_step = net(xt)
                a_val = float(a_raw_step[0].item())
                # Exploration noise: OU preferred if enabled, else Gaussian epsilon
                try:
                    if bool(args.ou_noise):
                        theta = float(args.ou_theta); dt = float(args.ou_dt)
                        sigma = max(float(args.explore_sigma_min), base_ou_sigma * (float(args.explore_sigma_decay) ** step_idx))
                        # mean-reverting to 0
                        ou_noise_val = ou_noise_val + theta * (0.0 - ou_noise_val) * dt + sigma * np.sqrt(dt) * float(np.random.randn())
                        a_val = a_val + ou_noise_val
                    elif float(args.explore_eps) > 0.0 and float(args.explore_sigma) > 0.0:
                        if float(np.random.rand()) < float(args.explore_eps):
                            decayed = max(float(args.explore_sigma_min), base_gauss_sigma * (float(args.explore_sigma_decay) ** step_idx))
                            decayed_open = max(float(args.explore_sigma_min), base_gauss_sigma_open * (float(args.explore_sigma_decay) ** step_idx))
                            sigma_eff = decayed_open if (current_units != 0) else decayed
                            a_val = a_val + float(np.random.normal(loc=0.0, scale=sigma_eff))
                except Exception:
                    pass
                y_step = float(1.0 / (1.0 + np.exp(-a_val)))  # sigmoid(a_val) in (0,1)
                last_step_y = y_step
                try:
                    stats_y.append(y_step)
                except Exception:
                    pass
                last_step_v = v_step[0].item()
                # Smoothed y for decision thresholds
                if last_step_y_ema is None:
                    last_step_y_ema = y_step
                else:
                    alpha = float(args.y_ema_alpha)
                    last_step_y_ema = float(alpha * y_step + (1.0 - alpha) * last_step_y_ema)

                # Tick-level micro update to encourage movement and stabilize early learning
                if args.micro_update:
                    # Compute mid delta from FeatureBuilder window
                    if len(fb.mid_window) >= 2:
                        mid_delta = float(fb.mid_window[-1] - fb.mid_window[-2])
                    else:
                        mid_delta = 0.0
                    # Position-aware reward: positive when aligned with current exposure; small bias when flat
                    if current_units > 0:
                        r_micro = mid_delta
                    elif current_units < 0:
                        r_micro = -mid_delta
                    else:
                        r_micro = float(args.micro_flat_bias) * mid_delta
                    r_micro = r_micro * float(args.micro_reward_scale)
                    # Commitment bonus toward non-neutral outputs (use smoothed y when available)
                    y_for_commit = float(last_step_y_ema if last_step_y_ema is not None else y_step)
                    r_commit = float(args.commit_reward_coef) * float(abs(y_for_commit - 0.5))
                    r_total = r_micro + r_commit
                    # Critic target and advantage
                    opt.zero_grad()
                    _, a_live, v_live = net(xt)
                    v_pred = v_live[0]
                    target = torch.tensor(float(np.clip(r_total, -float(args.reward_clip), float(args.reward_clip))), dtype=torch.float32)
                    adv_micro = torch.clamp(target.detach() - v_pred.detach(), -float(args.adv_clip), float(args.adv_clip))
                    # Policy shaping target: encourage y toward 1 on upward price moves, toward 0 on downward moves
                    dist_live = torch.distributions.Bernoulli(logits=a_live)
                    action_like = torch.tensor(1.0 if mid_delta > 0 else 0.0, dtype=torch.float32)
                    logprob_live = dist_live.log_prob(action_like)
                    actor_loss_micro = -adv_micro * logprob_live.mean()
                    value_loss_micro = F.smooth_l1_loss(v_pred, target)
                    entropy_bonus = dist_live.entropy().mean()
                    loss_micro = float(args.micro_actor_coef) * actor_loss_micro \
                                 + float(args.micro_value_coef) * value_loss_micro \
                                 - float(args.entropy_coef) * entropy_bonus
                    # KL regularization toward pretrained policy (logit-space L2 proxy)
                    try:
                        if pretrained_net is not None and float(args.kl_coef) > 0.0:
                            with torch.no_grad():
                                _, a_pre, _ = pretrained_net(xt)
                            kl_pen = float(args.kl_coef) * torch.mean((a_live - a_pre) ** 2)
                            loss_micro = loss_micro + kl_pen
                    except Exception:
                        pass
                    loss_micro.backward()
                    # Track grad norm and KL penalty for stats
                    try:
                        last_grad_norm = _compute_grad_norm(net.parameters())
                    except Exception:
                        last_grad_norm = None
                    try:
                        last_kl_pen = float(kl_pen.detach().item()) if 'kl_pen' in locals() else None
                    except Exception:
                        last_kl_pen = None
                    nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                    opt.step()

                # DAgger-style supervised correction when uncertain
                try:
                    if (str(args.expert_baseline) != "none") and bool(args.dagger_supervise):
                        y_use = float(last_step_y_ema if last_step_y_ema is not None else y_step)
                        if abs(y_use - 0.5) <= float(args.uncertain_band):
                            # Use M5 closes if available else D1
                            def deque_close(bars: Deque[Dict[str, Any]]) -> List[float]:
                                return [float(b.get("c") or 0.0) for b in bars]
                            closes = deque_close(ccache.m5) if len(ccache.m5) >= 20 else deque_close(ccache.d1)
                            label = combined_label(
                                closes,
                                method=str(args.expert_baseline),
                                fast=int(args.expert_fast),
                                slow=int(args.expert_slow),
                                neutral_band=float(args.expert_neutral_band),
                            )
                            # Supervise only if expert is confident enough
                            if label is not None and abs(float(label) - 0.5) > float(args.expert_confidence_threshold):
                                opt.zero_grad()
                                _, a_live2, _ = net(xt)
                                # BCEWithLogits toward expert label
                                target = torch.tensor(float(label), dtype=torch.float32)
                                bce = nn.BCEWithLogitsLoss()
                                # Adaptive dagger weight decay
                                cur_w = max(float(args.dagger_weight_min), float(args.dagger_weight) * (float(args.dagger_weight_decay) ** step_idx))
                                loss_sup = cur_w * bce(a_live2[0], target)
                                loss_sup.backward()
                                # Track dagger weight and grad norm
                                last_dagger_w = float(cur_w)
                                try:
                                    last_grad_norm = _compute_grad_norm(net.parameters())
                                except Exception:
                                    pass
                                nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                                opt.step()
                                print(json.dumps({"type": "SUPERVISE", "y": round(y_step, 4), "label": float(label)}), flush=True)
                except Exception:
                    pass

                # Natural close detection and training update
                if st.open and current_units == 0 and st.entry_nav is not None and last_nav > 0:
                    G = (last_nav - st.entry_nav) / st.entry_nav * cfg.reward_scale
                    if G < 0:
                        G = G * cfg.neg_reward_coef
                    if cfg.reward_clip > 0:
                        G = float(np.clip(G, -cfg.reward_clip, cfg.reward_clip))
                    with torch.no_grad():
                        _, _, v_entry = net(torch.from_numpy(st.entry_features[None, :]).float())
                        v_entry_val = float(v_entry.item())
                    advantage = G - v_entry_val
                    if cfg.adv_clip > 0:
                        advantage = float(np.clip(advantage, -cfg.adv_clip, cfg.adv_clip))
                    # A2C update: action was long(1)/short(0) at entry using Bernoulli on actor logit
                    opt.zero_grad()
                    _, a_raw_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                    dist = torch.distributions.Bernoulli(logits=a_raw_t)
                    action_tensor = torch.tensor(float(st.entry_action_long or 0), dtype=torch.float32)
                    logprob = dist.log_prob(action_tensor)
                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                    # KL regularization against pretrained policy on entry state
                    try:
                        if pretrained_net is not None and float(args.kl_coef) > 0.0:
                            with torch.no_grad():
                                _, a_pre_entry, _ = pretrained_net(torch.from_numpy(st.entry_features[None, :]).float())
                            kl_pen2 = float(args.kl_coef) * torch.mean((a_raw_t - a_pre_entry) ** 2)
                            loss = loss + kl_pen2
                    except Exception:
                        pass
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    print(json.dumps({"type": "EP_END", "reward": G, "adv": advantage}), flush=True)
                    st = TradeState(open=False)
                    last_step_y = None
                    last_step_v = None

                # Decision logic (with EMA smoothing, consecutive gating, and min-hold ticks)
                pos_long = current_units > 0
                pos_short = current_units < 0
                if st.open:
                    ticks_open += 1
                else:
                    ticks_open = 0

                y_use = float(last_step_y_ema if last_step_y_ema is not None else y_step)
                # Curriculum warmup thresholds
                use_enter_long = float(cfg.enter_long_thresh)
                use_exit_long = float(cfg.exit_long_thresh)
                use_enter_short = float(cfg.enter_short_thresh)
                use_exit_short = float(cfg.exit_short_thresh)
                if (time.time() - start_ts) <= float(args.warmup_seconds):
                    use_enter_long = float(args.warmup_enter_long)
                    use_exit_long = float(args.warmup_exit_long)
                    use_enter_short = float(args.warmup_enter_short)
                    use_exit_short = float(args.warmup_exit_short)

                if pos_long:
                    # track exit condition
                    if y_use < use_exit_long:
                        consec_exit_long += 1
                    else:
                        consec_exit_long = 0
                    can_exit = (ticks_open >= int(args.min_hold_ticks)) and (consec_exit_long >= int(args.consec_exit))
                    if can_exit:
                        try:
                            order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                       units=-current_units, tp_pips=None, sl_pips=None,
                                                       anchor=None, client_tag="scal-thr", client_comment="exit long",
                                                       fifo_safe=False, fifo_adjust=False)
                            current_units = refresh_units(api, account_id, args.instrument)
                            print(json.dumps({"type": "EXIT_LONG", "order": _summarize_order(order), "y": round(y_step,4)}), flush=True)
                            # Immediate episodic A2C update on close
                            if st.entry_features is not None and st.entry_nav is not None and last_nav > 0:
                                try:
                                    nav_now = fetch_nav(api, account_id) or last_nav
                                    G = (nav_now - st.entry_nav) / st.entry_nav * cfg.reward_scale
                                    if G < 0:
                                        G = G * cfg.neg_reward_coef
                                    if cfg.reward_clip > 0:
                                        G = float(np.clip(G, -cfg.reward_clip, cfg.reward_clip))
                                    with torch.no_grad():
                                        _, _, v_entry = net(torch.from_numpy(st.entry_features[None, :]).float())
                                        v_entry_val = float(v_entry.item())
                                    advantage = G - v_entry_val
                                    if cfg.adv_clip > 0:
                                        advantage = float(np.clip(advantage, -cfg.adv_clip, cfg.adv_clip))
                                    opt.zero_grad()
                                    _, a_raw_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                                    dist = torch.distributions.Bernoulli(logits=a_raw_t)
                                    action_tensor = torch.tensor(1.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                                    try:
                                        if pretrained_net is not None and float(args.kl_coef) > 0.0:
                                            with torch.no_grad():
                                                _, a_pre_entry, _ = pretrained_net(torch.from_numpy(st.entry_features[None, :]).float())
                                            kl_pen2 = float(args.kl_coef) * torch.mean((a_raw_t - a_pre_entry) ** 2)
                                            loss = loss + kl_pen2
                                    except Exception:
                                        pass
                                    loss.backward()
                                    try:
                                        last_grad_norm = _compute_grad_norm(net.parameters())
                                    except Exception:
                                        pass
                                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                                    opt.step()
                                    print(json.dumps({"type": "EP_END", "reward": G, "adv": advantage}), flush=True)
                                except Exception:
                                    pass
                            st = TradeState(open=False)
                            last_step_y = None
                            last_step_v = None
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                elif pos_short:
                    if y_use > use_exit_short:
                        consec_exit_short += 1
                    else:
                        consec_exit_short = 0
                    can_exit = (ticks_open >= int(args.min_hold_ticks)) and (consec_exit_short >= int(args.consec_exit))
                    if can_exit:
                        try:
                            order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                       units=-current_units, tp_pips=None, sl_pips=None,
                                                       anchor=None, client_tag="scal-thr", client_comment="exit short",
                                                       fifo_safe=False, fifo_adjust=False)
                            current_units = refresh_units(api, account_id, args.instrument)
                            print(json.dumps({"type": "EXIT_SHORT", "order": _summarize_order(order), "y": round(y_step,4)}), flush=True)
                            # Immediate episodic A2C update on close
                            if st.entry_features is not None and st.entry_nav is not None and last_nav > 0:
                                try:
                                    nav_now = fetch_nav(api, account_id) or last_nav
                                    G = (nav_now - st.entry_nav) / st.entry_nav * cfg.reward_scale
                                    if G < 0:
                                        G = G * cfg.neg_reward_coef
                                    if cfg.reward_clip > 0:
                                        G = float(np.clip(G, -cfg.reward_clip, cfg.reward_clip))
                                    with torch.no_grad():
                                        _, _, v_entry = net(torch.from_numpy(st.entry_features[None, :]).float())
                                        v_entry_val = float(v_entry.item())
                                    advantage = G - v_entry_val
                                    if cfg.adv_clip > 0:
                                        advantage = float(np.clip(advantage, -cfg.adv_clip, cfg.adv_clip))
                                    opt.zero_grad()
                                    _, a_raw_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                                    dist = torch.distributions.Bernoulli(logits=a_raw_t)
                                    action_tensor = torch.tensor(0.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                                    try:
                                        if pretrained_net is not None and float(args.kl_coef) > 0.0:
                                            with torch.no_grad():
                                                _, a_pre_entry, _ = pretrained_net(torch.from_numpy(st.entry_features[None, :]).float())
                                            kl_pen2 = float(args.kl_coef) * torch.mean((a_raw_t - a_pre_entry) ** 2)
                                            loss = loss + kl_pen2
                                    except Exception:
                                        pass
                                    loss.backward()
                                    try:
                                        last_grad_norm = _compute_grad_norm(net.parameters())
                                    except Exception:
                                        pass
                                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                                    opt.step()
                                    print(json.dumps({"type": "EP_END", "reward": G, "adv": advantage}), flush=True)
                                except Exception:
                                    pass
                            st = TradeState(open=False)
                            last_step_y = None
                            last_step_v = None
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                elif (not pos_long and not pos_short):
                    if y_use > use_enter_long:
                        consec_enter_long += 1
                        consec_enter_short = 0
                    elif y_use < use_enter_short:
                        consec_enter_short += 1
                        consec_enter_long = 0
                    else:
                        consec_enter_long = 0
                        consec_enter_short = 0

                    if consec_enter_long >= int(args.consec_enter):
                        try:
                            order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                       units=cfg.units, tp_pips=None, sl_pips=None,
                                                       anchor=None, client_tag="scal-thr", client_comment="enter long",
                                                       fifo_safe=False, fifo_adjust=False)
                            current_units = refresh_units(api, account_id, args.instrument)
                            st.open = True
                            st.entry_features = x.copy()
                            st.entry_action_long = 1
                            st.entry_logit = float(np.log(y_step + 1e-12) - np.log(1.0 - y_step + 1e-12)) if 0.0 < y_step < 1.0 else (10.0 if y_step >= 1.0 else -10.0)
                            st.entry_v = float(last_step_v) if last_step_v is not None else 0.0
                            st.entry_nav = last_nav
                            print(json.dumps({
                                "type": "ENTER_LONG",
                                "units": int(cfg.units),
                                "y": round(float(y_step), 4),
                                "v": round(float(st.entry_v), 4),
                                "order": _summarize_order(order),
                            }), flush=True)
                            # reset counters after trade
                            consec_enter_long = consec_enter_short = consec_exit_long = consec_exit_short = 0
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                    elif consec_enter_short >= int(args.consec_enter):
                        try:
                            order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                       units=-cfg.units, tp_pips=None, sl_pips=None,
                                                       anchor=None, client_tag="scal-thr", client_comment="enter short",
                                                       fifo_safe=False, fifo_adjust=False)
                            current_units = refresh_units(api, account_id, args.instrument)
                            st.open = True
                            st.entry_features = x.copy()
                            st.entry_action_long = 0
                            st.entry_logit = float(np.log(y_step + 1e-12) - np.log(1.0 - y_step + 1e-12)) if 0.0 < y_step < 1.0 else (10.0 if y_step >= 1.0 else -10.0)
                            st.entry_v = float(last_step_v) if last_step_v is not None else 0.0
                            st.entry_nav = last_nav
                            print(json.dumps({
                                "type": "ENTER_SHORT",
                                "units": int(-cfg.units),
                                "y": round(float(y_step), 4),
                                "v": round(float(st.entry_v), 4),
                                "order": _summarize_order(order),
                            }), flush=True)
                            consec_enter_long = consec_enter_short = consec_exit_long = consec_exit_short = 0
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)

                # Autosave
                if cfg.autosave_secs > 0 and (tnow - getattr(main, "_last_save", 0.0)) >= cfg.autosave_secs:
                    try:
                        base_path = Path(cfg.model_path)
                        if getattr(args, "save_with_timestamp", False):
                            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            suffix = ''.join(base_path.suffixes)
                            name = f"{base_path.stem}_{ts}{suffix if suffix else ''}"
                            save_path = base_path.with_name(name)
                        else:
                            save_path = base_path
                        os.makedirs(save_path.parent, exist_ok=True)
                        torch.save({
                            "model": net.state_dict(),
                            "opt": opt.state_dict(),
                            "cfg": cfg.__dict__,
                        }, str(save_path))
                        main._last_save = tnow  # type: ignore[attr-defined]
                        print(json.dumps({"type": "SAVED", "path": str(save_path)}), flush=True)
                    except Exception as e:
                        print(json.dumps({"type": "SAVE_ERROR", "error": str(e)}), flush=True)
                # Step increment for decays
                step_idx += 1
        except Exception as exc:
            print(json.dumps({"type": "STREAM_ERROR", "error": str(exc)}), flush=True)
            time.sleep(5.0)
            continue


if __name__ == "__main__":
    def time_cyclical_features(dt: Optional[datetime]) -> np.ndarray:
        if dt is None:
            dt = datetime.now(timezone.utc)
        minute = dt.minute
        hour = dt.hour
        dow = dt.weekday()
        dom = dt.day
        moy = dt.month
        def sc(val: float, period: float) -> List[float]:
            ang = 2.0 * np.pi * (val / period)
            return [float(np.sin(ang)), float(np.cos(ang))]
        out: List[float] = []
        out += sc(minute, 60.0)
        out += sc(hour, 24.0)
        out += sc(dow, 7.0)
        out += sc(dom - 1, 31.0)
        out += sc(moy - 1, 12.0)
        return np.array(out, dtype=np.float32)
    main()
