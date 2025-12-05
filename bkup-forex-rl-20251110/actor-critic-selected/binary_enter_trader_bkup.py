#!/usr/bin/env python3
"""
Binary-Enter Actor-Critic (long-only) - Tick Stream, Single Instrument

- Single Bernoulli actor head: enter or skip.
- Fixed long units and fixed hold seconds; no TP/SL attached.
- When a position closes (deadline-based or external), compute episodic reward from NAV change
  and perform an A2C update using the entry state's features.
- Reuses feature engineering and streaming structure from actor_critic.py.

Environment:
  Requires OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY in environment.
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
import random
import atexit
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


@dataclass
class Config:
    instrument: str
    environment: str = "practice"
    units: int = 100
    min_units: int = 10
    order_cooldown: float = 5.0
    feature_ticks: int = 240
    reward_scale: float = 10000.0
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    lr: float = 1e-3
    entropy_coef: float = 0.001
    autosave_secs: float = 120.0
    hold_sec: float = 30.0
    model_path: str = ""
    enter_threshold: float = 0.5
    decision_mode: str = "threshold"  # or "sample"
    explore_eps: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_clip: float = 1.0
    adv_clip: float = 5.0
    gamma: float = 0.99
    micro_update: bool = True
    micro_reward_scale: float = 100.0
    micro_actor_coef: float = 0.05
    micro_value_coef: float = 0.1
    flatten_on_start: bool = True
    flatten_on_exit: bool = True


# ---------- Feature engineering (copied and trimmed from actor_critic.py) ----------

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
        self.h1: Deque[Dict[str, Any]] = deque(maxlen=h1_len)
        self.d1: Deque[Dict[str, Any]] = deque(maxlen=d_len)
        self.w1: Deque[Dict[str, Any]] = deque(maxlen=w_len)
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

    def backfill(self, h1_count: int, d_count: int, w_count: int) -> None:
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

    def maybe_refresh(self, h1_refresh_secs: float, d_refresh_secs: float, w_refresh_secs: float) -> None:
        now = time.time()
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
        if len(r) > 0:
            obv_tick = float(np.sum(np.sign(r[-60:]))) / 60.0
        else:
            obv_tick = 0.0
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


def time_cyclical_features(dt: Optional[datetime]) -> np.ndarray:
    """Return 10-dim cyclical time features using sin/cos encodings.
    Features: minute-of-hour, hour-of-day, day-of-week, day-of-month, month-of-year.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    minute = dt.minute  # 0..59
    hour = dt.hour      # 0..23
    dow = dt.weekday()  # 0..6 (Mon=0)
    dom = dt.day        # 1..31
    moy = dt.month      # 1..12

    def sc(val: float, period: float) -> List[float]:
        ang = 2.0 * np.pi * (val / period)
        return [float(np.sin(ang)), float(np.cos(ang))]

    out: List[float] = []
    out += sc(minute, 60.0)
    out += sc(hour, 24.0)
    out += sc(dow, 7.0)
    out += sc(dom - 1, 31.0)  # 0..30
    out += sc(moy - 1, 12.0)  # 0..11
    return np.array(out, dtype=np.float32)


# ---------- Network ----------

class ActorCriticBinaryNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, value_maxabs: float = 5.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.actor_logit = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.value_maxabs = float(value_maxabs)
        # Initialize with small head weights and zero biases for stability
        def _init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        import math
        # Initialize encoder normally
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                _init_weights(layer)
        # Initialize heads to zeros to start neutral (p≈0.5, v≈0)
        nn.init.zeros_(self.actor_logit.weight)
        nn.init.zeros_(self.actor_logit.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.norm(h)
        logit = self.actor_logit(h).squeeze(-1)
        logit = torch.clamp(logit, -5.0, 5.0)
        v_raw = self.critic(h).squeeze(-1)
        v = torch.tanh(v_raw) * self.value_maxabs
        return h, logit, v


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


def flatten_position(api: API, account_id: str, instrument: str) -> Optional[Dict[str, Any]]:
    """Close any open net position for instrument by placing a market order of -current_units.
    Returns order summary if an order was sent, otherwise None.
    """
    units = refresh_units(api, account_id, instrument)
    if units != 0:
        try:
            order = place_market_order(
                api=api, account_id=account_id, instrument=instrument,
                units=-units, tp_pips=None, sl_pips=None,
                anchor=None, client_tag="bin-ac", client_comment="auto flatten",
                fifo_safe=False, fifo_adjust=False,
            )
            return _summarize_order(order)
        except Exception as exc:
            print(json.dumps({"type": "FLATTEN_ERROR", "error": str(exc)}), flush=True)
    return None


def _summarize_order(order_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary of an order placement result to reduce log verbosity."""
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


@dataclass
class TradeState:
    open: bool = False
    last_order_time: float = 0.0
    entry_features: Optional[np.ndarray] = None
    entry_action: Optional[int] = None  # 1 if entered
    entry_logit: Optional[float] = None
    entry_v: Optional[float] = None
    entry_nav: Optional[float] = None
    deadline_ts: Optional[float] = None


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Binary-Enter Actor-Critic (long-only)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--min-units", type=int, default=10)
    parser.add_argument("--order-cooldown", type=float, default=5.0)
    parser.add_argument("--feature-ticks", type=int, default=240)
    parser.add_argument("--hold-sec", type=float, default=30.0)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/binary_enter_v001.pt")
    parser.add_argument("--decision-mode", choices=["threshold", "sample"], default="threshold")
    parser.add_argument("--enter-threshold", type=float, default=0.5)
    parser.add_argument("--explore-eps", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--micro-update", action="store_true")
    parser.add_argument("--no-micro-update", dest="micro_update", action="store_false")
    parser.set_defaults(micro_update=True)
    parser.add_argument("--micro-reward-scale", type=float, default=100.0)
    parser.add_argument("--micro-actor-coef", type=float, default=0.05)
    parser.add_argument("--micro-value-coef", type=float, default=0.1)
    parser.add_argument("--flatten-on-start", action="store_true")
    parser.add_argument("--no-flatten-on-start", dest="flatten_on_start", action="store_false")
    parser.set_defaults(flatten_on_start=True)
    parser.add_argument("--flatten-on-exit", action="store_true")
    parser.add_argument("--no-flatten-on-exit", dest="flatten_on_exit", action="store_false")
    parser.set_defaults(flatten_on_exit=True)
    args = parser.parse_args()

    cfg = Config(
        instrument=args.instrument,
        environment=args.environment,
        units=args.units,
        min_units=args.min_units,
        order_cooldown=args.order_cooldown,
        feature_ticks=args.feature_ticks,
        reward_scale=args.reward_scale,
        nav_poll_secs=args.nav_poll_secs,
        pos_refresh_secs=args.pos_refresh_secs,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        autosave_secs=args.autosave_secs,
        hold_sec=args.hold_sec,
        model_path=args.model_path,
        decision_mode=args.decision_mode,
        enter_threshold=args.enter_threshold,
        explore_eps=args.explore_eps,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        adv_clip=args.adv_clip,
        gamma=args.gamma,
        micro_update=args.micro_update,
        micro_reward_scale=args.micro_reward_scale,
        micro_actor_coef=args.micro_actor_coef,
        micro_value_coef=args.micro_value_coef,
        flatten_on_start=args.flatten_on_start,
        flatten_on_exit=args.flatten_on_exit,
    )

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials in env vars.")
    api = API(access_token=access_token, environment=args.environment)

    fb = FeatureBuilder(cfg.feature_ticks)
    ccache = CandleCache(api, args.instrument, h1_len=60, d_len=60, w_len=60)
    ccache.backfill(60, 60, 60)

    # Input dim: tick(19) + dom(28) + H1(16) + D(16) + W(16) + time(10) = 105
    input_dim = 105

    net = ActorCriticBinaryNet(input_dim=input_dim, hidden_dim=128, value_maxabs=5.0)
    net.train()
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    # Optional: flatten any residual position on start
    if cfg.flatten_on_start:
        summ = flatten_position(api, account_id, args.instrument)
        if summ is not None:
            print(json.dumps({"type": "AUTO_FLATTEN_START", "order": summ}), flush=True)

    # State
    st = TradeState(open=False)
    last_step_p: Optional[float] = None
    last_step_logit: Optional[float] = None  # kept for internal gating; no longer logged
    last_step_v: Optional[float] = None
    last_nav = fetch_nav(api, account_id) or 1.0
    last_nav_poll = time.time()
    last_pos_refresh = time.time()
    current_units = refresh_units(api, account_id, args.instrument)

    # Ensure flatten on exit if requested
    if cfg.flatten_on_exit:
        def _on_exit() -> None:
            try:
                summ2 = flatten_position(api, account_id, args.instrument)
                if summ2 is not None:
                    print(json.dumps({"type": "AUTO_FLATTEN_EXIT", "order": summ2}), flush=True)
            except Exception:
                pass
        atexit.register(_on_exit)

    # Resilient streaming loop with auto-reconnect
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
                    # Periodic refresh of candles
                    ccache.maybe_refresh(60.0, 300.0, 3600.0)
                    hb = {
                        "type": "HB",
                        "nav": round(float(last_nav), 6) if isinstance(last_nav, (int, float)) else last_nav,
                        "units": int(current_units),
                        "open": bool(st.open),
                    }
                    if last_step_p is not None:
                        hb["p"] = round(float(last_step_p), 4)
                    if last_step_v is not None:
                        hb["v"] = round(float(last_step_v), 4)
                    if st.deadline_ts is not None:
                        hb["t_to_deadline"] = max(0.0, round(st.deadline_ts - tnow, 1))
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
                x = np.concatenate([x_tick, x_dom, h1_feats, d1_feats, w1_feats, t_feats]).astype(np.float32)
                # Feature sanity check
                if not np.all(np.isfinite(x)):
                    print(json.dumps({"type": "WARN", "msg": "nonfinite features", "has_nan": bool(np.any(np.isnan(x))), "has_inf": bool(np.any(~np.isfinite(x))) }), flush=True)
                    continue
                xt = torch.from_numpy(x)[None, :]

                # Model inference each PRICE step
                net.train(False)
                with torch.no_grad():
                    _, logit_step, v_step = net(xt)
                last_step_p = torch.sigmoid(logit_step[0]).item()
                last_step_v = v_step[0].item()

                # Optional micro update at tick level to encourage movement
                if cfg.micro_update and x_tick is not None:
                    # Compute micro reward as signed mid-price delta direction depending on position
                    if len(fb.mid_window) >= 2:
                        price_delta = fb.mid_window[-1] - fb.mid_window[-2]
                    else:
                        price_delta = 0.0
                    # Scale to avoid tiny numbers; when flat, we "reward" being aligned with not losing money
                    if st.open:
                        r_micro = (price_delta) * cfg.micro_reward_scale
                    else:
                        r_micro = (-price_delta) * cfg.micro_reward_scale
                    # Critic target: immediate micro reward (no bootstrap to keep it simple/noise-robust)
                    opt.zero_grad()
                    _, logit_live, v_live = net(xt)
                    v_pred = v_live[0]
                    target = torch.tensor(float(np.clip(r_micro, -cfg.reward_clip, cfg.reward_clip)), dtype=torch.float32)
                    # Advantage for micro policy update uses v_pred as baseline
                    adv_micro = target.detach() - v_pred.detach()
                    adv_micro = torch.clamp(adv_micro, -cfg.adv_clip, cfg.adv_clip)
                    dist_live = torch.distributions.Bernoulli(logits=logit_live)
                    # When open, ideal action is to continue entering next time; when flat, ideal is to skip if price rises.
                    # We softly encourage probabilities using sign of r_micro.
                    action_like = torch.tensor(1.0 if r_micro > 0 else 0.0, dtype=torch.float32)
                    logprob_live = dist_live.log_prob(action_like)
                    actor_loss_micro = -adv_micro * logprob_live.mean()
                    value_loss_micro = F.smooth_l1_loss(v_pred, target)
                    entropy_bonus = dist_live.entropy().mean()
                    loss_micro = cfg.micro_actor_coef * actor_loss_micro + cfg.micro_value_coef * value_loss_micro - cfg.entropy_coef * entropy_bonus
                    loss_micro.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()

                # Close by deadline
                if st.open and st.deadline_ts is not None and tnow >= st.deadline_ts and current_units != 0:
                    try:
                        order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                   units=-current_units, tp_pips=None, sl_pips=None,
                                                   anchor=None, client_tag="bin-ac", client_comment="deadline close",
                                                   fifo_safe=False, fifo_adjust=False)
                        print(json.dumps({"type": "FORCE_CLOSE", "order": _summarize_order(order)}), flush=True)
                    except Exception as exc:
                        print(json.dumps({"error": str(exc)}), flush=True)
                    current_units = refresh_units(api, account_id, args.instrument)

                # Detect close and update
                if st.open and current_units == 0 and st.entry_nav is not None and last_nav > 0:
                    G = (last_nav - st.entry_nav) / st.entry_nav * cfg.reward_scale
                    # Stabilize reward and advantage
                    if cfg.reward_clip > 0:
                        G = float(np.clip(G, -cfg.reward_clip, cfg.reward_clip))
                    with torch.no_grad():
                        _, _, v_entry = net(torch.from_numpy(st.entry_features[None, :]).float())
                        v_entry_val = float(v_entry.item())
                    advantage = G - v_entry_val
                    if cfg.adv_clip > 0:
                        advantage = float(np.clip(advantage, -cfg.adv_clip, cfg.adv_clip))
                    # A2C update: action was 1 (entered)
                    opt.zero_grad()
                    _, logit_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                    dist = torch.distributions.Bernoulli(logits=logit_t)
                    logprob = dist.log_prob(torch.ones_like(logit_t))  # action=1
                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    print(json.dumps({"type": "EP_END", "reward": G, "adv": advantage}), flush=True)
                    st = TradeState(open=False)

                    # After update, force a re-evaluation step before allowing another entry
                    last_step_p = None
                    last_step_v = None
                    last_step_logit = None

                # Entry decision
                can_enter = (not st.open) and ((tnow - st.last_order_time) >= cfg.order_cooldown)
                if can_enter:
                    # Require at least one fresh STEP after any training update
                    if last_step_p is None:
                        continue
                    # Use last inference outputs
                    p_enter = float(last_step_p) if last_step_p is not None else 0.0
                    if cfg.decision_mode == "sample":
                        enter = bool(torch.bernoulli(torch.tensor(p_enter)).item() > 0.0)
                    else:
                        enter = (p_enter >= cfg.enter_threshold)
                    explored = False
                    if not enter and cfg.explore_eps > 0.0 and (random.random() < cfg.explore_eps):
                        enter = True
                        explored = True
                    # Prevent compounding net exposure: only enter when currently flat
                    if current_units != 0:
                        enter = False
                    if enter:
                        units = cfg.units
                        if units >= cfg.min_units:
                            try:
                                order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                           units=units, tp_pips=None, sl_pips=None,
                                                           anchor=None, client_tag="bin-ac", client_comment="enter long",
                                                           fifo_safe=False, fifo_adjust=False)
                                current_units = refresh_units(api, account_id, args.instrument)
                                st.open = True
                                st.last_order_time = tnow
                                st.entry_features = x.copy()
                                st.entry_action = 1
                                # store logit derived from p to avoid retracing
                                st.entry_logit = float(np.log(p_enter + 1e-12) - np.log(1.0 - p_enter + 1e-12)) if 0.0 < p_enter < 1.0 else (10.0 if p_enter >= 1.0 else -10.0)
                                st.entry_v = float(last_step_v) if last_step_v is not None else 0.0
                                st.entry_nav = last_nav
                                st.deadline_ts = tnow + cfg.hold_sec
                                print(json.dumps({
                                    "type": "ENTER",
                                    "units": int(units),
                                    "p": round(float(p_enter), 4),
                                    "v": round(float(st.entry_v), 4),
                                    "hold_sec": float(cfg.hold_sec),
                                    "decision": cfg.decision_mode,
                                    "threshold": round(float(cfg.enter_threshold), 3),
                                    "explore": explored,
                                    "order": _summarize_order(order),
                                }), flush=True)
                            except Exception as exc:
                                print(json.dumps({"error": str(exc)}), flush=True)
                    else:
                        # Optional: A2C update on skip? Omit; we only credit on episodes with entry
                        pass

                # Autosave
                if cfg.autosave_secs > 0 and (tnow - getattr(main, "_last_save", 0.0)) >= cfg.autosave_secs:
                    try:
                        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
                        torch.save({
                            "model": net.state_dict(),
                            "opt": opt.state_dict(),
                            "cfg": cfg.__dict__,
                        }, cfg.model_path)
                        main._last_save = tnow  # type: ignore[attr-defined]
                        print(json.dumps({"type": "SAVED", "path": cfg.model_path}), flush=True)
                    except Exception:
                        pass
        except Exception as exc:
            # Network/stream hiccup. Log and reconnect after a short backoff.
            print(json.dumps({"type": "STREAM_ERROR", "error": str(exc)}), flush=True)
            time.sleep(5.0)
            continue


if __name__ == "__main__":
    main()
