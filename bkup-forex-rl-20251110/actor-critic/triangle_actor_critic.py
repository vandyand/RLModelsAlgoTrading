#!/usr/bin/env python3
"""
Actor-Critic Triangle Trader (long-only) - Tick Stream, Single Instrument

- Three actor outputs define the triangle and entry gate:
  1) enter gate (Bernoulli) -> enter only if currently flat and gate=1
  2) triangle height in pips (min_pips..max_pips) via sigmoid(z_height)
  3) triangle width in candle widths (min_width..max_width) via sigmoid(z_width)

- Geometry:
  - Base (top) is horizontal at +height pips above entry mid-price
  - Apex (bottom) is at -height pips below entry mid-price
  - Base length is `width` in candle-widths of a chosen granularity
  - At entry, current price-time is the midpoint of the left edge
    => vertices relative to entry (x in candles, y in pips):
       TL = (-width/4, +height)
       TR = ( 3*width/4, +height)
       A  = (  width/4, -height)   # apex, centered horizontally under base

- Exit rule: while open, compute current point p=(elapsed_candles, price_offset_pips);
  if p is outside the triangle, close immediately. Episodic A2C update at close.

Environment:
  Requires OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY in environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.instruments as instruments_ep


# Reuse order and instrument helpers
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from streamer.orders import (  # type: ignore
    place_market_order,
    fetch_instrument_spec,
    calc_pip_size,
)


# ---------- Config ----------


@dataclass
class Config:
    instrument: str
    environment: str = "practice"
    # Sizing
    units: int = 100
    min_units: int = 10
    order_cooldown: float = 5.0
    # Features
    feature_ticks: int = 240
    # RL/optimization
    lr: float = 1e-3
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    actor_sigma: float = 0.3
    value_maxabs: float = 5.0
    device: str = "cpu"
    # Reward shaping + polling
    reward_scale: float = 10000.0
    reward_clip: float = 1.0
    adv_clip: float = 5.0
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    # Autosave
    autosave_secs: float = 120.0
    model_path: str = ""
    # Gate decision
    decision_mode: str = "threshold"  # or "sample"
    enter_threshold: float = 0.5
    # Triangle outputs ranges
    min_pips: float = 2.0
    max_pips: float = 15.0
    min_width: float = 5.0        # in candles
    max_width: float = 180.0      # in candles
    width_granularity: str = "M1"  # candle granularity for width in OANDA notation
    # Safety
    flatten_on_start: bool = True
    flatten_on_exit: bool = True
    # Exploration + micro shaping
    explore_eps: float = 0.0
    micro_update: bool = True
    micro_reward_scale: float = 100.0
    micro_actor_coef: float = 0.05
    micro_value_coef: float = 0.1
    # Debug and init bias
    log_debug: bool = False
    enter_bias_init: float = 0.0


# ---------- Utilities ----------


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def granularity_to_seconds(gran: str) -> float:
    g = (gran or "M1").upper()
    table = {
        "S5": 5,
        "S10": 10,
        "S15": 15,
        "S30": 30,
        "M1": 60,
        "M2": 120,
        "M4": 240,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H3": 10800,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "H12": 43200,
        "D": 86400,
        "W": 604800,
        "M": 2592000,
    }
    return float(table.get(g, 60))


# ---------- Feature engineering (adapted from actor_critic.py) ----------


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
        self.dom_depth = 5  # K=5 levels
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
        # Reduce warmup: require only 2 ticks; internal logic handles short windows.
        if len(self.mid_window) < 2:
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

        r1 = last_ratio(1)
        r5 = last_ratio(5)
        r20 = last_ratio(20)
        m5, s5 = roll_mean_std(5)
        m20, s20 = roll_mean_std(20)
        m60, s60 = roll_mean_std(60)
        if len(prices) < 20:
            sma20 = prices[-1]
            std20 = 1e-6
        else:
            sma20 = float(np.mean(prices[-20:]))
            std20 = float(np.std(prices[-20:]) + self.eps)
        z_sma20 = float((prices[-1] - sma20) / std20)
        if len(self.spread_window) >= 30:
            sp_arr = np.array(self.spread_window)
            sp = (sp_arr[-1] - sp_arr.mean()) / (sp_arr.std() + self.eps)
        else:
            sp = 0.0
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd = ema12 - ema26
        signal = ema(macd, 9)
        hist = macd - signal
        macd_last = float(macd[-1])
        signal_last = float(signal[-1])
        hist_last = float(hist[-1])
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
        while len(bid_px) < k:
            bid_px.append(0.0)
        while len(ask_px) < k:
            ask_px.append(0.0)
        while len(bid_liq) < k:
            bid_liq.append(0.0)
        while len(ask_liq) < k:
            ask_liq.append(0.0)

        sb = float(np.sum(bid_liq))
        sa = float(np.sum(ask_liq))
        total = sb + sa + eps
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
        return float(np.log((close[-1] + eps) / (close[-1 - k] + eps)))

    def std_ret(k: int) -> float:
        if len(r) < k:
            return 0.0
        return float(np.std(r[-k:]) + eps)

    tr_list: List[float] = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        tr_list.append(tr)
    atr20 = float(np.mean(tr_list[-20:])) if len(tr_list) >= 1 else 0.0

    def dist_max(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.max(close[-k:])) / (np.std(close[-k:]) + eps))

    def dist_min(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.min(close[-k:])) / (np.std(close[-k:]) + eps))

    if len(vol) >= 20:
        v20 = (vol[-1] - np.mean(vol[-20:])) / (np.std(vol[-20:]) + eps)
    else:
        v20 = 0.0

    ema12c = ema(close, 12)
    ema26c = ema(close, 26)
    macd = ema12c - ema26c
    signal = ema(macd, 9)
    hist = macd - signal
    macd_last = float(macd[-1])
    signal_last = float(signal[-1])
    hist_last = float(hist[-1])
    rsi14 = rsi(close, 14, eps)
    vol_of_vol = float(np.std(np.abs(r[-20:])) if len(r) >= 20 else 0.0)

    feats = np.array([
        last_ret(1), last_ret(5), last_ret(20),
        std_ret(5), std_ret(20), std_ret(60),
        atr20,
        dist_max(20), dist_min(20),
        v20,
        macd_last, signal_last, hist_last,
        rsi14 / 100.0,
        vol_of_vol,
    ], dtype=float)
    return np.clip(feats, -5.0, 5.0)


def time_cyclical_features(ts_str: str) -> np.ndarray:
    # Extract UTC components from OANDA RFC3339 timestamps
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
    minute = dt.minute
    hour = dt.hour
    dow = dt.weekday()
    dom = dt.day
    moy = dt.month

    def sc(val: int, period: float) -> List[float]:
        ang = 2.0 * np.pi * (float(val) / period)
        return [float(np.sin(ang)), float(np.cos(ang))]

    out: List[float] = []
    out += sc(minute, 60.0)
    out += sc(hour, 24.0)
    out += sc(dow, 7.0)
    out += sc(dom - 1, 31.0)
    out += sc(moy - 1, 12.0)
    return np.array(out, dtype=np.float32)


# ---------- Triangle geometry ----------


def triangle_vertices(height_pips: float, width_candles: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Relative to entry at (0, 0)
    w = float(width_candles)
    h = float(height_pips)
    tl = np.array([-w / 4.0, +h], dtype=float)
    tr = np.array([+3.0 * w / 4.0, +h], dtype=float)
    ap = np.array([+w / 4.0, -h], dtype=float)
    return ap, tl, tr


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    # Barycentric sign test
    def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))

    b1 = sign(p, a, b) < 0.0
    b2 = sign(p, b, c) < 0.0
    b3 = sign(p, c, a) < 0.0
    return (b1 == b2) and (b2 == b3)


# ---------- Network ----------


class ActorCriticTriangleNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, value_maxabs: float = 5.0) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Actor heads: 2 Gaussian means (height, width) and 1 Bernoulli logit (enter)
        self.actor_mu = nn.Linear(hidden_dim, 2)
        self.enter_logit = nn.Linear(hidden_dim, 1)
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
        self.value_maxabs = float(value_maxabs)

        # Initialization
        import math

        def _init_linear(m: nn.Module, gain: float = 1.0) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Encoder with Kaiming
        for layer in self.encoder:
            _init_linear(layer)
        # Actor mu: small initialized weights so mu starts near 0 but not constant
        _init_linear(self.actor_mu)
        with torch.no_grad():
            self.actor_mu.weight.mul_(0.05)
            if self.actor_mu.bias is not None:
                self.actor_mu.bias.zero_()
        # Enter and critic start neutral; bias for enter can be set via CLI later
        nn.init.zeros_(self.enter_logit.weight)
        nn.init.zeros_(self.enter_logit.bias)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.norm(h)
        mu = self.actor_mu(h)                # shape (B, 2)
        enter_logit = self.enter_logit(h).squeeze(-1)  # shape (B,)
        v_raw = self.critic(h).squeeze(-1)   # shape (B,)
        v = torch.tanh(v_raw) * self.value_maxabs
        return h, mu, enter_logit, v


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


def sample_gaussian(mu: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.random.normal(0.0, sigma, size=mu.shape)
    z = mu + eps
    return z, eps


def decode_triangle(z: np.ndarray, min_pips: float, max_pips: float, min_width: float, max_width: float) -> Dict[str, float]:
    z_h, z_w = float(z[0]), float(z[1])
    height_pips = min_pips + sigmoid(z_h) * max(0.0, max_pips - min_pips)
    width_candles = min_width + sigmoid(z_w) * max(0.0, max_width - min_width)
    return {
        "height_pips": float(height_pips),
        "width_candles": float(width_candles),
    }


@dataclass
class TradeState:
    open: bool = False
    last_order_time: float = 0.0
    entry_features: Optional[np.ndarray] = None
    entry_z: Optional[np.ndarray] = None
    entry_enter_logit: Optional[float] = None
    entry_v: Optional[float] = None
    entry_nav: Optional[float] = None
    entry_wall_ts: Optional[float] = None
    entry_mid: Optional[float] = None
    pip_size: Optional[float] = None
    height_pips: Optional[float] = None
    width_candles: Optional[float] = None


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-Critic Triangle Trader (long-only)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    # Sizing
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--min-units", type=int, default=10)
    parser.add_argument("--order-cooldown", type=float, default=5.0)
    # Features
    parser.add_argument("--feature-ticks", type=int, default=240)
    # Triangle outputs ranges
    parser.add_argument("--min-pips", type=float, default=2.0)
    parser.add_argument("--max-pips", type=float, default=15.0)
    parser.add_argument("--min-width", type=float, default=5.0, help="Min width (candles) of top base")
    parser.add_argument("--max-width", type=float, default=180.0, help="Max width (candles) of top base")
    parser.add_argument("--width-granularity", default="M1", help="Candle granularity used for width (e.g., S5, M1, M5, H1)")
    # RL/optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-sigma", type=float, default=0.3)
    parser.add_argument("--value-maxabs", type=float, default=5.0)
    # Reward shaping + polling
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--reward-clip", type=float, default=1.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    # Gate decision
    parser.add_argument("--decision-mode", choices=["threshold", "sample"], default="threshold")
    parser.add_argument("--enter-threshold", type=float, default=0.5)
    parser.add_argument("--explore-eps", type=float, default=0.0)
    # Micro shaping
    parser.add_argument("--micro-update", action="store_true")
    parser.add_argument("--no-micro-update", dest="micro_update", action="store_false")
    parser.set_defaults(micro_update=True)
    parser.add_argument("--micro-reward-scale", type=float, default=100.0)
    parser.add_argument("--micro-actor-coef", type=float, default=0.05)
    parser.add_argument("--micro-value-coef", type=float, default=0.1)
    # Autosave
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/triangle_ac_v001.pt")
    # Debug and init
    parser.add_argument("--log-debug", action="store_true")
    parser.add_argument("--enter-bias-init", type=float, default=0.0)
    # Safety
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
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        actor_sigma=args.actor_sigma,
        value_maxabs=args.value_maxabs,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        adv_clip=args.adv_clip,
        nav_poll_secs=args.nav_poll_secs,
        pos_refresh_secs=args.pos_refresh_secs,
        autosave_secs=args.autosave_secs,
        model_path=args.model_path,
        decision_mode=args.decision_mode,
        enter_threshold=args.enter_threshold,
        explore_eps=args.explore_eps,
        min_pips=args.min_pips,
        max_pips=args.max_pips,
        min_width=args.min_width,
        max_width=args.max_width,
        width_granularity=args.width_granularity,
        flatten_on_start=args.flatten_on_start,
        flatten_on_exit=args.flatten_on_exit,
        micro_update=args.micro_update,
        micro_reward_scale=args.micro_reward_scale,
        micro_actor_coef=args.micro_actor_coef,
        micro_value_coef=args.micro_value_coef,
        log_debug=bool(args.log_debug),
        enter_bias_init=args.enter_bias_init,
    )

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials in env vars.")
    api = API(access_token=access_token, environment=args.environment)

    # Instrument pip size for pip conversion
    try:
        spec = fetch_instrument_spec(api, account_id, args.instrument)
        pip_location = int(spec.get("pipLocation"))
        pip_size = calc_pip_size(pip_location)
    except Exception:
        pip_size = 0.0001

    # Feature builders
    fb = FeatureBuilder(cfg.feature_ticks)
    ccache = CandleCache(api, args.instrument, h1_len=60, d_len=60, w_len=60)
    ccache.backfill(60, 60, 60)

    # Model
    # input: tick(19) + dom(28) + H1(16) + D(16) + W(16) + time(10) = 105
    input_dim = 105
    net = ActorCriticTriangleNet(input_dim=input_dim, hidden_dim=128, value_maxabs=cfg.value_maxabs)
    net.train()
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    # Optional bias init to encourage initial entries
    try:
        if abs(float(cfg.enter_bias_init)) > 0.0:
            with torch.no_grad():
                net.enter_logit.bias.fill_(float(cfg.enter_bias_init))
    except Exception:
        pass

    # Optional: flatten on start
    st = TradeState(open=False)
    if cfg.flatten_on_start:
        try:
            units_now = refresh_units(api, account_id, args.instrument)
            if units_now != 0:
                order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                           units=-units_now, tp_pips=None, sl_pips=None,
                                           anchor=None, client_tag="tri-ac", client_comment="auto flatten",
                                           fifo_safe=False, fifo_adjust=False)
                print(json.dumps({"type": "FLATTEN_START", "order": order.get("response") if isinstance(order, dict) else None}), flush=True)
        except Exception:
            pass

    last_nav = fetch_nav(api, account_id)
    nav_estimate = last_nav or 1.0
    last_nav_update_time = time.time()
    last_pos_refresh_time = time.time()
    current_units = refresh_units(api, account_id, args.instrument)

    candle_sec = granularity_to_seconds(cfg.width_granularity)

    # Price stream
    stream = pricing.PricingStream(accountID=account_id, params={"instruments": args.instrument})

    # Last prediction snapshot for heartbeat
    last_p_enter: Optional[float] = None
    last_h_pips_mean: Optional[float] = None
    last_w_candles_mean: Optional[float] = None

    while True:
        try:
            for msg in api.request(stream):
                msg_type = msg.get("type")
                now_wall = time.time()

                if msg_type == "HEARTBEAT":
                    if (now_wall - last_nav_update_time) >= cfg.nav_poll_secs:
                        nav_now = fetch_nav(api, account_id)
                        if nav_now is not None and nav_now > 0:
                            nav_estimate = nav_now
                            last_nav = nav_now
                        last_nav_update_time = now_wall
                    if (now_wall - last_pos_refresh_time) >= cfg.pos_refresh_secs:
                        current_units = refresh_units(api, account_id, args.instrument)
                        last_pos_refresh_time = now_wall
                    try:
                        hb = {"type": "HEARTBEAT", "time": msg.get("time"), "nav": nav_estimate, "units": current_units}
                        if last_p_enter is not None:
                            hb.update({
                                "p_enter": float(last_p_enter),
                                "h_pips_mean": float(last_h_pips_mean) if last_h_pips_mean is not None else None,
                                "w_candles_mean": float(last_w_candles_mean) if last_w_candles_mean is not None else None,
                            })
                        print(json.dumps(hb), flush=True)
                    except Exception:
                        pass
                    # Refresh higher timeframe candles periodically for features
                    ccache.maybe_refresh(h1_refresh_secs=60.0, d_refresh_secs=300.0, w_refresh_secs=3600.0)
                    continue

                if msg_type != "PRICE":
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
                if bid is None and ask is None:
                    continue
                mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else (bid if bid is not None else ask)

                ts = msg.get("time") or ""

                # Optional minimal tick log
                if cfg.log_debug:
                    try:
                        print(json.dumps({"type": "PRICE", "time": ts, "mid": mid}), flush=True)
                    except Exception:
                        pass

                # Build features
                fb.update_tick(bid, ask)
                tick_feats = fb.compute_tick_features()
                dom_feats = fb.compute_dom_features(bids, asks)
                h1_feats = candle_features(ccache.h1)
                d_feats = candle_features(ccache.d1)
                w_feats = candle_features(ccache.w1)
                time_feats = time_cyclical_features(ts)

                if tick_feats is None:
                    # Fallback to zeros so we can still run inference/logging
                    tick_feats = np.zeros(19, dtype=np.float32)
                # Defensive: ensure each block has expected size; otherwise pad/trim
                def ensure(shape_arr: np.ndarray, n: int) -> np.ndarray:
                    arr = np.asarray(shape_arr, dtype=np.float32).reshape(-1)
                    if arr.size == n:
                        return arr
                    if arr.size > n:
                        return arr[:n]
                    pad = np.zeros(n - arr.size, dtype=np.float32)
                    return np.concatenate([arr, pad])

                tick_feats = ensure(tick_feats, 19)
                dom_feats = ensure(dom_feats, 28)
                h1_feats = ensure(h1_feats, 16)
                d_feats = ensure(d_feats, 16)
                w_feats = ensure(w_feats, 16)
                time_feats = ensure(time_feats, 10)

                x = np.concatenate([tick_feats, dom_feats, h1_feats, d_feats, w_feats, time_feats]).astype(np.float32)
                if x.shape[0] != 105:
                    # Safety: skip until full feature vector forms
                    if cfg.log_debug:
                        try:
                            print(json.dumps({
                                "type": "FEAT_SHAPE_MISMATCH",
                                "time": ts,
                                "shape": int(x.shape[0]),
                            }), flush=True)
                        except Exception:
                            pass
                    continue

                # Optional feature sanity log and immediate PRED snapshot
                try:
                    xt_pred = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    net.eval()
                    with torch.no_grad():
                        _, mu_pred, enter_logit_pred, _ = net(xt_pred)
                    mu_np = mu_pred[0].detach().cpu().numpy()
                    p_enter_now = float(torch.sigmoid(enter_logit_pred[0]).item())
                    means_decoded = decode_triangle(mu_np, cfg.min_pips, cfg.max_pips, cfg.min_width, cfg.max_width)
                    last_p_enter = p_enter_now
                    last_h_pips_mean = float(means_decoded["height_pips"])
                    last_w_candles_mean = float(means_decoded["width_candles"])
                    if cfg.log_debug:
                        print(json.dumps({
                            "type": "PRED",
                            "time": ts,
                            "mid": mid,
                            "feat_ok": bool(np.all(np.isfinite(x))),
                            "p_enter": p_enter_now,
                            "mu_h": float(mu_np[0]) if mu_np.shape[0] >= 1 else None,
                            "mu_w": float(mu_np[1]) if mu_np.shape[0] >= 2 else None,
                            "h_pips_mean": last_h_pips_mean,
                            "w_candles_mean": last_w_candles_mean,
                        }), flush=True)
                except Exception as exc:
                    if cfg.log_debug:
                        try:
                            print(json.dumps({
                                "type": "PRED_ERROR",
                                "time": ts,
                                "error": str(exc),
                                "x_shape": int(x.shape[0]),
                            }), flush=True)
                        except Exception:
                            pass

                # Micro update to shape gate and critic even when episodes are sparse
                if cfg.micro_update:
                    try:
                        xt = torch.from_numpy(x).float().unsqueeze(0)
                        _, _, enter_logit_live, v_live = net(xt)
                        v_pred = v_live[0]
                        # Price delta proxy
                        if len(fb.mid_window) >= 2:
                            price_delta = float(fb.mid_window[-1] - fb.mid_window[-2])
                        else:
                            price_delta = 0.0
                        r_micro = (price_delta if st.open else -price_delta) * float(cfg.micro_reward_scale)
                        target = torch.tensor(float(np.clip(r_micro, -cfg.reward_clip, cfg.reward_clip)), dtype=torch.float32)
                        adv_micro = (target.detach() - v_pred.detach())
                        adv_micro = torch.clamp(adv_micro, -cfg.adv_clip, cfg.adv_clip)
                        dist_live = torch.distributions.Bernoulli(logits=enter_logit_live)
                        action_like = torch.tensor(1.0 if r_micro > 0 else 0.0, dtype=torch.float32)
                        logprob_live = dist_live.log_prob(action_like)
                        actor_loss_micro = -adv_micro * logprob_live.mean()
                        value_loss_micro = F.smooth_l1_loss(v_pred, target)
                        entropy_bonus = dist_live.entropy().mean()
                        loss_micro = (
                            float(cfg.micro_actor_coef) * actor_loss_micro
                            + float(cfg.micro_value_coef) * value_loss_micro
                            - float(cfg.entropy_coef) * entropy_bonus
                        )
                        opt.zero_grad()
                        loss_micro.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                        opt.step()
                    except Exception:
                        pass

                # (PRED snapshot already computed above)

                # Exit check if triangle is active
                if st.open and current_units != 0 and st.entry_wall_ts is not None and st.entry_mid is not None and st.height_pips is not None and st.width_candles is not None:
                    elapsed_sec = max(0.0, now_wall - float(st.entry_wall_ts))
                    x_candles = elapsed_sec / candle_sec
                    y_pips = (mid - float(st.entry_mid)) / float(st.pip_size or pip_size)
                    ap, tl, tr = triangle_vertices(height_pips=float(st.height_pips), width_candles=float(st.width_candles))
                    p = np.array([x_candles, y_pips], dtype=float)
                    inside = point_in_triangle(p, ap, tl, tr)
                    if not inside:
                        try:
                            order = place_market_order(
                                api=api, account_id=account_id, instrument=args.instrument,
                                units=-current_units, tp_pips=None, sl_pips=None,
                                anchor=None, client_tag="tri-ac", client_comment="triangle exit",
                                fifo_safe=False, fifo_adjust=False,
                            )
                            print(json.dumps({
                                "type": "TRI_EXIT_OUTSIDE",
                                "time": ts,
                                "elapsed_candles": x_candles,
                                "price_offset_pips": y_pips,
                                "height_pips": float(st.height_pips),
                                "width_candles": float(st.width_candles),
                                "order": (order.get("response") if isinstance(order, dict) else None),
                            }), flush=True)
                        except Exception:
                            pass
                        current_units = refresh_units(api, account_id, args.instrument)

                # Detect episode close by flat and perform A2C update
                if st.open and current_units == 0 and st.entry_nav is not None and nav_estimate > 0:
                    G = float((nav_estimate - st.entry_nav) / st.entry_nav) * cfg.reward_scale
                    if cfg.reward_clip > 0:
                        G = float(np.clip(G, -cfg.reward_clip, cfg.reward_clip))

                    # Learning step
                    try:
                        if st.entry_features is not None and st.entry_z is not None and st.entry_enter_logit is not None and st.entry_v is not None:
                            xt = torch.tensor(st.entry_features, dtype=torch.float32).unsqueeze(0)
                            with torch.no_grad():
                                _, mu_t, enter_logit_t, v_t = net(xt)
                            # Log-probs
                            sigma = float(cfg.actor_sigma)
                            z = st.entry_z.astype(np.float32)
                            mu = mu_t[0].detach().numpy().astype(np.float32)
                            # Gaussian logprob for z (2 dims)
                            logprob_gauss = -0.5 * np.sum(((z - mu) ** 2) / (sigma ** 2) + np.log(2 * np.pi * (sigma ** 2)))
                            # Bernoulli logprob for enter=1
                            enter_logit = float(st.entry_enter_logit)
                            enter_prob = 1.0 / (1.0 + np.exp(-enter_logit))
                            logprob_enter = float(np.log(max(1e-12, enter_prob)))  # we only update on enter action
                            logprob = float(logprob_gauss + logprob_enter)

                            advantage = float(G - float(st.entry_v))
                            if cfg.adv_clip > 0:
                                advantage = float(np.clip(advantage, -cfg.adv_clip, cfg.adv_clip))

                            # Backprop with current features to allow critic update
                            net.train()
                            h_t, mu_t2, enter_logit_t2, v_t2 = net(xt)
                            # Recompute logprobs in torch for autograd
                            z_t = torch.tensor(z, dtype=torch.float32)
                            mu_t2v = mu_t2[0]
                            sigma_t = torch.tensor(sigma, dtype=torch.float32)
                            logprob_gauss_t = -0.5 * torch.sum(((z_t - mu_t2v) ** 2) / (sigma_t ** 2) + torch.log(2 * torch.tensor(np.pi) * (sigma_t ** 2)))
                            enter_prob_t = torch.sigmoid(enter_logit_t2[0])
                            logprob_enter_t = torch.log(torch.clamp(enter_prob_t, min=1e-12))
                            logprob_t = logprob_gauss_t + logprob_enter_t

                            advantage_t = torch.tensor(advantage, dtype=torch.float32)
                            G_t = torch.tensor(G, dtype=torch.float32)
                            v_pred = v_t2[0]

                            actor_loss = -advantage_t * logprob_t
                            critic_loss = cfg.value_coef * 0.5 * (G_t - v_pred) ** 2
                            # Entropy bonus (Gaussian + Bernoulli)
                            entropy_gauss = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma_t ** 2)))
                            entropy_bern = - (enter_prob_t * torch.log(torch.clamp(enter_prob_t, min=1e-12)) + (1 - enter_prob_t) * torch.log(torch.clamp(1 - enter_prob_t, min=1e-12)))
                            loss = actor_loss + critic_loss + cfg.entropy_coef * (entropy_gauss + entropy_bern)

                            opt.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                            opt.step()

                            print(json.dumps({"type": "TRI_CLOSED", "time": ts, "reward": G, "adv": advantage}), flush=True)
                    except Exception as exc:
                        print(json.dumps({"type": "UPDATE_ERROR", "error": str(exc)}), flush=True)

                    # Reset state
                    st = TradeState(open=False)

                # If flat, consider entering
                can_enter = (current_units == 0) and (not st.open) and ((now_wall - st.last_order_time) >= cfg.order_cooldown)
                if can_enter:
                    try:
                        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                        net.train()
                        _, mu_t, enter_logit_t, v_t = net(xt)
                        mu = mu_t[0].detach().numpy()
                        enter_prob = float(torch.sigmoid(enter_logit_t[0]).item())
                        # Decide to enter
                        if cfg.decision_mode == "threshold":
                            enter = (enter_prob >= cfg.enter_threshold)
                        else:
                            enter = (np.random.rand() < enter_prob)
                        explored = False
                        if not enter and float(cfg.explore_eps) > 0.0 and (random.random() < float(cfg.explore_eps)):
                            enter = True
                            explored = True
                        if not enter:
                            if cfg.log_debug:
                                print(json.dumps({"type": "SKIP_ENTER", "time": ts, "enter_prob": enter_prob}), flush=True)
                        else:
                            # Sample z for (height, width) — allow optional heavy exploration early
                            z, _ = sample_gaussian(mu, cfg.actor_sigma)
                            outs = decode_triangle(z, cfg.min_pips, cfg.max_pips, cfg.min_width, cfg.max_width)
                            height_pips = float(outs["height_pips"]) if float(outs["height_pips"]) > 0 else float(cfg.min_pips)
                            width_candles = float(outs["width_candles"]) if float(outs["width_candles"]) > 0 else float(cfg.min_width)

                            units = int(round(cfg.units))
                            if units >= cfg.min_units:
                                try:
                                    order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                               units=units, tp_pips=None, sl_pips=None,
                                                               anchor=None, client_tag="tri-ac", client_comment="triangle open",
                                                               fifo_safe=False, fifo_adjust=False)
                                    current_units = refresh_units(api, account_id, args.instrument)
                                    st.open = True
                                    st.last_order_time = now_wall
                                    st.entry_features = x.copy()
                                    st.entry_z = z.astype(np.float32)
                                    st.entry_enter_logit = float(np.log(enter_prob + 1e-12) - np.log(1.0 - enter_prob + 1e-12) if 0.0 < enter_prob < 1.0 else (10.0 if enter_prob >= 1.0 else -10.0))
                                    st.entry_v = float(v_t[0].item())
                                    st.entry_nav = nav_estimate
                                    st.entry_wall_ts = now_wall
                                    st.entry_mid = mid
                                    st.pip_size = float(pip_size)
                                    st.height_pips = height_pips
                                    st.width_candles = width_candles
                                    print(json.dumps({
                                        "type": "TRI_OPEN",
                                        "time": ts,
                                        "units": units,
                                        "height_pips": height_pips,
                                        "width_candles": width_candles,
                                        "width_granularity": cfg.width_granularity,
                                        "enter_prob": enter_prob,
                                        "explore": explored,
                                        "order": (order.get("response") if isinstance(order, dict) else None),
                                    }), flush=True)
                                except Exception as exc:
                                    print(json.dumps({"error": str(exc)}), flush=True)
                    except Exception:
                        pass

                # Autosave
                if cfg.autosave_secs > 0 and (now_wall - getattr(main, "_last_save", 0.0)) >= cfg.autosave_secs:
                    try:
                        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
                        torch.save({
                            "model": net.state_dict(),
                            "opt": opt.state_dict(),
                            "cfg": cfg.__dict__,
                        }, cfg.model_path)
                        main._last_save = now_wall  # type: ignore[attr-defined]
                        print(json.dumps({"type": "SAVED", "path": cfg.model_path}), flush=True)
                    except Exception:
                        pass

        except KeyboardInterrupt:
            print("Interrupted. Exiting.", flush=True)
            break
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
