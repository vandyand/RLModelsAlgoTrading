#!/usr/bin/env python3
"""
Multi-20 Trailing-Stop Actor-Critic Trader (bi-directional) - Live OANDA

- Trades up to 20 instruments concurrently (defaults to DEFAULT_OANDA_20)
- Shared feature stack reused from scalar-threshold trader (tick+DOM+candles)
- Policy outputs discrete actions {-1: short, 0: flat, +1: long}
- Exits are handled exclusively via volatility-aware trailing stops
- Reward per episode = direction-aware normalized price delta * reward_scale

Requirements: OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import atexit
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments_ep
import oandapyV20.endpoints.accounts as accounts_ep
import oandapyV20.endpoints.orders as orders_ep
import oandapyV20.endpoints.positions as positions_ep
import oandapyV20.endpoints.transactions as transactions_ep

# Repo path
TRAILING_MODE_ALIASES = {
    "atr": "atr",
    "atr_vol": "atr",
    "vol": "atr",
    "volatility": "atr",
    "pip": "pip",
    "pips": "pip",
    "pip_fixed": "pip",
    "fixed_pip": "pip",
    "tick": "tick_vol",
    "tick_vol": "tick_vol",
    "tickvol": "tick_vol",
    "tick-vol": "tick_vol",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FOREX_DIR = os.path.dirname(os.path.dirname(__file__))
if FOREX_DIR not in sys.path:
    sys.path.append(FOREX_DIR)
# Instruments default universe
try:
    from forex_rl.actor_critic.multi20_offline_actor_critic import DEFAULT_OANDA_20  # type: ignore
except Exception:
    DEFAULT_OANDA_20 = [
        "EUR_USD","USD_JPY","GBP_USD","AUD_USD","USD_CHF",
        "USD_CAD","NZD_USD","EUR_JPY","GBP_JPY","EUR_GBP",
        "EUR_CHF","EUR_AUD","EUR_CAD","GBP_CHF","AUD_JPY",
        "AUD_CHF","CAD_JPY","NZD_JPY","GBP_AUD","AUD_NZD",
    ]


def pip_size(symbol: str) -> float:
    symbol = (symbol or "").upper()
    if symbol.endswith("JPY") or symbol.startswith("JPY_"):
        return 0.01
    return 0.0001


def canonical_trailing_mode(value: Optional[str]) -> str:
    if value is None:
        return "atr"
    key = str(value).strip().lower()
    return TRAILING_MODE_ALIASES.get(key, "atr")


def quantize_distance(value: float, precision: int) -> float:
    if precision < 0:
        return value
    step = 10 ** (-precision)
    rounded = round(round(value / step) * step, precision)
    return max(step, rounded)

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
            self.m1.clear(); [self.m1.append(b) for b in bars]
            self.last_m1_fetch = time.time()
        if m5_count > 0:
            bars = self._fetch("M5", m5_count)
            self.m5.clear(); [self.m5.append(b) for b in bars]
            self.last_m5_fetch = time.time()
        if h1_count > 0:
            bars = self._fetch("H1", h1_count)
            self.h1.clear(); [self.h1.append(b) for b in bars]
            self.last_h1_fetch = time.time()
        if d_count > 0:
            bars = self._fetch("D", d_count)
            self.d1.clear(); [self.d1.append(b) for b in bars]
            self.last_d_fetch = time.time()
        if w_count > 0:
            bars = self._fetch("W", w_count)
            self.w1.clear(); [self.w1.append(b) for b in bars]
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


def atr_from_bars(bars: Deque[Dict[str, Any]], period: int) -> float:
    if len(bars) < period + 1:
        return 0.0
    close = [b.get("c") or 0.0 for b in bars]
    high = [b.get("h") or 0.0 for b in bars]
    low = [b.get("l") or 0.0 for b in bars]
    trs: List[float] = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        trs.append(tr)
    if len(trs) < 1:
        return 0.0
    return float(np.mean(trs[-period:]))


def tick_volatility(fb: FeatureBuilder, window: int = 120) -> float:
    if len(fb.mid_window) < 10:
        return 0.0
    arr = np.array(list(fb.mid_window)[-window:], dtype=float)
    if len(arr) < 2:
        return 0.0
    returns = np.diff(np.log(arr))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns) * arr[-1])

class PerInstrumentEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 192), nn.ReLU(),
            nn.Linear(192, 192), nn.ReLU(),
            nn.Linear(192, 192), nn.ReLU(),
            nn.Linear(192, embed_dim), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.norm(h)


class HeadMLP(nn.Module):
    def __init__(self, in_dim: int, head_hidden: int = 96, value_maxabs: float = 2.0, value_bounded: bool = True) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(head_hidden, 3)
        self.critic = nn.Linear(head_hidden, 1)
        self.value_maxabs = float(value_maxabs)
        self.value_bounded = bool(value_bounded)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        logits = self.actor(h)
        raw_v = self.critic(h).squeeze(-1)
        v = (torch.tanh(raw_v) * self.value_maxabs) if self.value_bounded else raw_v
        return logits, v


class MultiInstrumentTrailingNet(nn.Module):
    def __init__(self, per_inst_dim: int, num_instruments: int, embed_dim: int = 96, context_dim: int = 384, head_hidden: int = 96, value_maxabs: float = 2.0, value_bounded: bool = True) -> None:
        super().__init__()
        self.num_instruments = num_instruments
        self.per_inst_dim = per_inst_dim
        self.local_encoder = PerInstrumentEncoder(per_inst_dim, embed_dim)
        self.context = nn.Sequential(
            nn.Linear(embed_dim * num_instruments, context_dim), nn.ReLU(),
            nn.Linear(context_dim, context_dim), nn.ReLU(),
            nn.Linear(context_dim, context_dim), nn.ReLU(),
        )
        self.context_norm = nn.LayerNorm(context_dim)
        self.heads = nn.ModuleList([
            HeadMLP(context_dim + embed_dim, head_hidden, value_maxabs=value_maxabs, value_bounded=value_bounded)
            for _ in range(num_instruments)
        ])

    def forward(self, x_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x_concat.size(0)
        feat = x_concat[:, : self.num_instruments * self.per_inst_dim]
        fn = self.num_instruments
        local_in = feat.view(B * fn, self.per_inst_dim)
        emb = self.local_encoder(local_in)
        emb_bn = emb.view(B, fn, -1)
        context_in = emb_bn.reshape(B, fn * emb_bn.size(-1))
        ctx = self.context_norm(self.context(context_in))
        logits_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        for i in range(fn):
            hi = torch.cat([ctx, emb_bn[:, i, :]], dim=-1)
            logits_i, v_i = self.heads[i](hi)
            logits_list.append(logits_i)
            values_list.append(v_i)
        logits = torch.stack(logits_list, dim=1)
        values = torch.stack(values_list, dim=1)
        return emb_bn, logits, values


class FlatTrailingNet(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int = 768, value_maxabs: float = 2.0, value_bounded: bool = True) -> None:
        super().__init__()
        self.num_instruments = num_instruments
        self.value_maxabs = float(value_maxabs)
        self.value_bounded = bool(value_bounded)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backbone_norm = nn.LayerNorm(hidden)
        self.actor = nn.Linear(hidden, num_instruments * 3)
        self.value = nn.Linear(hidden, num_instruments)

    def forward(self, x_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone_norm(self.backbone(x_concat))
        logits = self.actor(h).view(-1, self.num_instruments, 3)
        raw_v = self.value(h)
        values = (torch.tanh(raw_v) * self.value_maxabs) if self.value_bounded else raw_v
        return torch.empty(0, device=x_concat.device), logits, values

def _summarize_order(order_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(order_obj, dict):
        return {}
    try:
        resp = (order_obj or {}).get("response") or order_obj
        create = resp.get("orderCreateTransaction", {})
        fill = resp.get("orderFillTransaction", {})
        instrument = fill.get("instrument") or create.get("instrument")
        out = {
            "instrument": instrument,
            "order_id": create.get("id"),
            "fill_id": fill.get("id"),
            "time": fill.get("time") or create.get("time"),
            "units": fill.get("units") or create.get("units"),
            "price": fill.get("price"),
            "pl": fill.get("pl") or fill.get("realizedPL"),
            "balance": fill.get("accountBalance"),
            "reason": fill.get("reason") or create.get("reason"),
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception:
        try:
            return {"raw": str(order_obj)[:300]}
        except Exception:
            return {"raw": "<unprintable>"}


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


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _fetch_instrument_meta(api: API, account_id: str, instruments: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    chunk: List[str] = []
    for inst in instruments:
        if inst:
            chunk.append(inst)
        if len(chunk) == 10:
            _load_instrument_chunk(api, account_id, chunk, out)
            chunk = []
    if chunk:
        _load_instrument_chunk(api, account_id, chunk, out)
    return out


def _load_instrument_chunk(api: API, account_id: str, chunk: List[str], out: Dict[str, Dict[str, float]]) -> None:
    params = {"instruments": ",".join(chunk)}
    req = accounts_ep.AccountInstruments(accountID=account_id, params=params)
    try:
        resp = api.request(req)
    except Exception:
        return
    for spec in resp.get("instruments", []):
        name = spec.get("name") or spec.get("instrument")
        if not name:
            continue
        try:
            precision = int(spec.get("displayPrecision", 5))
        except Exception:
            precision = 5
        try:
            min_trailing = float(spec.get("minimumTrailingStopDistance") or 0.0)
        except Exception:
            min_trailing = 0.0
        out[str(name)] = {"precision": precision, "min_trailing": min_trailing}


def _oanda_account_summary(api: API, account_id: str) -> Tuple[Optional[float], Optional[int]]:
    try:
        resp = api.request(accounts_ep.AccountSummary(accountID=account_id))
        account = resp.get("account", resp)
        nav_val = account.get("NAV") or account.get("balance")
        last_txn = account.get("lastTransactionID")
        nav = float(nav_val) if nav_val is not None else None
        last_id = int(last_txn) if last_txn is not None else None
        return nav, last_id
    except Exception:
        return None, None


def _oanda_refresh_positions(
    api: API,
    account_id: str,
    instruments: List[str],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    units: Dict[str, int] = {inst: 0 for inst in instruments}
    pnl: Dict[str, float] = {inst: 0.0 for inst in instruments}
    try:
        resp = api.request(positions_ep.OpenPositions(accountID=account_id))
    except Exception:
        return units, pnl
    for p in resp.get("positions", []):
        inst = str(p.get("instrument") or "")
        if inst not in units:
            continue
        long_side = p.get("long") or {}
        short_side = p.get("short") or {}
        long_units = _safe_float(long_side.get("units"))
        short_units = _safe_float(short_side.get("units"))
        units[inst] = int(round(long_units + short_units))
        pnl[inst] = _safe_float(long_side.get("unrealizedPL")) + _safe_float(short_side.get("unrealizedPL"))
    return units, pnl


def _oanda_place_market_order_with_trailing(
    api: API,
    account_id: str,
    instrument: str,
    units: int,
    trailing_distance: float,
    display_precision: int,
    client_tag: Optional[str],
    client_comment: Optional[str],
) -> Dict[str, Any]:
    if units == 0:
        return {}
    order_payload: Dict[str, Any] = {
        "order": {
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "instrument": instrument,
            "units": str(int(units)),
        }
    }
    if client_tag or client_comment:
        order_payload["order"]["clientExtensions"] = {}
        if client_tag:
            order_payload["order"]["clientExtensions"]["tag"] = client_tag
        if client_comment:
            order_payload["order"]["clientExtensions"]["comment"] = client_comment
    if trailing_distance > 0.0:
        precision = max(0, int(display_precision))
        distance_str = f"{float(trailing_distance):.{precision}f}"
        order_payload["order"]["trailingStopLossOnFill"] = {
            "distance": distance_str,
            "timeInForce": "GTC",
        }
    req = orders_ep.OrderCreate(accountID=account_id, data=order_payload)
    return api.request(req)


def _oanda_close_position_all(api: API, account_id: str, instrument: str, units: int) -> Optional[Dict[str, Any]]:
    if units == 0:
        return None
    data = {"longUnits": "ALL"} if units > 0 else {"shortUnits": "ALL"}
    req = positions_ep.PositionClose(accountID=account_id, instrument=instrument, data=data)
    return api.request(req)


def _oanda_fetch_transactions_since(
    api: API,
    account_id: str,
    last_transaction_id: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if last_transaction_id is None:
        return [], None
    try:
        req = transactions_ep.TransactionsSinceID(accountID=account_id, params={"id": str(int(last_transaction_id))})
        resp = api.request(req)
    except Exception:
        return [], last_transaction_id
    txns = resp.get("transactions", [])
    new_last = resp.get("lastTransactionID")
    try:
        last_val = int(new_last)
    except Exception:
        last_val = last_transaction_id
    return txns, last_val

@dataclass
class TradeState:
    open: bool = False
    entry_features: Optional[np.ndarray] = None
    entry_action_idx: Optional[int] = None
    entry_direction: int = 0
    entry_nav: Optional[float] = None
    entry_mid: Optional[float] = None
    entry_fill_price: Optional[float] = None
    entry_step: Optional[int] = None
    entry_ts: Optional[float] = None
    trade_id: Optional[str] = None
    trailing_distance: Optional[float] = None
    order_id: Optional[str] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


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


def compute_trailing_distance(
    mode: str,
    instrument: str,
    fb: FeatureBuilder,
    cache: CandleCache,
    atr_period: int,
    atr_mult: float,
    pip_distance: float,
    tick_vol_mult: float,
    min_distance_pips: float,
) -> float:
    psize = pip_size(instrument)
    min_abs = max(min_distance_pips * psize, psize)
    try:
        if mode == "pip":
            return max(min_abs, psize * pip_distance)
        if mode == "tick_vol":
            vol = tick_volatility(fb)
            if vol <= 0:
                vol = psize * pip_distance
            return max(min_abs, vol * tick_vol_mult)
        atr_val = atr_from_bars(cache.m1, atr_period)
        if atr_val <= 0:
            atr_val = psize * pip_distance
        return max(min_abs, atr_val * atr_mult)
    except Exception:
        return min_abs


def normalized_trade_reward(entry_px: float, exit_px: float, direction: int, scale: float, neg_coef: float) -> float:
    if entry_px <= 0 or exit_px <= 0 or direction == 0:
        return 0.0
    if direction > 0:
        raw = (exit_px - entry_px) / (exit_px + entry_px)
    else:
        raw = (entry_px - exit_px) / (exit_px + entry_px)
    reward = raw * scale
    if reward < 0:
        reward *= neg_coef
    return float(reward)

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-20 Trailing Stop Trader")
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-id", type=int, default=1)
    parser.add_argument("--instruments", default=",".join(DEFAULT_OANDA_20))
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--account-suffix", type=int, default=None)
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"), help="OANDA REST token")
    parser.add_argument("--client-tag", default="multi20-trailing")
    parser.add_argument("--client-comment", default="multi20_trailing_trader")
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--feature-ticks", type=int, default=240)
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
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--reward-transform", choices=["none", "exp", "expm1"], default="none")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--value-maxabs", type=float, default=2.0)
    parser.add_argument("--value-target-scale", type=float, default=1.0)
    parser.add_argument("--value-bounded", action="store_true")
    parser.add_argument("--no-value-bounded", dest="value_bounded", action="store_false")
    parser.set_defaults(value_bounded=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--arch", choices=["flat", "modular"], default="flat")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--flat-hidden", type=int, default=512)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/multi20_trailing_v001.pt")
    parser.add_argument("--from-checkpoint", action="store_true")
    parser.add_argument("--flatten-on-start", action="store_true")
    parser.add_argument("--no-flatten-on-start", dest="flatten_on_start", action="store_false")
    parser.set_defaults(flatten_on_start=True)
    parser.add_argument("--flatten-on-exit", action="store_true")
    parser.add_argument("--no-flatten-on-exit", dest="flatten_on_exit", action="store_false")
    parser.set_defaults(flatten_on_exit=True)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    parser.add_argument("--transactions-poll-secs", type=float, default=5.0)
    parser.add_argument("--train-stats-secs", type=float, default=30.0)
    parser.add_argument("--epochs-per-close", type=int, default=1)
    parser.add_argument("--trajectory-max-steps", type=int, default=64, help="Number of pre-entry decisions to replay when a trade closes.")
    parser.add_argument("--adv-norm-eps", type=float, default=1e-6, help="Stability constant for advantage normalization.")
    parser.add_argument("--replay-shuffle", action="store_true", help="Shuffle stored trajectory steps before each replay epoch.")
    parser.add_argument("--no-replay-shuffle", dest="replay_shuffle", action="store_false")
    parser.set_defaults(replay_shuffle=True)
    # Trailing stop controls
    parser.add_argument("--trailing-mode", choices=["atr", "pip", "tick_vol"], default="atr")
    parser.add_argument("--trailing-atr-period", type=int, default=20)
    parser.add_argument("--trailing-atr-mult", type=float, default=1.0)
    parser.add_argument("--trailing-pip-distance", type=float, default=25.0)
    parser.add_argument("--trailing-tick-vol-mult", type=float, default=3.0)
    parser.add_argument("--trailing-min-distance-pips", type=float, default=5.0)
    parser.add_argument("--trailing-min-step-pips", type=float, default=1.5)

    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    cfg_path = str(args.config) if args.config else os.path.join(script_dir, f"multi20_trailing_config_{int(args.config_id):03d}.json")
    config_dict: Dict[str, Any] = {}
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config_dict = json.load(f)
    except Exception:
        config_dict = {}
    for k, v in (config_dict or {}).items():
        if hasattr(args, k):
            try:
                if getattr(args, k) == parser.get_default(k):
                    setattr(args, k, v)
            except Exception:
                pass
    try:
        print(json.dumps({"type": "CONFIG", "path": cfg_path, "loaded": bool(config_dict)}), flush=True)
    except Exception:
        pass

    normalized_mode = canonical_trailing_mode(getattr(args, "trailing_mode", "atr"))
    if normalized_mode != getattr(args, "trailing_mode", None):
        try:
            print(json.dumps({"type": "TRAILING_MODE_NORMALIZED", "original": getattr(args, "trailing_mode", None), "normalized": normalized_mode}), flush=True)
        except Exception:
            pass
    args.trailing_mode = normalized_mode
    try:
        print(json.dumps({
            "type": "TRAILING_CONFIG",
            "mode": args.trailing_mode,
            "atr_period": int(args.trailing_atr_period),
            "atr_mult": float(args.trailing_atr_mult),
            "pip_distance": float(args.trailing_pip_distance),
            "tick_vol_mult": float(args.trailing_tick_vol_mult),
            "min_distance_pips": float(args.trailing_min_distance_pips),
        }), flush=True)
    except Exception:
        pass

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()][:20]
    if len(instruments) == 0:
        instruments = DEFAULT_OANDA_20
    num_inst = len(instruments)

    base_account = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = args.access_token or os.environ.get("OANDA_DEMO_KEY")
    if not base_account or not access_token:
        raise RuntimeError("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY or pass --account-id/--access-token.")
    if args.account_suffix is not None:
        if len(str(base_account)) < 3:
            raise RuntimeError("Account id must be at least 3 chars to use --account-suffix")
        account_id = str(base_account)[:-3] + f"{int(args.account_suffix):03d}"
    else:
        account_id = str(base_account)
    api = API(access_token=access_token, environment=args.environment)
    instrument_meta = _fetch_instrument_meta(api, account_id, instruments)

    fb: Dict[str, FeatureBuilder] = {inst: FeatureBuilder(args.feature_ticks) for inst in instruments}
    cc: Dict[str, CandleCache] = {inst: CandleCache(api, inst, h1_len=args.h1_bars, d_len=args.d_bars, w_len=args.w_bars) for inst in instruments}
    for inst in instruments:
        cc[inst].backfill(args.m1_bars, args.m5_bars, args.h1_bars, args.d_bars, args.w_bars)

    dom_extra = 0
    per_inst_dim = 19 + 28 + dom_extra + (16 * 5) + 3
    input_dim = per_inst_dim * num_inst + 10
    if str(args.arch) == "flat":
        net = FlatTrailingNet(input_dim=input_dim, num_instruments=num_inst, hidden=int(args.flat_hidden), value_maxabs=float(args.value_maxabs), value_bounded=bool(args.value_bounded))
    else:
        net = MultiInstrumentTrailingNet(
            per_inst_dim=per_inst_dim,
            num_instruments=num_inst,
            embed_dim=int(args.embed_dim),
            context_dim=int(args.context_dim),
            head_hidden=int(args.head_hidden),
            value_maxabs=float(args.value_maxabs),
            value_bounded=bool(args.value_bounded),
        )
    opt = optim.Adam(net.parameters(), lr=float(args.lr))
    net.train()

    if args.model_path:
        if str(args.model_path) == "forex-rl/actor-critic/checkpoints/multi20_trailing_v001.pt":
            args.model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "multi20_trailing_v001.pt")
    try:
        if args.model_path and (bool(args.from_checkpoint) or os.path.exists(str(args.model_path))):
            ckpt = torch.load(str(args.model_path), map_location="cpu")
            if isinstance(ckpt, dict):
                net.load_state_dict(ckpt.get("model", ckpt))
                opt_state = ckpt.get("opt")
                if opt_state:
                    opt.load_state_dict(opt_state)
            else:
                net.load_state_dict(ckpt)
            print(json.dumps({"type": "LOADED", "path": str(args.model_path)}), flush=True)
    except Exception as exc:
        print(json.dumps({"type": "LOAD_WARN", "error": str(exc)}), flush=True)

    st: Dict[str, TradeState] = {inst: TradeState(open=False) for inst in instruments}
    inst_index: Dict[str, int] = {inst: idx for idx, inst in enumerate(instruments)}
    trade_id_map: Dict[str, str] = {}
    last_logits: Dict[str, Optional[np.ndarray]] = {inst: None for inst in instruments}
    last_values: Dict[str, Optional[float]] = {inst: None for inst in instruments}
    last_probs: Dict[str, Optional[List[float]]] = {inst: None for inst in instruments}
    last_features: Dict[str, np.ndarray] = {inst: np.zeros(per_inst_dim, dtype=np.float32) for inst in instruments}
    traj_max = max(1, int(args.trajectory_max_steps))
    pre_entry_buffers: Dict[str, Deque[Dict[str, Any]]] = {inst: deque(maxlen=traj_max) for inst in instruments}
    adv_stats: Dict[str, Dict[str, float]] = {
        inst: {"count": 0.0, "mean": 0.0, "m2": 1.0} for inst in instruments
    }

    last_nav, last_transaction_id = _oanda_account_summary(api, account_id)
    if last_nav is None:
        last_nav = 1.0
    if last_transaction_id is None:
        last_transaction_id = 0
    last_nav_poll = time.time()
    last_pos_refresh = time.time()
    last_txn_poll = time.time()
    units_map, pos_pnl_map = _oanda_refresh_positions(api, account_id, instruments)

    if args.flatten_on_start:
        for inst in instruments:
            try:
                u = units_map.get(inst, 0)
                if u != 0:
                    resp = _oanda_close_position_all(api, account_id, inst, u)
                    print(json.dumps({"type": "AUTO_FLATTEN_START", "instrument": inst, "units": u, "response": resp}), flush=True)
            except Exception:
                pass
        units_map, pos_pnl_map = _oanda_refresh_positions(api, account_id, instruments)

    if args.flatten_on_exit:
        def _on_exit() -> None:
            try:
                refreshed_units, _ = _oanda_refresh_positions(api, account_id, instruments)
                for inst in instruments:
                    u = refreshed_units.get(inst, 0)
                    if u != 0:
                        resp = _oanda_close_position_all(api, account_id, inst, u)
                        print(json.dumps({"type": "AUTO_FLATTEN_EXIT", "instrument": inst, "units": u, "response": resp}), flush=True)
            except Exception:
                pass
        atexit.register(_on_exit)

    stats_reward: Dict[str, Deque[float]] = {inst: deque(maxlen=1000) for inst in instruments}
    stats_duration: Dict[str, Deque[float]] = {inst: deque(maxlen=500) for inst in instruments}
    last_stats_emit = time.time()
    last_grad_norm: Optional[float] = None
    rew_ema: Dict[str, float] = {inst: 1.0 for inst in instruments}

    def _make_traj_step(feat: np.ndarray, action_index: int) -> Dict[str, Any]:
        return {
            "features": np.array(feat, dtype=np.float32).copy(),
            "action_idx": int(action_index),
        }

    def _value_target_from_reward(reward_val: float) -> float:
        if bool(args.value_bounded):
            return float(np.tanh(reward_val / max(1e-6, float(args.value_target_scale))) * float(args.value_maxabs))
        return reward_val

    def _normalize_advantage(inst: str, adv_val: float) -> float:
        stats = adv_stats.get(inst)
        if stats is None:
            return adv_val
        count = stats["count"]
        mean = stats["mean"]
        m2 = max(stats["m2"], float(args.adv_norm_eps))
        if count >= 2.0:
            variance = m2 / max(1.0, count - 1.0)
            std = max(float(args.adv_norm_eps), math.sqrt(max(variance, 0.0)))
            norm_adv = (adv_val - mean) / std
        else:
            norm_adv = adv_val
        count += 1.0
        delta = adv_val - mean
        mean += delta / count
        delta2 = adv_val - mean
        m2 += delta * delta2
        stats["count"] = count
        stats["mean"] = mean
        stats["m2"] = max(m2, float(args.adv_norm_eps))
        return float(norm_adv)

    def _handle_trade_close(inst: str, trade_id: str, closed: Dict[str, Any], tx: Dict[str, Any]) -> None:
        nonlocal last_grad_norm
        ts = st.get(inst)
        if ts is None or not ts.open:
            trade_id_map.pop(trade_id, None)
            return
        exit_price = _safe_float(closed.get("price"))
        if exit_price <= 0:
            exit_price = _safe_float(tx.get("price"))
        if exit_price <= 0 and fb[inst].mid_window:
            exit_price = float(fb[inst].mid_window[-1])
        entry_price = float(ts.entry_fill_price or ts.entry_mid or exit_price)
        reward_raw = normalized_trade_reward(
            entry_px=entry_price,
            exit_px=float(exit_price),
            direction=int(ts.entry_direction),
            scale=float(args.reward_scale),
            neg_coef=float(args.neg_reward_coef),
        )
        reward_proc = reward_raw
        if float(args.reward_clip) > 0.0:
            reward_proc = float(np.clip(reward_proc, -float(args.reward_clip), float(args.reward_clip)))
        if str(args.reward_transform) == "expm1":
            reward_proc = float(np.expm1(reward_proc))
        elif str(args.reward_transform) == "exp":
            reward_proc = float(np.exp(reward_proc))
        rew_ema_prev = rew_ema.get(inst, 1.0)
        rew_ema_new = 0.99 * rew_ema_prev + 0.01 * max(1e-6, abs(reward_proc))
        rew_ema[inst] = max(1e-6, rew_ema_new)
        reward_norm = reward_proc / rew_ema[inst]
        idx = inst_index.get(inst, 0)
        action_idx = ts.entry_action_idx if ts.entry_action_idx is not None else (2 if ts.entry_direction > 0 else 0)
        epochs_close = max(1, int(args.epochs_per_close))
        traj_steps = list(ts.trajectory) if ts.trajectory else []
        if not traj_steps:
            fallback_feat = ts.entry_features
            if fallback_feat is None:
                fallback_feat = last_features.get(inst, np.zeros(per_inst_dim, dtype=np.float32))
            traj_steps = [_make_traj_step(fallback_feat, action_idx)]
        total_steps = len(traj_steps)
        gamma_val = max(0.0, float(args.gamma))
        traj_with_discounts: List[Tuple[Dict[str, Any], float]] = []
        for step_pos, step_payload in enumerate(traj_steps):
            power = max(0, total_steps - 1 - step_pos)
            discount = gamma_val ** power if gamma_val > 0.0 else 1.0
            traj_with_discounts.append((step_payload, discount))

        base_traj = list(traj_with_discounts)
        for _ in range(epochs_close):
            step_iter = list(base_traj)
            if bool(args.replay_shuffle) and len(step_iter) > 1:
                random.shuffle(step_iter)
            for step_payload, discount in step_iter:
                features = step_payload.get("features")
                step_action_idx = int(step_payload.get("action_idx", action_idx))
                if features is None:
                    continue
                opt.zero_grad()
                x_entry = torch.from_numpy(np.array(features, dtype=np.float32))[None, :]
                _, logits_all2, values_all2 = net(x_entry)
                logits_close = logits_all2[0, idx]
                dist_close = torch.distributions.Categorical(logits=logits_close)
                v_pred = values_all2[0, idx]
                action_tensor = torch.tensor(step_action_idx, dtype=torch.long)
                logprob = dist_close.log_prob(action_tensor)
                reward_component = reward_norm * discount
                v_target = _value_target_from_reward(reward_component)
                raw_advantage = v_target - float(v_pred.detach().item())
                norm_advantage = _normalize_advantage(inst, raw_advantage)
                actor_loss = -torch.tensor(norm_advantage, dtype=torch.float32) * logprob
                critic_loss = F.smooth_l1_loss(v_pred, torch.tensor(v_target, dtype=torch.float32))
                entropy_bonus = dist_close.entropy()
                loss = actor_loss + float(args.value_coef) * critic_loss - float(args.entropy_coef) * entropy_bonus
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                opt.step()
                last_grad_norm = _compute_grad_norm(net.parameters())

        dur_sec = max(0.0, float(time.time() - float(ts.entry_ts or time.time())))
        stats_reward[inst].append(float(reward_raw))
        stats_duration[inst].append(float(dur_sec))
        trade_id_map.pop(trade_id, None)
        units_map[inst] = 0
        pos_pnl_map[inst] = 0.0
        if inst in pre_entry_buffers:
            pre_entry_buffers[inst].clear()
        st[inst] = TradeState(open=False)
        print(json.dumps({
            "type": "EP_END",
            "instrument": inst,
            "reward": float(reward_raw),
            "reward_norm": float(reward_norm),
            "v_target": float(v_target),
            "duration_sec": float(dur_sec),
            "entry_px": float(entry_price),
            "exit_px": float(exit_price),
            "reason": tx.get("reason"),
            "trade_id": trade_id,
            "transaction_id": tx.get("id"),
        }), flush=True)

    def _process_transactions(txns: List[Dict[str, Any]]) -> None:
        for tx in txns:
            trades_closed = list(tx.get("tradesClosed") or [])
            trades_closed += list(tx.get("tradesReduced") or [])
            if not trades_closed:
                continue
            for closed in trades_closed:
                trade_id = str(closed.get("tradeID") or "")
                if not trade_id:
                    continue
                inst = trade_id_map.get(trade_id) or tx.get("instrument")
                if not inst or inst not in st:
                    continue
                _handle_trade_close(inst, trade_id, closed, tx)

    step_idx = 0
    stream = pricing.PricingStream(accountID=account_id, params={"instruments": ",".join(instruments)})
    while True:
        try:
            for msg in api.request(stream):
                tnow = time.time()
                mtype = msg.get("type")
                if mtype == "HEARTBEAT":
                    if (tnow - last_nav_poll) >= float(args.nav_poll_secs):
                        nv, _ = _oanda_account_summary(api, account_id)
                        if nv is not None:
                            last_nav = nv
                        last_nav_poll = tnow
                    if (tnow - last_pos_refresh) >= float(args.pos_refresh_secs):
                        units_map, pos_pnl_map = _oanda_refresh_positions(api, account_id, instruments)
                        last_pos_refresh = tnow
                    if (tnow - last_txn_poll) >= float(args.transactions_poll_secs):
                        txns, last_transaction_id = _oanda_fetch_transactions_since(api, account_id, last_transaction_id)
                        if txns:
                            _process_transactions(txns)
                        last_txn_poll = tnow
                    for inst in instruments:
                        cc[inst].maybe_refresh(args.m1_refresh_secs, args.m5_refresh_secs, args.h1_refresh_secs, args.d_refresh_secs, args.w_refresh_secs)
                    if (tnow - last_stats_emit) >= float(args.train_stats_secs):
                        try:
                            summary = {
                                inst: {
                                    "reward": (float(np.mean(list(stats_reward[inst]))) if len(stats_reward[inst]) > 0 else None),
                                    "duration_sec": (float(np.mean(list(stats_duration[inst]))) if len(stats_duration[inst]) > 0 else None),
                                    "prob": last_probs.get(inst),
                                    "value": last_values.get(inst),
                                }
                                for inst in instruments
                            }
                            print(json.dumps({"type": "TRAIN_STATS", "grad_norm": (round(float(last_grad_norm),4) if last_grad_norm else None), "summary": summary}), flush=True)
                        except Exception:
                            pass
                        last_stats_emit = tnow
                    continue

                if mtype != "PRICE":
                    continue

                inst = msg.get("instrument")
                if inst not in fb:
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

                mid = fb[inst].update_tick(bid, ask)
                x_tick = fb[inst].compute_tick_features()
                x_dom_raw = fb[inst].compute_dom_features(bids, asks)
                dom_c = 5.0
                x_dom = dom_c * np.tanh(x_dom_raw / max(1e-9, dom_c))
                m1_feats = candle_features(cc[inst].m1)
                m5_feats = candle_features(cc[inst].m5)
                h1_feats = candle_features(cc[inst].h1)
                d1_feats = candle_features(cc[inst].d1)
                w1_feats = candle_features(cc[inst].w1)
                if x_tick is None:
                    continue

                try:
                    if units_map.get(inst, 0) != 0:
                        dur_sec = float(max(0.0, (time.time() - float(st[inst].entry_ts or time.time()))))
                        direction_feat = 1.0 if units_map.get(inst, 0) > 0 else -1.0
                        cur_reward = float(pos_pnl_map.get(inst, 0.0)) * float(args.reward_scale)
                    else:
                        dur_sec = 0.0
                        direction_feat = 0.0
                        cur_reward = 0.0
                except Exception:
                    dur_sec = 0.0
                    direction_feat = 0.0
                    cur_reward = 0.0

                last_features[inst] = np.concatenate([
                    x_tick, x_dom,
                    m1_feats, m5_feats, h1_feats, d1_feats, w1_feats,
                    np.array([dur_sec, direction_feat, cur_reward], dtype=np.float32)
                ]).astype(np.float32)

                try:
                    ts_str = msg.get("time") or msg.get("timestamp")
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                t_feats = time_cyclical_features(dt)
                X_list = [last_features[inst2] for inst2 in instruments]
                X_list.append(t_feats)
                x_full = np.concatenate(X_list).astype(np.float32)
                if not np.all(np.isfinite(x_full)):
                    continue
                xt = torch.from_numpy(x_full)[None, :]

                net.eval()
                with torch.no_grad():
                    _, logits_all, values_all = net(xt)
                idx = instruments.index(inst)
                logits_inst = logits_all[0, idx]
                values_inst = values_all[0, idx]
                last_logits[inst] = logits_inst.cpu().numpy()
                last_values[inst] = float(values_inst.item())
                probs = torch.softmax(logits_inst, dim=-1)
                last_probs[inst] = probs.detach().cpu().numpy().tolist()

                action_idx = None
                action_dir = 0
                if not st[inst].open:
                    dist_live = torch.distributions.Categorical(logits=logits_inst)
                    action_idx = int(dist_live.sample().item())
                    action_dir = [-1, 0, 1][action_idx]
                    if int(args.trajectory_max_steps) > 0:
                        pre_entry_buffers[inst].append(_make_traj_step(x_full, action_idx))

                def enter_position(direction: int, action_index: int) -> None:
                    units_delta = int(args.units) * (1 if direction > 0 else -1)
                    meta = instrument_meta.get(inst, {})
                    disp_prec = int(meta.get("precision", 5))
                    distance_raw = compute_trailing_distance(
                        mode=str(args.trailing_mode),
                        instrument=inst,
                        fb=fb[inst],
                        cache=cc[inst],
                        atr_period=int(args.trailing_atr_period),
                        atr_mult=float(args.trailing_atr_mult),
                        pip_distance=float(args.trailing_pip_distance),
                        tick_vol_mult=float(args.trailing_tick_vol_mult),
                        min_distance_pips=float(args.trailing_min_distance_pips),
                    )
                    min_trailing = float(meta.get("min_trailing", 0.0))
                    distance_quantized = quantize_distance(float(distance_raw), disp_prec)
                    distance_final = max(min_trailing, distance_quantized) if min_trailing > 0.0 else distance_quantized
                    try:
                        order = _oanda_place_market_order_with_trailing(
                            api=api,
                            account_id=account_id,
                            instrument=inst,
                            units=units_delta,
                            trailing_distance=float(distance_final),
                            display_precision=disp_prec,
                            client_tag=str(args.client_tag),
                            client_comment=str(args.client_comment),
                        )
                    except Exception as exc:
                        print(json.dumps({"type": "ENTER_ERROR", "instrument": inst, "error": str(exc)}), flush=True)
                        return
                    units_map[inst] = units_map.get(inst, 0) + units_delta
                    pos_pnl_map[inst] = 0.0
                    st[inst].open = True
                    st[inst].entry_features = x_full.copy()
                    st[inst].entry_action_idx = action_index
                    st[inst].entry_direction = direction
                    st[inst].entry_nav = last_nav
                    st[inst].entry_mid = float(mid)
                    st[inst].entry_step = int(step_idx)
                    st[inst].entry_ts = float(time.time())
                    st[inst].trailing_distance = float(distance_final)
                    st[inst].trajectory = list(pre_entry_buffers[inst])
                    pre_entry_buffers[inst].clear()
                    st[inst].trajectory.append(_make_traj_step(x_full, action_index))
                    fill = (order or {}).get("orderFillTransaction") or {}
                    st[inst].entry_fill_price = _safe_float(fill.get("price")) or float(mid)
                    trades_opened = list(fill.get("tradesOpened") or [])
                    trade_id = None
                    if trades_opened:
                        trade_id = trades_opened[0].get("tradeID")
                    elif fill.get("tradeOpened"):
                        trade_id = (fill.get("tradeOpened") or {}).get("tradeID")
                    if trade_id:
                        st[inst].trade_id = str(trade_id)
                        trade_id_map[str(trade_id)] = inst
                    st[inst].order_id = fill.get("orderID") or fill.get("orderId")
                    print(json.dumps({
                        "type": "ENTER",
                        "instrument": inst,
                        "direction": direction,
                        "units": int(units_delta),
                        "distance": float(distance_final),
                        "distance_meta": {
                            "mode": str(args.trailing_mode),
                            "raw": float(distance_raw),
                            "quantized": float(distance_quantized),
                            "instrument_min": float(min_trailing),
                            "final": float(distance_final),
                        },
                        "prob": last_probs[inst],
                        "order": _summarize_order(order),
                        "trade_id": st[inst].trade_id,
                    }), flush=True)

                if not st[inst].open and action_idx is not None:
                    if action_dir == 1:
                        enter_position(1, action_idx)
                    elif action_dir == -1:
                        enter_position(-1, action_idx)

                if float(args.autosave_secs) > 0 and (tnow - getattr(main, "_last_save", 0.0)) >= float(args.autosave_secs):
                    try:
                        os.makedirs(os.path.dirname(str(args.model_path)), exist_ok=True)
                        payload = {"model": net.state_dict(), "opt": opt.state_dict(), "meta": {"per_inst_dim": per_inst_dim, "num_instruments": num_inst}}
                        torch.save(payload, str(args.model_path))
                        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        base_dir = os.path.dirname(str(args.model_path))
                        base_name = os.path.splitext(os.path.basename(str(args.model_path)))[0]
                        ts_path = os.path.join(base_dir, f"{base_name}_{ts}.pt")
                        torch.save(payload, ts_path)
                        main._last_save = tnow  # type: ignore[attr-defined]
                        print(json.dumps({"type": "SAVED", "path": str(args.model_path), "timestamped_path": ts_path}), flush=True)
                    except Exception:
                        pass

                step_idx += 1
        except Exception as exc:
            print(json.dumps({"type": "STREAM_ERROR", "error": str(exc)}), flush=True)
            time.sleep(5.0)
            continue

if __name__ == "__main__":
    main()
