#!/usr/bin/env python3
"""
Multi-20 Scalar-Threshold Actor-Critic (bi-directional) - Live OANDA

- Trades up to 20 instruments concurrently (defaults to DEFAULT_OANDA_20)
- Shared local encoder per instrument slice -> per-instrument embedding
- Global context trunk on concatenated embeddings
- Per-instrument heads (small 2-layer MLP) outputting scalar logit and value
- Threshold mapping per instrument (enter/exit) with hysteresis, EMA smoothing
- Tick-level optional micro-updates per active instrument
- Episodic update on each instrument's close, tiered bonuses, exp/expm1 transforms

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
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import requests

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments_ep

# Repo path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FOREX_DIR = os.path.dirname(os.path.dirname(__file__))
if FOREX_DIR not in sys.path:
    sys.path.append(FOREX_DIR)
try:
    import broker_ipc  # type: ignore
except Exception:
    broker_ipc = None  # type: ignore

# Instruments default universe
try:
    from forex_rl.actor_critic.multi20_offline_actor_critic import DEFAULT_OANDA_20  # type: ignore
except Exception:
    # Fallback common 20 FX pairs
    DEFAULT_OANDA_20 = [
        "EUR_USD","USD_JPY","GBP_USD","AUD_USD","USD_CHF",
        "USD_CAD","NZD_USD","EUR_JPY","GBP_JPY","EUR_GBP",
        "EUR_CHF","EUR_AUD","EUR_CAD","GBP_CHF","AUD_JPY",
        "AUD_CHF","CAD_JPY","NZD_JPY","GBP_AUD","AUD_NZD",
    ]


@dataclass
class Config:
    instruments: List[str]
    environment: str = "practice"
    units: int = 100
    feature_ticks: int = 240
    reward_scale: float = 10000.0
    neg_reward_coef: float = 0.1
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    lr: float = 1e-3
    entropy_coef: float = 0.01
    autosave_secs: float = 120.0
    model_path: str = "forex-rl/actor-critic/checkpoints/multi20_threshold_v001.pt"
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_clip: float = 0.0  # 0 disables clipping
    adv_clip: float = 5.0
    gamma: float = 0.99
    # Thresholds
    enter_long_thresh: float = 0.6
    exit_long_thresh: float = 0.55
    enter_short_thresh: float = 0.4
    exit_short_thresh: float = 0.45
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


# ---------- Feature engineering (copied/trimmed from scalar_threshold_trader.py) ----------

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
        # Prefer an external HTTP candle cache if configured.
        base_url = os.environ.get("CANDLE_CACHE_BASE_URL")
        # By default we do NOT fall back to direct OANDA when the cache is configured,
        # to avoid silently reintroducing upstream load. Opt-in via CANDLE_CACHE_FALLBACK_TO_OANDA=1.
        fallback_allowed = str(os.environ.get("CANDLE_CACHE_FALLBACK_TO_OANDA", "0")).lower() in ("1", "true", "yes", "y")
        if base_url:
            try:
                url = f"{str(base_url).rstrip('/')}/v3/instruments/{self.instrument}/candles"
                params = {"granularity": granularity, "count": int(count), "price": "M"}
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                out: List[Dict[str, Any]] = []
                for c in data.get("candles", []):
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
            except Exception as exc:
                if not fallback_allowed:
                    raise RuntimeError(f"CANDLE_CACHE_BASE_URL request failed: {exc}") from exc
                # Otherwise, fall through to direct OANDA REST as a backup.
        try:
            req = instruments_ep.InstrumentsCandles(
                instrument=self.instrument,
                params={\"granularity\": granularity, \"count\": count, \"price\": \"M\"},
            )
            resp = self.api.request(req)
            out2: List[Dict[str, Any]] = []
            for c in resp.get(\"candles\", []):
                mid = c.get(\"mid\") or {}
                out2.append({
                    \"time\": c.get(\"time\"),
                    \"complete\": bool(c.get(\"complete\", False)),
                    \"o\": float(mid.get(\"o\")) if mid.get(\"o\") is not None else None,
                    \"h\": float(mid.get(\"h\")) if mid.get(\"h\") is not None else None,
                    \"l\": float(mid.get(\"l\")) if mid.get(\"l\") is not None else None,
                    \"c\": float(mid.get(\"c\")) if mid.get(\"c\") is not None else None,
                    \"v\": float(c.get(\"volume\", 0.0)),
                })
            return out2
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


# ---------- Network ----------

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
    def __init__(self, in_dim: int, head_hidden: int = 96, value_maxabs: float = 2.0, logit_maxabs: float = 2.0, value_bounded: bool = True) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(head_hidden, 1)
        self.critic = nn.Linear(head_hidden, 1)
        self.value_maxabs = float(value_maxabs)
        self.logit_maxabs = float(logit_maxabs)
        self.value_bounded = bool(value_bounded)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        a = torch.tanh(self.actor(h).squeeze(-1)) * self.logit_maxabs
        raw_v = self.critic(h).squeeze(-1)
        v = (torch.tanh(raw_v) * self.value_maxabs) if self.value_bounded else raw_v
        return a, v


class MultiInstrumentThresholdNet(nn.Module):
    def __init__(self, per_inst_dim: int, num_instruments: int, embed_dim: int = 96, context_dim: int = 384, head_hidden: int = 96, value_maxabs: float = 2.0, logit_maxabs: float = 2.0, value_bounded: bool = True) -> None:
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
        self.heads = nn.ModuleList([HeadMLP(context_dim + embed_dim, head_hidden, value_maxabs=value_maxabs, logit_maxabs=logit_maxabs, value_bounded=value_bounded) for _ in range(num_instruments)])

    def forward(self, x_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_concat: (B, num_instruments * per_inst_dim + 10_time)
        B = x_concat.size(0)
        # split out time(10) at end
        feat = x_concat[:, : self.num_instruments * self.per_inst_dim]
        # reshape to (B*N, per_inst_dim) to apply shared local encoder
        fn = self.num_instruments
        local_in = feat.view(B * fn, self.per_inst_dim)
        emb = self.local_encoder(local_in)  # (B*N, embed)
        emb_bn = emb.view(B, fn, -1)
        context_in = emb_bn.reshape(B, fn * emb_bn.size(-1))
        ctx = self.context_norm(self.context(context_in))
        # Per-head outputs
        logits_list: List[torch.Tensor] = []
        values_list: List[torch.Tensor] = []
        for i in range(fn):
            hi = torch.cat([ctx, emb_bn[:, i, :]], dim=-1)
            a_i, v_i = self.heads[i](hi)
            logits_list.append(a_i)
            values_list.append(v_i)
        logits = torch.stack(logits_list, dim=1)  # (B, N)
        values = torch.stack(values_list, dim=1)  # (B, N)
        return emb_bn, logits, values


# Flat architecture: one big MLP from concatenated features to N logits and N values
class FlatThresholdNet(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int = 768, value_maxabs: float = 2.0, logit_maxabs: float = 2.0, value_bounded: bool = True) -> None:
        super().__init__()
        self.num_instruments = num_instruments
        self.value_maxabs = float(value_maxabs)
        self.logit_maxabs = float(logit_maxabs)
        self.value_bounded = bool(value_bounded)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.backbone_norm = nn.LayerNorm(hidden)
        self.actor = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, num_instruments)

    def forward(self, x_concat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone_norm(self.backbone(x_concat))
        logits = torch.tanh(self.actor(h)) * self.logit_maxabs
        raw_v = self.value(h)
        values = (torch.tanh(raw_v) * self.value_maxabs) if self.value_bounded else raw_v
        # return a dummy embedding tensor for interface compatibility
        return torch.empty(0, device=x_concat.device), logits, values

# ---------- Helpers ----------

def _summarize_order(order_obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(order_obj, dict):
        return {}
    # Tall-borker orders return {"order_id": ...}
    if "order_id" in order_obj and len(order_obj) == 1:
        return {"order_id": order_obj.get("order_id")}
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


def _broker_fetch_symbol_map(client: "broker_ipc.BrokerIPCClient") -> Tuple[Dict[str, int], Dict[int, str]]:
    try:
        resp = client.get_instruments()
    except Exception as exc:
        raise RuntimeError(f"broker get_instruments error: {exc}") from exc

    if not resp.ok:
        raise RuntimeError(resp.error or "get_instruments_failed")

    rows = resp.data or []
    sym_to_id: Dict[str, int] = {}
    id_to_sym: Dict[int, str] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            inst_id = int(row.get("id"))
        except Exception:
            continue
        sym_to_id[symbol] = inst_id
        id_to_sym[inst_id] = symbol
    return sym_to_id, id_to_sym


def _broker_positions_snapshot(
    client: "broker_ipc.BrokerIPCClient",
    account_id: int,
    id_to_symbol: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    try:
        resp = client.get_positions(account_id)
        if not resp.ok:
            raise RuntimeError(resp.error or "get_positions_failed")
        rows = resp.data or []
    except Exception as exc:
        raise RuntimeError(f"broker get_positions error: {exc}") from exc
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        try:
            inst_id = int(row.get("instrument_id"))
        except Exception:
            inst_id = None
        qty_raw = float(row.get("quantity") or 0.0)
        side = str(row.get("side") or "").lower()
        signed_qty = qty_raw if side == "long" else (-qty_raw if side == "short" else qty_raw)
        unreal = float(row.get("unrealized_pnl") or 0.0)
        if not symbol and inst_id is not None and id_to_symbol:
            symbol = id_to_symbol.get(inst_id, "")
        if not symbol:
            continue
        payload = {"units": signed_qty, "unrealized": unreal, "instrument_id": inst_id}
        out[symbol] = payload
    return out


def _refresh_units_and_pnl(
    client: "broker_ipc.BrokerIPCClient",
    account_id: int,
    instruments: List[str],
    id_to_symbol: Optional[Dict[int, str]] = None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    snapshot = _broker_positions_snapshot(client, account_id, id_to_symbol)
    units: Dict[str, int] = {}
    pnl: Dict[str, float] = {}
    for symbol in instruments:
        data = snapshot.get(symbol.upper(), {"units": 0.0, "unrealized": 0.0})
        units[symbol] = int(round(data.get("units", 0.0)))
        pnl[symbol] = float(data.get("unrealized", 0.0))
    return units, pnl


def _broker_fetch_nav(client: "broker_ipc.BrokerIPCClient", account_id: int) -> Optional[float]:
    try:
        resp = client.get_account(account_id)
        if not resp.ok:
            raise RuntimeError(resp.error or "get_account_failed")
        data = resp.data or {}
        nav = data.get("equity") or data.get("balance")
        return float(nav) if nav is not None else None
    except Exception:
        return None


def _broker_place_order(
    client: "broker_ipc.BrokerIPCClient",
    account_id: int,
    symbol: str,
    units: float,
    sim_slippage_bps: float,
    sim_fee_perc: float,
    sim_fee_fixed: float,
) -> Optional[Dict[str, Any]]:
    if units == 0:
        return None
    side = "buy" if units > 0 else "sell"
    quantity = abs(float(units))
    try:
        resp = client.place_order(
            account_id=account_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="market",
            time_in_force="GTC",
            sim_slippage_bps=sim_slippage_bps,
            sim_fee_perc=sim_fee_perc,
            sim_fee_fixed=sim_fee_fixed,
        )
    except Exception as exc:
        raise RuntimeError(f"broker order failed: {exc}") from exc
    if not resp.ok:
        raise RuntimeError(f"broker order failed: {resp.error} ({resp.raw})")
    return resp.data or {}


@dataclass
class TradeState:
    open: bool = False
    entry_features: Optional[np.ndarray] = None
    entry_action_long: Optional[int] = None
    entry_logit: Optional[float] = None
    entry_v: Optional[float] = None
    entry_nav: Optional[float] = None
    entry_mid: Optional[float] = None
    entry_fill_price: Optional[float] = None
    entry_step: Optional[int] = None
    entry_ts: Optional[float] = None


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-20 Scalar Threshold Trader")
    # Config system
    parser.add_argument("--config", default=None, help="Path to JSON config to load; CLI overrides config")
    parser.add_argument("--config-id", type=int, default=1, help="Config ID to load (multi20_threshold_config_{ID:03d}.json)")
    parser.add_argument("--instruments", default=",".join(DEFAULT_OANDA_20), help="Comma-separated OANDA instruments (<=20)")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--account-suffix", type=int, default=None)
    parser.add_argument("--broker-account-id", type=int, default=int(os.environ.get("BROKER_ACCOUNT_ID", "1")), help="Tall-borker account id for execution")
    parser.add_argument("--broker-socket", default=os.environ.get("PRAGMAGEN_IPC_SOCKET", "/run/pragmagen/pragmagen.sock"), help="Tall-borker IPC socket path")
    parser.add_argument("--broker-sim-slippage-bps", type=float, default=float(os.environ.get("BROKER_SIM_SLIPPAGE_BPS", "0")), help="Simulated slippage (basis points) applied by tall-borker")
    parser.add_argument("--broker-sim-fee-perc", type=float, default=float(os.environ.get("BROKER_SIM_FEE_PERC", "0")), help="Simulated percentage fee applied by tall-borker")
    parser.add_argument("--broker-sim-fee-fixed", type=float, default=float(os.environ.get("BROKER_SIM_FEE_FIXED", "0")), help="Simulated fixed fee applied by tall-borker")
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
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/multi20_threshold_v001.pt")
    parser.add_argument("--from-checkpoint", action="store_true", help="Load weights from the (resolved) model path at startup")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Decision & anti-churn
    parser.add_argument("--y-ema-alpha", type=float, default=0.3)
    parser.add_argument("--min-hold-ticks", type=int, default=6)
    parser.add_argument("--consec-enter", type=int, default=1)
    parser.add_argument("--consec-exit", type=int, default=1)
    parser.add_argument("--enter-long-thresh", type=float, default=0.6)
    parser.add_argument("--exit-long-thresh", type=float, default=0.55)
    parser.add_argument("--enter-short-thresh", type=float, default=0.4)
    parser.add_argument("--exit-short-thresh", type=float, default=0.45)
    parser.add_argument("--flatten-on-start", action="store_true")
    parser.add_argument("--no-flatten-on-start", dest="flatten_on_start", action="store_false")
    parser.set_defaults(flatten_on_start=True)
    parser.add_argument("--flatten-on-exit", action="store_true")
    parser.add_argument("--no-flatten-on-exit", dest="flatten_on_exit", action="store_false")
    parser.set_defaults(flatten_on_exit=True)
    # Exploration
    parser.add_argument("--ou-noise", action="store_true")
    parser.add_argument("--ou-theta", type=float, default=0.15)
    parser.add_argument("--ou-sigma", type=float, default=0.2)
    parser.add_argument("--ou-dt", type=float, default=1.0)
    parser.add_argument("--explore-eps", type=float, default=0.05)
    parser.add_argument("--explore-sigma", type=float, default=0.6)
    parser.add_argument("--explore-sigma-open", type=float, default=0.3)
    parser.add_argument("--explore-sigma-decay", type=float, default=0.9999)
    parser.add_argument("--explore-sigma-min", type=float, default=0.02)
    # Reward transform & tiers
    parser.add_argument("--reward-transform", choices=["exp", "expm1"], default="expm1")
    parser.add_argument("--close-bonus-base", type=float, default=1.0)
    parser.add_argument("--close-bonus-direction", type=float, default=10.0)
    parser.add_argument("--close-bonus-positive", type=float, default=100.0)
    parser.add_argument("--tier-combine", choices=["add", "mul"], default="add")
    parser.add_argument("--direction-eps", type=float, default=0.0)
    # Micro update controls (per-instrument tick shaping)
    parser.add_argument("--micro-update", action="store_true")
    parser.add_argument("--no-micro-update", dest="micro_update", action="store_false")
    parser.set_defaults(micro_update=True)
    parser.add_argument("--micro-reward-scale", type=float, default=100.0)
    parser.add_argument("--micro-actor-coef", type=float, default=0.02)
    parser.add_argument("--micro-value-coef", type=float, default=0.05)
    parser.add_argument("--micro-flat-bias", type=float, default=0.2)
    parser.add_argument("--commit-reward-coef", type=float, default=0.001)
    # Regularization
    parser.add_argument("--logit-l2", type=float, default=0.0, help="L2 penalty on actor logits to avoid collapse")
    # Architecture and network sizes
    parser.add_argument("--arch", choices=["flat", "modular"], default="flat")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--flat-hidden", type=int, default=512)
    parser.add_argument("--value-maxabs", type=float, default=2.0)
    parser.add_argument("--value-bounded", action="store_true", help="Bound critic output with tanh to ±value-maxabs")
    parser.add_argument("--no-value-bounded", dest="value_bounded", action="store_false")
    parser.set_defaults(value_bounded=True)
    parser.add_argument("--logit-maxabs", type=float, default=2.0)
    # Reward normalization (per-instrument EMA of |reward|)
    parser.add_argument("--reward-ema-beta", type=float, default=0.99)
    parser.add_argument("--reward-norm-eps", type=float, default=1e-3)
    # Value target scaling and duration shaping
    parser.add_argument("--value-target-scale", type=float, default=1.0)
    parser.add_argument("--dur-peak-ticks", type=int, default=20)
    parser.add_argument("--dur-sigma", type=float, default=10.0)
    parser.add_argument("--dur-weight", type=float, default=0.2)
    parser.add_argument("--solely-duration-reward", action="store_true")
    parser.add_argument("--reward-mid-delta", action="store_true", help="Use simple midprice delta reward (scaled) on close")
    parser.add_argument("--reward-mid-mult", type=float, default=1e6, help="Magnitude multiplier for mid-delta raw reward before reward-scale")
    # DOM normalization controls
    parser.add_argument("--dom-soft-c", type=float, default=5.0, help="Soft tanh scaling constant for DOM features")
    parser.add_argument("--dom-ema-norm", action="store_true", help="Append EMA z-scores of DOM features per instrument")
    parser.add_argument("--dom-ema-beta", type=float, default=0.001)
    parser.add_argument("--dom-ema-eps", type=float, default=1e-6)
    parser.add_argument("--dom-liq-log-k", type=float, default=10000.0, help="Scale for log1p compression of DOM liquidity sums")
    # Stats
    parser.add_argument("--train-stats-secs", type=float, default=30.0)
    parser.add_argument("--epochs-per-close", type=int, default=1, help="Number of optimization epochs to run per close event")
    # Reward shaping toggles
    parser.add_argument("--min1-reward", action="store_true", help="Floor episodic reward G to at least 1.0 before normalization")

    args = parser.parse_args()

    # Load and merge JSON config (CLI overrides)
    script_dir = os.path.dirname(__file__)
    cfg_path = str(args.config) if args.config else os.path.join(script_dir, f"multi20_threshold_config_{int(args.config_id):03d}.json")
    config_dict: Dict[str, Any] = {}
    try:
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config_dict = json.load(f)
    except Exception:
        config_dict = {}
    # Merge: if value equals parser default, replace with config value; otherwise keep CLI
    try:
        for k, v in (config_dict or {}).items():
            if hasattr(args, k):
                try:
                    if getattr(args, k) == parser.get_default(k):
                        setattr(args, k, v)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        print(json.dumps({"type": "CONFIG", "path": cfg_path, "loaded": bool(config_dict), "keys": sorted(list(config_dict.keys())) if config_dict else []}), flush=True)
    except Exception:
        pass

    # Resolve model path to an absolute, robust default if user didn't override
    try:
        if getattr(args, 'model_path', None) is not None:
            default_mp = "forex-rl/actor-critic/checkpoints/multi20_threshold_v001.pt"
            if str(args.model_path) == default_mp:
                args.model_path = os.path.join(script_dir, "checkpoints", "multi20_threshold_v001.pt")
    except Exception:
        pass

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()][:20]
    if len(instruments) == 0:
        instruments = DEFAULT_OANDA_20
    num_inst = len(instruments)

    base_account = args.account_id or os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not base_account or not access_token:
        raise RuntimeError("Missing OANDA credentials in env vars.")
    if args.account_suffix is not None:
        if len(str(base_account)) < 3:
            raise RuntimeError("Account id must be at least 3 chars to use --account-suffix")
        account_id = str(base_account)[:-3] + f"{int(args.account_suffix):03d}"
    else:
        account_id = str(base_account)
    api = API(access_token=access_token, environment=args.environment)
    if broker_ipc is None:
        raise RuntimeError("Local tall-borker IPC client unavailable. Ensure forex-rl/broker_ipc.py can import broker_ipc.")
    broker_account_id = int(args.broker_account_id)
    broker_client = broker_ipc.BrokerIPCClient(socket_path=str(args.broker_socket))
    broker_sim_slippage = float(args.broker_sim_slippage_bps)
    broker_sim_fee_perc = float(args.broker_sim_fee_perc)
    broker_sim_fee_fixed = float(args.broker_sim_fee_fixed)

    sym_map_raw, inst_to_symbol = _broker_fetch_symbol_map(broker_client)
    symbol_to_inst = {sym.upper(): inst_id for sym, inst_id in sym_map_raw.items()}
    missing_after = [sym for sym in instruments if sym.upper() not in symbol_to_inst]
    if missing_after:
        raise RuntimeError(f"Tall-borker is missing instrument definitions for: {missing_after}")

    # Per-instrument builders and caches
    fb: Dict[str, FeatureBuilder] = {inst: FeatureBuilder(args.feature_ticks) for inst in instruments}
    cc: Dict[str, CandleCache] = {inst: CandleCache(api, inst, h1_len=args.h1_bars, d_len=args.d_bars, w_len=args.w_bars) for inst in instruments}
    for inst in instruments:
        cc[inst].backfill(args.m1_bars, args.m5_bars, args.h1_bars, args.d_bars, args.w_bars)

    # Per instrument feature sizes: tick(19) + dom(28 [+28 if dom-ema-norm]) + 5*16 + trade_state(3)
    dom_extra = 28 if bool(args.dom_ema_norm) else 0
    per_inst_dim = 19 + 28 + dom_extra + (16 * 5) + 3
    input_dim = per_inst_dim * num_inst + 10  # time features once

    if str(args.arch) == "flat":
        net = FlatThresholdNet(input_dim=input_dim, num_instruments=num_inst, hidden=int(args.flat_hidden), value_maxabs=float(args.value_maxabs), logit_maxabs=float(args.logit_maxabs), value_bounded=bool(args.value_bounded))
        # Zero-init actor/critic for neutral start (y≈0.5, v≈0)
        with torch.no_grad():
            nn.init.zeros_(net.actor.weight); nn.init.zeros_(net.actor.bias)
            nn.init.zeros_(net.value.weight); nn.init.zeros_(net.value.bias)
    else:
        net = MultiInstrumentThresholdNet(per_inst_dim=per_inst_dim, num_instruments=num_inst, embed_dim=int(args.embed_dim), context_dim=int(args.context_dim), head_hidden=int(args.head_hidden), value_maxabs=float(args.value_maxabs), logit_maxabs=float(args.logit_maxabs), value_bounded=bool(args.value_bounded))
        with torch.no_grad():
            for head in net.heads:
                nn.init.zeros_(head.actor.weight); nn.init.zeros_(head.actor.bias)
                nn.init.zeros_(head.critic.weight); nn.init.zeros_(head.critic.bias)
    opt = optim.Adam(net.parameters(), lr=float(args.lr))
    net.train()

    # Optionally load checkpoint if present
    try:
        if args.model_path and (bool(args.from_checkpoint) or os.path.exists(str(args.model_path))):
            ckpt = torch.load(str(args.model_path), map_location="cpu")
            if isinstance(ckpt, dict):
                try:
                    net.load_state_dict(ckpt.get("model", ckpt))
                except Exception:
                    net.load_state_dict(ckpt)
                try:
                    opt_state = ckpt.get("opt")
                    if opt_state:
                        opt.load_state_dict(opt_state)
                except Exception:
                    pass
                print(json.dumps({"type": "LOADED", "path": str(args.model_path)}), flush=True)
        elif bool(args.from_checkpoint):
            print(json.dumps({"type": "LOAD_WARN", "msg": "--from-checkpoint set but file not found", "path": str(args.model_path)}), flush=True)
    except Exception:
        pass

    # State trackers per instrument
    st: Dict[str, TradeState] = {inst: TradeState(open=False) for inst in instruments}
    last_y: Dict[str, Optional[float]] = {inst: None for inst in instruments}
    last_v: Dict[str, Optional[float]] = {inst: None for inst in instruments}
    last_y_ema: Dict[str, Optional[float]] = {inst: None for inst in instruments}
    ou_noise_val: Dict[str, float] = {inst: 0.0 for inst in instruments}
    last_features: Dict[str, np.ndarray] = {inst: np.zeros(per_inst_dim, dtype=np.float32) for inst in instruments}
    # DOM EMA state per instrument (mean/var for 28-dim DOM)
    dom_mean_map: Dict[str, np.ndarray] = {inst: np.zeros(28, dtype=np.float32) for inst in instruments}
    dom_var_map: Dict[str, np.ndarray] = {inst: np.ones(28, dtype=np.float32) for inst in instruments}

    # Account / units / nav
    last_nav = _broker_fetch_nav(broker_client, broker_account_id) or 1.0
    last_nav_poll = time.time()
    last_pos_refresh = time.time()
    units_map: Dict[str, int] = {inst: 0 for inst in instruments}
    pos_pnl_map: Dict[str, float] = {inst: 0.0 for inst in instruments}
    try:
        refreshed_units, refreshed_pnl = _refresh_units_and_pnl(
            broker_client, broker_account_id, instruments, inst_to_symbol
        )
        units_map.update(refreshed_units)
        pos_pnl_map.update(refreshed_pnl)
    except Exception:
        pass

    # Stats trackers
    stats_y: Dict[str, Deque[float]] = {inst: deque(maxlen=600) for inst in instruments}
    # Trade duration trackers (rolling)
    dur_ticks_hist: Dict[str, Deque[int]] = {inst: deque(maxlen=500) for inst in instruments}
    dur_sec_hist: Dict[str, Deque[float]] = {inst: deque(maxlen=500) for inst in instruments}
    # Episodic reward trackers (rolling)
    rew_hist: Dict[str, Deque[float]] = {inst: deque(maxlen=1000) for inst in instruments}
    rew_overall: Deque[float] = deque(maxlen=20000)
    last_stats_emit = time.time()
    last_grad_norm: Optional[float] = None
    # Reward normalization state per instrument
    rew_ema: Dict[str, float] = {inst: 1.0 for inst in instruments}

    # Flatten on start optionally (all instruments)
    if args.flatten_on_start:
        for inst in instruments:
            try:
                u = units_map.get(inst, 0)
                if u != 0:
                    order = _broker_place_order(broker_client, broker_account_id, inst, -u, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                    units_map[inst] = 0
                    pos_pnl_map[inst] = 0.0
                    print(json.dumps({"type": "AUTO_FLATTEN_START", "instrument": inst, "order": _summarize_order(order)}), flush=True)
            except Exception:
                pass

    # Flatten on exit at process end
    if args.flatten_on_exit:
        def _on_exit() -> None:
            try:
                refreshed_units, _ = _refresh_units_and_pnl(
                    broker_client, broker_account_id, instruments, inst_to_symbol
                )
                for inst in instruments:
                    u = refreshed_units.get(inst, 0)
                    if u != 0:
                        order = _broker_place_order(broker_client, broker_account_id, inst, -u, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                        print(json.dumps({"type": "AUTO_FLATTEN_EXIT", "instrument": inst, "order": _summarize_order(order)}), flush=True)
            except Exception:
                pass
        atexit.register(_on_exit)

    # OU decay base sigmas
    base_ou_sigma = float(args.ou_sigma)
    base_gauss_sigma = float(args.explore_sigma)
    base_gauss_sigma_open = float(args.explore_sigma_open)
    step_idx: int = 0

    # Stream across all instruments
    stream = pricing.PricingStream(accountID=account_id, params={"instruments": ",".join(instruments)})
    while True:
        try:
            for msg in api.request(stream):
                tnow = time.time()
                mtype = msg.get("type")
                if mtype == "HEARTBEAT":
                    if (tnow - last_nav_poll) >= float(args.nav_poll_secs):
                        nv = _broker_fetch_nav(broker_client, broker_account_id)
                        if nv is not None:
                            last_nav = nv
                        last_nav_poll = tnow
                    if (tnow - last_pos_refresh) >= float(args.pos_refresh_secs):
                        try:
                            refreshed_units, refreshed_pnl = _refresh_units_and_pnl(
                                broker_client, broker_account_id, instruments, inst_to_symbol
                            )
                            units_map.update(refreshed_units)
                            pos_pnl_map.update(refreshed_pnl)
                        except Exception:
                            pass
                        last_pos_refresh = tnow
                    # refresh candles for all
                    for inst in instruments:
                        cc[inst].maybe_refresh(args.m1_refresh_secs, args.m5_refresh_secs, args.h1_refresh_secs, args.d_refresh_secs, args.w_refresh_secs)
                    # Emit per-instrument HB summaries
                    hb_all: List[Dict[str, Any]] = []
                    for inst in instruments:
                        hb: Dict[str, Any] = {
                            "type": "HB",
                            "instrument": inst,
                            "nav": round(float(last_nav), 6) if isinstance(last_nav, (int, float)) else last_nav,
                            "units": int(units_map.get(inst, 0)),
                            "open": bool(st[inst].open),
                        }
                        try:
                            if fb[inst].mid_window:
                                hb["mid"] = round(float(fb[inst].mid_window[-1]), 5)
                        except Exception:
                            pass
                        if last_y.get(inst) is not None:
                            hb["y"] = round(float(last_y[inst] or 0.0), 4)
                        if last_v.get(inst) is not None:
                            hb["v"] = round(float(last_v[inst] or 0.0), 4)
                        hb_all.append(hb)
                    print(json.dumps({"type": "HB_ALL", "list": hb_all}), flush=True)
                    if (tnow - last_stats_emit) >= float(getattr(args, 'train_stats_secs', 30.0)):
                        try:
                            y_mean = {inst: (float(np.mean(list(stats_y[inst])) ) if len(stats_y[inst])>0 else None) for inst in instruments}
                            y_std = {inst: (float(np.std(list(stats_y[inst])) ) if len(stats_y[inst])>0 else None) for inst in instruments}
                            avg_dur_sec = {inst: (float(np.mean(list(dur_sec_hist[inst]))) if len(dur_sec_hist[inst])>0 else None) for inst in instruments}
                            avg_dur_ticks = {inst: (float(np.mean(list(dur_ticks_hist[inst]))) if len(dur_ticks_hist[inst])>0 else None) for inst in instruments}
                            avg_reward_inst = {inst: (float(np.mean(list(rew_hist[inst]))) if len(rew_hist[inst])>0 else None) for inst in instruments}
                            avg_reward_all = float(np.mean(list(rew_overall))) if len(rew_overall)>0 else None
                            print(json.dumps({"type": "TRAIN_STATS", "grad_norm": (round(float(last_grad_norm),4) if last_grad_norm is not None else None), "y_mean": y_mean, "y_std": y_std, "avg_dur_sec": avg_dur_sec, "avg_dur_ticks": avg_dur_ticks, "avg_reward_inst": avg_reward_inst, "avg_reward_all": avg_reward_all}), flush=True)
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
                # Log-compress liquidity sums (indices 4,5: sb, sa)
                try:
                    x_dom_proc = x_dom_raw.copy()
                    k = float(args.dom_liq_log_k)
                    x_dom_proc[4] = np.log1p(max(0.0, x_dom_proc[4]) / max(1e-9, k))
                    x_dom_proc[5] = np.log1p(max(0.0, x_dom_proc[5]) / max(1e-9, k))
                except Exception:
                    x_dom_proc = x_dom_raw
                # Soft squash to preserve ordering but bound scale
                c = float(args.dom_soft_c)
                x_dom_soft = c * np.tanh(x_dom_proc / max(1e-9, c))
                # Optional EMA z-scores appended
                if bool(args.dom_ema_norm):
                    m = dom_mean_map[inst]
                    v = dom_var_map[inst]
                    beta = float(args.dom_ema_beta)
                    eps = float(args.dom_ema_eps)
                    # EMA update (per-feature)
                    delta = x_dom_proc - m
                    m_new = (1.0 - beta) * m + beta * x_dom_proc
                    dom_mean_map[inst] = m_new.astype(np.float32)
                    v_new = (1.0 - beta) * v + beta * (x_dom_proc - m_new) * (x_dom_proc - m)
                    dom_var_map[inst] = np.maximum(v_new, eps).astype(np.float32)
                    z = (x_dom_proc - m_new) / np.sqrt(dom_var_map[inst] + eps)
                    x_dom = np.concatenate([x_dom_soft, z.astype(np.float32)])
                else:
                    x_dom = x_dom_soft.astype(np.float32)
                m1_feats = candle_features(cc[inst].m1)
                m5_feats = candle_features(cc[inst].m5)
                h1_feats = candle_features(cc[inst].h1)
                d1_feats = candle_features(cc[inst].d1)
                w1_feats = candle_features(cc[inst].w1)
                if x_tick is None:
                    continue
                # Trade-state features: duration (sec), direction, current implied reward
                try:
                    if units_map.get(inst, 0) != 0:
                        dur_sec = float(max(0.0, (time.time() - float(st[inst].entry_ts or time.time()))))
                        direction = 1.0 if units_map.get(inst, 0) > 0 else -1.0
                        cur_reward = 0.0
                        if bool(args.reward_mid_delta):
                            try:
                                entry_mid = float(st[inst].entry_mid or 0.0)
                                mid_now = float(mid) if isinstance(mid, (int, float)) else (float(fb[inst].mid_window[-1]) if fb[inst].mid_window else entry_mid)
                                if entry_mid > 0.0 and mid_now > 0.0:
                                    denom = max(1e-9, (mid_now + entry_mid))
                                    raw = direction * ((mid_now - entry_mid) / denom)
                                    cur_reward = float(raw) * float(args.reward_mid_mult) * float(args.reward_scale)
                            except Exception:
                                cur_reward = 0.0
                        elif bool(args.solely_duration_reward):
                            if dur_sec <= 150.0:
                                cur_reward = -1.0 + 2.0 * (dur_sec / 150.0)
                            elif dur_sec <= 600.0:
                                cur_reward = 1.0 - 2.0 * ((dur_sec - 150.0) / 450.0)
                            else:
                                cur_reward = -1.0
                        else:
                            try:
                                raw = float(pos_pnl_map.get(inst, 0.0)) * float(args.reward_scale)
                                if raw < 0:
                                    raw = raw * float(args.neg_reward_coef)
                                cur_reward = float(raw)
                            except Exception:
                                cur_reward = 0.0
                    else:
                        dur_sec = 0.0
                        direction = 0.0
                        cur_reward = 0.0
                except Exception:
                    dur_sec = 0.0
                    direction = 0.0
                    cur_reward = 0.0
                # Update latest vector for this instrument (append trade-state features)
                last_features[inst] = np.concatenate([
                    x_tick, x_dom, m1_feats, m5_feats, h1_feats, d1_feats, w1_feats,
                    np.array([dur_sec, direction, cur_reward], dtype=np.float32)
                ]).astype(np.float32)

                # Build full input across instruments (per_inst_dim * N + time(10))
                try:
                    ts_str = msg.get("time") or msg.get("timestamp")
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if ts_str else datetime.now(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                t_feats = time_cyclical_features(dt)
                X_list: List[np.ndarray] = []
                for inst2 in instruments:
                    X_list.append(last_features[inst2])
                X_list.append(t_feats)
                x_full = np.concatenate(X_list).astype(np.float32)
                if not np.all(np.isfinite(x_full)):
                    continue
                xt = torch.from_numpy(x_full)[None, :]

                # Forward
                net.train(False)
                with torch.no_grad():
                    _, logits_all, values_all = net(xt)
                idx = instruments.index(inst)
                a_val = float(logits_all[0, idx].item())
                v_val = float(values_all[0, idx].item())

                # Exploration per instrument
                try:
                    if bool(args.ou_noise):
                        theta = float(args.ou_theta); dt_ = float(args.ou_dt)
                        sigma = max(float(args.explore_sigma_min), base_ou_sigma * (float(args.explore_sigma_decay) ** step_idx))
                        ou_noise_val[inst] = ou_noise_val[inst] + theta * (0.0 - ou_noise_val[inst]) * dt_ + sigma * np.sqrt(dt_) * float(np.random.randn())
                        a_val = a_val + ou_noise_val[inst]
                    elif float(args.explore_eps) > 0.0 and float(args.explore_sigma) > 0.0:
                        if float(np.random.rand()) < float(args.explore_eps):
                            decayed = max(float(args.explore_sigma_min), base_gauss_sigma * (float(args.explore_sigma_decay) ** step_idx))
                            decayed_open = max(float(args.explore_sigma_min), base_gauss_sigma_open * (float(args.explore_sigma_decay) ** step_idx))
                            sigma_eff = decayed_open if (units_map.get(inst, 0) != 0) else decayed
                            a_val = a_val + float(np.random.normal(loc=0.0, scale=sigma_eff))
                except Exception:
                    pass

                y_step = float(1.0 / (1.0 + np.exp(-a_val)))
                last_y[inst] = y_step
                last_v[inst] = v_val
                try:
                    stats_y[inst].append(y_step)
                except Exception:
                    pass
                # EMA
                if last_y_ema[inst] is None:
                    last_y_ema[inst] = y_step
                else:
                    alpha = float(args.y_ema_alpha)
                    last_y_ema[inst] = float(alpha * y_step + (1.0 - alpha) * float(last_y_ema[inst]))

                # Micro update for this instrument
                if bool(args.micro_update):
                    if len(fb[inst].mid_window) >= 2:
                        mid_delta = float(fb[inst].mid_window[-1] - fb[inst].mid_window[-2])
                    else:
                        mid_delta = 0.0
                    u_now = units_map.get(inst, 0)
                    if u_now > 0:
                        r_micro = mid_delta
                    elif u_now < 0:
                        r_micro = -mid_delta
                    else:
                        r_micro = 0.2 * mid_delta
                    r_micro = r_micro * float(args.micro_reward_scale)
                    y_for_commit = float(last_y_ema[inst] if last_y_ema[inst] is not None else y_step)
                    r_commit = float(args.commit_reward_coef) * float(abs(y_for_commit - 0.5))
                    r_total = r_micro + r_commit

                    opt.zero_grad()
                    emb_bn, logits_all_live, values_all_live = net(xt)
                    v_pred = values_all_live[0, idx]
                    _r_clipped = float(r_total if float(args.reward_clip) <= 0.0 else np.clip(r_total, -float(args.reward_clip), float(args.reward_clip)))
                    _r_transformed = float(np.expm1(_r_clipped) if str(args.reward_transform) == "expm1" else np.exp(_r_clipped))
                    target = torch.tensor(_r_transformed, dtype=torch.float32)
                    adv_micro = torch.clamp(target.detach() - v_pred.detach(), -float(args.adv_clip), float(args.adv_clip))
                    dist_live = torch.distributions.Bernoulli(logits=logits_all_live[0, idx])
                    action_like = torch.tensor(1.0 if mid_delta > 0 else 0.0, dtype=torch.float32)
                    logprob_live = dist_live.log_prob(action_like)
                    actor_loss_micro = -adv_micro * logprob_live
                    value_loss_micro = F.smooth_l1_loss(v_pred, target)
                    entropy_bonus = dist_live.entropy()
                    # Optional actor logit L2 regularization
                    reg = 0.0
                    try:
                        if float(args.logit_l2) > 0.0:
                            reg = float(args.logit_l2) * (logits_all_live[0, idx] ** 2)
                    except Exception:
                        reg = 0.0
                    loss_micro = float(args.micro_actor_coef) * actor_loss_micro + float(args.micro_value_coef) * value_loss_micro - float(args.entropy_coef) * entropy_bonus + reg
                    loss_micro.backward()
                    try:
                        last_grad_norm = _compute_grad_norm(net.parameters())
                    except Exception:
                        last_grad_norm = None
                    nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                    opt.step()

                # Decision logic
                pos_long = units_map.get(inst, 0) > 0
                pos_short = units_map.get(inst, 0) < 0
                if st[inst].open:
                    st_ticks_open = 1  # unused placeholder for compatibility

                y_use = float(last_y_ema[inst] if last_y_ema[inst] is not None else y_step)
                use_enter_long = float(args.enter_long_thresh)
                use_exit_long = float(args.exit_long_thresh)
                use_enter_short = float(args.enter_short_thresh)
                use_exit_short = float(args.exit_short_thresh)

                # Gating counters per instrument (kept minimal)
                if pos_long:
                    can_exit = (use_exit_long is not None) and (y_use < use_exit_long)
                    if can_exit and (units_map.get(inst, 0) != 0):
                        try:
                            current_units = units_map.get(inst, 0)
                            order = _broker_place_order(broker_client, broker_account_id, inst, -current_units, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                            units_map[inst] = 0
                            pos_pnl_map[inst] = 0.0
                            print(json.dumps({"type": "EXIT_LONG", "instrument": inst, "order": _summarize_order(order), "y": round(y_step,4)}), flush=True)
                            # Episodic update using fill pl
                            try:
                                summ = _summarize_order(order)
                                if bool(args.reward_mid_delta):
                                    # Simple midprice delta normalized by scale, then reward_scale
                                    exit_mid = float(fb[inst].mid_window[-1]) if fb[inst].mid_window else float(mid)
                                    entry_mid = float(st[inst].entry_mid or exit_mid)
                                    sign = 1.0 if (st[inst].entry_action_long == 1) else -1.0
                                    denom = max(1e-9, (exit_mid + entry_mid))
                                    G_raw = sign * ((exit_mid - entry_mid) / denom)
                                    G_clip = 0.0
                                    G_transformed = float(G_raw) * float(args.reward_mid_mult) * float(args.reward_scale)
                                elif bool(args.solely_duration_reward):
                                    # duration-only triangular reward: -1 at 0s -> +1 at 150s -> -1 at 600s
                                    dur_sec = max(0.0, float(time.time() - float(st[inst].entry_ts or time.time())))
                                    if dur_sec <= 150.0:
                                        G_transformed = -1.0 + 2.0 * (dur_sec / 150.0)
                                    elif dur_sec <= 600.0:
                                        G_transformed = 1.0 - 2.0 * ((dur_sec - 150.0) / 450.0)
                                    else:
                                        G_transformed = -1.0
                                    G_clip = G_raw = 0.0
                                else:
                                    pl = float(summ.get("pl") or 0.0)
                                    G_raw = float(pl) * float(args.reward_scale)
                                    if G_raw < 0:
                                        G_raw = G_raw * float(args.neg_reward_coef)
                                    G_clip = float(G_raw if float(args.reward_clip) <= 0.0 else np.clip(G_raw, -float(args.reward_clip), float(args.reward_clip)))
                                    G_transformed = float(np.expm1(G_clip) if str(args.reward_transform) == "expm1" else np.exp(G_clip))
                                mid_now = float(fb[inst].mid_window[-1]) if fb[inst].mid_window else None
                                direction_ok = False
                                if (st[inst].entry_fill_price is not None) and (summ.get("price") is not None):
                                    try:
                                        exit_px = float(summ.get("price"))
                                        if st[inst].entry_action_long == 1:
                                            direction_ok = exit_px > float(st[inst].entry_fill_price)
                                        else:
                                            direction_ok = exit_px < float(st[inst].entry_fill_price)
                                    except Exception:
                                        direction_ok = False
                                if bool(args.reward_mid_delta):
                                    G = float(G_transformed)
                                    tier_add = 0.0
                                else:
                                    tier_add = float(args.close_bonus_base) + (float(args.close_bonus_direction) if direction_ok else 0.0) + (float(args.close_bonus_positive) if (G_raw > 0) else 0.0)
                                    G = float(G_transformed * (1.0 + tier_add)) if str(args.tier_combine) == "mul" else float(G_transformed + tier_add)
                                # Duration bell-curve bonus
                                try:
                                    dur = int(step_idx - int(st[inst].entry_step or step_idx))
                                except Exception:
                                    dur = 0
                                if float(args.dur_sigma) > 0.0:
                                    z = (float(dur) - float(args.dur_peak_ticks)) / float(args.dur_sigma)
                                    G += float(args.dur_weight) * float(np.exp(-0.5 * (z ** 2)))
                                # Optional floor: remove negative/small rewards (applied to final G)
                                if bool(args.min1_reward):
                                    G = float(max(1.0, float(G)))
                                # Per-instrument reward normalization
                                if bool(args.reward_mid_delta):
                                    G_use = float(G)
                                else:
                                    s_prev = float(rew_ema.get(inst, 1.0))
                                    s_new = float(args.reward_ema_beta) * s_prev + (1.0 - float(args.reward_ema_beta)) * float(abs(G))
                                    rew_ema[inst] = max(float(args.reward_norm_eps), s_new)
                                    G_use = float(G / rew_ema[inst])
                                # Map to critic range
                                if bool(args.value_bounded):
                                    v_target = float(np.tanh(G_use / max(1e-6, float(args.value_target_scale))) * float(args.value_maxabs))
                                else:
                                    v_target = float(G_use)
                                epochs_close = max(1, int(getattr(args, "epochs_per_close", 1)))
                                for _ in range(epochs_close):
                                    opt.zero_grad()
                                    x_entry = torch.from_numpy(st[inst].entry_features.astype(np.float32))[None, :]
                                    _, logits_all2, values_all2 = net(x_entry)
                                    v_t = values_all2[0, idx]
                                    dist = torch.distributions.Bernoulli(logits=logits_all2[0, idx])
                                    action_tensor = torch.tensor(1.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    advantage = v_target - float(v_t.detach().item())
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob
                                    critic_loss = F.smooth_l1_loss(v_t, torch.tensor(v_target, dtype=torch.float32))
                                    reg = 0.0
                                    try:
                                        if float(args.logit_l2) > 0.0:
                                            reg = float(args.logit_l2) * (logits_all2[0, idx] ** 2)
                                    except Exception:
                                        reg = 0.0
                                    loss = actor_loss + float(args.value_coef) * critic_loss - float(args.entropy_coef) * dist.entropy() + reg
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                                    opt.step()
                                # Track duration stats
                                try:
                                    dur_ticks_hist[inst].append(int(dur))
                                    dur_sec_val = max(0.0, float(time.time() - float(st[inst].entry_ts or time.time())))
                                    dur_sec_hist[inst].append(float(dur_sec_val))
                                except Exception:
                                    pass
                                # Track episode reward
                                try:
                                    rew_hist[inst].append(float(G))
                                    rew_overall.append(float(G))
                                except Exception:
                                    pass
                                # Track episode reward
                                try:
                                    rew_hist[inst].append(float(G))
                                    rew_overall.append(float(G))
                                except Exception:
                                    pass
                                avg_rew_inst = None
                                avg_rew_all = None
                                try:
                                    avg_rew_inst = float(np.mean(list(rew_hist[inst]))) if len(rew_hist[inst]) > 0 else None
                                except Exception:
                                    avg_rew_inst = None
                                try:
                                    avg_rew_all = float(np.mean(list(rew_overall))) if len(rew_overall) > 0 else None
                                except Exception:
                                    avg_rew_all = None
                                print(json.dumps({"type": "EP_END", "instrument": inst, "reward": float(G), "reward_norm": float(G_use), "v_target": float(v_target), "mode": ("mid_delta" if bool(args.reward_mid_delta) else ("duration" if bool(args.solely_duration_reward) else "mixed")), "reward_transformed": float(G_transformed), "reward_pre_transform": float(G_clip), "reward_raw": float(G_raw), "direction_ok": bool(direction_ok), "tier_bonus": float(tier_add), "duration_ticks": int(dur) if 'dur' in locals() else None, "avg_reward_inst": avg_rew_inst, "avg_reward_all": avg_rew_all}), flush=True)
                            except Exception:
                                pass
                            st[inst] = TradeState(open=False)
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                elif pos_short:
                    can_exit = (use_exit_short is not None) and (y_use > use_exit_short)
                    if can_exit and (units_map.get(inst, 0) != 0):
                        try:
                            current_units = units_map.get(inst, 0)
                            order = _broker_place_order(broker_client, broker_account_id, inst, -current_units, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                            units_map[inst] = 0
                            pos_pnl_map[inst] = 0.0
                            print(json.dumps({"type": "EXIT_SHORT", "instrument": inst, "order": _summarize_order(order), "y": round(y_step,4)}), flush=True)
                            # Episodic update using fill pl
                            try:
                                summ = _summarize_order(order)
                                if bool(args.reward_mid_delta):
                                    exit_mid = float(fb[inst].mid_window[-1]) if fb[inst].mid_window else float(mid)
                                    entry_mid = float(st[inst].entry_mid or exit_mid)
                                    sign = 1.0 if (st[inst].entry_action_long == 1) else -1.0
                                    denom = max(1e-9, (exit_mid + entry_mid))
                                    G_raw = sign * ((exit_mid - entry_mid) / denom)
                                    G_clip = 0.0
                                    G_transformed = float(G_raw) * float(args.reward_mid_mult) * float(args.reward_scale)
                                elif bool(args.solely_duration_reward):
                                    dur_sec = max(0.0, float(time.time() - float(st[inst].entry_ts or time.time())))
                                    if dur_sec <= 150.0:
                                        G_transformed = -1.0 + 2.0 * (dur_sec / 150.0)
                                    elif dur_sec <= 600.0:
                                        G_transformed = 1.0 - 2.0 * ((dur_sec - 150.0) / 450.0)
                                    else:
                                        G_transformed = -1.0
                                    G_clip = G_raw = 0.0
                                else:
                                    pl = float(summ.get("pl") or 0.0)
                                    G_raw = float(pl) * float(args.reward_scale)
                                    if G_raw < 0:
                                        G_raw = G_raw * float(args.neg_reward_coef)
                                    G_clip = float(G_raw if float(args.reward_clip) <= 0.0 else np.clip(G_raw, -float(args.reward_clip), float(args.reward_clip)))
                                    G_transformed = float(np.expm1(G_clip) if str(args.reward_transform) == "expm1" else np.exp(G_clip))
                                direction_ok = False
                                if (st[inst].entry_fill_price is not None) and (summ.get("price") is not None):
                                    try:
                                        exit_px = float(summ.get("price"))
                                        if st[inst].entry_action_long == 1:
                                            direction_ok = exit_px > float(st[inst].entry_fill_price)
                                        else:
                                            direction_ok = exit_px < float(st[inst].entry_fill_price)
                                    except Exception:
                                        direction_ok = False
                                if bool(args.reward_mid_delta):
                                    G = float(G_transformed)
                                    tier_add = 0.0
                                else:
                                    tier_add = float(args.close_bonus_base) + (float(args.close_bonus_direction) if direction_ok else 0.0) + (float(args.close_bonus_positive) if (G_raw > 0) else 0.0)
                                    G = float(G_transformed * (1.0 + tier_add)) if str(args.tier_combine) == "mul" else float(G_transformed + tier_add)
                                try:
                                    dur = int(step_idx - int(st[inst].entry_step or step_idx))
                                except Exception:
                                    dur = 0
                                if float(args.dur_sigma) > 0.0:
                                    z = (float(dur) - float(args.dur_peak_ticks)) / float(args.dur_sigma)
                                    G += float(args.dur_weight) * float(np.exp(-0.5 * (z ** 2)))
                                # Optional floor: remove negative/small rewards (applied to final G)
                                if bool(args.min1_reward):
                                    G = float(max(1.0, float(G)))
                                if bool(args.reward_mid_delta):
                                    G_use = float(G)
                                else:
                                    s_prev = float(rew_ema.get(inst, 1.0))
                                    s_new = float(args.reward_ema_beta) * s_prev + (1.0 - float(args.reward_ema_beta)) * float(abs(G))
                                    rew_ema[inst] = max(float(args.reward_norm_eps), s_new)
                                    G_use = float(G / rew_ema[inst])
                                if bool(args.value_bounded):
                                    v_target = float(np.tanh(G_use / max(1e-6, float(args.value_target_scale))) * float(args.value_maxabs))
                                else:
                                    v_target = float(G_use)
                                epochs_close = max(1, int(getattr(args, "epochs_per_close", 1)))
                                for _ in range(epochs_close):
                                    opt.zero_grad()
                                    x_entry = torch.from_numpy(st[inst].entry_features.astype(np.float32))[None, :]
                                    _, logits_all2, values_all2 = net(x_entry)
                                    v_t = values_all2[0, idx]
                                    dist = torch.distributions.Bernoulli(logits=logits_all2[0, idx])
                                    action_tensor = torch.tensor(0.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    advantage = v_target - float(v_t.detach().item())
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob
                                    critic_loss = F.smooth_l1_loss(v_t, torch.tensor(v_target, dtype=torch.float32))
                                    reg = 0.0
                                    try:
                                        if float(args.logit_l2) > 0.0:
                                            reg = float(args.logit_l2) * (logits_all2[0, idx] ** 2)
                                    except Exception:
                                        reg = 0.0
                                    loss = actor_loss + float(args.value_coef) * critic_loss - float(args.entropy_coef) * dist.entropy() + reg
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                                    opt.step()
                                # Track episode reward
                                try:
                                    rew_hist[inst].append(float(G))
                                    rew_overall.append(float(G))
                                except Exception:
                                    pass
                                avg_rew_inst = None
                                avg_rew_all = None
                                try:
                                    avg_rew_inst = float(np.mean(list(rew_hist[inst]))) if len(rew_hist[inst]) > 0 else None
                                except Exception:
                                    avg_rew_inst = None
                                try:
                                    avg_rew_all = float(np.mean(list(rew_overall))) if len(rew_overall) > 0 else None
                                except Exception:
                                    avg_rew_all = None
                                print(json.dumps({"type": "EP_END", "instrument": inst, "reward": float(G), "reward_norm": float(G_use), "v_target": float(v_target), "mode": ("mid_delta" if bool(args.reward_mid_delta) else ("duration" if bool(args.solely_duration_reward) else "mixed")), "reward_transformed": float(G_transformed), "reward_pre_transform": float(G_clip), "reward_raw": float(G_raw), "direction_ok": bool(direction_ok), "tier_bonus": float(tier_add), "duration_ticks": int(dur) if 'dur' in locals() else None, "avg_reward_inst": avg_rew_inst, "avg_reward_all": avg_rew_all}), flush=True)
                            except Exception:
                                pass
                            st[inst] = TradeState(open=False)
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                else:
                    # No position: entry checks
                    if y_use > use_enter_long:
                        # enter long
                        try:
                            units_delta = int(args.units)
                            order = _broker_place_order(broker_client, broker_account_id, inst, units_delta, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                            units_map[inst] = units_map.get(inst, 0) + units_delta
                            pos_pnl_map[inst] = 0.0
                            st[inst].open = True
                            st[inst].entry_features = x_full.copy()
                            st[inst].entry_action_long = 1
                            st[inst].entry_logit = float(np.log(y_step + 1e-12) - np.log(1.0 - y_step + 1e-12)) if 0.0 < y_step < 1.0 else (10.0 if y_step >= 1.0 else -10.0)
                            st[inst].entry_v = float(v_val)
                            st[inst].entry_nav = last_nav
                            st[inst].entry_mid = float(mid) if isinstance(mid, (int, float)) else None
                            st[inst].entry_step = int(step_idx)
                            st[inst].entry_ts = float(time.time())
                            try:
                                summ = _summarize_order(order)
                                st[inst].entry_fill_price = float(summ.get("price")) if summ.get("price") is not None else None
                            except Exception:
                                st[inst].entry_fill_price = None
                            print(json.dumps({"type": "ENTER_LONG", "instrument": inst, "units": int(args.units), "y": round(float(y_step),4), "v": round(float(v_val),4), "order": _summarize_order(order)}), flush=True)
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)
                    elif y_use < use_enter_short:
                        # enter short
                        try:
                            units_delta = -int(args.units)
                            order = _broker_place_order(broker_client, broker_account_id, inst, units_delta, broker_sim_slippage, broker_sim_fee_perc, broker_sim_fee_fixed)
                            units_map[inst] = units_map.get(inst, 0) + units_delta
                            pos_pnl_map[inst] = 0.0
                            st[inst].open = True
                            st[inst].entry_features = x_full.copy()
                            st[inst].entry_action_long = 0
                            st[inst].entry_logit = float(np.log(y_step + 1e-12) - np.log(1.0 - y_step + 1e-12)) if 0.0 < y_step < 1.0 else (10.0 if y_step >= 1.0 else -10.0)
                            st[inst].entry_v = float(v_val)
                            st[inst].entry_nav = last_nav
                            st[inst].entry_mid = float(mid) if isinstance(mid, (int, float)) else None
                            st[inst].entry_step = int(step_idx)
                            st[inst].entry_ts = float(time.time())
                            try:
                                summ = _summarize_order(order)
                                st[inst].entry_fill_price = float(summ.get("price")) if summ.get("price") is not None else None
                            except Exception:
                                st[inst].entry_fill_price = None
                            print(json.dumps({"type": "ENTER_SHORT", "instrument": inst, "units": int(-args.units), "y": round(float(y_step),4), "v": round(float(v_val),4), "order": _summarize_order(order)}), flush=True)
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)

                # Autosave
                if float(args.autosave_secs) > 0 and (tnow - getattr(main, "_last_save", 0.0)) >= float(args.autosave_secs):
                    try:
                        os.makedirs(os.path.dirname(str(args.model_path)), exist_ok=True)
                        payload = {
                            "model": net.state_dict(),
                            "opt": opt.state_dict(),
                            "meta": {"per_inst_dim": per_inst_dim, "num_instruments": num_inst},
                        }
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


if __name__ == "__main__":
    main()
