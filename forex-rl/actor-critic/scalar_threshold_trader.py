#!/usr/bin/env python3
"""
Scalar-Threshold Actor-Critic (bi-directional) - Tick Stream, Single Instrument

- Single scalar actor head in [0,1): maps to enter/exit decisions via thresholds
  - If y > 0.8 => enter long (+units) when flat
  - If y < 0.6 => exit long (flatten)
  - If y < 0.2 => enter short (-units) when flat
  - If y > 0.4 => exit short (flatten)
- Fixed position size (?units), no TP/SL, no time-based closes.
- Episodic reward at close credited to entry (A2C update). Negative rewards are
  down-weighted by neg_reward_coef (default 0.1).
- Reuses feature engineering and streaming structure from binary_enter_trader.py.
- Includes multi-horizon candle features: M1, M5, H1, D, W.

Environment:
  Requires OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY in environment.
"""
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
    feature_ticks: int = 240
    reward_scale: float = 10000.0
    neg_reward_coef: float = 0.1
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    lr: float = 1e-3
    entropy_coef: float = 0.001
    autosave_secs: float = 120.0
    model_path: str = ""
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_clip: float = 1.0
    adv_clip: float = 5.0
    gamma: float = 0.99
    # Thresholds for scalar decision (more frequent trading)
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
    entry_mid: Optional[float] = None


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scalar-Threshold Actor-Critic (bi-directional)")
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/scalar_threshold_v001.pt")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Reward clipping: default 0.0 disables clipping; set >0 to enable
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
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
    parser.add_argument("--enter-long-thresh", type=float, default=0.6)
    parser.add_argument("--exit-long-thresh", type=float, default=0.55)
    parser.add_argument("--enter-short-thresh", type=float, default=0.4)
    parser.add_argument("--exit-short-thresh", type=float, default=0.45)
    # Reward transform and tiered bonuses (episodic)
    parser.add_argument("--reward-transform", choices=["exp", "expm1"], default="expm1")
    parser.add_argument("--close-bonus-base", type=float, default=1.0)
    parser.add_argument("--close-bonus-direction", type=float, default=10.0)
    parser.add_argument("--close-bonus-positive", type=float, default=100.0)
    parser.add_argument("--tier-combine", choices=["add", "mul"], default="add")
    parser.add_argument("--direction-eps", type=float, default=0.0)
    # Training stats heartbeat
    parser.add_argument("--train-stats-secs", type=float, default=30.0)
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
        feature_ticks=args.feature_ticks,
        reward_scale=args.reward_scale,
        neg_reward_coef=args.neg_reward_coef,
        nav_poll_secs=args.nav_poll_secs,
        pos_refresh_secs=args.pos_refresh_secs,
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

    # Input dim: tick(19) + dom(28) + M1(16) + M5(16) + H1(16) + D(16) + W(16) + time(10) = 137
    input_dim = 137

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
    current_units = refresh_units(api, account_id, args.instrument)
    # Reward and stats trackers
    stats_y: Deque[float] = deque(maxlen=600)
    last_stats_emit: float = time.time()
    last_grad_norm: Optional[float] = None
    last_r_micro: Optional[float] = None
    last_r_commit: Optional[float] = None
    last_r_total: Optional[float] = None
    last_r_target: Optional[float] = None
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
                    ccache.maybe_refresh(cfg.m1_refresh_secs, cfg.m5_refresh_secs, cfg.h1_refresh_secs, cfg.d_refresh_secs, cfg.w_refresh_secs)
                    hb: Dict[str, Any] = {
                        "type": "HB",
                        "nav": round(float(last_nav), 6) if isinstance(last_nav, (int, float)) else last_nav,
                        "units": int(current_units),
                        "open": bool(st.open),
                    }
                    # Include latest mid if available
                    try:
                        if fb.mid_window:
                            hb["mid"] = round(float(fb.mid_window[-1]), 5)
                    except Exception:
                        pass
                    if last_step_y is not None:
                        hb["y"] = round(float(last_step_y), 4)
                    if last_step_v is not None:
                        hb["v"] = round(float(last_step_v), 4)
                    if last_r_micro is not None:
                        hb["r_micro"] = round(float(last_r_micro), 6)
                    if last_r_commit is not None:
                        hb["r_commit"] = round(float(last_r_commit), 6)
                    if last_r_total is not None:
                        hb["r_total"] = round(float(last_r_total), 6)
                    if last_r_target is not None:
                        hb["r_exp"] = round(float(last_r_target), 6)
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
                x = np.concatenate([x_tick, x_dom, m1_feats, m5_feats, h1_feats, d1_feats, w1_feats, t_feats]).astype(np.float32)
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
                        theta = float(args.ou_theta); sigma = float(args.ou_sigma); dt = float(args.ou_dt)
                        # mean-reverting to 0
                        ou_noise_val = ou_noise_val + theta * (0.0 - ou_noise_val) * dt + sigma * np.sqrt(dt) * float(np.random.randn())
                        a_val = a_val + ou_noise_val
                    elif float(args.explore_eps) > 0.0 and float(args.explore_sigma) > 0.0:
                        if float(np.random.rand()) < float(args.explore_eps):
                            sigma_eff = float(args.explore_sigma_open) if (current_units != 0) else float(args.explore_sigma)
                            a_val = a_val + float(np.random.normal(loc=0.0, scale=sigma_eff))
                except Exception:
                    pass
                y_step = float(1.0 / (1.0 + np.exp(-a_val)))  # sigmoid(a_val) in (0,1)
                last_step_y = y_step
                last_step_v = v_step[0].item()
                try:
                    stats_y.append(y_step)
                except Exception:
                    pass
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
                    # Reward transform (exp or expm1) applied after optional clipping
                    _r_clipped = float(r_total if float(args.reward_clip) <= 0.0 else np.clip(r_total, -float(args.reward_clip), float(args.reward_clip)))
                    _r_transformed = float(np.expm1(_r_clipped) if str(args.reward_transform) == "expm1" else np.exp(_r_clipped))
                    target = torch.tensor(_r_transformed, dtype=torch.float32)
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
                    loss_micro.backward()
                    # Track grad norm and last rewards
                    try:
                        last_grad_norm = _compute_grad_norm(net.parameters())
                    except Exception:
                        last_grad_norm = None
                    try:
                        last_r_micro = float(r_micro)
                        last_r_commit = float(r_commit)
                        last_r_total = float(r_total)
                        last_r_target = float(_r_transformed)
                    except Exception:
                        pass
                    nn.utils.clip_grad_norm_(net.parameters(), float(args.max_grad_norm))
                    opt.step()

                # Natural close detection and training update
                if st.open and current_units == 0 and st.entry_nav is not None and last_nav > 0:
                    G_raw = (last_nav - st.entry_nav) / st.entry_nav * cfg.reward_scale
                    if G_raw < 0:
                        G_raw = G_raw * cfg.neg_reward_coef
                    G_clip = float(G_raw if float(cfg.reward_clip) <= 0.0 else np.clip(G_raw, -float(cfg.reward_clip), float(cfg.reward_clip)))
                    # Reward transform (exp or expm1)
                    G_transformed = float(np.expm1(G_clip) if str(args.reward_transform) == "expm1" else np.exp(G_clip))
                    # Tiered bonuses
                    # Determine direction correctness using last mid vs entry_mid
                    try:
                        mid_now = float(fb.mid_window[-1]) if fb.mid_window else None
                    except Exception:
                        mid_now = None
                    direction_ok = False
                    if (st.entry_mid is not None) and (mid_now is not None):
                        if (st.entry_action_long == 1):
                            direction_ok = (mid_now - st.entry_mid) > float(args.direction_eps)
                        else:
                            direction_ok = (st.entry_mid - mid_now) > float(args.direction_eps)
                    tier_add = float(args.close_bonus_base) \
                               + (float(args.close_bonus_direction) if direction_ok else 0.0) \
                               + (float(args.close_bonus_positive) if (G_raw > 0) else 0.0)
                    if str(args.tier_combine) == "mul":
                        G = float(G_transformed * (1.0 + tier_add))
                    else:
                        G = float(G_transformed + tier_add)
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
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    print(json.dumps({
                        "type": "EP_END",
                        "reward": float(G),
                        "reward_transformed": float(G_transformed),
                        "reward_pre_transform": float(G_clip),
                        "reward_raw": float(G_raw),
                        "direction_ok": bool(direction_ok),
                        "tier_bonus": float(tier_add),
                        "combine": str(args.tier_combine),
                        "adv": float(advantage),
                    }), flush=True)
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
                            # Immediate episodic update on close
                            if st.entry_features is not None and st.entry_nav is not None and last_nav > 0:
                                try:
                                    nav_now = fetch_nav(api, account_id) or last_nav
                                    G_raw = (nav_now - st.entry_nav) / st.entry_nav * cfg.reward_scale
                                    if G_raw < 0:
                                        G_raw = G_raw * cfg.neg_reward_coef
                                    G_clip = float(G_raw if float(cfg.reward_clip) <= 0.0 else np.clip(G_raw, -float(cfg.reward_clip), float(cfg.reward_clip)))
                                    G_transformed = float(np.expm1(G_clip) if str(args.reward_transform) == "expm1" else np.exp(G_clip))
                                    # Direction correctness vs entry_mid
                                    mid_now = float(fb.mid_window[-1]) if fb.mid_window else None
                                    direction_ok = False
                                    if (st.entry_mid is not None) and (mid_now is not None):
                                        direction_ok = (mid_now - st.entry_mid) > float(args.direction_eps)
                                    tier_add = float(args.close_bonus_base) \
                                               + (float(args.close_bonus_direction) if direction_ok else 0.0) \
                                               + (float(args.close_bonus_positive) if (G_raw > 0) else 0.0)
                                    G = float(G_transformed * (1.0 + tier_add)) if str(args.tier_combine) == "mul" else float(G_transformed + tier_add)
                                    opt.zero_grad()
                                    _, a_raw_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                                    dist = torch.distributions.Bernoulli(logits=a_raw_t)
                                    action_tensor = torch.tensor(1.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    advantage = G - v_t[0].detach().item()
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                                    opt.step()
                                    print(json.dumps({
                                        "type": "EP_END",
                                        "reward": float(G),
                                        "reward_transformed": float(G_transformed),
                                        "reward_pre_transform": float(G_clip),
                                        "reward_raw": float(G_raw),
                                        "direction_ok": bool(direction_ok),
                                        "tier_bonus": float(tier_add),
                                        "combine": str(args.tier_combine),
                                        "adv": float(advantage),
                                    }), flush=True)
                                except Exception:
                                    pass
                            st = TradeState(open=False)
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
                            # Immediate episodic update on close
                            if st.entry_features is not None and st.entry_nav is not None and last_nav > 0:
                                try:
                                    nav_now = fetch_nav(api, account_id) or last_nav
                                    G_raw = (nav_now - st.entry_nav) / st.entry_nav * cfg.reward_scale
                                    if G_raw < 0:
                                        G_raw = G_raw * cfg.neg_reward_coef
                                    G_clip = float(G_raw if float(cfg.reward_clip) <= 0.0 else np.clip(G_raw, -float(cfg.reward_clip), float(cfg.reward_clip)))
                                    G_transformed = float(np.expm1(G_clip) if str(args.reward_transform) == "expm1" else np.exp(G_clip))
                                    # Direction correctness vs entry_mid
                                    mid_now = float(fb.mid_window[-1]) if fb.mid_window else None
                                    direction_ok = False
                                    if (st.entry_mid is not None) and (mid_now is not None):
                                        direction_ok = (st.entry_mid - mid_now) > float(args.direction_eps)
                                    tier_add = float(args.close_bonus_base) \
                                               + (float(args.close_bonus_direction) if direction_ok else 0.0) \
                                               + (float(args.close_bonus_positive) if (G_raw > 0) else 0.0)
                                    G = float(G_transformed * (1.0 + tier_add)) if str(args.tier_combine) == "mul" else float(G_transformed + tier_add)
                                    opt.zero_grad()
                                    _, a_raw_t, v_t = net(torch.from_numpy(st.entry_features[None, :]).float())
                                    dist = torch.distributions.Bernoulli(logits=a_raw_t)
                                    action_tensor = torch.tensor(0.0, dtype=torch.float32)
                                    logprob = dist.log_prob(action_tensor)
                                    advantage = G - v_t[0].detach().item()
                                    actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob.mean()
                                    critic_loss = F.smooth_l1_loss(v_t[0], torch.tensor(G, dtype=torch.float32))
                                    loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * dist.entropy().mean()
                                    loss.backward()
                                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                                    opt.step()
                                    print(json.dumps({
                                        "type": "EP_END",
                                        "reward": float(G),
                                        "reward_transformed": float(G_transformed),
                                        "reward_pre_transform": float(G_clip),
                                        "reward_raw": float(G_raw),
                                        "direction_ok": bool(direction_ok),
                                        "tier_bonus": float(tier_add),
                                        "combine": str(args.tier_combine),
                                        "adv": float(advantage),
                                    }), flush=True)
                                except Exception:
                                    pass
                            st = TradeState(open=False)
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
                            st.entry_mid = float(mid) if isinstance(mid, (int, float)) else None
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
                            st.entry_mid = float(mid) if isinstance(mid, (int, float)) else None
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
