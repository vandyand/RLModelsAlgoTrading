#!/usr/bin/env python3
"""
Actor-Critic Box Trader v0.0.1 (single instrument scaffolding)
- Tick + DOM + REST H1/D candle features (verbose)
- Box trade: enter gate, direction, magnitude, tp/sl pips, max hold time
- Episodic reward at close credited to entry (A2C update)
- Designed to evolve into multi-instrument with shared trunk/attention later
"""
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

from oandapyV20 import API
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.instruments as instruments_ep

# Reuse order helper for market order with TP/SL
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from streamer.orders import place_market_order  # type: ignore


@dataclass
class Config:
    instruments: List[str]
    max_units: int = 500
    min_units: int = 10
    order_cooldown: float = 7.0
    min_tp_pips: float = 2.0
    max_tp_pips: float = 12.0
    min_sl_pips: float = 2.0
    max_sl_pips: float = 12.0
    min_trade_sec: float = 10.0
    max_trade_sec: float = 600.0
    feature_ticks: int = 240
    reward_scale: float = 10000.0
    nav_poll_secs: float = 10.0
    pos_refresh_secs: float = 15.0
    lr: float = 1e-3
    actor_sigma: float = 0.3
    entropy_coef: float = 0.001
    device: str = "cpu"
    autosave_secs: float = 120.0
    model_path: str = ""
    # REST candles
    h1_bars: int = 60
    d_bars: int = 60
    w_bars: int = 60
    h1_refresh_secs: float = 60.0
    d_refresh_secs: float = 300.0
    w_refresh_secs: float = 3600.0


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
        self.dom_depth = 5  # fixed K=5
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
        # spread z-score
        if len(self.spread_window) >= 30:
            sp_arr = np.array(self.spread_window)
            sp = (sp_arr[-1] - sp_arr.mean()) / (sp_arr.std() + self.eps)
        else:
            sp = 0.0
        # EMA/MACD-like
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd = ema12 - ema26
        signal = ema(macd, 9)
        hist = macd - signal
        macd_last = float(macd[-1]); signal_last = float(signal[-1]); hist_last = float(hist[-1])
        # RSI(14)
        rsi14 = rsi(prices, 14, self.eps)
        # Vol-of-vol: std of last 20 absolute returns
        vol_of_vol = float(np.std(np.abs(r[-20:])) if len(r) >= 20 else 0.0)
        # OBV proxy at tick-level: cumulative sign of returns (normalized)
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
            s60,           # realized vol proxy
            sp,            # spread z-score
            macd_last, signal_last, hist_last,
            rsi14 / 100.0,  # scale to 0..1
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
        return float(np.log((close[-1] + eps)/(close[-1 - k] + eps)))

    def std_ret(k: int) -> float:
        if len(r) < k:
            return 0.0
        return float(np.std(r[-k:]) + eps)

    # ATR(20)
    tr_list: List[float] = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_list.append(tr)
    atr20 = float(np.mean(tr_list[-20:])) if len(tr_list) >= 1 else 0.0

    # Breakout distances
    def dist_max(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.max(close[-k:]))/(np.std(close[-k:]) + eps))

    def dist_min(k: int) -> float:
        if len(close) < k:
            return 0.0
        return float((close[-1] - np.min(close[-k:]))/(np.std(close[-k:]) + eps))

    # Volume z-score(20)
    if len(vol) >= 20:
        v20 = (vol[-1] - np.mean(vol[-20:]))/(np.std(vol[-20:]) + eps)
    else:
        v20 = 0.0

    # EMA/MACD on closes
    ema12c = ema(close, 12); ema26c = ema(close, 26)
    macd = ema12c - ema26c
    signal = ema(macd, 9)
    hist = macd - signal
    macd_last = float(macd[-1]); signal_last = float(signal[-1]); hist_last = float(hist[-1])
    # RSI(14)
    rsi14 = rsi(close, 14, eps)
    # Vol-of-vol: std of abs returns over last 20 bars
    vol_of_vol = float(np.std(np.abs(r[-20:])) if len(r) >= 20 else 0.0)
    # OBV: cumulative based on volume and close direction; z-score over last 20
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


class ActorCriticNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, 6)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.actor(h)
        v = self.critic(h).squeeze(-1)
        return h, mu, v


@dataclass
class BoxState:
    open: bool = False
    last_order_time: float = 0.0
    entry_features: Optional[np.ndarray] = None
    entry_z: Optional[np.ndarray] = None
    entry_v: Optional[float] = None
    entry_nav: Optional[float] = None
    deadline_ts: Optional[float] = None


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


def sample_actions(mu: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.random.normal(0.0, sigma, size=mu.shape)
    z = mu + eps
    return z, eps


def decode_actions(z: np.ndarray, cfg: Config) -> Dict[str, float]:
    direction = float(np.tanh(z[0]))
    magnitude = float(1.0 / (1.0 + np.exp(-z[1])))
    tp = cfg.min_tp_pips + float(1.0 / (1.0 + np.exp(-z[2]))) * max(0.0, cfg.max_tp_pips - cfg.min_tp_pips)
    sl = cfg.min_sl_pips + float(1.0 / (1.0 + np.exp(-z[3]))) * max(0.0, cfg.max_sl_pips - cfg.min_sl_pips)
    max_time = cfg.min_trade_sec + float(1.0 / (1.0 + np.exp(-z[4]))) * max(0.0, cfg.max_trade_sec - cfg.min_trade_sec)
    enter_prob = float(1.0 / (1.0 + np.exp(-z[5])))
    return {
        "direction": direction,
        "magnitude": magnitude,
        "tp_pips": tp,
        "sl_pips": sl,
        "max_time_sec": max_time,
        "enter_prob": enter_prob,
        "enter": enter_prob >= 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-Critic Box Trader v0.0.1 (single instrument)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", default="practice", choices=["practice", "live"])
    parser.add_argument("--max-units", type=int, default=500)
    parser.add_argument("--min-units", type=int, default=10)
    parser.add_argument("--order-cooldown", type=float, default=7.0)
    parser.add_argument("--min-tp-pips", type=float, default=2.0)
    parser.add_argument("--max-tp-pips", type=float, default=12.0)
    parser.add_argument("--min-sl-pips", type=float, default=2.0)
    parser.add_argument("--max-sl-pips", type=float, default=12.0)
    parser.add_argument("--min-trade-sec", type=float, default=10.0)
    parser.add_argument("--max-trade-sec", type=float, default=600.0)
    parser.add_argument("--feature-ticks", type=int, default=240)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    parser.add_argument("--h1-refresh-secs", type=float, default=60.0)
    parser.add_argument("--d-refresh-secs", type=float, default=300.0)
    parser.add_argument("--w-refresh-secs", type=float, default=3600.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--actor-sigma", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--autosave-secs", type=float, default=120.0)
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/ac_v001.pt")
    args = parser.parse_args()

    cfg = Config(
        instruments=[args.instrument],
        max_units=args.max_units,
        min_units=args.min_units,
        order_cooldown=args.order_cooldown,
        min_tp_pips=args.min_tp_pips,
        max_tp_pips=args.max_tp_pips,
        min_sl_pips=args.min_sl_pips,
        max_sl_pips=args.max_sl_pips,
        min_trade_sec=args.min_trade_sec,
        max_trade_sec=args.max_trade_sec,
        feature_ticks=args.feature_ticks,
        reward_scale=args.reward_scale,
        nav_poll_secs=args.nav_poll_secs,
        pos_refresh_secs=args.pos_refresh_secs,
        lr=args.lr,
        actor_sigma=args.actor_sigma,
        entropy_coef=args.entropy_coef,
        device="cpu",
        autosave_secs=args.autosave_secs,
        model_path=args.model_path,
        w_refresh_secs=args.w_refresh_secs,
    )

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials in env vars.")
    api = API(access_token=access_token, environment=args.environment)

    # Builders and caches
    fb = FeatureBuilder(cfg.feature_ticks)
    ccache = CandleCache(api, args.instrument, cfg.h1_bars, cfg.d_bars, cfg.w_bars)
    ccache.backfill(cfg.h1_bars, cfg.d_bars, cfg.w_bars)

    # Input dim: tick(19) + dom(8+4K=28) + H1(16) + D(16) + W(16) = 95
    input_dim = 95

    net = ActorCriticNet(input_dim=input_dim)
    net.train()
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    # State
    box = BoxState(open=False)
    last_nav = fetch_nav(api, account_id) or 1.0
    last_nav_poll = time.time()
    last_pos_refresh = time.time()
    current_units = refresh_units(api, account_id, args.instrument)

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
            # Periodically refresh REST candles
            ccache.maybe_refresh(args.h1_refresh_secs, args.d_refresh_secs, args.w_refresh_secs)
            print(json.dumps({"type": "HB", "nav": last_nav, "units": current_units}), flush=True)
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
        # Candle features
        h1_feats = candle_features(ccache.h1)
        d1_feats = candle_features(ccache.d1)
        w1_feats = candle_features(ccache.w1)
        if x_tick is None:
            continue
        x = np.concatenate([x_tick, x_dom, h1_feats, d1_feats, w1_feats]).astype(np.float32)
        xt = torch.from_numpy(x)[None, :]

        # Close by deadline
        if box.open and box.deadline_ts is not None and tnow >= box.deadline_ts and current_units != 0:
            try:
                order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                           units=-current_units, tp_pips=None, sl_pips=None,
                                           anchor=None, client_tag="ac-box", client_comment="deadline close",
                                           fifo_safe=False, fifo_adjust=False)
                print(json.dumps({"type": "FORCE_CLOSE", "order": order}), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
            current_units = refresh_units(api, account_id, args.instrument)

        # Detect natural close
        if box.open and current_units == 0 and box.entry_nav is not None and last_nav > 0:
            G = (last_nav - box.entry_nav) / box.entry_nav * cfg.reward_scale
            with torch.no_grad():
                _, _, v_entry = net(torch.from_numpy(box.entry_features[None, :]).float())
                v_entry_val = float(v_entry.item())
            advantage = G - v_entry_val
            # A2C update
            opt.zero_grad()
            _, mu_t, v_t = net(torch.from_numpy(box.entry_features[None, :]).float())
            mu = mu_t[0]
            z = torch.from_numpy(box.entry_z.astype(np.float32))
            sigma = cfg.actor_sigma
            logprob = -0.5 * torch.sum(((z - mu) / sigma) ** 2) - 6 * torch.log(torch.tensor(sigma))
            actor_loss = -torch.tensor(advantage, dtype=torch.float32) * logprob
            critic_loss = 0.5 * (torch.tensor(G, dtype=torch.float32) - v_t[0]) ** 2
            entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))
            loss = actor_loss + critic_loss + cfg.entropy_coef * entropy
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            print(json.dumps({"type": "BOX_CLOSED", "reward": G, "adv": advantage}), flush=True)
            box = BoxState(open=False)

        # Entry
        can_enter = (not box.open) and ((tnow - box.last_order_time) >= cfg.order_cooldown)
        if can_enter:
            net.train()
            _, mu_t, v_t = net(xt)
            mu = mu_t[0].detach().numpy()
            z, _ = sample_actions(mu, cfg.actor_sigma)
            outs = decode_actions(z, cfg)
            if outs["enter"]:
                units = int(round(outs["magnitude"] * cfg.max_units))
                units = units if outs["direction"] >= 0 else -units
                if abs(units) >= cfg.min_units:
                    try:
                        order = place_market_order(api=api, account_id=account_id, instrument=args.instrument,
                                                   units=units, tp_pips=outs["tp_pips"], sl_pips=outs["sl_pips"],
                                                   anchor=None, client_tag="ac-box", client_comment="box open",
                                                   fifo_safe=False, fifo_adjust=False)
                        current_units = refresh_units(api, account_id, args.instrument)
                        box.open = True
                        box.last_order_time = tnow
                        box.entry_features = x.copy()
                        box.entry_z = z.copy()
                        box.entry_v = float(v_t.item())
                        box.entry_nav = last_nav
                        box.deadline_ts = tnow + outs["max_time_sec"]
                        print(json.dumps({"type": "BOX_OPEN", "units": units, "tp": outs["tp_pips"], "sl": outs["sl_pips"], "tmax": outs["max_time_sec"], "order": order}), flush=True)
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


if __name__ == "__main__":
    main()
