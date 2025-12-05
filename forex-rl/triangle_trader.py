#!/usr/bin/env python3
"""
Triangle Strategy (long-only) - Tick Stream, Single Instrument

- Three-head linear policy per instrument:
  0: delta_pips in [min_pips, max_pips] via sigmoid
  1: time_sec in [min_time, max_time] via sigmoid
  2: enter gate probability in (0, 1) via sigmoid (enter iff >= 0.5)

- Geometry: Treat price-time as a 2D plane (x=time, y=price in pips).
  The current point at entry is the midpoint of one side of an equilateral triangle whose apex
  is at (time_sec, delta_pips) relative to the entry point after scaling time by `time_to_pips_scale`.
  While the position is open, we compute the current (x,y) relative to entry; if the point leaves
  the triangle, we immediately close the position. No traditional TP/SL orders are attached.

- Online episodic update: when a trade closes (position returns to flat), we compute an episode
  return from NAV change and update the policy weights with a simple bandit-style ascent using
  the head-wise derivatives of the decoding transforms.

- Heavy references: structure mirrors multi_instrument_* tick loop and box policy update, but with 3 heads.

Environment:
  Requires OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY in environment.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing

# Make repo root importable so we can reuse utilities in streamer/
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from streamer.orders import place_market_order, fetch_instrument_spec, calc_pip_size  # type: ignore


# ---------- Feature engineering (ticks -> compact feature vector) ----------

def compute_features_from_mid_series(mids: List[float]) -> Optional[np.ndarray]:
    """Compute features from a rolling series of mid prices (ticks).

    Mirrors the compact features used by online tick traders.
    """
    if len(mids) < 60:
        return None
    prices = np.array(mids, dtype=float)
    logp = np.log(prices)
    r = np.diff(logp)

    def last_ratio(k: int) -> float:
        if len(prices) <= k:
            return 0.0
        return float(np.log(prices[-1] / prices[-1 - k]))

    def roll_mean_std(window: int) -> Tuple[float, float]:
        if len(r) < window:
            return 0.0, 0.0
        seg = r[-window:]
        return float(np.mean(seg)), float(np.std(seg) + 1e-12)

    r1 = last_ratio(1)
    r5 = last_ratio(5)
    r20 = last_ratio(20)

    m5, s5 = roll_mean_std(5)
    m20, s20 = roll_mean_std(20)
    m60, s60 = roll_mean_std(60)

    if len(prices) < 20:
        sma20 = prices[-1]
        price_std20 = 1e-6
    else:
        sma20 = float(np.mean(prices[-20:]))
        price_std20 = float(np.std(prices[-20:]) + 1e-12)
    z_price_sma20 = float((prices[-1] - sma20) / price_std20)

    realized_vol_60 = s60

    features = np.array([
        1.0,
        r1, r5, r20,
        m5, s5,
        m20, s20,
        m60, s60,
        z_price_sma20,
        realized_vol_60,
    ], dtype=float)
    return np.clip(features, -5.0, 5.0)


# ---------- OANDA helpers ----------

def fetch_nav(api: API, account_id: str) -> Optional[float]:
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        nav = resp.get("account", {}).get("NAV")
        return float(nav) if nav is not None else None
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


# ---------- Triangle policy ----------


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class TrianglePolicy:
    """Three-head linear policy.

    Heads (logits -> bounded outputs):
      0: delta_pips = min_pips + sigmoid(z0) * (max_pips - min_pips)
      1: time_sec   = min_time + sigmoid(z1) * (max_time - min_time)
      2: enter_prob = sigmoid(z2) -> boolean gate
    """

    def __init__(self, num_features: int, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        # weights shape (3, num_features)
        self.weights = np.zeros((3, num_features), dtype=float)

    def act(
        self,
        features: np.ndarray,
        min_pips: float,
        max_pips: float,
        min_time: float,
        max_time: float,
        noise_sigma: float = 0.0,
    ) -> Dict[str, float]:
        z = self.weights @ features  # shape (3,)
        if noise_sigma > 0.0:
            z = z + np.random.normal(0.0, noise_sigma, size=z.shape)
        s0 = _sigmoid(float(z[0]))
        s1 = _sigmoid(float(z[1]))
        s2 = _sigmoid(float(z[2]))
        delta_pips = min_pips + s0 * max(0.0, max_pips - min_pips)
        time_sec = min_time + s1 * max(0.0, max_time - min_time)
        enter_prob = s2
        enter_flag = (enter_prob >= 0.5)
        return {
            "delta_pips": float(delta_pips),
            "time_sec": float(time_sec),
            "enter_prob": float(enter_prob),
            "enter": enter_flag,
            "z0": float(z[0]),
            "z1": float(z[1]),
            "z2": float(z[2]),
        }

    def update(
        self,
        features: np.ndarray,
        entry_logits: np.ndarray,
        reward: float,
        min_pips: float,
        max_pips: float,
        min_time: float,
        max_time: float,
    ) -> None:
        # Derivatives of outputs wrt logits at entry
        dz = np.zeros(3, dtype=float)
        s0 = _sigmoid(float(entry_logits[0])); dz[0] = s0 * (1.0 - s0) * max(0.0, max_pips - min_pips)
        s1 = _sigmoid(float(entry_logits[1])); dz[1] = s1 * (1.0 - s1) * max(0.0, max_time - min_time)
        s2 = _sigmoid(float(entry_logits[2])); dz[2] = s2 * (1.0 - s2)
        for h in range(3):
            self.weights[h, :] += self.learning_rate * reward * dz[h] * features


# ---------- Persistence ----------

def save_policy(path: str, policy: Optional[TrianglePolicy], noise_sigma: float) -> None:
    try:
        if policy is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "weights": policy.weights.tolist(),
            "noise_sigma": float(noise_sigma),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_policy(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


# ---------- Main loop ----------

@dataclass
class State:
    open: bool = False
    entry_features: Optional[np.ndarray] = None
    entry_logits: Optional[np.ndarray] = None
    entry_nav: Optional[float] = None
    last_order_time: float = 0.0
    # Triangle/entry geometry context
    entry_wall_ts: Optional[float] = None
    entry_mid: Optional[float] = None
    pip_size: Optional[float] = None
    apex_time_sec: Optional[float] = None
    apex_delta_pips: Optional[float] = None


# ---------- Triangle geometry ----------

def _rotate90(vec: np.ndarray) -> np.ndarray:
    # Rotate (x, y) by +90 degrees => (-y, x)
    return np.array([-vec[1], vec[0]], dtype=float)


def triangle_vertices_equilateral(apex_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vertices (A=apex, B, C) of an equilateral triangle.

    Base midpoint is at origin; apex is at apex_v. Base endpoints lie along the
    direction perpendicular to apex_v, with base length L = 2/sqrt(3) * |apex_v|.
    """
    v = np.array(apex_v, dtype=float)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-9:
        # Degenerate; return tiny triangle around origin to avoid division by zero
        eps = 1e-6
        return v, np.array([eps, 0.0]), np.array([-eps, 0.0])
    L = (2.0 / np.sqrt(3.0)) * v_norm
    u_hat = _rotate90(v) / v_norm  # perpendicular unit vector
    B = (L / 2.0) * u_hat
    C = -(L / 2.0) * u_hat
    A = v
    return A, B, C


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    # Barycentric sign test
    def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))

    b1 = sign(p, a, b) < 0.0
    b2 = sign(p, b, c) < 0.0
    b3 = sign(p, c, a) < 0.0
    return (b1 == b2) and (b2 == b3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triangle long-only trader (tick stream)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--max-units", type=int, default=1000, help="Max absolute units for long entry")
    parser.add_argument("--min-units", type=int, default=5, help="Minimum units to place an order")
    parser.add_argument("--units-frac", type=float, default=0.5, help="Fraction of max-units to use on enter (0..1)")
    parser.add_argument("--order-cooldown", type=float, default=5.0)
    parser.add_argument("--feature-ticks", type=int, default=240)
    # Triangle outputs ranges
    parser.add_argument("--min-pips", type=float, default=2.0)
    parser.add_argument("--max-pips", type=float, default=15.0)
    parser.add_argument("--min-time-sec", type=float, default=5.0)
    parser.add_argument("--max-time-sec", type=float, default=900.0)
    parser.add_argument("--time-to-pips-scale", type=float, default=1.0, help="Scale seconds into pip-units for triangle metric (pip_units_per_second)")
    # Rewarding
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--nav-poll-secs", type=float, default=10.0)
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0)
    # Exploration + persistence
    parser.add_argument("--noise-sigma", type=float, default=0.2)
    parser.add_argument("--noise-decay", type=float, default=0.999)
    parser.add_argument("--noise-min", type=float, default=0.02)
    parser.add_argument("--autosave-secs", type=float, default=60.0)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--log-debug", action="store_true")
    args = parser.parse_args()

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY.")

    api = API(access_token=access_token, environment=args.environment)

    # Policy + model path
    policy: Optional[TrianglePolicy] = None
    if args.model_path:
        model_path = args.model_path
    else:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, f"triangle_trader_{args.instrument}_TICK.json")
    last_save_ts: float = 0.0

    # State
    st = State(open=False)
    last_nav = fetch_nav(api, account_id)
    nav_estimate = last_nav or 1.0
    last_nav_update_time = time.time()
    last_pos_refresh_time = time.time()
    current_units = refresh_units(api, account_id, args.instrument)
    # Instrument pip size for pip conversion
    try:
        spec = fetch_instrument_spec(api, account_id, args.instrument)
        pip_location = int(spec.get("pipLocation"))
        pip_size = calc_pip_size(pip_location)
    except Exception:
        pip_size = 0.0001  # sensible default for many FX majors
    noise_sigma = args.noise_sigma

    # Tick stream
    stream = pricing.PricingStream(accountID=account_id, params={"instruments": args.instrument})

    # Rolling mids for features
    from collections import deque
    mid_window = deque(maxlen=args.feature_ticks)

    while True:
        try:
            for msg in api.request(stream):
                msg_type = msg.get("type")
                if msg_type == "HEARTBEAT":
                    now_wall = time.time()
                    if (now_wall - last_nav_update_time) >= args.nav_poll_secs:
                        nav_now = fetch_nav(api, account_id)
                        if nav_now is not None and nav_now > 0:
                            nav_estimate = nav_now
                            last_nav = nav_now
                        last_nav_update_time = now_wall
                    if (now_wall - last_pos_refresh_time) >= args.pos_refresh_secs:
                        current_units = refresh_units(api, account_id, args.instrument)
                        last_pos_refresh_time = now_wall
                    try:
                        print(json.dumps({"type": "HEARTBEAT", "time": msg.get("time"), "nav": nav_estimate, "units": current_units}), flush=True)
                    except Exception:
                        pass
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

                ts = msg.get("time")
                try:
                    print(json.dumps({"type": "PRICE", "time": ts, "bid": bid, "ask": ask, "mid": mid}), flush=True)
                except Exception:
                    pass

                mid_window.append(mid)
                now_wall = time.time()

                # Triangle boundary check: if outside, close immediately
                if st.open and current_units != 0 and st.entry_wall_ts is not None and st.entry_mid is not None and st.apex_time_sec is not None and st.apex_delta_pips is not None:
                    # Current point in triangle coordinates
                    elapsed_sec = max(0.0, now_wall - float(st.entry_wall_ts))
                    y_pips = (mid - float(st.entry_mid)) / float(st.pip_size or pip_size)
                    p = np.array([elapsed_sec * args.time_to_pips_scale, y_pips], dtype=float)
                    apex_v = np.array([float(st.apex_time_sec) * args.time_to_pips_scale, float(st.apex_delta_pips)], dtype=float)
                    A, B, C = triangle_vertices_equilateral(apex_v)
                    inside = point_in_triangle(p, A, B, C)
                    if not inside:
                        try:
                            order = place_market_order(
                                api=api, account_id=account_id, instrument=args.instrument,
                                units=-current_units, tp_pips=None, sl_pips=None,
                                anchor=None, client_tag="tri-rl", client_comment="triangle exit",
                                fifo_safe=False, fifo_adjust=False,
                            )
                            print(json.dumps({
                                "type": "TRI_EXIT_OUTSIDE",
                                "time": ts,
                                "elapsed_sec": elapsed_sec,
                                "price_offset_pips": y_pips,
                                "apex_time_sec": float(st.apex_time_sec),
                                "apex_delta_pips": float(st.apex_delta_pips),
                                "order": (order.get("response") if isinstance(order, dict) else None),
                            }), flush=True)
                        except Exception as _:
                            pass
                        current_units = refresh_units(api, account_id, args.instrument)

                # Detect close by flat
                if st.open and current_units == 0 and st.entry_nav is not None and nav_estimate > 0:
                    ep_return = (nav_estimate - st.entry_nav) / st.entry_nav
                    shaped_reward = ep_return * args.reward_scale
                    if st.entry_features is not None and st.entry_logits is not None and policy is not None:
                        policy.update(
                            st.entry_features,
                            st.entry_logits,
                            shaped_reward,
                            args.min_pips, args.max_pips,
                            args.min_time_sec, args.max_time_sec,
                        )
                    print(json.dumps({"type": "TRI_CLOSED", "time": ts, "reward": shaped_reward}), flush=True)
                    # Clear state
                    st = State(open=False)

                # Build features and maybe act if flat
                features = None
                if len(mid_window) >= 60:
                    features = compute_features_from_mid_series(list(mid_window))
                    if features is not None and policy is None:
                        policy = TrianglePolicy(num_features=features.shape[0], learning_rate=0.05)
                        # Attempt to load checkpoint
                        ckpt = load_policy(model_path)
                        if ckpt:
                            try:
                                if isinstance(ckpt.get("weights"), list):
                                    arr = np.array(ckpt.get("weights"), dtype=float)
                                    if arr.ndim == 2 and arr.shape[1] == features.shape[0] and arr.shape[0] == 3:
                                        policy.weights = arr
                                if isinstance(ckpt.get("noise_sigma"), (int, float)):
                                    noise_sigma = float(ckpt.get("noise_sigma"))
                            except Exception:
                                pass

                # Consider entering long if flat
                if current_units == 0 and (not st.open) and features is not None and policy is not None and (now_wall - st.last_order_time) >= args.order_cooldown:
                    outs = policy.act(
                        features,
                        min_pips=args.min_pips, max_pips=args.max_pips,
                        min_time=args.min_time_sec, max_time=args.max_time_sec,
                        noise_sigma=noise_sigma,
                    )
                    if not outs.get("enter", False):
                        try:
                            print(json.dumps({"type": "SKIP_ENTER", "time": ts, "enter_prob": outs.get("enter_prob")}), flush=True)
                        except Exception:
                            pass
                    else:
                        units = int(round(max(args.min_units, args.max_units * max(0.0, min(1.0, args.units_frac)))))
                        try:
                            order = place_market_order(
                                api=api, account_id=account_id, instrument=args.instrument,
                                units=units, tp_pips=None, sl_pips=None,
                                anchor=None, client_tag="tri-rl", client_comment="triangle open",
                                fifo_safe=False, fifo_adjust=False,
                            )
                            st.open = True
                            st.entry_features = features.copy()
                            st.entry_logits = np.array([outs["z0"], outs["z1"], outs["z2"]], dtype=float)
                            st.entry_nav = nav_estimate
                            st.entry_wall_ts = now_wall
                            st.entry_mid = mid
                            st.pip_size = float(pip_size)
                            st.apex_time_sec = float(outs["time_sec"]) if float(outs["time_sec"]) > 0 else float(args.min_time_sec)
                            st.apex_delta_pips = float(outs["delta_pips"]) if float(outs["delta_pips"]) > 0 else float(args.min_pips)
                            st.last_order_time = now_wall
                            current_units = refresh_units(api, account_id, args.instrument)
                            print(json.dumps({
                                "type": "TRI_OPEN",
                                "time": ts,
                                "units": units,
                                "apex_time_sec": float(st.apex_time_sec),
                                "apex_delta_pips": float(st.apex_delta_pips),
                                "time_to_pips_scale": args.time_to_pips_scale,
                                "enter_prob": float(outs["enter_prob"]),
                                "order": (order.get("response") if isinstance(order, dict) else None),
                            }), flush=True)
                        except Exception as exc:
                            print(json.dumps({"error": str(exc)}), flush=True)

                # Noise anneal and autosave
                noise_sigma = max(args.noise_min, noise_sigma * args.noise_decay)
                if args.autosave_secs > 0 and (now_wall - last_save_ts) >= args.autosave_secs:
                    save_policy(model_path, policy, noise_sigma)
                    last_save_ts = now_wall

        except KeyboardInterrupt:
            print("Interrupted. Exiting.", flush=True)
            try:
                save_policy(model_path, policy, noise_sigma)
            except Exception:
                pass
            break
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
            time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
