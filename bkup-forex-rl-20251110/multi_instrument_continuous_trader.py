import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

# Make repo root importable so we can reuse utilities in streamer/
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Reuse existing helper for market order placement (TP/SL optional)
from streamer.orders import place_market_order  # type: ignore


ACTIONS: List[int] = [-1, 0, 1]


@dataclass
class BanditConfig:
    epsilon: float = 0.2        # initial exploration rate
    min_epsilon: float = 0.02   # floor for exploration
    epsilon_decay: float = 0.999  # multiplicative decay per step
    alpha: float = 0.1          # unused in contextual mode, reserved


class ContextualSoftmaxPolicy:
    def __init__(self, num_features: int, learning_rate: float = 0.1) -> None:
        # We maintain weights per action: shape (num_actions, num_features)
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.zeros((len(ACTIONS), num_features), dtype=float)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def action_probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.weights @ features  # shape (num_actions,)
        return self._softmax(logits)

    def select_action(self, features: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        # Epsilon-greedy over softmax probabilities
        probs = self.action_probabilities(features)
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action_index = int(np.argmax(probs))
            action = ACTIONS[action_index]
        return action, probs

    def update_on_episode(self, features: np.ndarray, chosen_action: int, episode_return: float) -> None:
        # REINFORCE single-step update: grad = (one_hot(a) - probs) ⊗ x
        probs = self.action_probabilities(features)
        one_hot = np.zeros(len(ACTIONS))
        action_index = ACTION_TO_INDEX[chosen_action]
        one_hot[action_index] = 1.0
        grad = (one_hot - probs)[:, None] * features[None, :]  # shape (num_actions, num_features)
        self.weights += self.learning_rate * episode_return * grad


class ContinuousLinearPolicy:
    """Deterministic continuous policy: a = tanh(w^T x) in [-1, 1].

    Online update with a simple bandit-like gradient ascent:
      w <- w + lr * reward * a * (1 - a^2) * x
    This pushes the policy to increase magnitude in the direction that led to positive reward.
    """

    def __init__(self, num_features: int, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_features, dtype=float)

    def act(self, features: np.ndarray) -> float:
        z = float(self.weights @ features)
        return float(np.tanh(z))

    def update(self, features: np.ndarray, action_value: float, reward: float) -> None:
        # grad of tanh is (1 - a^2)
        grad = (1.0 - action_value * action_value) * features
        self.weights += self.learning_rate * reward * action_value * grad


def fetch_nav(api: API, account_id: str) -> Optional[float]:
    try:
        resp = api.request(accounts.AccountSummary(accountID=account_id))
        nav = resp.get("account", {}).get("NAV")
        return float(nav) if nav is not None else None
    except Exception:
        return None


def get_open_position_direction(api: API, account_id: str, instrument: str) -> int:
    """Return -1 for net short, 1 for net long, 0 if flat for instrument."""
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for pos in resp.get("positions", []):
            if pos.get("instrument") != instrument:
                continue
            long_units = float(pos.get("long", {}).get("units") or 0.0)
            short_units = float(pos.get("short", {}).get("units") or 0.0)
            net = long_units + short_units
            if net > 0:
                return 1
            if net < 0:
                return -1
            return 0
    except Exception:
        pass
    return 0


def get_net_units(api: API, account_id: str, instrument: str) -> int:
    try:
        resp = api.request(positions.OpenPositions(accountID=account_id))
        for pos in resp.get("positions", []):
            if pos.get("instrument") != instrument:
                continue
            long_units = float(pos.get("long", {}).get("units") or 0.0)
            short_units = float(pos.get("short", {}).get("units") or 0.0)
            net = long_units + short_units
            return int(round(net))
    except Exception:
        pass
    return 0


def place_units_delta_order(
    api: API,
    account_id: str,
    instrument: str,
    delta_units: int,
) -> Optional[Dict[str, object]]:
    if delta_units == 0:
        return None
    try:
        # No TP/SL; this is a pure position size adjustment
        result = place_market_order(
            api=api,
            account_id=account_id,
            instrument=instrument,
            units=delta_units,
            tp_pips=None,
            sl_pips=None,
            anchor=None,
            client_tag="online-rl",
            client_comment="online sizing order",
            fifo_safe=False,
            fifo_adjust=False,
        )
        return result
    except Exception as exc:
        return {"error": str(exc)}


def summarize_order_result(order_result: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not order_result or not isinstance(order_result, dict):
        return None
    resp = order_result.get("response") if isinstance(order_result.get("response"), dict) else None
    if not isinstance(resp, dict):
        return {"error": order_result.get("error")}
    fill = resp.get("orderFillTransaction") if isinstance(resp.get("orderFillTransaction"), dict) else None
    create = resp.get("orderCreateTransaction") if isinstance(resp.get("orderCreateTransaction"), dict) else None
    if isinstance(fill, dict):
        return {
            "order_id": fill.get("orderID"),
            "fill_id": fill.get("id"),
            "instrument": fill.get("instrument"),
            "units": fill.get("units"),
            "price": fill.get("price"),
            "time": fill.get("time"),
        }
    if isinstance(create, dict):
        return {
            "order_id": create.get("id"),
            "instrument": create.get("instrument"),
            "units": create.get("units"),
            "time": create.get("time"),
        }
    return None


ACTION_TO_INDEX: Dict[int, int] = {-1: 0, 0: 1, 1: 2}


def save_policy(path: str, policy: Optional["ContinuousLinearPolicy"], noise_sigma: float) -> None:
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
        # Best-effort; ignore persistence errors
        pass


def load_policy(path: str) -> Optional[Dict[str, object]]:
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


def compute_features_from_mid_series(mids: List[float]) -> Optional[np.ndarray]:
    """Compute features from a rolling series of mid prices (ticks)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Online continuous sizing trader (tick stream, multi-instrument)")
    parser.add_argument("--instruments", default="EUR_USD,USD_JPY", help="Comma-separated instruments to trade")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--max-units", type=int, default=1000, help="Max absolute position units for target sizing")
    parser.add_argument("--deadband-units", type=int, default=10, help="Skip orders when adjustment is below this threshold")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing for target sizing (0..1; higher=stickier)")
    parser.add_argument("--order-cooldown", type=float, default=2.0, help="Minimum seconds between orders")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for continuous policy")
    parser.add_argument("--feature-ticks", type=int, default=240, help="Number of recent mid ticks to compute features")
    parser.add_argument("--nav-poll-secs", type=float, default=10.0, help="Interval seconds to refresh NAV estimate")
    parser.add_argument("--pos-refresh-secs", type=float, default=15.0, help="Interval seconds to refresh net position units")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Sleep between stream yields when needed")
    parser.add_argument("--log-debug", action="store_true", help="Print full debug logs every step")
    parser.add_argument("--noise-sigma", type=float, default=0.2, help="Stddev of Gaussian exploration noise on policy logit")
    parser.add_argument("--noise-decay", type=float, default=0.999, help="Multiplicative decay of noise per step")
    parser.add_argument("--noise-min", type=float, default=0.02, help="Minimum noise stddev")
    parser.add_argument("--model-path", default="", help="Path to save/load policy checkpoint (JSON)")
    parser.add_argument("--autosave-secs", type=float, default=60.0, help="Autosave interval seconds (0 disables)")
    parser.add_argument("--reward-scale", type=float, default=1000.0, help="Multiply rewards by this factor for learning stability")
    parser.add_argument("--baseline-beta", type=float, default=0.99, help="EMA baseline for advantage: b<-beta*b+(1-beta)*r")
    parser.add_argument("--rebate-on-fill", action="store_true", help="Offset spread hit by adding half-spread cost fraction to reward on fills")
    args = parser.parse_args()

    account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
    access_token = os.environ.get("OANDA_DEMO_KEY")
    if not account_id or not access_token:
        raise RuntimeError("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY.")

    api = API(access_token=access_token, environment=args.environment)

    config = BanditConfig(
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        alpha=args.alpha,
    )

    # Initialize policy with feature dimension (computed below on first pass)
    discrete_policy: Optional[ContextualSoftmaxPolicy] = None  # kept for reference if needed
    policy: Optional[ContinuousLinearPolicy] = None

    last_candle_time: Optional[str] = None
    last_nav: Optional[float] = None
    last_action: int = 0
    # Track for one-step delayed updates
    last_features: Optional[np.ndarray] = None
    last_action_value: Optional[float] = None
    prev_pos_dir: int = 0
    last_order_time: float = 0.0
    smoothed_target: float = 0.0
    noise_sigma: float = args.noise_sigma
    # Note: no heartbeat-driven logging; we print on each PRICE tick
    # Instruments list
    instruments_list = [s.strip() for s in (args.instruments or "").split(",") if s.strip()]
    if not instruments_list:
        raise RuntimeError("No instruments provided")

    # Model persistence
    if args.model_path:
        model_path = args.model_path
    else:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, f"online_trader_{'-'.join(instruments_list)}_TICK.json")
    last_save_ts: float = 0.0
    # Reward baseline
    baseline_r: float = 0.0

    # Warm-up NAV and position
    last_nav = fetch_nav(api, account_id)
    nav_estimate = last_nav or 1.0
    last_nav_update_time = time.time()
    # Per-instrument state
    from collections import deque
    class InstState:
        def __init__(self) -> None:
            self.mid_window = deque(maxlen=args.feature_ticks)
            self.last_mid: Optional[float] = None
            self.current_units: int = 0
            self.last_features: Optional[np.ndarray] = None
            self.last_action_value: Optional[float] = None
            self.smoothed_target: float = 0.0
            self.last_order_time: float = 0.0
            self.policy: Optional[ContinuousLinearPolicy] = None

    states: Dict[str, InstState] = {inst: InstState() for inst in instruments_list}

    # Initial positions refresh for all instruments
    def refresh_all_positions() -> None:
        try:
            resp = api.request(positions.OpenPositions(accountID=account_id))
            by_inst = {p.get("instrument"): p for p in resp.get("positions", [])}
            for inst in instruments_list:
                p = by_inst.get(inst) or {}
                long_units = float((p.get("long") or {}).get("units") or 0.0)
                short_units = float((p.get("short") or {}).get("units") or 0.0)
                states[inst].current_units = int(round(long_units + short_units))
        except Exception:
            pass

    refresh_all_positions()
    last_pos_refresh_time = time.time()

    # Rolling series for features
    # Rolling series contained per instrument in states

    # Stream setup
    stream_request = pricing.PricingStream(accountID=account_id, params={"instruments": ",".join(instruments_list)})

    while True:
        try:
            for msg in api.request(stream_request):
                msg_type = msg.get("type")
                if msg_type == "HEARTBEAT":
                    # Periodic NAV/position refresh on heartbeats
                    now_wall = time.time()
                    if (now_wall - last_nav_update_time) >= args.nav_poll_secs:
                        nav_now = fetch_nav(api, account_id)
                        if nav_now is not None and nav_now > 0:
                            nav_estimate = nav_now
                            last_nav = nav_now
                        last_nav_update_time = now_wall
                    if (now_wall - last_pos_refresh_time) >= args.pos_refresh_secs:
                        refresh_all_positions()
                        last_pos_refresh_time = now_wall
                    # Print heartbeat so user can see liveness
                    hb = {
                        "time": msg.get("time"),
                        "type": "HEARTBEAT",
                        "price": None,
                        "nav": nav_estimate,
                        "current_units": {inst: states[inst].current_units for inst in instruments_list},
                    }
                    print(json.dumps(hb, default=str), flush=True)
                    continue

                if msg_type != "PRICE":
                    continue

                instrument = msg.get("instrument")
                if instrument not in states:
                    # Debug: instrument mismatch
                    try:
                        print(json.dumps({"type": "PRICE_OTHER", "instrument": instrument, "want": instruments_list, "time": msg.get("time")}), flush=True)
                    except Exception:
                        pass
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
                if bid is not None and ask is not None:
                    mid = (bid + ask) / 2.0
                else:
                    mid = bid if bid is not None else ask  # best effort

                ts = msg.get("time")
                # Print raw tick for visibility
                try:
                    print(json.dumps({"type": "PRICE", "time": ts, "instrument": instrument, "bid": bid, "ask": ask, "mid": mid}), flush=True)
                except Exception:
                    pass
                st = states[instrument]
                st.mid_window.append(mid)

                # NAV refresh by time
                now_wall = time.time()
                if (now_wall - last_nav_update_time) >= args.nav_poll_secs:
                    nav_now = fetch_nav(api, account_id)
                    if nav_now is not None and nav_now > 0:
                        nav_estimate = nav_now
                        last_nav = nav_now
                    last_nav_update_time = now_wall
                if (now_wall - last_pos_refresh_time) >= args.pos_refresh_secs:
                    refresh_all_positions()
                    last_pos_refresh_time = now_wall

                # Compute pseudo reward from price change and current units
                raw_step_reward = 0.0
                if st.last_mid is not None and nav_estimate > 0:
                    price_change = mid - st.last_mid
                    raw_step_reward = (st.current_units * price_change) / nav_estimate
                # Immediate step log for visibility
                try:
                    print(json.dumps({
                        "type": "STEP",
                        "time": ts,
                        "mid": mid,
                        "instrument": instrument,
                        "units": st.current_units,
                        "price_change": (mid - st.last_mid) if st.last_mid is not None else None,
                        "raw_reward": raw_step_reward,
                        "scaled_reward": raw_step_reward * args.reward_scale
                    }), flush=True)
                except Exception:
                    pass

                # Build features
                features = None
                if len(st.mid_window) >= 60:
                    features = compute_features_from_mid_series(list(st.mid_window))
                    if features is not None and st.policy is None:
                        st.policy = ContinuousLinearPolicy(num_features=features.shape[0], learning_rate=args.lr)
                        # Attempt to load checkpoint
                        ckpt = load_policy(model_path)
                        if ckpt:
                            try:
                                weights_map = ckpt.get("weights_by_instrument") if isinstance(ckpt.get("weights_by_instrument"), dict) else None
                                loaded_weights = None
                                if weights_map and instrument in weights_map:
                                    loaded_weights = weights_map[instrument]
                                elif isinstance(ckpt.get("weights"), list) and instrument in instruments_list:
                                    # Back-compat: single-head
                                    loaded_weights = ckpt.get("weights")
                                if isinstance(loaded_weights, list) and len(loaded_weights) == features.shape[0]:
                                    st.policy.weights = np.array(loaded_weights, dtype=float)
                                # else ignore mismatched/empty
                                if isinstance(ckpt.get("noise_sigma"), (int, float)):
                                    noise_sigma = float(ckpt.get("noise_sigma"))
                            except Exception:
                                pass

                # Policy action
                action_value: float = 0.0
                if st.policy is not None and features is not None:
                    # Ensure weight dimension matches features
                    if st.policy.weights.shape[0] != features.shape[0]:
                        st.policy.weights = np.zeros(features.shape[0], dtype=float)
                    z = float(st.policy.weights @ features)
                    if noise_sigma > 0.0:
                        z += float(np.random.normal(0.0, noise_sigma))
                    action_value = float(np.tanh(z))
                else:
                    # Exploratory action during warm-up before policy/features ready
                    if noise_sigma > 0.0:
                        action_value = float(np.tanh(np.random.normal(0.0, noise_sigma)))

                # Smooth and compute delta units
                st.smoothed_target = (args.smoothing * st.smoothed_target) + ((1.0 - args.smoothing) * action_value)
                target_units = int(round(st.smoothed_target * args.max_units))
                delta_units = target_units - st.current_units

                # Possibly place order
                order_result: Optional[Dict[str, object]] = None
                if abs(delta_units) >= args.deadband_units and (now_wall - st.last_order_time) >= args.order_cooldown:
                    order_result = place_units_delta_order(
                        api=api,
                        account_id=account_id,
                        instrument=instrument,
                        delta_units=delta_units,
                    )
                    st.last_order_time = now_wall
                    # After order, refresh position on next iteration via pos_refresh
                    try:
                        refresh_all_positions()
                    except Exception:
                        pass
                else:
                    # Debug: why no order
                    try:
                        print(json.dumps({
                            "type": "NO_ORDER",
                            "time": ts,
                            "instrument": instrument,
                            "delta_units": delta_units,
                            "deadband": args.deadband_units,
                            "cooldown_ok": (now_wall - st.last_order_time) >= args.order_cooldown
                        }), flush=True)
                    except Exception:
                        pass

                # ---- Per-tick reward shaping, logging, and learning (INSIDE stream loop) ----
                fill_rebate_frac = 0.0
                if args.rebate_on_fill and isinstance(order_result, dict):
                    resp = order_result.get("response") if isinstance(order_result.get("response"), dict) else None
                    fill = resp.get("orderFillTransaction") if isinstance(resp, dict) and isinstance(resp.get("orderFillTransaction"), dict) else None
                    if isinstance(fill, dict):
                        try:
                            half_spread_cost = float(fill.get("halfSpreadCost", 0.0))
                            balance = float(fill.get("accountBalance", nav_estimate or 0.0))
                            if balance > 0.0 and half_spread_cost > 0.0:
                                fill_rebate_frac = half_spread_cost / balance
                        except Exception:
                            fill_rebate_frac = 0.0

                shaped_reward = (raw_step_reward + fill_rebate_frac) * args.reward_scale

                if args.log_debug:
                    log = {
                        "time": ts,
                        "instrument": instrument,
                        "price": mid,
                        "nav": nav_estimate,
                        "reward": shaped_reward,
                        "raw_reward": raw_step_reward,
                        "action_value": action_value,
                        "smoothed_target": st.smoothed_target,
                        "current_units": st.current_units,
                        "target_units": target_units,
                        "delta_units": delta_units,
                        "noise_sigma": noise_sigma,
                        "order": summarize_order_result(order_result),
                    }
                else:
                    log = {
                        "time": ts,
                        "instrument": instrument,
                        "price": mid,
                        "nav": nav_estimate,
                        "reward": shaped_reward,
                        "action_value": round(action_value, 4),
                        "target_units": target_units,
                        "current_units": st.current_units,
                        "delta_units": delta_units,
                        "order": summarize_order_result(order_result) if order_result is not None else None,
                    }
                print(json.dumps(log, default=str), flush=True)

                # Learning update
                if st.last_features is not None and st.last_action_value is not None and st.policy is not None:
                    baseline_r = (args.baseline_beta * baseline_r) + ((1.0 - args.baseline_beta) * shaped_reward)
                    advantage = shaped_reward - baseline_r
                    st.policy.update(st.last_features, st.last_action_value, advantage)

                # Advance state
                st.last_mid = mid
                if features is not None:
                    st.last_features = features
                    st.last_action_value = action_value
                noise_sigma = max(args.noise_min, noise_sigma * args.noise_decay)
                if args.autosave_secs > 0 and (now_wall - last_save_ts) >= args.autosave_secs:
                    # Save per-instrument weights
                    try:
                        weights_by_instrument = {
                            inst: (states[inst].policy.weights.tolist() if states[inst].policy is not None else [])
                            for inst in instruments_list
                        }
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "weights_by_instrument": weights_by_instrument,
                                "noise_sigma": float(noise_sigma),
                            }, f)
                    except Exception:
                        pass
                    last_save_ts = now_wall
                # ---- end per-tick block ----

            # Remove stale single-instrument logging block

            # Remove stale single-instrument tail block

        except KeyboardInterrupt:
            print("Interrupted. Exiting.", flush=True)
            # Best-effort save on exit
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
