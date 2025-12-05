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

# Make repo root importable so we can reuse utilities in streamer/
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Reuse existing helper for market order placement (TP/SL optional)
from streamer.orders import place_market_order  # type: ignore
from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore


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


def compute_features_from_candles(candles: List[Dict[str, float]]) -> Optional[np.ndarray]:
    """Compute a fixed feature vector from recent candles.

    candles: list of dicts with keys: timestamp, open, high, low, close, volume
    Returns a feature vector np.ndarray of shape (D,) or None if insufficient data.
    """
    if len(candles) < 61:
        return None
    closes = np.array([c["close"] for c in candles], dtype=float)
    # Per-minute log returns
    log_prices = np.log(closes)
    r = np.diff(log_prices)  # length N-1

    def last_ratio(k: int) -> float:
        if len(closes) <= k:
            return 0.0
        return float(np.log(closes[-1] / closes[-1 - k]))

    # Rolling stats helpers over returns
    def roll_mean_std(window: int) -> Tuple[float, float]:
        if len(r) < window:
            return 0.0, 0.0
        segment = r[-window:]
        return float(np.mean(segment)), float(np.std(segment) + 1e-12)

    # r_1, r_5, r_15 over closes (multi-step log return)
    r1 = last_ratio(1)
    r5 = last_ratio(5)
    r15 = last_ratio(15)

    # Rolling mean and std of 1-min returns over windows
    m5, s5 = roll_mean_std(5)
    m15, s15 = roll_mean_std(15)
    m60, s60 = roll_mean_std(60)

    # SMA20 z-score for price
    if len(closes) < 20:
        sma20 = closes[-1]
        price_std20 = 1e-6
    else:
        sma20 = float(np.mean(closes[-20:]))
        price_std20 = float(np.std(closes[-20:]) + 1e-12)
    z_price_sma20 = float((closes[-1] - sma20) / price_std20)

    # Realized volatility (std of returns over last 60)
    realized_vol_60 = s60

    # Assemble features; include bias term
    features = np.array([
        1.0,            # bias
        r1, r5, r15,
        m5, s5,
        m15, s15,
        m60, s60,
        z_price_sma20,
        realized_vol_60,
    ], dtype=float)
    # Clip extreme values
    features = np.clip(features, -5.0, 5.0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Online continuous sizing trader for OANDA (EUR_USD)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--granularity", default="S5", help="Candle granularity, e.g., S5|S10|S15|S30|M1|M5...")
    parser.add_argument("--max-units", type=int, default=1000, help="Max absolute position units for target sizing")
    parser.add_argument("--deadband-units", type=int, default=10, help="Skip orders when adjustment is below this threshold")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing for target sizing (0..1; higher=stickier)")
    parser.add_argument("--order-cooldown", type=float, default=2.0, help="Minimum seconds between orders")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for continuous policy")
    parser.add_argument("--feature-window", type=int, default=120, help="Number of recent candles to compute features")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval for new M1 candle")
    parser.add_argument("--log-heartbeat-secs", type=float, default=60.0, help="Print a compact heartbeat at this interval when no orders (set 0 to log every step)")
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

    # Candle source (M1)
    adapter = OandaRestCandlesAdapter(
        instrument=args.instrument,
        granularity=args.granularity,
        environment=args.environment,
        access_token=access_token,
    )

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
    last_heartbeat_ts: float = 0.0
    # Model persistence
    if args.model_path:
        model_path = args.model_path
    else:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(models_dir, f"online_trader_{args.instrument}_{args.granularity}.json")
    last_save_ts: float = 0.0
    # Reward baseline
    baseline_r: float = 0.0

    # Warm-up NAV
    last_nav = fetch_nav(api, account_id)

    while True:
        try:
            # Fetch last complete candle; detect new minute by timestamp change
            recent = list(adapter.fetch(count=2))
            if not recent:
                time.sleep(max(1.0, args.poll_seconds))
                continue
            latest = recent[-1]
            ts = latest["timestamp"]
            if ts == last_candle_time:
                # Optional per-step heartbeat even when no new candle
                if args.log_heartbeat_secs <= 0:
                    hb = {
                        "time": ts,
                        "price": latest["close"],
                        "nav": fetch_nav(api, account_id),
                        "reward": 0.0,
                        "action_value": None,
                        "target_units": None,
                        "current_units": get_net_units(api, account_id, args.instrument),
                    }
                    print(json.dumps(hb, default=str))
                time.sleep(max(1.0, args.poll_seconds))
                continue

            # Compute raw reward from NAV change over last step
            nav_now = fetch_nav(api, account_id)
            raw_step_reward = 0.0
            if last_nav is not None and nav_now is not None and last_nav > 0:
                raw_step_reward = (nav_now - last_nav) / last_nav

            # Position state
            pos_dir = get_open_position_direction(api, account_id, args.instrument)
            flat = (pos_dir == 0)

            # Fetch recent candles for features
            recent_candles = list(adapter.fetch(count=max(61, args.feature_window)))
            features = compute_features_from_candles(recent_candles) if recent_candles else None
            if features is not None and policy is None:
                policy = ContinuousLinearPolicy(num_features=features.shape[0], learning_rate=args.lr)
                # Attempt to load checkpoint
                ckpt = load_policy(model_path)
                if ckpt and isinstance(ckpt.get("weights"), list):
                    weights_list = ckpt.get("weights")
                    if isinstance(weights_list, list) and len(weights_list) == features.shape[0]:
                        try:
                            policy.weights = np.array(weights_list, dtype=float)
                            if isinstance(ckpt.get("noise_sigma"), (int, float)):
                                noise_sigma = float(ckpt.get("noise_sigma"))
                        except Exception:
                            pass

            # Policy action: continuous target in [-1, 1]
            action_value: float = 0.0
            if features is not None and policy is not None:
                z = float(policy.weights @ features)
                if noise_sigma > 0.0:
                    z += float(np.random.normal(0.0, noise_sigma))
                action_value = float(np.tanh(z))

            # Smooth the target to reduce churn
            smoothed_target = (args.smoothing * smoothed_target) + ((1.0 - args.smoothing) * action_value)

            # Compute desired units and delta
            target_units = int(round(smoothed_target * args.max_units))
            current_units = get_net_units(api, account_id, args.instrument)
            delta_units = target_units - current_units

            order_result: Optional[Dict[str, object]] = None
            now_ts = time.time()
            if abs(delta_units) >= args.deadband_units and (now_ts - last_order_time) >= args.order_cooldown:
                order_result = place_units_delta_order(
                    api=api,
                    account_id=account_id,
                    instrument=args.instrument,
                    delta_units=delta_units,
                )
                last_order_time = now_ts

            # Optional rebate: add back fraction of half-spread cost on fills to neutralize immediate negative NAV on entry
            fill_rebate_frac = 0.0
            if args.rebate_on_fill and isinstance(order_result, dict):
                resp = order_result.get("response") if isinstance(order_result.get("response"), dict) else None
                fill = resp.get("orderFillTransaction") if isinstance(resp, dict) and isinstance(resp.get("orderFillTransaction"), dict) else None
                if isinstance(fill, dict):
                    try:
                        half_spread_cost = float(fill.get("halfSpreadCost", 0.0))
                        balance = float(fill.get("accountBalance", nav_now or 0.0))
                        if balance > 0.0 and half_spread_cost > 0.0:
                            fill_rebate_frac = half_spread_cost / balance
                    except Exception:
                        fill_rebate_frac = 0.0

            shaped_reward = (raw_step_reward + fill_rebate_frac) * args.reward_scale

            # Logging: reduce verbosity
            now_wall = time.time()
            if args.log_debug:
                log = {
                    "time": ts,
                    "instrument": args.instrument,
                    "price": latest["close"],
                    "nav": nav_now,
                    "reward": shaped_reward,
                    "raw_reward": raw_step_reward,
                    "action_value": action_value,
                    "smoothed_target": smoothed_target,
                    "current_units": current_units,
                    "target_units": target_units,
                    "delta_units": delta_units,
                    "position_dir": pos_dir,
                    "noise_sigma": noise_sigma,
                    "order": summarize_order_result(order_result),
                }
                print(json.dumps(log, default=str))
            else:
                if order_result is not None:
                    # Print compact order event
                    log = {
                        "time": ts,
                        "price": latest["close"],
                        "nav": nav_now,
                        "reward": shaped_reward,
                        "target_units": target_units,
                        "delta_units": delta_units,
                        "order": summarize_order_result(order_result),
                    }
                    print(json.dumps(log, default=str))
                elif (args.log_heartbeat_secs <= 0) or ((now_wall - last_heartbeat_ts) >= args.log_heartbeat_secs):
                    # Periodic heartbeat
                    hb = {
                        "time": ts,
                        "price": latest["close"],
                        "nav": nav_now,
                        "reward": shaped_reward,
                        "action_value": round(action_value, 4),
                        "target_units": target_units,
                        "current_units": current_units,
                    }
                    print(json.dumps(hb, default=str))
                    last_heartbeat_ts = now_wall

            # Online policy update using one-step reward and previous features/action (correct temporal credit)
            if last_features is not None and last_action_value is not None and policy is not None:
                # Advantage with EMA baseline
                baseline_r = (args.baseline_beta * baseline_r) + ((1.0 - args.baseline_beta) * shaped_reward)
                advantage = shaped_reward - baseline_r
                policy.update(last_features, last_action_value, advantage)

            # Advance
            last_nav = nav_now
            last_candle_time = ts
            prev_pos_dir = pos_dir
            if features is not None:
                last_features = features
                last_action_value = action_value

            # Decay exploration rate
            config.epsilon = max(
                config.min_epsilon,
                config.epsilon * config.epsilon_decay,
            )

            # Anneal exploration noise
            noise_sigma = max(args.noise_min, noise_sigma * args.noise_decay)

            # Autosave model
            if args.autosave_secs > 0 and (now_wall - last_save_ts) >= args.autosave_secs:
                save_policy(model_path, policy, noise_sigma)
                last_save_ts = now_wall

        except KeyboardInterrupt:
            print("Interrupted. Exiting.")
            # Best-effort save on exit
            try:
                save_policy(model_path, policy, noise_sigma)
            except Exception:
                pass
            break
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
