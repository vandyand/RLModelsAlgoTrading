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

# Reuse existing helpers for TP/SL order placement
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


def place_action_order(
    api: API,
    account_id: str,
    instrument: str,
    action: int,
    units_abs: int,
    tp_pips: Optional[float],
    sl_pips: Optional[float],
) -> Optional[Dict[str, object]]:
    if action == 0:
        return None
    units = units_abs if action > 0 else -units_abs
    # Use default anchoring (ask for buy, bid for sell) by passing None,
    # or you can switch to "mid" if desired.
    anchor: Optional[str] = None
    try:
        result = place_market_order(
            api=api,
            account_id=account_id,
            instrument=instrument,
            units=units,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            anchor=anchor,
            client_tag="online-rl",
            client_comment="online bandit order",
            fifo_safe=False,
            fifo_adjust=False,
        )
        return result
    except Exception as exc:
        return {"error": str(exc)}


ACTION_TO_INDEX: Dict[int, int] = {-1: 0, 0: 1, 1: 2}


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
    parser = argparse.ArgumentParser(description="Online tri-state bandit trader for OANDA (EUR_USD M1)")
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--granularity", default="S5", help="Candle granularity, e.g., S5|S10|S15|S30|M1|M5...")
    parser.add_argument("--units", type=int, default=1000, help="Trade size in units (absolute value)")
    parser.add_argument("--tp-pips", type=float, default=3.0)
    parser.add_argument("--sl-pips", type=float, default=3.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for contextual policy")
    parser.add_argument("--feature-window", type=int, default=120, help="Number of recent candles to compute features")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval for new M1 candle")
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

    # Initialize contextual policy with feature dimension (computed below on first pass)
    policy: Optional[ContextualSoftmaxPolicy] = None

    last_candle_time: Optional[str] = None
    last_nav: Optional[float] = None
    last_action: int = 0
    # Episode tracking: attribute all PnL during a position to the entry action
    holding_action: Optional[int] = None
    entry_nav: Optional[float] = None
    entry_features: Optional[np.ndarray] = None
    prev_pos_dir: int = 0

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
                time.sleep(max(1.0, args.poll_seconds))
                continue

            # Compute reward from NAV change over last minute
            nav_now = fetch_nav(api, account_id)
            step_reward = 0.0
            if last_nav is not None and nav_now is not None and last_nav > 0:
                step_reward = (nav_now - last_nav) / last_nav

            # Position state
            pos_dir = get_open_position_direction(api, account_id, args.instrument)
            flat = (pos_dir == 0)

            # Fetch recent candles for features
            recent_candles = list(adapter.fetch(count=max(61, args.feature_window)))
            features = compute_features_from_candles(recent_candles) if recent_candles else None
            if features is not None and policy is None:
                policy = ContextualSoftmaxPolicy(num_features=features.shape[0], learning_rate=args.lr)

            # If a held position has just closed, credit cumulative episode return to the entry action
            order_result: Optional[Dict[str, object]] = None
            if (
                holding_action is not None
                and flat
                and prev_pos_dir != 0
                and entry_nav is not None
                and nav_now is not None
                and entry_nav > 0
                and entry_features is not None
                and policy is not None
            ):
                episode_return = (nav_now - entry_nav) / entry_nav
                policy.update_on_episode(entry_features, holding_action, episode_return)
                holding_action = None
                entry_nav = None
                entry_features = None

            # Choose next action (policy requires features); when holding a position, output 0 by design
            action: int = 0
            probs: Optional[np.ndarray] = None
            if flat and features is not None and policy is not None:
                action, probs = policy.select_action(features, config.epsilon)

            # If flat and action proposes entering, place order and start a new episode
            if flat and action in (-1, 1):
                order_result = place_action_order(
                    api=api,
                    account_id=account_id,
                    instrument=args.instrument,
                    action=action,
                    units_abs=args.units,
                    tp_pips=args.tp_pips,
                    sl_pips=args.sl_pips,
                )
                # On entry, mark episode start using current NAV
                if nav_now is not None:
                    holding_action = action
                    entry_nav = nav_now
                    entry_features = features

            # Log step summary
            log = {
                "time": ts,
                "instrument": args.instrument,
                "price": latest["close"],
                "nav": nav_now,
                "reward": step_reward,
                "action": action,
                "last_action": last_action,
                "position_dir": pos_dir,
                "epsilon": config.epsilon,
                "probs": probs.tolist() if probs is not None else None,
                "holding_action": holding_action,
                "order_result": order_result,
            }
            print(json.dumps(log, default=str))

            # Advance
            last_action = action
            last_nav = nav_now
            last_candle_time = ts
            prev_pos_dir = pos_dir

            # Decay exploration rate
            config.epsilon = max(
                config.min_epsilon,
                config.epsilon * config.epsilon_decay,
            )

        except KeyboardInterrupt:
            print("Interrupted. Exiting.")
            break
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
