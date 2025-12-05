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
FOREX_DIR = os.path.dirname(__file__)
if FOREX_DIR not in sys.path:
    sys.path.append(FOREX_DIR)

# Reuse existing helper for market order placement (TP/SL optional)
from streamer.orders import place_market_order  # type: ignore
try:
    # Local paper broker IPC client (tall-borker)
    import broker_ipc  # type: ignore
except Exception:
    broker_ipc = None  # type: ignore
try:
    # Forex FRB consumer (shared-memory live market data)
    import frb_feed  # type: ignore
except Exception:
    frb_feed = None  # type: ignore


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


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class MultiHeadBoxPolicy:
    """Six-head linear policy for box trading per instrument.

    Heads (logits -> bounded outputs):
      0: direction in [-1, 1] via tanh
      1: magnitude in (0, 1) via sigmoid
      2: tp pips via sigmoid scaled to [min_tp, max_tp]
      3: sl pips via sigmoid scaled to [min_sl, max_sl]
      4: max time seconds via sigmoid scaled to [min_time, max_time]
      5: enter probability in (0, 1) via sigmoid (boolean gate)
    """

    def __init__(self, num_features: int, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        # weights shape (6, num_features)
        self.weights = np.zeros((6, num_features), dtype=float)

    def act(self, features: np.ndarray,
            min_tp: float, max_tp: float,
            min_sl: float, max_sl: float,
            min_time: float, max_time: float,
            noise_sigma: float = 0.0) -> Dict[str, float]:
        z = self.weights @ features  # shape (6,)
        if noise_sigma > 0.0:
            z = z + np.random.normal(0.0, noise_sigma, size=z.shape)
        direction = float(np.tanh(z[0]))
        magnitude = _sigmoid(float(z[1]))
        tp = min_tp + _sigmoid(float(z[2])) * max(0.0, max_tp - min_tp)
        sl = min_sl + _sigmoid(float(z[3])) * max(0.0, max_sl - min_sl)
        max_time = min_time + _sigmoid(float(z[4])) * max(0.0, max_time - min_time)
        enter_prob = _sigmoid(float(z[5]))
        enter_flag = (enter_prob >= 0.5)
        return {
            "direction": direction,
            "magnitude": magnitude,
            "tp_pips": tp,
            "sl_pips": sl,
            "max_time_sec": max_time,
            "enter_prob": enter_prob,
            "enter": enter_flag,
            "z0": float(z[0]),
            "z1": float(z[1]),
            "z2": float(z[2]),
            "z3": float(z[3]),
            "z4": float(z[4]),
            "z5": float(z[5]),
        }

    def update(self, features: np.ndarray, entry_logits: np.ndarray, reward: float,
               min_tp: float, max_tp: float,
               min_sl: float, max_sl: float,
               min_time: float, max_time: float) -> None:
        # Compute derivatives of outputs wrt logits at entry
        dz = np.zeros(6, dtype=float)
        # direction tanh
        a0 = float(np.tanh(entry_logits[0]))
        dz[0] = (1.0 - a0 * a0)
        # magnitude sigmoid
        s1 = _sigmoid(float(entry_logits[1]))
        dz[1] = s1 * (1.0 - s1)
        # tp/sl/time sigmoid scaled
        s2 = _sigmoid(float(entry_logits[2])); scale2 = max(0.0, max_tp - min_tp)
        s3 = _sigmoid(float(entry_logits[3])); scale3 = max(0.0, max_sl - min_sl)
        s4 = _sigmoid(float(entry_logits[4])); scale4 = max(0.0, max_time - min_time)
        dz[2] = s2 * (1.0 - s2) * scale2
        dz[3] = s3 * (1.0 - s3) * scale3
        dz[4] = s4 * (1.0 - s4) * scale4
        # enter head
        s5 = _sigmoid(float(entry_logits[5]))
        dz[5] = s5 * (1.0 - s5)

        # Gradient ascent: w_h <- w_h + lr * reward * dz[h] * x
        for h in range(6):
            self.weights[h, :] += self.learning_rate * reward * dz[h] * features


class SharedBoxPolicy:
    def __init__(self, learning_rate: float = 0.05) -> None:
        self.learning_rate = learning_rate
        self.trunk: Optional[np.ndarray] = None  # shape (H, D) we set H=D
        self.heads: Dict[str, np.ndarray] = {}   # instrument -> (6, H)

    def ensure_head(self, instrument: str, feature_dim: int) -> None:
        if self.trunk is None:
            self.trunk = np.zeros((feature_dim, feature_dim), dtype=float)
        if instrument not in self.heads:
            self.heads[instrument] = np.zeros((6, self.trunk.shape[0]), dtype=float)

    def act(self, instrument: str, features: np.ndarray,
            min_tp: float, max_tp: float,
            min_sl: float, max_sl: float,
            min_time: float, max_time: float,
            noise_sigma: float = 0.0) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        assert self.trunk is not None and instrument in self.heads
        latent = self.trunk @ features  # shape (H,)
        z = self.heads[instrument] @ latent  # shape (6,)
        if noise_sigma > 0.0:
            z = z + np.random.normal(0.0, noise_sigma, size=z.shape)
        # decode heads
        direction = float(np.tanh(z[0]))
        magnitude = _sigmoid(float(z[1]))
        tp = min_tp + _sigmoid(float(z[2])) * max(0.0, max_tp - min_tp)
        sl = min_sl + _sigmoid(float(z[3])) * max(0.0, max_sl - min_sl)
        max_time_s = min_time + _sigmoid(float(z[4])) * max(0.0, max_time - min_time)
        enter_prob = _sigmoid(float(z[5]))
        enter_flag = (enter_prob >= 0.5)
        outs = {
            "direction": direction,
            "magnitude": magnitude,
            "tp_pips": tp,
            "sl_pips": sl,
            "max_time_sec": max_time_s,
            "enter_prob": enter_prob,
            "enter": enter_flag,
        }
        return outs, z, latent

    def update(self, instrument: str, features: np.ndarray, entry_logits: np.ndarray, entry_latent: np.ndarray, reward: float,
               min_tp: float, max_tp: float, min_sl: float, max_sl: float, min_time: float, max_time: float) -> None:
        assert self.trunk is not None and instrument in self.heads
        head = self.heads[instrument]
        # compute dz like MultiHeadBoxPolicy
        dz = np.zeros(6, dtype=float)
        a0 = float(np.tanh(entry_logits[0])); dz[0] = (1.0 - a0 * a0)
        s1 = _sigmoid(float(entry_logits[1])); dz[1] = s1 * (1.0 - s1)
        s2 = _sigmoid(float(entry_logits[2])); dz[2] = s2 * (1.0 - s2) * max(0.0, max_tp - min_tp)
        s3 = _sigmoid(float(entry_logits[3])); dz[3] = s3 * (1.0 - s3) * max(0.0, max_sl - min_sl)
        s4 = _sigmoid(float(entry_logits[4])); dz[4] = s4 * (1.0 - s4) * max(0.0, max_time - min_time)
        s5 = _sigmoid(float(entry_logits[5])); dz[5] = s5 * (1.0 - s5)
        # update head and trunk
        for h in range(6):
            head[h, :] += self.learning_rate * reward * dz[h] * entry_latent
            # trunk gradient: outer(head[h,:], features)
            self.trunk[:, :] += self.learning_rate * reward * dz[h] * np.outer(head[h, :], features)

    def save(self, path: str, noise_sigma: float, instruments: List[str]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data: Dict[str, Any] = {"noise_sigma": float(noise_sigma)}
            if self.trunk is not None:
                data["shared_trunk"] = self.trunk.tolist()
            data["heads_by_instrument"] = {
                inst: (self.heads[inst].tolist() if inst in self.heads else []) for inst in instruments
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load(self, path: str, feature_dim: int) -> Optional[float]:
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            trunk = data.get("shared_trunk")
            if isinstance(trunk, list):
                arr = np.array(trunk, dtype=float)
                if arr.ndim == 2 and arr.shape == (feature_dim, feature_dim):
                    self.trunk = arr
            heads = data.get("heads_by_instrument")
            if isinstance(heads, dict):
                for inst, mat in heads.items():
                    try:
                        arr = np.array(mat, dtype=float)
                        if arr.ndim == 2 and arr.shape == (6, feature_dim):
                            self.heads[inst] = arr
                    except Exception:
                        continue
            noise_sigma = data.get("noise_sigma")
            return float(noise_sigma) if isinstance(noise_sigma, (int, float)) else None
        except Exception:
            return None

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
    parser.add_argument("--broker", choices=["oanda", "ipc"], default="oanda", help="Execution broker: OANDA API or local IPC")
    parser.add_argument("--broker-account-id", type=int, default=int(os.environ.get("BROKER_ACCOUNT_ID", "1")), help="Local broker account id (IPC)")
    parser.add_argument("--ipc-socket", default=os.environ.get("PRAGMAGEN_IPC_SOCKET", "/run/pragmagen/pragmagen.sock"), help="Unix socket path for local broker")
    parser.add_argument("--market-data", choices=["oanda", "frb"], default=None, help="Market data source; defaults to broker-appropriate")
    parser.add_argument("--frb-path", default=os.environ.get("FRB_PATH", "/dev/shm/market_data"), help="FRB shared-memory path for forex data")
    parser.add_argument("--max-units", type=int, default=1000, help="Max absolute position units per instrument")
    parser.add_argument("--min-units", type=int, default=5, help="Minimum absolute units to place a box trade")
    parser.add_argument("--order-cooldown", type=float, default=5.0, help="Minimum seconds between orders per instrument")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for policy")
    parser.add_argument("--feature-ticks", type=int, default=240, help="Number of recent mid ticks to compute features")
    # Box parameters
    parser.add_argument("--min-tp-pips", type=float, default=2.0)
    parser.add_argument("--max-tp-pips", type=float, default=15.0)
    parser.add_argument("--min-sl-pips", type=float, default=2.0)
    parser.add_argument("--max-sl-pips", type=float, default=15.0)
    parser.add_argument("--min-trade-sec", type=float, default=5.0)
    parser.add_argument("--max-trade-sec", type=float, default=900.0)
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
    # Local broker simulation cost controls (optional; default zero for curriculum)
    parser.add_argument("--sim-slippage-bps", type=float, default=float(os.environ.get("SIM_SLIPPAGE_BPS", "0")), help="Simulated slippage in basis points (local broker); overridden by ramp if enabled")
    parser.add_argument("--sim-fee-perc", type=float, default=float(os.environ.get("SIM_FEE_PERC", "0")), help="Simulated percentage fee, e.g., 0.001 for 0.1%")
    parser.add_argument("--sim-fee-fixed", type=float, default=float(os.environ.get("SIM_FEE_FIXED", "0")), help="Simulated fixed fee per order")
    # Slippage ramp
    parser.add_argument("--slip-ramp-start-bps", type=float, default=float(os.environ.get("SLIP_RAMP_START_BPS", "0")), help="Ramp start slippage (bps)")
    parser.add_argument("--slip-ramp-target-bps", type=float, default=float(os.environ.get("SLIP_RAMP_TARGET_BPS", "1")), help="Ramp target slippage (bps)")
    parser.add_argument("--slip-ramp-days", type=float, default=float(os.environ.get("SLIP_RAMP_DAYS", "5")), help="Ramp duration in days")
    parser.add_argument("--slip-ramp-epoch-ts", type=float, default=float(os.environ.get("SLIP_RAMP_EPOCH_TS", "0")), help="Ramp epoch UNIX ts; 0 means now")
    args = parser.parse_args()

    # Decide market data default based on broker if not explicitly set
    if args.market_data is None:
        market_data = "oanda" if args.broker == "oanda" else "frb"
    else:
        market_data = args.market_data

    # Broker setup
    api: Optional[API] = None
    ipc_client: Optional["broker_ipc.BrokerIPCClient"] = None
    account_id: Optional[str] = None
    if args.broker == "oanda":
        account_id = os.environ.get("OANDA_DEMO_ACCOUNT_ID")
        access_token = os.environ.get("OANDA_DEMO_KEY")
        if not account_id or not access_token:
            raise RuntimeError("Missing OANDA credentials. Set OANDA_DEMO_ACCOUNT_ID and OANDA_DEMO_KEY.")
        api = API(access_token=access_token, environment=args.environment)
    else:
        if broker_ipc is None:
            raise RuntimeError("Local broker IPC client unavailable. Ensure forex-rl/broker_ipc.py is importable.")
        ipc_client = broker_ipc.BrokerIPCClient(socket_path=args.ipc_socket)

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
    def fetch_nav_oanda(api: API, account_id: str) -> Optional[float]:
        return fetch_nav(api, account_id)

    def fetch_nav_ipc(client: "broker_ipc.BrokerIPCClient", account_id_int: int) -> Optional[float]:
        try:
            r = client.get_account_derived(account_id_int)
            jd = r.data or {}
            for k in ("equity", "nav", "NAV", "Equity", "NetAssetValue"):
                v = jd.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            # Fallback to base account fields
            r2 = client.get_account(account_id_int)
            jd2 = r2.data or {}
            for k in ("equity", "balance", "NAV"):
                v = jd2.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            pass
        return None

    last_nav = fetch_nav_oanda(api, account_id) if api and account_id else (fetch_nav_ipc(ipc_client, args.broker_account_id) if ipc_client else None)
    nav_estimate = last_nav or 1.0
    last_nav_update_time = time.time()
    # Per-instrument state
    from collections import deque
    class InstState:
        def __init__(self) -> None:
            self.mid_window = deque(maxlen=args.feature_ticks)
            self.last_mid: Optional[float] = None
            self.current_units: int = 0
            self.last_order_time: float = 0.0
            # Box policy and entry state
            self.policy: Optional[MultiHeadBoxPolicy] = None
            self.entry_features: Optional[np.ndarray] = None
            self.entry_logits: Optional[np.ndarray] = None
            self.entry_nav: Optional[float] = None
            self.deadline_ts: Optional[float] = None
            self.open: bool = False

    states: Dict[str, InstState] = {inst: InstState() for inst in instruments_list}

    # Initial positions refresh for all instruments
    def refresh_all_positions() -> None:
        if api and account_id:
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
        elif ipc_client is not None:
            try:
                pr = ipc_client.get_positions(args.broker_account_id)
                plist = pr.data or []
                # Heuristic: each position has {symbol, side, quantity} or {symbol, qty, net}
                # Build net by instrument name as provided in instruments_list (uppercased)
                by_sym = {}
                for pos in plist:
                    sym = str(pos.get("symbol", "")).upper()
                    qty = pos.get("net")
                    if not isinstance(qty, (int, float)):
                        # Try long/short or side/quantity
                        q = 0.0
                        try:
                            long_q = float(pos.get("long") or 0.0)
                            short_q = float(pos.get("short") or 0.0)
                            q = long_q - short_q
                        except Exception:
                            side = str(pos.get("side", "")).lower()
                            quantity = float(pos.get("quantity") or pos.get("qty") or 0.0)
                            q = quantity if side == "buy" else (-quantity if side == "sell" else 0.0)
                        qty = q
                    by_sym[sym] = int(round(float(qty)))
                for inst in instruments_list:
                    states[inst].current_units = int(by_sym.get(inst.upper(), 0))
            except Exception:
                pass

    refresh_all_positions()
    last_pos_refresh_time = time.time()

    # Rolling series for features
    # Rolling series contained per instrument in states

    if market_data == "oanda":
        # Stream setup via OANDA PricingStream
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

                    # STEP log for visibility
                    try:
                        print(json.dumps({
                            "type": "STEP",
                            "time": ts,
                            "mid": mid,
                            "instrument": instrument,
                            "units": st.current_units,
                            "open": st.open,
                        }), flush=True)
                    except Exception:
                        pass

                    # Build features
                    features = None
                    if len(st.mid_window) >= 60:
                        features = compute_features_from_mid_series(list(st.mid_window))
                        if features is not None and st.policy is None:
                            st.policy = MultiHeadBoxPolicy(num_features=features.shape[0], learning_rate=args.lr)
                            # Attempt to load checkpoint
                            ckpt = load_policy(model_path)
                            if ckpt:
                                try:
                                    weights_map = ckpt.get("weights_by_instrument") if isinstance(ckpt.get("weights_by_instrument"), dict) else None
                                    loaded_weights = None
                                    if weights_map and instrument in weights_map:
                                        loaded_weights = weights_map[instrument]
                                    if isinstance(loaded_weights, list):
                                        arr = np.array(loaded_weights, dtype=float)
                                        if arr.ndim == 2 and arr.shape[1] == features.shape[0] and arr.shape[0] == 6:
                                            st.policy.weights = arr
                                    # else ignore mismatched/empty
                                    if isinstance(ckpt.get("noise_sigma"), (int, float)):
                                        noise_sigma = float(ckpt.get("noise_sigma"))
                                except Exception:
                                    pass

                    # Close on deadline
                    if st.open and st.deadline_ts is not None and time.time() >= st.deadline_ts and st.current_units != 0:
                        try:
                            order_result = place_units_delta_order(api, account_id, instrument, delta_units=-st.current_units)
                            print(json.dumps({"type": "FORCE_CLOSE", "instrument": instrument, "time": ts, "units": -st.current_units, "order": summarize_order_result(order_result)}), flush=True)
                        except Exception as _:
                            pass
                        refresh_all_positions()

                    # Detect box close by flat after being open
                    if st.open and st.current_units == 0 and st.entry_nav is not None and nav_estimate > 0:
                        # Episode reward
                        ep_return = (nav_estimate - st.entry_nav) / st.entry_nav
                        shaped_reward = ep_return * args.reward_scale
                        # Update per-instrument head with entry logits
                        if st.entry_features is not None and st.entry_logits is not None and st.policy is not None:
                            st.policy.update(
                                st.entry_features,
                                st.entry_logits,
                                shaped_reward,
                                args.min_tp_pips,
                                args.max_tp_pips,
                                args.min_sl_pips,
                                args.max_sl_pips,
                                args.min_trade_sec,
                                args.max_trade_sec,
                            )
                        print(json.dumps({"type": "BOX_CLOSED", "instrument": instrument, "time": ts, "reward": shaped_reward}), flush=True)
                        # Clear state
                        st.open = False
                        st.entry_features = None
                        st.entry_logits = None
                        st.entry_nav = None
                        st.deadline_ts = None

                    # If flat, consider opening a new box
                    if st.current_units == 0 and features is not None and st.policy is not None and (time.time() - st.last_order_time) >= args.order_cooldown:
                        # Compute box outputs
                        outs = st.policy.act(features,
                                             min_tp=args.min_tp_pips, max_tp=args.max_tp_pips,
                                             min_sl=args.min_sl_pips, max_sl=args.max_sl_pips,
                                             min_time=args.min_trade_sec, max_time=args.max_trade_sec,
                                             noise_sigma=noise_sigma)
                        if not outs.get("enter", False):
                            # Gate is off; skip opening
                            print(json.dumps({"type": "SKIP_ENTER", "instrument": instrument, "time": ts, "enter_prob": outs.get("enter_prob")}), flush=True)
                            st.last_mid = mid
                            continue
                        # Determine units
                        direction = outs["direction"]
                        magnitude = outs["magnitude"]
                        units = int(round(magnitude * args.max_units))
                        units = units if direction >= 0 else -units
                        if abs(units) >= args.min_units:
                            try:
                                order_result = place_market_order(
                                    api=api,
                                    account_id=account_id,
                                    instrument=instrument,
                                    units=units,
                                    tp_pips=outs["tp_pips"],
                                    sl_pips=outs["sl_pips"],
                                    anchor=None,
                                    client_tag="box-rl",
                                    client_comment="box trade",
                                    fifo_safe=False,
                                    fifo_adjust=False,
                                )
                                st.last_order_time = time.time()
                                st.entry_features = features.copy()
                                st.entry_logits = np.array([outs["z0"], outs["z1"], outs["z2"], outs["z3"], outs["z4"], outs["z5"]], dtype=float)
                                st.entry_nav = nav_estimate
                                st.deadline_ts = time.time() + float(outs["max_time_sec"])
                                st.open = True
                                print(json.dumps({"type": "BOX_OPEN", "instrument": instrument, "time": ts, "units": units, "tp_pips": outs["tp_pips"], "sl_pips": outs["sl_pips"], "max_time_sec": outs["max_time_sec"], "order": summarize_order_result(order_result)}), flush=True)
                                # Refresh positions
                                refresh_all_positions()
                            except Exception as exc:
                                print(json.dumps({"error": str(exc)}), flush=True)

                    # Note: delta-units sizing is disabled in box mode. The section below is removed.

                    # Advance last mid
                    st.last_mid = mid
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
    else:
        # FRB-driven market data with local IPC broker
        if frb_feed is None:
            raise RuntimeError("FRB consumer unavailable. Ensure forex-rl/frb_feed.py is importable.")
        if ipc_client is None:
            raise RuntimeError("IPC broker client not initialized.")
        consumer = frb_feed.FRBConsumerFX(path=args.frb_path, start_at_head=True)
        # Slippage ramp init
        from slippage import linear_ramp_bps  # type: ignore
        ramp_epoch = time.time() if float(args.slip_ramp_epoch_ts) == 0.0 else float(args.slip_ramp_epoch_ts)

        # Synchronous streaming over FRB ticker events
        for instrument, fields in consumer.stream(instruments_list):
            try:
                mid = float(fields.get("mid")) if fields and ("mid" in fields) else None
            except Exception:
                mid = None
            if mid is None:
                continue
            ts = time.time()

            try:
                print(json.dumps({"type": "PRICE", "time": ts, "instrument": instrument, "mid": mid}), flush=True)
            except Exception:
                pass

            st = states[instrument]
            st.mid_window.append(mid)

            # Periodic NAV refresh (best-effort)
            now_wall = time.time()
            if (now_wall - last_nav_update_time) >= args.nav_poll_secs:
                nav_now = fetch_nav_ipc(ipc_client, args.broker_account_id)
                if nav_now is not None and nav_now > 0:
                    nav_estimate = nav_now
                    last_nav = nav_now
                last_nav_update_time = now_wall

            # STEP log
            try:
                print(json.dumps({
                    "type": "STEP",
                    "time": ts,
                    "mid": mid,
                    "instrument": instrument,
                    "units": st.current_units,
                    "open": st.open,
                    "nav": nav_estimate,
                }), flush=True)
            except Exception:
                pass

            # Features and policy init
            features = None
            if len(st.mid_window) >= 60:
                features = compute_features_from_mid_series(list(st.mid_window))
                if features is not None and st.policy is None:
                    st.policy = MultiHeadBoxPolicy(num_features=features.shape[0], learning_rate=args.lr)
                    # Attempt to load checkpoint
                    ckpt = load_policy(model_path)
                    if ckpt:
                        try:
                            weights_map = ckpt.get("weights_by_instrument") if isinstance(ckpt.get("weights_by_instrument"), dict) else None
                            loaded_weights = None
                            if weights_map and instrument in weights_map:
                                loaded_weights = weights_map[instrument]
                            if isinstance(loaded_weights, list):
                                arr = np.array(loaded_weights, dtype=float)
                                if arr.ndim == 2 and arr.shape[1] == features.shape[0] and arr.shape[0] == 6:
                                    st.policy.weights = arr
                            if isinstance(ckpt.get("noise_sigma"), (int, float)):
                                noise_sigma = float(ckpt.get("noise_sigma"))
                        except Exception:
                            pass

            # Deadline-based close
            if st.open and st.deadline_ts is not None and time.time() >= st.deadline_ts and st.current_units != 0:
                try:
                    # Market close by placing opposite side order with same absolute qty
                    side = "sell" if st.current_units > 0 else "buy"
                    qty = abs(st.current_units)
                    cur_slip_bps = linear_ramp_bps(
                        start_bps=float(args.slip_ramp_start_bps),
                        target_bps=float(args.slip_ramp_target_bps),
                        start_epoch_ts=ramp_epoch,
                        ramp_days=float(args.slip_ramp_days),
                        now_ts=time.time(),
                    )
                    r = ipc_client.place_order(
                        account_id=args.broker_account_id,
                        symbol=instrument,
                        side=side,
                        quantity=float(qty),
                        order_type="market",
                        limit_price=None,
                        time_in_force="GTC",
                        sim_slippage_bps=float(cur_slip_bps if cur_slip_bps is not None else args.sim_slippage_bps),
                        sim_fee_perc=float(args.sim_fee_perc),
                        sim_fee_fixed=float(args.sim_fee_fixed),
                    )
                    st.current_units = 0
                    print(json.dumps({"type": "FORCE_CLOSE", "instrument": instrument, "time": ts, "units": -qty, "order": r.data}), flush=True)
                except Exception:
                    pass

            # Detect box close by flat
            if st.open and st.current_units == 0 and st.entry_nav is not None and nav_estimate > 0:
                ep_return = (nav_estimate - st.entry_nav) / st.entry_nav
                shaped_reward = ep_return * args.reward_scale
                if st.entry_features is not None and st.entry_logits is not None and st.policy is not None:
                    st.policy.update(
                        st.entry_features,
                        st.entry_logits,
                        shaped_reward,
                        args.min_tp_pips,
                        args.max_tp_pips,
                        args.min_sl_pips,
                        args.max_sl_pips,
                        args.min_trade_sec,
                        args.max_trade_sec,
                    )
                print(json.dumps({"type": "BOX_CLOSED", "instrument": instrument, "time": ts, "reward": shaped_reward}), flush=True)
                st.open = False
                st.entry_features = None
                st.entry_logits = None
                st.entry_nav = None
                st.deadline_ts = None

            # Consider opening
            if st.current_units == 0 and features is not None and st.policy is not None and (time.time() - st.last_order_time) >= args.order_cooldown:
                outs = st.policy.act(
                    features,
                    min_tp=args.min_tp_pips, max_tp=args.max_tp_pips,
                    min_sl=args.min_sl_pips, max_sl=args.max_sl_pips,
                    min_time=args.min_trade_sec, max_time=args.max_trade_sec,
                    noise_sigma=noise_sigma,
                )
                if not outs.get("enter", False):
                    print(json.dumps({"type": "SKIP_ENTER", "instrument": instrument, "time": ts, "enter_prob": outs.get("enter_prob")}), flush=True)
                    st.last_mid = mid
                    noise_sigma = max(args.noise_min, noise_sigma * args.noise_decay)
                    continue
                direction = outs["direction"]
                magnitude = outs["magnitude"]
                units = int(round(magnitude * args.max_units))
                units = units if direction >= 0 else -units
                if abs(units) >= args.min_units:
                    try:
                        side = "buy" if units > 0 else "sell"
                        qty = abs(units)
                        cur_slip_bps = linear_ramp_bps(
                            start_bps=float(args.slip_ramp_start_bps),
                            target_bps=float(args.slip_ramp_target_bps),
                            start_epoch_ts=ramp_epoch,
                            ramp_days=float(args.slip_ramp_days),
                            now_ts=time.time(),
                        )
                        r = ipc_client.place_order(
                            account_id=args.broker_account_id,
                            symbol=instrument,
                            side=side,
                            quantity=float(qty),
                            order_type="market",
                            limit_price=None,
                            time_in_force="GTC",
                            sim_slippage_bps=float(cur_slip_bps if cur_slip_bps is not None else args.sim_slippage_bps),
                            sim_fee_perc=float(args.sim_fee_perc),
                            sim_fee_fixed=float(args.sim_fee_fixed),
                        )
                        st.current_units = units
                        st.last_order_time = time.time()
                        st.entry_features = features.copy()
                        st.entry_logits = np.array([outs["z0"], outs["z1"], outs["z2"], outs["z3"], outs["z4"], outs["z5"]], dtype=float)
                        st.entry_nav = nav_estimate
                        st.deadline_ts = time.time() + float(outs["max_time_sec"])
                        st.open = True
                        print(json.dumps({"type": "BOX_OPEN", "instrument": instrument, "time": ts, "units": units, "tp_pips": outs["tp_pips"], "sl_pips": outs["sl_pips"], "max_time_sec": outs["max_time_sec"], "order": r.data}), flush=True)
                    except Exception as exc:
                        print(json.dumps({"error": str(exc)}), flush=True)

            # Advance
            st.last_mid = mid
            noise_sigma = max(args.noise_min, noise_sigma * args.noise_decay)
            if args.autosave_secs > 0 and (now_wall - last_save_ts) >= args.autosave_secs:
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


if __name__ == "__main__":
    main()
