#!/usr/bin/env python3
"""
Offline Actor-Critic for USD-quoted Alpaca Crypto (via candle cache)
--------------------------------------------------------------------

- Universe: fixed set of Alpaca-listed cryptocurrencies quoted in USD:

    AAVE/USD, AVAX/USD, BAT/USD, BCH/USD, BTC/USD, CRV/USD, DOGE/USD,
    DOT/USD, ETH/USD, GRT/USD, LINK/USD, LTC/USD, SHIB/USD, SUSHI/USD,
    UNI/USD, XRP/USD, XTZ/USD, YFI/USD

- Data source: forex-rl/candle_cache_service.py exposed via HTTP, which
  proxies to Alpaca (stocks & crypto) and OANDA (FX) but normalizes all
  candles into an OANDA-compatible JSON shape.
- Horizon: 1-day step on daily OHLCV data.
- Reward: Sharpe-like portfolio reward based on per-instrument
  contributions: mean(contrib) / std(contrib).
- Actions: continuous in [-1, 1] per instrument, interpreted as
  normalized target nominal position sizes (later scaled in live
  trading by account NAV / risk budget).

Usage example (from forex-rl root):

  # Ensure candle_cache_service is running and has Alpaca crypto access:
  CANDLE_CACHE_BASE_URL=http://127.0.0.1:9100 \\
  python -m actor-critic.crypto_offline_actor_critic \\
      --start 2020-01-01 --end 2025-12-31 \\
      --epochs 8

Notes:
- This script assumes the candle_cache_service is running and has
  access to Alpaca API keys for crypto.
- We only use daily (D) candles and do not mix in FX or equities here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Repo imports: use the existing OANDA REST adapter which can point at
# the local candle cache when CANDLE_CACHE_BASE_URL is set.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore
from wfo.adapters.tcn_scalar_threshold import TCNScalarHead  # type: ignore


DEFAULT_CRYPTO_SYMBOLS: List[str] = [
    "AAVE/USD",
    "AVAX/USD",
    "BAT/USD",
    "BCH/USD",
    "BTC/USD",
    "CRV/USD",
    "DOGE/USD",
    "DOT/USD",
    "ETH/USD",
    "GRT/USD",
    "LINK/USD",
    "LTC/USD",
    "SHIB/USD",
    "SUSHI/USD",
    "UNI/USD",
    "XRP/USD",
    "XTZ/USD",
    "YFI/USD",
]


# ---------- Config ----------


@dataclass
class Config:
    # Universe and data
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_CRYPTO_SYMBOLS))
    environment: str = "practice"
    # Candle cache base URL (OANDA-compatible HTTP interface)
    candle_cache_base_url: str = "http://127.0.0.1:9100"
    # Date range (inclusive)
    start: str = "2019-01-01"
    end: Optional[str] = None  # inclusive end; if None, use today-1

    # Model
    ae_hidden: List[int] = None  # retained for CLI compatibility but unused
    ae_latent_dim: int = 64  # retained for CLI compatibility but unused
    # MLP architectures: lists of hidden sizes per layer.
    policy_hidden: List[int] = None
    value_hidden: List[int] = None

    # Trading semantics
    # max_notional is a scale factor applied to actions in [-1, 1];
    # in live trading this will be mapped to actual dollar notionals.
    max_notional: float = 1.0
    action_mapping: str = "continuous"  # we default to long-only continuous sizing

    # Training
    batch_size: int = 128
    ae_epochs: int = 5  # retained for CLI compatibility but unused
    epochs: int = 6
    gamma: float = 0.99
    actor_sigma: float = 0.3
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    adv_clip: float = 5.0
    reward_scale: float = 1.0

    # TCN auxiliary model (next-bar movement predictor); trained separately.
    tcn_lookback: int = 64

    # Misc
    seed: int = 42
    autosave_secs: float = 120.0
    model_path: str = "forex-rl/actor-critic/checkpoints/crypto_offline_ac.pt"

    # Reward choice (see equities_offline_actor_critic.Config for details).
    sharpe_like: bool = False

    def __post_init__(self) -> None:
        if self.ae_hidden is None:
            # No autoencoder is used, but keep a default for any downstream
            # tooling that might still inspect this field.
            self.ae_hidden = [2048, 512, 128]
        if self.policy_hidden is None:
            self.policy_hidden = [256, 256]
        if self.value_hidden is None:
            self.value_hidden = [256, 256]


# ---------- Time features ----------


def time_cyclical_features_from_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return cyclical time features for each date in index.

    For daily data we only keep:
      - day-of-week, day-of-month, month-of-year (sin/cos for each).
    Minute/hour encodings are dropped as they carry no information on D bars.
    """
    idx = pd.to_datetime(index, utc=True).tz_convert("UTC")

    dow = idx.weekday.values
    dom = idx.day.values - 1
    moy = idx.month.values - 1

    def sc(vals: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
        ang = 2.0 * np.pi * (vals.astype(float) / period)
        return np.sin(ang), np.cos(ang)

    dw_s, dw_c = sc(dow, 7.0)
    dm_s, dm_c = sc(dom, 31.0)
    my_s, my_c = sc(moy, 12.0)

    df = pd.DataFrame(
        {
            "time_dow_sin": dw_s,
            "time_dow_cos": dw_c,
            "time_dom_sin": dm_s,
            "time_dom_cos": dm_c,
            "time_moy_sin": my_s,
            "time_moy_cos": my_c,
        },
        index=index,
    )
    return df.astype(np.float32)


# ---------- OHLCV feature engineering ----------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean() + eps
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ohlcv_features(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute a 16-dim feature set per row from OHLCV DataFrame with columns
    ['open','high','low','close','volume'].
    """
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get("close", "close")].astype(float)
    high = df[cols.get("high", "high")].astype(float)
    low = df[cols.get("low", "low")].astype(float)
    vol = df[cols.get("volume", "volume")].astype(float)

    logc = np.log(np.maximum(close, eps))
    r = logc.diff()

    def last_ret(k: int) -> pd.Series:
        return logc - logc.shift(k)

    def std_ret(k: int) -> pd.Series:
        return r.rolling(k).std().fillna(0.0)

    # ATR(20)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = tr.rolling(20).mean().fillna(0.0)

    # Breakout distances
    roll_max20 = close.rolling(20).max()
    roll_min20 = close.rolling(20).min()
    roll_std20 = close.rolling(20).std().fillna(1e-6)
    dist_max20 = (close - roll_max20) / (roll_std20 + eps)
    dist_min20 = (close - roll_min20) / (roll_std20 + eps)

    # Volume z-score
    v_mean20 = vol.rolling(20).mean()
    v_std20 = vol.rolling(20).std().fillna(1e-6)
    v20 = (vol - v_mean20) / (v_std20 + eps)

    # MACD on closes (EMA 12/26, signal 9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal

    rsi14 = _rsi(close, 14, eps)
    vol_of_vol = r.abs().rolling(20).std().fillna(0.0)

    # OBV (tick-direction based using daily closes)
    direction = close.diff()
    obv_step = np.where(direction > 0, vol, np.where(direction < 0, -vol, 0.0))
    obv = pd.Series(obv_step, index=close.index).cumsum()
    obv_mean20 = obv.rolling(20).mean()
    obv_std20 = obv.rolling(20).std().fillna(1e-6)
    obv_z = (obv - obv_mean20) / (obv_std20 + eps)

    feats = pd.DataFrame(
        {
            "ret_1": last_ret(1),
            "ret_5": last_ret(5),
            "ret_20": last_ret(20),
            "std_5": std_ret(5),
            "std_20": std_ret(20),
            "std_60": std_ret(60),
            "atr20": atr20,
            "dist_max20": dist_max20,
            "dist_min20": dist_min20,
            "vol_z20": v20,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
            "rsi14_norm": (rsi14 / 100.0),
            "vol_of_vol20": vol_of_vol,
            "obv_z20": obv_z,
        },
        index=df.index,
    )
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats.astype(np.float32)


# ---------- Data loading ----------


def fetch_crypto_ohlcv(
    symbol: str,
    granularity: str,
    start: str,
    end: Optional[str],
    environment: str,
) -> pd.DataFrame:
    """Fetch OHLCV for a crypto symbol via the candle cache / OANDA adapter."""
    adapter = OandaRestCandlesAdapter(
        instrument=symbol,
        granularity=granularity,
        environment=environment,
        access_token=None,  # hitting candle cache; upstream keys live there
    )
    # Inclusive end date
    if end is None:
        from datetime import datetime, timezone, timedelta

        end = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

    rows: List[Dict[str, Any]] = list(
        adapter.fetch_range(
            from_time=f"{start}T00:00:00Z",
            to_time=f"{end}T23:59:59Z",
            batch=5000,
        )
    )
    if not rows:
        raise RuntimeError(f"No candles fetched for {symbol} {granularity} between {start} and {end}")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    # Normalize to date index for daily data
    if granularity == "D":
        df.index = df.index.normalize()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def build_crypto_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Return (X_features, R_crypto_returns, dates_index) for the crypto universe."""
    daily_feats: List[pd.DataFrame] = []
    close_for_returns: Dict[str, pd.Series] = {}

    # Ensure adapter points at candle cache
    os.environ.setdefault("CANDLE_CACHE_BASE_URL", cfg.candle_cache_base_url)

    # Fetch daily series per symbol
    for sym in cfg.symbols:
        try:
            d_df = fetch_crypto_ohlcv(sym, "D", cfg.start, cfg.end, cfg.environment)
        except RuntimeError as e:
            # Skip symbols with no data in the requested window; emit a lightweight log line.
            print(
                json.dumps(
                    {
                        "status": "skip_symbol_no_data",
                        "symbol": sym,
                        "start": cfg.start,
                        "end": cfg.end,
                        "reason": str(e),
                    }
                ),
                flush=True,
            )
            continue
        # Ensure strictly unique daily index to avoid downstream reindex errors
        # when multiple bars map to the same normalized date (e.g. overlapping imports).
        if not d_df.index.is_unique:
            dup_count = int(d_df.index.duplicated().sum())
            print(
                json.dumps(
                    {
                        "status": "crypto_symbol_drop_duplicates",
                        "symbol": sym,
                        "dropped_rows": dup_count,
                    }
                ),
                flush=True,
            )
            d_df = d_df[~d_df.index.duplicated(keep="last")]

        # Per-symbol raw coverage logging (before intersection, after de-dup)
        if not d_df.empty:
            first_ts = d_df.index[0]
            last_ts = d_df.index[-1]
            print(
                json.dumps(
                    {
                        "status": "crypto_symbol_coverage",
                        "symbol": sym,
                        "rows": int(len(d_df)),
                        "start_eff": first_ts.isoformat(),
                        "end_eff": last_ts.isoformat(),
                    }
                ),
                flush=True,
            )
        d_feats = compute_ohlcv_features(d_df).add_prefix(f"CRYPTO_{sym}_D_")
        daily_feats.append(d_feats)
        close_for_returns[sym] = d_df["close"]

    if not daily_feats:
        raise RuntimeError(
            f"No usable symbols with candles for window {cfg.start}..{cfg.end}; "
            f"check crypto symbol list or date range."
        )

    # Align all daily features on intersection of dates
    base_index = daily_feats[0].index
    for f in daily_feats[1:]:
        base_index = base_index.intersection(f.index)

    # Reindex all frames to base_index and concat
    blocks: List[pd.DataFrame] = []
    for frm in daily_feats:
        blocks.append(frm.reindex(base_index).fillna(0.0))

    # Time features
    t_feats = time_cyclical_features_from_index(base_index)
    blocks.append(t_feats)

    X = pd.concat(blocks, axis=1).astype(np.float32)

    # Returns (next-day log returns per symbol)
    r_cols: Dict[str, pd.Series] = {}
    for sym, cls in close_for_returns.items():
        close = cls.reindex(base_index).ffill()
        r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        r_next = r.shift(-1).fillna(0.0)
        r_cols[sym] = r_next
    R = pd.DataFrame(r_cols, index=base_index).astype(np.float32)

    return X, R, base_index


def _build_tcn_features_from_checkpoint(
    X: pd.DataFrame,
    cfg: Config,
    symbols: List[str],
    ckpt_path: str,
    device: torch.device,
) -> pd.DataFrame:
    """Load pre-trained TCN from checkpoint and build per-symbol features.

    The checkpoint is expected to contain at least:
      - 'state_dict': model parameters
      - 'in_channels': input channel count
      - 'tcn_lookback': context window length
      - optional TCN hyperparameters (hidden_channels, kernel_size, num_blocks, dropout)
    """
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"TCN checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels"))
    L = int(ckpt.get("tcn_lookback", cfg.tcn_lookback))
    hidden_channels = int(ckpt.get("hidden_channels", 32))
    kernel_size = int(ckpt.get("kernel_size", 5))
    num_blocks = int(ckpt.get("num_blocks", 3))
    dropout = float(ckpt.get("dropout", 0.1))

    model = TCNScalarHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    idx = X.index
    n = len(idx)
    if n <= L + 2:
        out = pd.DataFrame(index=idx)
        for sym in symbols:
            col_name = f"TCN_{sym}_pred_next_ret"
            out[col_name] = 0.0
        return out.astype(np.float32)

    out_feats: Dict[str, pd.Series] = {}
    for sym in symbols:
        prefix = f"CRYPTO_{sym}_D_"
        feat_cols = [c for c in X.columns if c.startswith(prefix)]
        if not feat_cols:
            continue
        F_sym = X[feat_cols].values.astype(np.float32)
        mean = F_sym.mean(axis=0, keepdims=True)
        std = F_sym.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        F_norm = (F_sym - mean) / std

        preds = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for t in range(L, n - 1):
                x_seq = F_norm[t - L : t, :].T  # (C, L)
                x_t = torch.tensor(x_seq[None, ...], dtype=torch.float32, device=device)
                y_hat = model(x_t).item()
                preds[t] = float(y_hat)
        col_name = f"TCN_{sym}_pred_next_ret"
        out_feats[col_name] = pd.Series(preds, index=idx)

    out_df = pd.DataFrame(out_feats, index=idx)
    out_df = out_df.astype(np.float32).fillna(0.0)
    return out_df


def _build_mlp(input_dim: int, hidden_sizes: List[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


# ---------- Policy / Value ----------


class ActorCriticMulti(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_instruments: int,
        policy_hidden: List[int],
        value_hidden: List[int],
    ) -> None:
        super().__init__()
        self.policy = _build_mlp(input_dim, policy_hidden, num_instruments)
        self.value = _build_mlp(input_dim, value_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # No separate encoder/autoencoder: operate directly on standardized features.
        z = x
        mu = self.policy(z)
        v = self.value(z).squeeze(-1)
        return z, mu, v


# ---------- Training utilities ----------


def standardize_fit(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean())
        s = float(X[c].std())
        if s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


def tensorize(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)


def train_actor_critic(
    X_train: pd.DataFrame,
    R_train: pd.DataFrame,
    X_val: pd.DataFrame,
    R_val: pd.DataFrame,
    cfg: Config,
    device: torch.device,
) -> Dict[str, Any]:
    num_inst = R_train.shape[1]
    model = ActorCriticMulti(
        input_dim=X_train.shape[1],
        num_instruments=num_inst,
        policy_hidden=list(cfg.policy_hidden),
        value_hidden=list(cfg.value_hidden),
    ).to(device)

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    X_tr = tensorize(X_train).to(device)
    R_tr = tensorize(R_train).to(device)
    X_va = tensorize(X_val).to(device)
    R_va = tensorize(R_val).to(device)

    sigma = float(cfg.actor_sigma)
    log_sigma_const = float(np.log(sigma + 1e-8))

    def step_epoch(Xb: torch.Tensor, Rb: torch.Tensor, train: bool) -> Dict[str, float]:
        if train:
            model.train()
        else:
            model.eval()
        T = Xb.size(0)
        # Continuous sizing by default; we still keep a position vector
        # so that we can extend to hysteresis-style mapping later if needed.
        pos_vec = torch.zeros(Rb.size(1), dtype=torch.float32, device=Xb.device)

        total_loss = 0.0
        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        steps = 0

        for t in range(0, T - 1):
            s_t = Xb[t : t + 1]
            s_tp1 = Xb[t + 1 : t + 2]
            r_tp1 = Rb[t + 1]  # next-day per-instrument returns

            _, mu_t, v_t = model(s_t)
            with torch.no_grad():
                _, _, v_tp1 = model(s_tp1)

            # Sample pre-squash action from Normal(mu, sigma)
            eps = torch.randn_like(mu_t)
            pre = mu_t + sigma * eps
            a_t = torch.tanh(pre)  # in (-1, 1), before mapping to notionals

            # Continuous mapping: allow both long and short notionals by scaling
            # tanh outputs directly into [-max_notional, max_notional].
            if (cfg.action_mapping or "continuous") == "continuous":
                pos = a_t[0] * float(cfg.max_notional)
                pos_vec = pos  # track current targets for potential extensions
            else:
                # Threshold hysteresis; mirror multi20 style but on notionals.
                # Here we already use symmetric long/short bands in [-max_notional, max_notional].
                a = a_t[0]
                maxn = float(cfg.max_notional)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxn), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxn), pos_vec)
                pos = pos_vec

            contrib = pos * r_tp1
            mean_c = torch.mean(contrib)
            std_c = torch.std(contrib)
            downside = torch.clamp(contrib, max=0.0)
            downside_var = torch.mean(downside**2)
            downside_std = torch.sqrt(downside_var + 1e-8)
            if cfg.sharpe_like:
                risk_metric = mean_c / (std_c + 1e-8)
            else:
                risk_metric = mean_c / (downside_std + 1e-8)
            r_scalar = cfg.reward_scale * risk_metric

            adv = (r_scalar + cfg.gamma * v_tp1[0] - v_t[0]).detach()
            if cfg.adv_clip > 0:
                adv = torch.clamp(adv, -cfg.adv_clip, cfg.adv_clip)

            # Log-prob of Normal (pre-squash); ignore tanh correction for simplicity
            logprob = -0.5 * torch.sum(
                ((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const,
                dim=1,
            )
            entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma**2)))

            actor_loss = -adv * logprob.mean()
            value_target = (r_scalar + cfg.gamma * v_tp1[0]).detach()
            value_loss = cfg.value_coef * 0.5 * (value_target - v_t[0]) ** 2
            loss = actor_loss + value_loss - cfg.entropy_coef * entropy

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

            total_loss += float(loss.item())
            total_actor += float(actor_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)
            total_reward += float(r_scalar.item())
            steps += 1

        return {
            "loss": total_loss / max(1, steps),
            "actor": total_actor / max(1, steps),
            "value": total_value / max(1, steps),
            "entropy": total_entropy / max(1, steps),
            "reward": total_reward / max(1, steps),
        }

    for epoch in range(cfg.epochs):
        tr = step_epoch(X_tr, R_tr, train=True)
        va = step_epoch(X_va, R_va, train=False)
        print(
            json.dumps(
                {
                    "phase": "train",
                    "epoch": epoch + 1,
                    **{f"tr_{k}": v for k, v in tr.items()},
                    **{f"va_{k}": v for k, v in va.items()},
                }
            ),
            flush=True,
        )

    # Final greedy validation (no sampling)
    model.eval()
    with torch.no_grad():
        T = X_va.size(0)
        cum = 0.0
        pos_vec = torch.zeros(R_va.size(1), dtype=torch.float32, device=X_va.device)
        for t in range(0, T - 1):
            s_t = X_va[t : t + 1]
            _, mu_t, _ = model(s_t)
            a_t = torch.tanh(mu_t)
            r_tp1 = R_va[t + 1]
            if (cfg.action_mapping or "continuous") == "continuous":
                pos = a_t[0] * float(cfg.max_notional)
                pos_vec = pos
            else:
                a = a_t[0]
                maxn = float(cfg.max_notional)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxn), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxn), pos_vec)
                pos = pos_vec
            contrib = pos * r_tp1
            mean_c = torch.mean(contrib)
            std_c = torch.std(contrib)
            downside = torch.clamp(contrib, max=0.0)
            downside_var = torch.mean(downside**2)
            downside_std = torch.sqrt(downside_var + 1e-8)
            if cfg.sharpe_like:
                risk_metric = mean_c / (std_c + 1e-8)
            else:
                risk_metric = mean_c / (downside_std + 1e-8)
            cum += float(risk_metric.item())
        avg_daily = cum / max(1, T - 1)
        print(json.dumps({"phase": "eval", "val_avg_daily_reward": avg_daily}), flush=True)

    return {"model": model, "val_avg_daily_reward": float(avg_daily)}


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Actor-Critic for USD-quoted Alpaca crypto via candle cache."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_CRYPTO_SYMBOLS),
        help="Comma-separated list of crypto symbols (default: fixed 22 USD-quoted Alpaca cryptos).",
    )
    parser.add_argument(
        "--environment",
        choices=["practice", "live"],
        default="practice",
        help="Environment tag for upstream (practice strongly recommended).",
    )
    parser.add_argument(
        "--candle-cache-base-url",
        default=os.environ.get("CANDLE_CACHE_BASE_URL", "http://127.0.0.1:9100"),
        help="Base URL for candle_cache_service (OANDA-compatible).",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end")
    parser.add_argument("--ae-latent", type=int, default=64)  # ignored (for backward CLI compatibility)
    parser.add_argument("--ae-epochs", type=int, default=5)  # ignored (for backward CLI compatibility)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-sigma", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--sharpe-like",
        action="store_true",
        help="Use Sharpe-like reward (mean/std) instead of Sortino-like (mean/downside_std).",
    )
    parser.add_argument(
        "--policy-hidden",
        default="256,256",
        help="Comma-separated hidden sizes for policy MLP (e.g. 256,256,128).",
    )
    parser.add_argument(
        "--value-hidden",
        default="256,256",
        help="Comma-separated hidden sizes for value MLP (e.g. 256,256).",
    )
    parser.add_argument(
        "--max-notional",
        type=float,
        default=1.0,
        help="Scale for actions in [-1,1] mapped to target notionals.",
    )
    parser.add_argument(
        "--action-mapping",
        choices=["continuous"],
        default="continuous",
        help="How to map actor outputs to target notionals (crypto is long-only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        default="forex-rl/actor-critic/checkpoints/crypto_offline_ac.pt",
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument(
        "--use-tcn",
        action="store_true",
        help="When set, load a pre-trained TCN checkpoint and append its features; "
        "by default, use only raw OHLCV-derived + time features.",
    )
    args = parser.parse_args()

    # Parse symbols from CLI
    symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided; use --symbols or rely on defaults.")

    # Parse MLP architectures from comma-separated CLI strings
    def _parse_hidden(s: str, default: List[int]) -> List[int]:
        try:
            vals = [int(x.strip()) for x in (s or "").split(",") if x.strip()]
            return vals or list(default)
        except Exception:
            return list(default)

    policy_hidden = _parse_hidden(args.policy_hidden, [256, 256])
    value_hidden = _parse_hidden(args.value_hidden, [256, 256])

    cfg = Config(
        symbols=symbols,
        environment=args.environment,
        candle_cache_base_url=args.candle_cache_base_url,
        start=args.start,
        end=args.end,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        actor_sigma=float(args.actor_sigma),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        adv_clip=float(args.adv_clip),
        reward_scale=float(args.reward_scale),
        model_path=args.model_path,
        seed=int(args.seed),
        max_notional=float(args.max_notional),
        action_mapping=str(args.action_mapping),
        policy_hidden=policy_hidden,
        value_hidden=value_hidden,
        sharpe_like=bool(args.sharpe_like),
    )

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataset
    print(
        json.dumps(
            {
                "status": "fetch_data",
                "symbols": cfg.symbols,
                "start": cfg.start,
                "end": cfg.end,
                "candle_cache_base_url": cfg.candle_cache_base_url,
            }
        ),
        flush=True,
    )
    X, R, dates = build_crypto_dataset(cfg)

    # Time-based train/val split index (used for AC to avoid leakage)
    n = len(X)
    split = int(n * 0.8)
    if split < 1:
        split = n  # all-train, no-val when window is very short

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally load pre-trained TCN (trained separately) and append its outputs as features.
    if args.use_tcn:
        tcn_ckpt_path = os.path.join(FX_ROOT, "actor-critic", "checkpoints", "crypto_tcn.pt")
        tcn_feats = _build_tcn_features_from_checkpoint(
            X, cfg, cfg.symbols, ckpt_path=tcn_ckpt_path, device=device
        )
        X_aug = pd.concat([X, tcn_feats], axis=1)
    else:
        X_aug = X

    # Standardize (optionally TCN-augmented) features
    Xn, stats = standardize_fit(X_aug)

    # Time-based train/val split on augmented features
    X_train = Xn.iloc[:split]
    R_train = R.iloc[:split]
    X_val = Xn.iloc[split:]
    R_val = R.iloc[split:]

    # Train actor-critic directly on standardized + TCN-augmented features (no autoencoder)
    ac_meta = train_actor_critic(X_train, R_train, X_val, R_val, cfg, device)

    # Save checkpoint: policy + critic + stats + config meta
    model = ac_meta["model"]
    ckpt = {
        "config": asdict(cfg),
        "feature_stats": stats,
        "state_dict": model.state_dict(),
        "dates": [d.isoformat() for d in dates],
        "symbols": cfg.symbols,
        "input_dim": int(X_train.shape[1]),
    }

    model_path = cfg.model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp_path = model_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, model_path)
    print(json.dumps({"status": "saved", "model_path": model_path}), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()

