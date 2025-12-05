#!/usr/bin/env python3
"""
Multi-Asset Offline Actor-Critic (20 FX instruments + ETF features)
- Offline training only (historical candles); no live/order placement
- Daily step; reward = mean/vol of portfolio contributions (Sharpe-like)
- Massive feature set from:
  * FX: Daily OHLCV-derived features per instrument (weekly/hourly disabled by default)
  * ETFs: Daily OHLCV-derived features for a configurable list (yfinance; 15-ticker core by default)
  * Cyclical time(10) features
- Dimensionality reduction via an Autoencoder; encoder used as state trunk
- Actor produces continuous actions in [-1, 1] per FX instrument
  which are treated as raw position sizes scaled by --max-units (default 100).

Requirements:
  - OANDA REST credentials for FX historical data (practice/live):
    OANDA_DEMO_ACCOUNT_ID, OANDA_DEMO_KEY (or pass via CLI)
  - yfinance for ETF candles

Notes:
  - This script builds a training dataset, pre-trains an autoencoder,
    then trains an A2C-style actor-critic purely on historical sequences.
  - Value function provides TD(0) baseline; actor uses diagonal Gaussian policy
    (tanh-squashed action applied to returns; log-prob computed pre-squash, correction ignored).
  - By default, only Daily + Weekly features are enabled to keep runtime reasonable.
    You can extend to H1 and weekly in a follow-up iteration.

Outputs:
  - Checkpoint with encoder + policy + critic, feature normalization stats, and config

Run example:
  python forex-rl/actor-critic/multi20_offline_actor_critic.py \
      --instruments EUR_USD,USD_JPY,GBP_USD,USD_CHF,AUD_USD,USD_CAD,NZD_USD,EUR_JPY,EUR_GBP,GBP_JPY,EUR_AUD,EUR_CAD,AUD_JPY,CAD_JPY,GBP_CAD,NZD_JPY,CHF_JPY,EUR_NZD,GBP_AUD,GBP_NZD \
      --start 2020-01-01 --end 2025-08-31 --epochs 8 --ae-epochs 6 --use-all-etfs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Repo imports (OANDA REST adapter for historical candles)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
# Also add forex-rl folder to import local module with hyphenated parent
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore

# yfinance for ETF candles
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # will error later if ETFs requested


# ---------- Config ----------


@dataclass
class Config:
    instruments: List[str]
    environment: str = "practice"
    account_id: Optional[str] = None
    access_token: Optional[str] = None
    # Data range
    start: str = "2019-01-01"
    end: Optional[str] = None  # inclusive end date (YYYY-MM-DD); if None, use today-1
    # Features
    include_weekly: bool = False  # default to daily only
    include_hourly: bool = False  # reserved for later
    use_all_etfs: bool = False
    etf_tickers: Optional[List[str]] = None  # overrides use_all_etfs if provided
    # AE/Model
    ae_hidden: List[int] = None  # filled in post-init
    ae_latent_dim: int = 64
    policy_hidden: int = 256
    value_hidden: int = 256
    # Trading semantics
    max_units: int = 100  # scale factor for actions in [-1,1]
    # Action mapping: threshold (default) or continuous
    action_mapping: str = "threshold"
    # Training
    batch_size: int = 128
    ae_epochs: int = 5
    epochs: int = 6
    gamma: float = 0.99
    actor_sigma: float = 0.3
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    adv_clip: float = 5.0
    reward_scale: float = 1.0
    # Misc
    seed: int = 42
    autosave_secs: float = 120.0
    model_path: str = "forex-rl/actor-critic/checkpoints/multi20_offline_ac.pt"

    def __post_init__(self) -> None:
        if self.ae_hidden is None:
            self.ae_hidden = [2048, 512, 128]


# ---------- Defaults ----------

# Optimal high-liquidity, low-spread OANDA pairs
DEFAULT_OANDA_20: List[str] = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF",
    "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
    "EUR_CHF", "EUR_AUD", "EUR_CAD", "GBP_CHF", "AUD_JPY",
    "AUD_CHF", "CAD_JPY", "NZD_JPY", "GBP_AUD", "AUD_NZD",
]
DEFAULT_INSTRUMENTS_CSV = ",".join(DEFAULT_OANDA_20)


# ---------- Utility: ETF universe ----------


def get_etf_universe(use_all: bool) -> List[str]:
    """Return ETF list. If use_all is True, return the full 115+ list;
    otherwise return a compact diversified 15-ticker core set by default.
    """
    all_tickers = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VTV', 'VUG', 'IJR', 'IJH', 'IVV', 'VOO', 'XLK', 'VGT',
        'FTEC', 'SOXX', 'ARKK', 'ARKQ', 'CLOU', 'CYBR', 'VEA', 'IEFA', 'EFA', 'VT', 'VXUS',
        'SCHF', 'EWJ', 'FEZ', 'VWO', 'EEM', 'IEMG', 'SCHE', 'EWZ', 'FXI', 'ASHR', 'KWEB', 'XLF',
        'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC', 'VDE', 'DBC', 'PDBC',
        'USCI', 'DJP', 'GSG', 'COMT', 'GLD', 'IAU', 'SLV', 'PPLT', 'PALL', 'GLTR', 'GDX', 'GDXJ',
        'USO', 'UCO', 'UGA', 'BNO', 'USL', 'DBO', 'OIL', 'OILK', 'UNG', 'BOIL', 'KOLD', 'FCG',
        'CPER', 'DBB', 'PICK', 'REMX', 'DBA', 'CORN', 'WEAT', 'SOYB', 'JO', 'NIB', 'AGG', 'BND',
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK', 'TIP', 'VCIT', 'VNQ', 'IYR', 'XLRE', 'SCHH',
        'USRT', 'RWR', 'VNQI', 'VXX', 'UVXY', 'VIXY', 'SVXY', 'VIXM', 'VXZ', 'UUP', 'UDN', 'FXE',
        'FXY', 'FXB', 'FXC', 'TAIL', 'PHDG', 'NUSI', 'QYLD', 'JEPI'
    ]
    if use_all:
        return all_tickers
    # Compact diversified core (15 tickers): broad US, tech, intl dev/em, bonds, commodities, REITs, USD
    return [
        'SPY',  # US broad
        'QQQ',  # Tech/growth tilt
        'VEA',  # Intl developed
        'VWO',  # Emerging markets
        'AGG',  # US aggregate bonds
        'TLT',  # Long treasuries
        'HYG',  # High yield
        'GLD',  # Gold
        'DBC',  # Broad commodities
        'VNQ',  # US REITs
        'UUP',  # US Dollar index proxy
        'XLE',  # Energy
        'XLV',  # Healthcare
        'XLF',  # Financials
        'XLK',  # Technology
    ]


# ---------- Time features ----------


def time_cyclical_features_from_index(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return 10-dim cyclical time features for each date in index.
    Encodings: minute, hour, day-of-week, day-of-month, month-of-year (sin/cos for each).
    For daily data, minute/hour are effectively 0/constant but retained for compatibility.
    """
    idx = pd.to_datetime(index, utc=True).tz_convert('UTC')

    minute = idx.minute.values
    hour = idx.hour.values
    dow = idx.weekday.values
    dom = idx.day.values - 1
    moy = idx.month.values - 1

    def sc(vals: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
        ang = 2.0 * np.pi * (vals.astype(float) / period)
        return np.sin(ang), np.cos(ang)

    m_s, m_c = sc(minute, 60.0)
    h_s, h_c = sc(hour, 24.0)
    dw_s, dw_c = sc(dow, 7.0)
    dm_s, dm_c = sc(dom, 31.0)
    my_s, my_c = sc(moy, 12.0)

    df = pd.DataFrame({
        'time_min_sin': m_s, 'time_min_cos': m_c,
        'time_hr_sin': h_s, 'time_hr_cos': h_c,
        'time_dow_sin': dw_s, 'time_dow_cos': dw_c,
        'time_dom_sin': dm_s, 'time_dom_cos': dm_c,
        'time_moy_sin': my_s, 'time_moy_cos': my_c,
    }, index=index)
    return df.astype(np.float32)


# ---------- Feature engineering ----------


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
    """Compute a 16-dim feature set per row from OHLCV DataFrame with columns
    ['open','high','low','close','volume'].
    Returns DataFrame with the same index and 16 feature columns.
    """
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get('close', 'close')].astype(float)
    high = df[cols.get('high', 'high')].astype(float)
    low = df[cols.get('low', 'low')].astype(float)
    vol = df[cols.get('volume', 'volume')].astype(float)

    logc = np.log(np.maximum(close, eps))
    r = logc.diff()

    def last_ret(k: int) -> pd.Series:
        return (logc - logc.shift(k))

    def std_ret(k: int) -> pd.Series:
        return r.rolling(k).std().fillna(0.0)

    # ATR(20)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
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

    feats = pd.DataFrame({
        'ret_1': last_ret(1),
        'ret_5': last_ret(5),
        'ret_20': last_ret(20),
        'std_5': std_ret(5),
        'std_20': std_ret(20),
        'std_60': std_ret(60),
        'atr20': atr20,
        'dist_max20': dist_max20,
        'dist_min20': dist_min20,
        'vol_z20': v20,
        'macd': macd,
        'macd_signal': signal,
        'macd_hist': hist,
        'rsi14_norm': (rsi14 / 100.0),
        'vol_of_vol20': vol_of_vol,
        'obv_z20': obv_z,
    }, index=df.index)
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats.astype(np.float32)


# ---------- Data loading ----------


def fetch_fx_ohlcv(instrument: str, granularity: str, start: str, end: Optional[str], environment: str, access_token: Optional[str]) -> pd.DataFrame:
    """Fetch OHLCV for an FX instrument from OANDA REST.
    Returns DataFrame with columns [open, high, low, close, volume] indexed by UTC date (normalized to midnight for D/W; hourly timestamps for H1).
    """
    if access_token is None:
        raise RuntimeError("Access token required for OANDA REST (set OANDA_DEMO_KEY or pass --access-token)")
    adapter = OandaRestCandlesAdapter(
        instrument=instrument,
        granularity=granularity,
        environment=environment,
        access_token=access_token,
    )
    if end is None:
        # yfinance convention end exclusive; here we keep inclusive; adapter will stop at to_time
        from datetime import datetime, timezone, timedelta
        end = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

    rows: List[Dict[str, Any]] = list(adapter.fetch_range(from_time=f"{start}T00:00:00Z", to_time=f"{end}T23:59:59Z", batch=5000))
    if not rows:
        raise RuntimeError(f"No candles fetched for {instrument} {granularity} between {start} and {end}")
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
    # Normalize index to date for D/W data
    if granularity in ("D", "W"):
        df.index = df.index.normalize()
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)


def fetch_etf_ohlcv(tickers: List[str], start: str, end: Optional[str]) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Please install yfinance.")
    # yfinance end is exclusive; make it inclusive by bumping a day when end provided
    if end is None:
        from datetime import datetime, timezone, timedelta
        end_excl = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    else:
        try:
            end_excl = (pd.to_datetime(end).tz_localize('UTC') + pd.Timedelta(days=1)).date().isoformat()
        except Exception:
            # Fallback: string add one day via pandas
            end_excl = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()
    data = yf.download(tickers, start=start, end=end_excl, auto_adjust=True, progress=False, threads=True)
    out: Dict[str, pd.DataFrame] = {}
    # data is a DataFrame with MultiIndex columns if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        for tkr in tickers:
            sub = pd.DataFrame({
                'open': data['Open'][tkr],
                'high': data['High'][tkr],
                'low': data['Low'][tkr],
                'close': data['Close'][tkr],
                'volume': data['Volume'][tkr],
            })
            sub.index = pd.to_datetime(sub.index, utc=True).normalize()
            out[tkr] = sub.dropna()
    else:
        # Single ticker case
        tkr = tickers[0]
        sub = pd.DataFrame({
            'open': data['Open'], 'high': data['High'], 'low': data['Low'], 'close': data['Close'], 'volume': data['Volume']
        })
        sub.index = pd.to_datetime(sub.index, utc=True).normalize()
        out[tkr] = sub.dropna()
    return out


# ---------- Dataset builder ----------


def build_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_features, R_fx_returns, dates_index)
    - X_features: full feature matrix with columns from FX (D/W) and ETF features and time features
    - R_fx_returns: DataFrame of daily log returns per FX instrument (next-day returns aligned to state at t)
    - dates_index: Date index (daily) for states
    """
    # FX Daily + Weekly
    fx_daily_feats: List[pd.DataFrame] = []
    fx_weekly_feats: List[pd.DataFrame] = []
    fx_close_for_returns: Dict[str, pd.Series] = {}

    # Fetch daily/weekly series per instrument
    for inst in cfg.instruments:
        d_df = fetch_fx_ohlcv(inst, "D", cfg.start, cfg.end, cfg.environment, cfg.access_token)
        d_feats = compute_ohlcv_features(d_df)
        d_feats = d_feats.add_prefix(f"FX_{inst}_D_")
        fx_daily_feats.append(d_feats)
        fx_close_for_returns[inst] = d_df['close']
        if cfg.include_weekly:
            w_df = fetch_fx_ohlcv(inst, "W", cfg.start, cfg.end, cfg.environment, cfg.access_token)
            w_feats = compute_ohlcv_features(w_df).add_prefix(f"FX_{inst}_W_")
            # Forward-fill weekly features to daily dates (align on week ending date)
            w_feats_ff = w_feats.reindex(d_feats.index).ffill().fillna(0.0)
            fx_weekly_feats.append(w_feats_ff)

    # Align all FX daily features on intersection of dates
    base_index = fx_daily_feats[0].index
    for f in fx_daily_feats[1:]:
        base_index = base_index.intersection(f.index)
    if cfg.include_weekly:
        for f in fx_weekly_feats:
            base_index = base_index.intersection(f.index)

    # ETFs (do NOT constrain base_index by ETF availability; forward-fill instead)
    etf_feats_frames: List[pd.DataFrame] = []
    etf_tickers = cfg.etf_tickers if (cfg.etf_tickers and len(cfg.etf_tickers) > 0) else get_etf_universe(cfg.use_all_etfs)
    if etf_tickers:
        etf_map = fetch_etf_ohlcv(etf_tickers, cfg.start, cfg.end)
        for tkr, df in etf_map.items():
            feats = compute_ohlcv_features(df).add_prefix(f"ETF_{tkr}_")
            etf_feats_frames.append(feats)
            # Don't intersect base_index with ETF dates; some ETFs post later.

    # Reindex all frames to base_index and concat
    blocks: List[pd.DataFrame] = []
    for frm in fx_daily_feats:
        blocks.append(frm.reindex(base_index).fillna(0.0))
    if cfg.include_weekly:
        for frm in fx_weekly_feats:
            blocks.append(frm.reindex(base_index).ffill().fillna(0.0))
    for frm in etf_feats_frames:
        # Forward-fill ETF features to the FX date so inference advances at 5pm ET even if ETFs post later
        blocks.append(frm.reindex(base_index).ffill().fillna(0.0))

    # Time features
    t_feats = time_cyclical_features_from_index(base_index)
    blocks.append(t_feats)

    X = pd.concat(blocks, axis=1).astype(np.float32)

    # Returns (next-day log returns per FX instrument) ? used for per-instrument contributions and Sharpe-like portfolio reward
    r_cols: Dict[str, pd.Series] = {}
    for inst, cls in fx_close_for_returns.items():
        close = cls.reindex(base_index).ffill()
        r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # Align next-day return to today's state (reward at t+1 for action at t)
        r_next = r.shift(-1).fillna(0.0)
        r_cols[inst] = r_next
    R = pd.DataFrame(r_cols, index=base_index).astype(np.float32)

    # Drop first/last if needed to ensure no leakage; we already shifted R
    return X, R, base_index


# ---------- Autoencoder ----------


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        enc_layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        enc_layers += [nn.Linear(last, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers += [nn.Linear(last, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


# ---------- Policy / Value ----------


class ActorCriticMulti(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int, num_instruments: int, policy_hidden: int = 256, value_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, policy_hidden), nn.ReLU(),
            nn.Linear(policy_hidden, num_instruments),
        )
        self.value = nn.Sequential(
            nn.Linear(latent_dim, value_hidden), nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(self.training):
            z = self.encoder(x)
        mu = self.policy(z)
        v = self.value(z).squeeze(-1)
        return z, mu, v


# ---------- Training ----------


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


def pretrain_autoencoder(X_train: pd.DataFrame, input_dim: int, cfg: Config, device: torch.device) -> Tuple[AutoEncoder, Dict[str, Any]]:
    ae = AutoEncoder(input_dim=input_dim, hidden_dims=cfg.ae_hidden, latent_dim=cfg.ae_latent_dim).to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(tensorize(X_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    mse = nn.MSELoss()

    for epoch in range(cfg.ae_epochs):
        ae.train()
        total = 0.0
        n = 0
        for (xb,) in loader:
            xb = xb.to(device)
            _, recon = ae(xb)
            loss = mse(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(json.dumps({"phase": "ae", "epoch": epoch + 1, "loss_mse": total / max(1, n)}), flush=True)
    return ae, {"input_dim": input_dim, "latent": cfg.ae_latent_dim}


def train_actor_critic(X_train: pd.DataFrame, R_train: pd.DataFrame, X_val: pd.DataFrame, R_val: pd.DataFrame, ae: AutoEncoder, cfg: Config, device: torch.device) -> Dict[str, Any]:
    num_inst = R_train.shape[1]
    model = ActorCriticMulti(encoder=ae.encoder, latent_dim=cfg.ae_latent_dim, num_instruments=num_inst, policy_hidden=cfg.policy_hidden, value_hidden=cfg.value_hidden).to(device)
    # Optionally freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

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
        # For threshold mapping we keep stateful positions across time
        pos_vec = torch.zeros(Rb.size(1), dtype=torch.float32, device=Xb.device)
        # We'll iterate sequentially: state at t, reward from R(t) (already shifted as next-day returns)
        total_loss = 0.0
        total_actor = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_reward = 0.0
        steps = 0
        for t in range(0, T - 1):
            s_t = Xb[t:t + 1]
            s_tp1 = Xb[t + 1:t + 2]
            r_tp1 = Rb[t + 1]  # next-day per-instrument returns

            _, mu_t, v_t = model(s_t)
            with torch.no_grad():
                _, _, v_tp1 = model(s_tp1)
            # Sample pre-squash action from Normal(mu, sigma)
            eps = torch.randn_like(mu_t)
            pre = mu_t + sigma * eps
            a_t = torch.tanh(pre)  # in [-1, 1]

            # Map actions to positions
            if (cfg.action_mapping or "threshold") == "continuous":
                # Continuous sizing
                pos = a_t[0] * float(cfg.max_units)
            else:
                # Threshold hysteresis with constant units
                a = a_t[0]
                maxu = float(cfg.max_units)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                # Exits
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                # Entries
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxu), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxu), pos_vec)
                pos = pos_vec
            # Per-instrument contributions: position * return
            contrib = pos * r_tp1
            # Sharpe-like reward: mean/vol across instruments for the day
            mean_c = torch.mean(contrib)
            std_c = torch.std(contrib)
            sharpe_like = mean_c / (std_c + 1e-8)
            r_scalar = cfg.reward_scale * sharpe_like

            # Advantage
            adv = (r_scalar + cfg.gamma * v_tp1[0] - v_t[0]).detach()
            if cfg.adv_clip > 0:
                adv = torch.clamp(adv, -cfg.adv_clip, cfg.adv_clip)

            # Log-prob of Normal (pre-squash); ignore tanh correction for simplicity
            logprob = -0.5 * torch.sum(((pre - mu_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const, dim=1)
            entropy = 0.5 * torch.sum(torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))

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
        print(json.dumps({"phase": "train", "epoch": epoch + 1, **{f"tr_{k}": v for k, v in tr.items()}, **{f"va_{k}": v for k, v in va.items()}}), flush=True)

    # Final greedy validation (no sampling)
    model.eval()
    with torch.no_grad():
        T = X_va.size(0)
        cum = 0.0
        pos_vec = torch.zeros(R_va.size(1), dtype=torch.float32, device=X_va.device)
        for t in range(0, T - 1):
            s_t = X_va[t:t + 1]
            _, mu_t, _ = model(s_t)
            a_t = torch.tanh(mu_t)
            r_tp1 = R_va[t + 1]
            if (cfg.action_mapping or "threshold") == "continuous":
                pos = a_t[0] * float(cfg.max_units)
            else:
                a = a_t[0]
                maxu = float(cfg.max_units)
                enter_long = (pos_vec == 0) & (a > 0.66)
                enter_short = (pos_vec == 0) & (a < -0.66)
                exit_long = (pos_vec > 0) & (a < 0.33)
                exit_short = (pos_vec < 0) & (a > -0.33)
                pos_vec = pos_vec.masked_fill(exit_long | exit_short, 0.0)
                pos_vec = torch.where(enter_long, torch.full_like(pos_vec, maxu), pos_vec)
                pos_vec = torch.where(enter_short, torch.full_like(pos_vec, -maxu), pos_vec)
                pos = pos_vec
            contrib = pos * r_tp1
            sharpe_like = torch.mean(contrib) / (torch.std(contrib) + 1e-8)
            cum += float(sharpe_like.item())
        avg_daily = cum / max(1, T - 1)
        print(json.dumps({"phase": "eval", "val_avg_daily_reward": avg_daily}), flush=True)

    return {"model": model}


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Actor-Critic for 20 FX instruments with ETF features")
    parser.add_argument("--instruments", default=DEFAULT_INSTRUMENTS_CSV, help="Comma-separated OANDA instruments (20 recommended)")
    parser.add_argument("--environment", choices=["practice", "live"], default="practice")
    parser.add_argument("--account-id", default=os.environ.get("OANDA_DEMO_ACCOUNT_ID"))
    parser.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end")
    parser.add_argument("--include-weekly", action="store_true")
    parser.add_argument("--no-include-weekly", dest="include_weekly", action="store_false")
    parser.set_defaults(include_weekly=False)
    parser.add_argument("--include-hourly", action="store_true")  # reserved for later extensions
    parser.add_argument("--use-all-etfs", action="store_true", help="Use the full 115+ ETF universe")
    parser.add_argument("--etf-tickers", default="", help="Comma-separated custom ETF tickers (overrides --use-all-etfs if provided)")
    parser.add_argument("--ae-latent", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-sigma", type=float, default=0.3)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--adv-clip", type=float, default=5.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--max-units", type=int, default=100, help="Units scaling for actions in [-1,1]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mapping", choices=["threshold", "continuous"], default="threshold", help="How to map actor outputs to positions")
    parser.add_argument("--model-path", default="forex-rl/actor-critic/checkpoints/multi20_offline_ac.pt")
    args = parser.parse_args()

    instruments = [s.strip() for s in (args.instruments or "").split(",") if s.strip()]
    if len(instruments) == 0:
        # Fallback to default universe
        instruments = DEFAULT_OANDA_20

    etf_list = [s.strip().upper() for s in (args.etf_tickers or "").split(",") if s.strip()]

    cfg = Config(
        instruments=instruments,
        environment=args.environment,
        account_id=args.account_id,
        access_token=args.access_token,
        start=args.start,
        end=args.end,
        include_weekly=bool(args.include_weekly),
        include_hourly=bool(args.include_hourly),
        use_all_etfs=bool(args.use_all_etfs),
        etf_tickers=etf_list if len(etf_list) > 0 else None,
        ae_latent_dim=int(args.ae_latent),
        ae_epochs=int(args.ae_epochs),
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
        max_units=int(args.max_units),
        action_mapping=str(args.action_mapping),
    )

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataset
    print(json.dumps({"status": "fetch_data", "instruments": cfg.instruments, "use_all_etfs": cfg.use_all_etfs, "etf_override": bool(cfg.etf_tickers), "start": cfg.start, "end": cfg.end}), flush=True)
    X, R, dates = build_dataset(cfg)

    # Standardize features
    Xn, stats = standardize_fit(X)

    # Train/val split (time-based)
    n = len(Xn)
    if n < 200:
        raise RuntimeError("Not enough data to train. Increase date range.")
    split = int(n * 0.8)
    X_train = Xn.iloc[:split]
    R_train = R.iloc[:split]
    X_val = Xn.iloc[split:]
    R_val = R.iloc[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder pretrain
    ae, ae_meta = pretrain_autoencoder(X_train, input_dim=X_train.shape[1], cfg=cfg, device=device)

    # Actor-Critic
    out = train_actor_critic(X_train, R_train, X_val, R_val, ae, cfg, device)
    model: ActorCriticMulti = out["model"]

    # Save checkpoint
    try:
        os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
        torch.save({
            "cfg": asdict(cfg),
            "feature_stats": stats,
            "ae_encoder_state": ae.encoder.state_dict(),
            "policy_state": model.policy.state_dict(),
            "value_state": model.value.state_dict(),
            "meta": {"input_dim": X_train.shape[1], "latent_dim": cfg.ae_latent_dim, "num_instruments": R.shape[1]},
        }, cfg.model_path)
        print(json.dumps({"saved": cfg.model_path}), flush=True)
    except Exception as exc:
        print(json.dumps({"save_error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
