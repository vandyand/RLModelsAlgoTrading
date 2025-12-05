#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Repo paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
FX_ROOT = os.path.join(REPO_ROOT, "forex-rl")
if FX_ROOT not in sys.path:
    sys.path.append(FX_ROOT)
# Add continuous-trader module dir (hyphenated) to sys.path
CT_ROOT = os.path.join(FX_ROOT, "continuous-trader")
if CT_ROOT not in sys.path:
    sys.path.append(CT_ROOT)

from oanda_rest_adapter import OandaRestCandlesAdapter  # type: ignore
import instruments as ct_instruments  # type: ignore
import rewards as ct_rewards  # type: ignore
import model as ct_model  # type: ignore


@dataclass
class Config:
    instruments_csv: Optional[str]
    environment: str  # kept for OANDA REST fallback; default 'practice'
    access_token: Optional[str]
    start: str
    end: Optional[str]
    include_h1: bool
    include_d: bool
    ae_latent: int
    ae_hidden: List[int]
    policy_hidden: int
    value_hidden: int
    batch_size: int
    ae_epochs: int
    epochs: int
    gamma: float
    actor_sigma: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    reward_name: str
    reward_scale: float
    model_path: str
    max_units: int
    seed: int
    feature_source: str = "exported"  # exported|compute
    feature_grans: List[str] = None  # set post-parse
    feature_dir: Optional[str] = None


def _load_local_ohlcv(instrument: str, granularity: str) -> Optional[pd.DataFrame]:
    """Load OHLCV locally if available.

    Search order:
      1) forex-rl/continuous-trader/data/{instrument}_{granularity}.csv
      2) If missing and granularity in {D,H1}: try resampling from M5 found in:
         - forex-rl/continuous-trader/data/{instrument}_M5.csv
         - forex-rl/ga-ness/data/{instrument}_M5.csv
    """
    ct_dir = os.path.join(FX_ROOT, "continuous-trader", "data")
    ga_dir = os.path.join(FX_ROOT, "ga-ness", "data")

    # Exact match first
    exact_path = os.path.join(ct_dir, f"{instrument}_{granularity}.csv")
    if os.path.exists(exact_path):
        try:
            df = pd.read_csv(exact_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
            cols = {c.lower(): c for c in df.columns}
            df = df[[cols.get("open","open"), cols.get("high","high"), cols.get("low","low"), cols.get("close","close"), cols.get("volume","volume")]]
            df.columns = ["open","high","low","close","volume"]
            if granularity in ("D", "W") and isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.normalize()
            return df.astype(float)
        except Exception:
            pass

    # Resample from M5 if needed
    if granularity in ("H1", "D"):
        for base_dir in (ct_dir, ga_dir):
            m5_path = os.path.join(base_dir, f"{instrument}_M5.csv")
            if not os.path.exists(m5_path):
                continue
            try:
                df = pd.read_csv(m5_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
                cols = {c.lower(): c for c in df.columns}
                df = df[[cols.get("open","open"), cols.get("high","high"), cols.get("low","low"), cols.get("close","close"), cols.get("volume","volume")]]
                df.columns = ["open","high","low","close","volume"]
                rule = '1H' if granularity == 'H1' else '1D'
                agg = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }
                out = df.resample(rule).agg(agg).dropna()
                if granularity == 'D':
                    out.index = out.index.normalize()
                return out.astype(float)
            except Exception:
                continue
    return None


def fetch_fx_range(instrument: str, granularity: str, start: str, end: Optional[str], environment: str, access_token: Optional[str]) -> pd.DataFrame:
    # Prefer local CSV cache if present
    local = _load_local_ohlcv(instrument, granularity)
    if local is not None:
        # Trim to range if provided
        if start:
            local = local[local.index >= pd.to_datetime(f"{start}T00:00:00Z")]
        if end:
            local = local[local.index <= pd.to_datetime(f"{end}T23:59:59Z")]
        return local

    # Fallback to REST
    adapter = OandaRestCandlesAdapter(instrument=instrument, granularity=granularity, environment=environment, access_token=access_token)
    rows = list(adapter.fetch_range(from_time=f"{start}T00:00:00Z", to_time=(f"{end}T23:59:59Z" if end else None), batch=5000))
    if not rows:
        raise RuntimeError(f"No candles for {instrument} {granularity} in range")
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    if granularity in ("D", "W"):
        df.index = df.index.normalize()
    return df[['open','high','low','close','volume']].astype(float)


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    close = df[cols.get('close', 'close')].astype(float)
    high = df[cols.get('high', 'high')].astype(float)
    low = df[cols.get('low', 'low')].astype(float)
    vol = df[cols.get('volume', 'volume')].astype(float)

    logc = np.log(np.maximum(close, 1e-12))
    r = logc.diff()

    def last_ret(k: int) -> pd.Series:
        return (logc - logc.shift(k))

    def std_ret(k: int) -> pd.Series:
        return r.rolling(k).std().fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean().fillna(0.0)

    roll_max20 = close.rolling(20).max()
    roll_min20 = close.rolling(20).min()
    roll_std20 = close.rolling(20).std().fillna(1e-6)
    dist_max20 = (close - roll_max20) / (roll_std20 + 1e-12)
    dist_min20 = (close - roll_min20) / (roll_std20 + 1e-12)

    v_mean20 = vol.rolling(20).mean()
    v_std20 = vol.rolling(20).std().fillna(1e-6)
    v20 = (vol - v_mean20) / (v_std20 + 1e-12)

    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    hist = macd - signal

    def rsi(close: pd.Series, period: int = 14, eps: float = 1e-12) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean() + eps
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi14 = rsi(close, 14) / 100.0
    vol_of_vol = r.abs().rolling(20).std().fillna(0.0)

    direction = close.diff()
    obv_step = np.where(direction > 0, vol, np.where(direction < 0, -vol, 0.0))
    obv = pd.Series(obv_step, index=close.index).cumsum()
    obv_mean20 = obv.rolling(20).mean()
    obv_std20 = obv.rolling(20).std().fillna(1e-6)
    obv_z = (obv - obv_mean20) / (obv_std20 + 1e-12)

    feats = pd.DataFrame({
        'ret_1': last_ret(1), 'ret_5': last_ret(5), 'ret_20': last_ret(20),
        'std_5': std_ret(5), 'std_20': std_ret(20), 'std_60': std_ret(60),
        'atr20': atr20,
        'dist_max20': dist_max20, 'dist_min20': dist_min20,
        'vol_z20': v20,
        'macd': macd, 'macd_signal': signal, 'macd_hist': hist,
        'rsi14_norm': rsi14,
        'vol_of_vol20': vol_of_vol,
        'obv_z20': obv_z,
    }, index=df.index)
    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats.astype(np.float32)


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


def build_dataset(cfg: Config, instruments: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build daily feature grid (D + optional H1) for many instruments; target is next-day EUR_USD return.

    We'll concatenate features across instruments and granularities; the target y is the next-day log return of EUR_USD.
    """
    frames: List[pd.DataFrame] = []
    eur_close: Optional[pd.Series] = None
    for inst in instruments:
        d_df = fetch_fx_range(inst, "D", cfg.start, cfg.end, cfg.environment, cfg.access_token)
        d_feats = compute_ohlcv_features(d_df).add_prefix(f"D_{inst}_")
        frames.append(d_feats)
        if inst == "EUR_USD":
            eur_close = d_df['close']
        if cfg.include_h1:
            try:
                h1_df = fetch_fx_range(inst, "H1", cfg.start, cfg.end, cfg.environment, cfg.access_token)
                # Resample to daily end-of-day by last observation
                h1_daily = h1_df.resample('1D').last().dropna()
                h1_feats = compute_ohlcv_features(h1_daily).add_prefix(f"H1_{inst}_")
                frames.append(h1_feats)
            except Exception:
                pass
    if eur_close is None:
        raise RuntimeError("EUR_USD not present in instrument list")

    # Align intersection of dates
    base_index = frames[0].index
    for f in frames[1:]:
        base_index = base_index.intersection(f.index)

    blocks: List[pd.DataFrame] = [frm.reindex(base_index).fillna(0.0) for frm in frames]
    X = pd.concat(blocks, axis=1).astype(np.float32)

    close = eur_close.reindex(base_index).ffill()
    r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = r.shift(-1).fillna(0.0)  # next-day EUR_USD return
    return X, y, base_index


def _safe_inst(inst: str) -> str:
    return inst.replace('/', '_')


def load_exported_feature_matrix(cfg: Config, instruments: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load exported feature CSVs per instrument and granularity, align to daily dates, and build EUR_USD next-day y.

    Defaults to using only D features to keep memory reasonable on small VMs.
    """
    feat_dir = cfg.feature_dir or os.path.join(FX_ROOT, "continuous-trader", "data", "features")
    grans = [g.strip().upper() for g in (cfg.feature_grans or ["D"]) if g.strip()]

    frames: List[pd.DataFrame] = []
    for gran in grans:
        for inst in instruments:
            safe = _safe_inst(inst)
            fpath = os.path.join(feat_dir, gran, f"{safe}_{gran}_features.csv")
            if not os.path.exists(fpath):
                # Skip missing combo
                continue
            df = pd.read_csv(fpath)
            # Expect index in first column
            if df.columns[0].lower() in ("timestamp", "date", "index"):
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], utc=True)
                df = df.set_index(df.columns[0]).sort_index()
            else:
                # Assume unnamed index
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.sort_index()
            if gran in ("H1", "S5", "M5", "M15", "M30"):
                # Downsample to daily last
                df = df.resample('1D').last().dropna(how='all')
            # Add column prefixes by instrument+gran for uniqueness
            df = df.add_prefix(f"{inst}_{gran}_")
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No exported features found under {feat_dir} for grans {grans}")

    # Align on intersection of daily dates
    base_index = frames[0].index
    for f in frames[1:]:
        base_index = base_index.intersection(f.index)
    blocks: List[pd.DataFrame] = [frm.reindex(base_index).fillna(0.0) for frm in frames]
    X = pd.concat(blocks, axis=1).astype(np.float32)

    # Build y from local D OHLCV of EUR_USD
    d_loc = _load_local_ohlcv("EUR_USD", "D")
    if d_loc is None:
        # fallback REST
        d_loc = fetch_fx_range("EUR_USD", "D", cfg.start, cfg.end, cfg.environment, cfg.access_token)
    d_loc.index = pd.to_datetime(d_loc.index, utc=True).normalize()
    close = d_loc['close'].reindex(base_index).ffill()
    r = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = r.shift(-1).fillna(0.0)
    return X, y, base_index


def tensorize(df: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(df.values, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline train AE + single-output actor-critic with configurable reward (target EUR_USD)")
    p.add_argument("--instruments-csv", default=None)
    # Environment/token kept optional; will prefer local CSVs if present
    p.add_argument("--environment", choices=["practice","live"], default="practice")
    p.add_argument("--access-token", default=os.environ.get("OANDA_DEMO_KEY"))
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    # H1 default enabled; no flag required; kept for compatibility if provided
    p.add_argument("--include-h1", action="store_true")
    p.add_argument("--ae-latent", type=int, default=64)
    p.add_argument("--ae-hidden", default="2048,512,128")
    p.add_argument("--policy-hidden", type=int, default=256)
    p.add_argument("--value-hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ae-epochs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-sigma", type=float, default=0.3)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--reward", default="pnl", help="Reward name: pnl|sortino|sharpe or custom via code")
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--max-units", type=int, default=100)
    p.add_argument("--model-path", default="forex-rl/continuous-trader/checkpoints/offline_eurusd.pt")
    p.add_argument("--features", choices=["exported","compute"], default="exported", help="Use exported feature CSVs or compute on the fly")
    p.add_argument("--feature-grans", default="D", help="Comma-separated feature granularities to include when using exported (e.g., D or M5,H1,D)")
    p.add_argument("--feature-dir", default=os.path.join(FX_ROOT, "continuous-trader", "data", "features"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ae_hidden = tuple(int(x) for x in (args.ae_hidden or "").split(',') if x.strip())

    cfg = Config(
        instruments_csv=args.instruments_csv,
        environment=args.environment,
        access_token=args.access_token,
        start=args.start,
        end=args.end,
        include_h1=True,
        include_d=True,
        ae_latent=int(args.ae_latent),
        ae_hidden=list(ae_hidden) if len(ae_hidden) > 0 else [2048, 512, 128],
        policy_hidden=int(args.policy_hidden),
        value_hidden=int(args.value_hidden),
        batch_size=int(args.batch_size),
        ae_epochs=int(args.ae_epochs),
        epochs=int(args.epochs),
        gamma=float(args.gamma),
        actor_sigma=float(args.actor_sigma),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        max_grad_norm=float(args.max_grad_norm),
        reward_name=str(args.reward),
        reward_scale=float(args.reward_scale),
        model_path=str(args.model_path),
        max_units=int(args.max_units),
        seed=int(args.seed),
        feature_source=str(args.features),
        feature_grans=[s.strip().upper() for s in (args.feature_grans or "D").split(',') if s.strip()],
        feature_dir=str(args.feature_dir),
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    instruments = ct_instruments.load_68(cfg.instruments_csv)
    if "EUR_USD" not in instruments:
        instruments = ["EUR_USD"] + instruments

    print(json.dumps({"status": "build_features", "source": cfg.feature_source, "grans": cfg.feature_grans, "count": len(instruments)}), flush=True)
    if cfg.feature_source == "exported":
        X, y, dates = load_exported_feature_matrix(cfg, instruments)
    else:
        X, y, dates = build_dataset(cfg, instruments)

    Xn, stats = standardize_fit(X)

    # Train/val split by time
    n = len(Xn)
    if n < 200:
        raise RuntimeError("Not enough data to train; extend date range")
    split = int(n * 0.8)
    X_train = Xn.iloc[:split]
    y_train = y.iloc[:split]
    X_val = Xn.iloc[split:]
    y_val = y.iloc[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autoencoder pretrain
    ae = ct_model.AutoEncoder(input_dim=X_train.shape[1], hidden_dims=tuple(cfg.ae_hidden), latent_dim=cfg.ae_latent).to(device)
    opt_ae = optim.Adam(ae.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(torch.tensor(X_train.values, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    mse = nn.MSELoss()
    for epoch in range(cfg.ae_epochs):
        ae.train()
        total = 0.0
        nobs = 0
        for (xb,) in dl:
            xb = xb.to(device)
            _, recon = ae(xb)
            loss = mse(recon, xb)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
            total += float(loss.item()) * xb.size(0)
            nobs += xb.size(0)
        print(json.dumps({"phase": "ae", "epoch": epoch + 1, "loss_mse": total / max(1, nobs)}), flush=True)

    # Actor-Critic on daily steps with configurable reward function applied to scaled action
    model = ct_model.ActorCriticSingle(encoder=ae.encoder, latent_dim=cfg.ae_latent, policy_hidden=cfg.policy_hidden, value_hidden=cfg.value_hidden).to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    reg = ct_rewards.RewardRegistry()
    reward_fn = reg.get(cfg.reward_name)

    def step_epoch(Xb: pd.DataFrame, yb: pd.Series, train: bool) -> Dict[str, float]:
        model.train(train)
        X_t = torch.tensor(Xb.values, dtype=torch.float32).to(device)
        y_np = yb.values.astype(np.float32)
        T = X_t.size(0)
        total = {"loss": 0.0, "actor": 0.0, "value": 0.0, "entropy": 0.0, "reward": 0.0}
        steps = 0
        sigma = float(cfg.actor_sigma)
        log_sigma_const = float(np.log(sigma + 1e-8))
        for t in range(0, T - 1):
            s_t = X_t[t:t + 1]
            s_tp1 = X_t[t + 1:t + 2]
            r_next = float(y_np[t + 1])

            a_t, logit_t, v_t = model(s_t)
            with torch.no_grad():
                _, _, v_tp1 = model(s_tp1)

            # Interpret action a in [0,1) as position gating: map to signed position via thresholds offline?
            # For offline reward shaping, let signed_pos = (a - 0.5) * 2 in [-1,1]
            signed_pos = (a_t - 0.5) * 2.0
            inst_contrib = signed_pos[0] * float(cfg.max_units) * r_next

            # Reward shaping via registry (Sortino-like default)
            ctx = ct_rewards.RewardContext(inst_return=float(inst_contrib))
            r_scalar = cfg.reward_scale * float(reward_fn(ctx))

            adv = (r_scalar + cfg.gamma * v_tp1[0] - v_t[0]).detach()
            # Gaussian policy over logit with fixed sigma
            eps = torch.randn_like(logit_t)
            pre = logit_t + sigma * eps
            # log-prob under Normal
            logprob = -0.5 * (((pre - logit_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const)
            entropy = 0.5 * torch.log(2 * torch.tensor(np.pi) * (sigma ** 2))

            actor_loss = -adv * logprob.mean()
            value_target = (r_scalar + cfg.gamma * v_tp1[0]).detach()
            value_loss = cfg.value_coef * 0.5 * (value_target - v_t[0]) ** 2
            loss = actor_loss + value_loss - cfg.entropy_coef * entropy

            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

            total["loss"] += float(loss.item())
            total["actor"] += float(actor_loss.item())
            total["value"] += float(value_loss.item())
            total["entropy"] += float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)
            total["reward"] += float(r_scalar)
            steps += 1
        for k in total:
            total[k] = total[k] / max(1, steps)
        return total

    for epoch in range(cfg.epochs):
        tr = step_epoch(X_train, y_train, train=True)
        va = step_epoch(X_val, y_val, train=False)
        print(json.dumps({"phase": "train", "epoch": epoch + 1, **{f"tr_{k}": v for k, v in tr.items()}, **{f"va_{k}": v for k, v in va.items()}}), flush=True)

    # Save
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    torch.save({
        "cfg": asdict(cfg),
        "feature_stats": stats,
        "ae_encoder_state": ae.encoder.state_dict(),
        "policy_state": model.policy.state_dict(),
        "value_state": model.value.state_dict(),
        "meta": {"input_dim": X_train.shape[1], "latent_dim": cfg.ae_latent, "num_instruments": 1},
    }, cfg.model_path)
    print(json.dumps({"saved": cfg.model_path}), flush=True)


if __name__ == "__main__":
    main()
