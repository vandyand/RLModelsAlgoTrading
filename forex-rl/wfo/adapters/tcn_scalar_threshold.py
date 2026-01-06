from __future__ import annotations

"""
Single-instrument TCN + scalar-threshold strategy adapter for WFO.

- Data source: HTTP candle cache (`candle_cache_service.py`) via OANDA-style
  `/v3/instruments/{instrument}/candles` endpoints.
- Instruments: currently single Forex pair (e.g. EUR_USD).
- Granularities: multi-timeframe OHLCV (e.g. M5,H1,D).
- Model:
  - Temporal ConvNet over stacked per-granularity log-return / candle-shape
    features.
  - Outputs one scalar z_t per base bar (approximately N(0,1) after training).
- Decision:
  - Map z_t to {-1,0,+1} positions using scalar thresholds with hysteresis
    (enter/exit long/short).
- Metrics:
  - Uses `ga-multi20.compute_portfolio_metrics` to compute Sharpe, cum_return,
    drawdown, etc., per validation window.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

# Reuse portfolio metrics helper
_ROOT = Path(__file__).resolve().parents[2]
_GA_MULTI20_DIR = _ROOT / "ga-multi20"
import sys

if str(_GA_MULTI20_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_MULTI20_DIR))

from fitness import compute_portfolio_metrics  # type: ignore
from backtester import _extract_trades as _bt_extract_trades  # type: ignore

# Optional flag to skip some heavier per-window diagnostics (e.g. equity R^2).
_LIGHT_METRICS = os.environ.get("WFO_LIGHT_METRICS", "0") == "1"


# -----------------------------------------------------------------------------
# Small utility: column-wise standardization (same contract as ac_multi20)
# -----------------------------------------------------------------------------


def _standardize_fit(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    Xn = X.copy()
    for c in X.columns:
        m = float(X[c].mean())
        s = float(X[c].std())
        if not math.isfinite(m):
            m = 0.0
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        stats[c] = (m, s)
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32), stats


def _standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if not math.isfinite(m):
            m = 0.0
        if not math.isfinite(s) or s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


# -----------------------------------------------------------------------------
# HTTP candle loader (talks to candle_cache_service)
# -----------------------------------------------------------------------------


def _to_oanda_iso(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _fetch_candles_http(
    base_url: str,
    instrument: str,
    granularity: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price: str = "M",
) -> pd.DataFrame:
    """Fetch candles via HTTP cache and return DataFrame indexed by UTC time.

    Columns: ['o','h','l','c','v'] as floats.

    OANDA enforces an upper bound on the implicit bar `count` for from/to
    requests (currently ~5000 bars). When a requested [start,end] range would
    exceed that, the adapter chunks the request into multiple smaller ranges
    and stitches them together before returning.
    """

    if start >= end:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"]).astype(float)

    # Approximate bar seconds for a subset of granularities we care about.
    gran = granularity.upper()
    gran_seconds = {
        "S5": 5,
        "S10": 10,
        "S15": 15,
        "S30": 30,
        "M1": 60,
        "M2": 120,
        "M4": 240,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H3": 10800,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "H12": 43200,
        "D": 86400,
    }
    sec = gran_seconds.get(gran, 300)
    max_bars = 4000  # stay comfortably under OANDA's implicit max count

    total_sec = max(0.0, (end - start).total_seconds())
    est_bars = total_sec / float(sec) if sec > 0 else 0.0

    def _fetch_single(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        base = base_url.rstrip("/")
        url = f"{base}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": gran,
            "from": _to_oanda_iso(s),
            "to": _to_oanda_iso(e),
            "price": price,
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach candle cache at {url} — is `candle_cache_service` running?"
            ) from exc
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP {resp.status_code} from candle cache for {instrument} {gran}: {resp.text[:200]}"
            )
        payload = resp.json()
        candles = payload.get("candles", [])
        records: List[Dict[str, Any]] = []
        for c in candles:
            mid = c.get("mid") or {}
            try:
                t = pd.Timestamp(c.get("time"))
            except Exception:
                continue
            records.append(
                {
                    "time": t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC"),
                    "o": float(mid.get("o")) if mid.get("o") is not None else np.nan,
                    "h": float(mid.get("h")) if mid.get("h") is not None else np.nan,
                    "l": float(mid.get("l")) if mid.get("l") is not None else np.nan,
                    "c": float(mid.get("c")) if mid.get("c") is not None else np.nan,
                    "v": float(c.get("volume", 0.0)),
                }
            )
        if not records:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"]).astype(float)
        df = pd.DataFrame.from_records(records).set_index("time").sort_index()
        return df.astype(float)

    # Single request is safe
    if est_bars <= max_bars or sec <= 0:
        return _fetch_single(start, end)

    # Chunk into smaller from/to windows
    dfs: List[pd.DataFrame] = []
    # Use chunk length slightly below max_bars to be conservative
    chunk_bars = max_bars - 100
    chunk_delta = pd.Timedelta(seconds=sec * chunk_bars)
    cur = start
    while cur < end:
        nxt = cur + chunk_delta
        if nxt > end:
            nxt = end
        df_part = _fetch_single(cur, nxt)
        if not df_part.empty:
            dfs.append(df_part)
        cur = nxt

    if not dfs:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"]).astype(float)
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    return df_all


# -----------------------------------------------------------------------------
# TCN model
# -----------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """1D causal conv with explicit left padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + x  # residual


class TCNScalarHead(nn.Module):
    """Small TCN that maps multi-feature history to one scalar per sample."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(_TCNBlock(hidden_channels, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        out = self.head(h).squeeze(-1)  # (B,)
        return out


class TCNActionHead(nn.Module):
    """TCN head that outputs logits over {-1, 0, +1} actions."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(_TCNBlock(hidden_channels, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        logits = self.head(h)  # (B, num_classes)
        return logits


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------


@dataclass
class TCNScalarConfig:
    instrument: str = "EUR_USD"
    grans: str = "M5,H1,D"
    base_gran: str = "M5"
    candle_cache_base: str = "http://127.0.0.1:9100"
    lookback_bars: int = 128  # history length per sample at base gran
    target_horizon: int = 4  # bars ahead (base gran) for forward return label
    epochs: int = 5
    lr: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 5
    num_blocks: int = 3
    dropout: float = 0.1
    # NOTE: Thresholds are in z-scored space where labels are ~N(0,1).
    # Long side uses positive bands; short side must use *negative* bands.
    # Symmetric defaults: enter on |z| > 0.8, exit on |z| < 0.4.
    enter_long: float = 0.8
    exit_long: float = 0.4
    enter_short: float = -0.8
    exit_short: float = -0.4
    trade_cost_bps: float = 0.0  # per flip, in basis points


class TCNScalarThresholdAdapter:
    """TCN + scalar-threshold strategy for single-instrument Forex."""

    name = "tcn-scalar-threshold"

    def __init__(
        self,
        *,
        instrument: str = "EUR_USD",
        grans: str = "M5,H1,D",
        base_gran: str = "M5",
        candle_cache_base: str = "",
        lookback_bars: int = 128,
        target_horizon: int = 4,
        epochs: int = 5,
        lr: float = 1e-3,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
        dropout: float = 0.1,
        enter_long: float = 0.8,
        exit_long: float = 0.6,
        enter_short: float = 0.2,
        exit_short: float = 0.4,
        trade_cost_bps: float = 0.0,
    ) -> None:
        self.cfg = TCNScalarConfig(
            instrument=instrument.upper(),
            grans=grans,
            base_gran=base_gran.upper(),
            candle_cache_base=candle_cache_base
            or os.environ.get("CANDLE_CACHE_BASE", "http://127.0.0.1:9100"),
            lookback_bars=int(lookback_bars),
            target_horizon=int(target_horizon),
            epochs=int(epochs),
            lr=float(lr),
            hidden_channels=int(hidden_channels),
            kernel_size=int(kernel_size),
            num_blocks=int(num_blocks),
            dropout=float(dropout),
            enter_long=float(enter_long),
            exit_long=float(exit_long),
            enter_short=float(enter_short),
            exit_short=float(exit_short),
            trade_cost_bps=float(trade_cost_bps),
        )
        self.grans: List[str] = [g.strip().upper() for g in self.cfg.grans.split(",") if g.strip()]
        if self.cfg.base_gran not in self.grans:
            self.grans = [self.cfg.base_gran] + self.grans
        self._train_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._label_mean: float = 0.0
        self._label_std: float = 1.0
        self._model: Optional[TCNScalarHead] = None
        # For global-trade logging
        self._last_positions: Optional[pd.DataFrame] = None
        self._last_trades: Optional[List[Any]] = None
        # For global-trade logging
        self._last_positions: Optional[pd.DataFrame] = None
        self._last_trades: Optional[List[Any]] = None

    # ------------------------------------------------------------------
    # Adapter API
    # ------------------------------------------------------------------

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch candles for [start,end] and build per-bar feature panel.

        - Uses base_gran bars as the time index.
        - For slower granularities, features are forward-filled to base index.
        """
        # Add margin for lookback and target horizon
        margin_bars = max(self.cfg.lookback_bars + self.cfg.target_horizon + 2, 256)
        if self.cfg.base_gran == "H1":
            margin = pd.Timedelta(hours=margin_bars)
        elif self.cfg.base_gran == "D":
            margin = pd.Timedelta(days=margin_bars)
        else:
            # assume M5-like base; 5 minutes per bar
            margin = pd.Timedelta(minutes=5 * margin_bars)
        start_ext = start - margin
        end_ext = end

        # Fetch per-granularity candles
        candles_by_gran: Dict[str, pd.DataFrame] = {}
        for g in self.grans:
            df = _fetch_candles_http(
                self.cfg.candle_cache_base,
                self.cfg.instrument,
                g,
                start_ext,
                end_ext,
            )
            candles_by_gran[g] = df

        base = candles_by_gran[self.cfg.base_gran]
        if base.empty:
            # Return empty DataFrames; WFO core will log zero rows.
            idx = pd.DatetimeIndex([], tz="UTC")
            return pd.DataFrame(index=idx), pd.DataFrame(index=idx, columns=[self.cfg.instrument])

        # Restrict base to requested [start,end]
        base = base.loc[(base.index >= start) & (base.index <= end)].copy()
        # If there are no candles inside the requested window, treat this as an
        # empty window rather than attempting to build returns from an empty
        # price series (which would raise on logp[0]).
        if base.empty:
            idx = pd.DatetimeIndex([], tz="UTC")
            return pd.DataFrame(index=idx), pd.DataFrame(index=idx, columns=[self.cfg.instrument])
        idx = base.index

        # Build features per granularity at base index.
        # IMPORTANT: we ALWAYS create the same feature set for every granularity
        # in self.grans, even if a particular candle DataFrame is empty for a
        # given window. This guarantees that the feature dimensionality (C) is
        # identical between train and validation windows, which is critical for
        # TCN-based models that fix their input channel count at init time.
        feat_cols: Dict[str, pd.Series] = {}
        eps = 1e-12
        for g in self.grans:
            df = candles_by_gran.get(g)
            if df is None or df.empty:
                # No data for this granularity in this window: fall back to
                # flat/zero features so that the column set remains stable.
                r_s = pd.Series(0.0, index=idx)
                feat_cols[f"{g}_ret"] = r_s
                feat_cols[f"{g}_range"] = pd.Series(0.0, index=idx)
                feat_cols[f"{g}_shape"] = pd.Series(0.5, index=idx)
                feat_cols[f"{g}_logv"] = pd.Series(0.0, index=idx)
                for win in (4, 16):
                    feat_cols[f"{g}_ret_mean_{win}"] = pd.Series(0.0, index=idx)
                    feat_cols[f"{g}_ret_std_{win}"] = pd.Series(0.0, index=idx)
                continue

            # Reindex to base index using last known candle
            df_g = df[["o", "h", "l", "c", "v"]].copy()
            df_g = df_g.sort_index().reindex(idx, method="ffill")
            close = df_g["c"].astype(float)
            # log return
            logp = np.log(np.clip(close.values, eps, None))
            r = np.diff(logp, prepend=logp[0])
            r_s = pd.Series(r, index=idx)
            feat_cols[f"{g}_ret"] = r_s
            # range volatility
            rng = (df_g["h"] - df_g["l"]) / np.clip(close, eps, None)
            feat_cols[f"{g}_range"] = rng
            # candle shape (body position within range)
            span = (df_g["h"] - df_g["l"]).replace(0.0, np.nan)
            shape = (close - df_g["l"]) / span
            shape = shape.fillna(0.5)
            feat_cols[f"{g}_shape"] = shape
            # log volume
            vol = np.log(df_g["v"].values + 1.0)
            feat_cols[f"{g}_logv"] = pd.Series(vol, index=idx)

            # Lightweight "grid-like" feature expansion: add a couple of
            # short/medium-horizon rolling statistics of returns to give the
            # TCN more context on recent trend and volatility, while keeping
            # everything strictly causal.
            for win in (4, 16):
                roll = r_s.rolling(window=win, min_periods=1)
                feat_cols[f"{g}_ret_mean_{win}"] = roll.mean()
                feat_cols[f"{g}_ret_std_{win}"] = roll.std().fillna(0.0)

        # ------------------------------------------------------------------
        # Time-based cyclical features on the base-granularity index.
        #
        # We add sin/cos encodings for:
        #   - minute-of-hour      (0..59)
        #   - hour-of-day         (0..23)
        #   - trading-day-of-week (Mon..Fri mapped to 0..4; Sat/Sun clamped)
        #   - day-of-month        (1..31, normalized by 31)
        #   - month-of-year       (1..12)
        #
        # This adds 5 * 2 = 10 features, taking us from 24 -> 34 channels
        # for the common M5,H1,H4 setup.
        # ------------------------------------------------------------------
        ts = idx
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        minute = ts.minute.to_numpy()
        hour = ts.hour.to_numpy()
        dom = ts.day.to_numpy()
        month = ts.month.to_numpy()
        dow = ts.dayofweek.to_numpy()  # 0=Mon .. 6=Sun

        # Normalize to [0,1) style fractions
        minute_frac = minute / 60.0
        hour_frac = hour / 24.0
        # Clamp day-of-week to trading days only so Friday is 4/5 rather than 4/7.
        dow_trading = np.clip(dow, 0, 4)
        dow_frac = dow_trading / 5.0
        dom_frac = (dom - 1) / 31.0  # approximate across month lengths
        month_frac = (month - 1) / 12.0

        def _add_time_feature(name: str, frac: np.ndarray) -> None:
            angle = 2.0 * np.pi * frac
            feat_cols[f"time_{name}_sin"] = pd.Series(np.sin(angle), index=idx)
            feat_cols[f"time_{name}_cos"] = pd.Series(np.cos(angle), index=idx)

        _add_time_feature("minute", minute_frac)
        _add_time_feature("hour", hour_frac)
        _add_time_feature("dow_trading", dow_frac)
        _add_time_feature("dom", dom_frac)
        _add_time_feature("month", month_frac)

        X = pd.DataFrame(feat_cols, index=idx).astype(np.float32)
        closes = pd.DataFrame({"close": base["c"].astype(float)}, index=idx)
        closes.columns = [self.cfg.instrument]
        return X, closes

    def _build_sequences(
        self,
        Xn: pd.DataFrame,
        closes: pd.DataFrame,
        lookback: int,
        target_horizon: int,
        train: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert per-bar features into TCN sequences and z-scored labels."""
        values = Xn.values  # (T, F)
        px = closes[self.cfg.instrument].astype(float).values  # (T,)
        T, F = values.shape
        L = lookback
        H = target_horizon
        xs: List[np.ndarray] = []
        ys: List[float] = []
        # We require enough history and room for target horizon
        for t in range(L, T - H):
            window = values[t - L : t, :]  # (L, F)
            ret = math.log(max(px[t + H], 1e-12) / max(px[t], 1e-12))
            xs.append(window.T)  # (F, L)
            ys.append(ret)
        if not xs:
            return np.zeros((0, F, L), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        X_arr = np.stack(xs).astype(np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        if train:
            mu = float(y_arr.mean())
            sigma = float(y_arr.std())
            if not math.isfinite(mu):
                mu = 0.0
            if not math.isfinite(sigma) or sigma < 1e-6:
                sigma = 1.0
            self._label_mean = mu
            self._label_std = sigma
        mu = self._label_mean
        sigma = self._label_std if self._label_std > 0.0 else 1.0
        y_z = (y_arr - mu) / sigma
        return X_arr, y_z.astype(np.float32)

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        if X.empty or closes.empty:
            self._model = None
            return None
        # Standardize features
        Xn, stats = _standardize_fit(X)
        self._train_stats = stats
        # Build train sequences
        X_arr, y_arr = self._build_sequences(
            Xn,
            closes,
            lookback=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
            train=True,
        )
        if X_arr.shape[0] == 0:
            self._model = None
            return None
        B, C, L = X_arr.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TCNScalarHead(
            in_channels=C,
            hidden_channels=self.cfg.hidden_channels,
            kernel_size=self.cfg.kernel_size,
            num_blocks=self.cfg.num_blocks,
            dropout=self.cfg.dropout,
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)
        model.train()
        for ep in range(self.cfg.epochs):
            # Single full-batch pass per epoch for now (windows are modest)
            opt.zero_grad(set_to_none=True)
            pred = model(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        self._model = model
        return model

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        # Edge cases
        if model is None or X.empty or closes.empty or self._train_stats is None:
            return {
                "cum_return": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_dd": 0.0,
                "trades": 0.0,
                "time_in_mkt": 0.0,
                "win_rate": 0.0,
                "win_loss": 0.0,
                "profit_factor": 0.0,
                "equity_r2": 0.0,
            }
        Xn = _standardize_apply(X, self._train_stats)
        X_arr, _ = self._build_sequences(
            Xn,
            closes,
            lookback=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
            train=False,
        )
        idx = closes.index
        T = len(idx)
        positions = pd.Series(0.0, index=idx, name=self.cfg.instrument)
        if X_arr.shape[0] == 0:
            # Flat; metrics will be zero; also zero trades => Sharpe = 0
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            m = compute_portfolio_metrics(closes, pos_df, 0.0)
            # clear last-trades
            self._last_positions = pos_df
            self._last_trades = []
            return {
                "cum_return": float(m.cum_return),
                "sharpe": 0.0,
                "bar_sharpe": float(m.sharpe),
                "sortino": float(m.sortino),
                "max_dd": float(m.max_drawdown),
                "trades": float(m.trades),
                "time_in_mkt": float(m.time_in_market),
                "win_rate": float(m.win_rate),
                "win_loss": float(m.win_loss_ratio),
                "profit_factor": float(m.profit_factor),
                "equity_r2": float(m.equity_r2),
                # Diagnostics for empty-sequence windows
                "z_mean": 0.0,
                "z_std": 0.0,
                "z_min": 0.0,
                "z_max": 0.0,
                "z_frac_gt_enter_long": 0.0,
                "z_frac_lt_enter_short": 0.0,
                "pos_frac_long": 0.0,
                "pos_frac_short": 0.0,
                "pos_frac_flat": 1.0,
            }

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
            z = model(X_t).cpu().numpy()  # (N_seq,)

        # Map from sequence index to bar index; sequences start at bar L and end at T - H - 1
        L = self.cfg.lookback_bars
        H = self.cfg.target_horizon
        start_bar = L
        # We built sequences up to T-H-1, but X_arr.shape[0] gives exact count
        for i, z_i in enumerate(z):
            t = start_bar + i
            if t >= T:
                break
            prev_pos = positions.iloc[t - 1] if t > 0 else 0.0
            new_pos = prev_pos
            # Hysteresis thresholds on scalar z (long: high z, short: low z)
            if prev_pos == 0.0:
                if z_i > self.cfg.enter_long:
                    new_pos = 1.0
                elif z_i < self.cfg.enter_short:
                    new_pos = -1.0
            elif prev_pos > 0.0:
                if z_i < self.cfg.exit_long:
                    new_pos = 0.0
            elif prev_pos < 0.0:
                if z_i > self.cfg.exit_short:
                    new_pos = 0.0
            positions.iloc[t] = new_pos

        pos_df = pd.DataFrame({self.cfg.instrument: positions})
        self._last_positions = pos_df
        trade_cost = float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0
        m = compute_portfolio_metrics(closes, pos_df, trade_cost)

        # ------------------------------------------------------------------
        # Diagnostics: distribution of z_t and positions on validation set
        # ------------------------------------------------------------------
        try:
            z_mean = float(np.mean(z)) if z.size > 0 else 0.0
            z_std = float(np.std(z)) if z.size > 0 else 0.0
            z_min = float(np.min(z)) if z.size > 0 else 0.0
            z_max = float(np.max(z)) if z.size > 0 else 0.0
            frac_gt_enter_long = float(np.mean(z > self.cfg.enter_long)) if z.size > 0 else 0.0
            frac_lt_enter_short = float(np.mean(z < self.cfg.enter_short)) if z.size > 0 else 0.0
        except Exception:
            z_mean = z_std = z_min = z_max = frac_gt_enter_long = frac_lt_enter_short = 0.0

        pos_arr = pos_df.to_numpy(dtype=np.float32, copy=False)
        if pos_arr.size > 0:
            pos_frac_long = float((pos_arr > 0).mean())
            pos_frac_short = float((pos_arr < 0).mean())
            pos_frac_flat = float((pos_arr == 0).mean())
        else:
            pos_frac_long = pos_frac_short = 0.0
            pos_frac_flat = 1.0

        # Derive trade-level stats (win rate, profit factor, equity R^2) via GA backtester helpers.
        trades = _bt_extract_trades(pos_df, closes, trade_cost)
        self._last_trades = trades
        closed_trades = len(trades)
        pf = 0.0
        wl = 0.0
        wr = 0.0
        r2 = 0.0
        trade_sharpe = 0.0
        if trades:
            profits = np.array([t.net_pnl for t in trades], dtype=float)
            gains = profits[profits > 0]
            losses = -profits[profits < 0]
            if losses.size > 0:
                pf = float(gains.sum() / max(1e-12, losses.sum()))
            elif gains.size > 0:
                pf = float("inf")
            n_win = int((profits > 0).sum())
            n_loss = int((profits < 0).sum())
            if n_loss > 0:
                wl = float(n_win / max(1, n_loss))
            elif n_win > 0:
                wl = float("inf")
            wr = float(n_win / max(1, len(profits)))

            # Trade-based Sharpe: mean(net_pnl) / std(net_pnl) over trades
            mu = float(profits.mean()) if profits.size > 0 else 0.0
            sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
            if sigma > 1e-12:
                trade_sharpe = mu / sigma
            else:
                trade_sharpe = 0.0

            # Equity curve and R^2 vs linear trend over time (skip when light metrics enabled)
            if not _LIGHT_METRICS:
                px = closes.to_numpy(dtype=np.float32, copy=False)
                pos_np = pos_df.to_numpy(dtype=np.float32, copy=False)
                ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
                strat_ret = (pos_np[1:, :] * ret).mean(axis=1)
                if strat_ret.size > 0:
                    eq = np.cumprod(1.0 + strat_ret)
                    x = np.arange(eq.size, dtype=np.float64)
                    y = eq.astype(np.float64)
                    x_mean = x.mean()
                    y_mean = y.mean()
                    cov = float(((x - x_mean) * (y - y_mean)).sum())
                    varx = float(((x - x_mean) ** 2).sum())
                    if varx > 1e-12:
                        beta = cov / varx
                        alpha = y_mean - beta * x_mean
                        yhat = alpha + beta * x
                        ss_res = float(((y - yhat) ** 2).sum())
                        ss_tot = float(((y - y_mean) ** 2).sum())
                        r2 = float(1.0 - ss_res / max(1e-12, ss_tot)) if ss_tot > 1e-12 else 0.0
            # Long/short breakdown
            longs = [t for t in trades if t.side > 0]
            shorts = [t for t in trades if t.side < 0]
            long_trades = len(longs)
            short_trades = len(shorts)
            long_win = len([t for t in longs if t.net_pnl > 0.0])
            long_loss = len([t for t in longs if t.net_pnl < 0.0])
            short_win = len([t for t in shorts if t.net_pnl > 0.0])
            short_loss = len([t for t in shorts if t.net_pnl < 0.0])
        else:
            long_trades = short_trades = long_win = long_loss = short_win = short_loss = 0

        return {
            "cum_return": float(m.cum_return),
            # Sharpe is now trade-based; keep bar-based Sharpe separately.
            "sharpe": float(trade_sharpe),
            "bar_sharpe": float(m.sharpe),
            "sortino": float(m.sortino),
            "max_dd": float(m.max_drawdown),
            # Use closed trade count rather than flip count so that
            # trades == long_trades + short_trades per window.
            "trades": float(closed_trades),
            "time_in_mkt": float(m.time_in_market),
            "win_rate": float(wr),
            "win_loss": float(wl),
            "profit_factor": float(pf),
            "equity_r2": float(r2),
            "long_trades": float(long_trades),
            "short_trades": float(short_trades),
            "long_win_trades": float(long_win),
            "long_loss_trades": float(long_loss),
            "short_win_trades": float(short_win),
            "short_loss_trades": float(short_loss),
            # New diagnostics to understand scalar behavior
            "z_mean": z_mean,
            "z_std": z_std,
            "z_min": z_min,
            "z_max": z_max,
            "z_frac_gt_enter_long": frac_gt_enter_long,
            "z_frac_lt_enter_short": frac_lt_enter_short,
            "pos_frac_long": pos_frac_long,
            "pos_frac_short": pos_frac_short,
            "pos_frac_flat": pos_frac_flat,
        }


# -----------------------------------------------------------------------------
# Action-classification adapter (supervised {-1,0,+1} policy)
# -----------------------------------------------------------------------------


@dataclass
class TCNActionConfig:
    instrument: str = "EUR_USD"
    grans: str = "M5,H1,D"
    base_gran: str = "M5"
    candle_cache_base: str = "http://127.0.0.1:9100"
    lookback_bars: int = 128
    target_horizon: int = 4
    epochs: int = 5
    lr: float = 1e-3
    hidden_channels: int = 32
    kernel_size: int = 5
    num_blocks: int = 3
    dropout: float = 0.1
    # Labeling thresholds on forward return (base_gran horizon)
    label_long_thresh: float = 0.0005  # ~5 bp
    label_short_thresh: float = -0.0005
    # Optional alternative: quantile-based labeling on forward returns.
    # When label_mode == "quantile", we ignore label_*_thresh and instead
    # compute per-window forward-return quantiles and use those.
    label_mode: str = "bps"  # "bps" | "quantile"
    label_long_quantile: float = 0.7  # e.g. 0.7..0.9
    label_short_quantile: float = 0.3  # e.g. 0.1..0.3
    trade_cost_bps: float = 0.0
    # Optional MLP policy-head layout for TCNActionNQAdapter-style adapters.
    # When provided, this is interpreted as a sequence of layer sizes for the
    # policy MLP operating on the backbone hidden state. For example:
    #   mlp_layers: [32, 16, 8]
    # yields:
    #   hidden_dim -> 32 -> 16 -> 8 -> 1
    # with ReLU + Dropout between all hidden layers and tanh on the final scalar
    # output. If None or empty, a single hidden layer of size hidden_channels is
    # used (the original behavior).
    mlp_layers: Optional[List[int]] = None
    # NQ-only debug flag: when True (only set in NQ configs), the NQ adapter
    # can override labels (e.g. force all-long) for plumbing tests.
    debug_force_all_long: bool = False
    # NQ-only hyperparameters for the causal NQ override in TCNActionNQAdapter.
    # - nq_ret_thresh: minimum absolute horizon log-return to consider trading.
    # - nq_vol_lookback: lookback (in base bars) for rolling volatility.
    # - nq_z_thresh: minimum absolute z-score (horizon return / vol) to trade.
    # - nq_trend_fast / nq_trend_slow: fast/slow EMA spans (in base bars) used
    #   for the higher-timeframe trend filter.
    # - nq_trend_eps: optional relative tolerance band around the slow EMA
    #   before declaring up/down trend.
    nq_ret_thresh: float = 0.0005
    nq_vol_lookback: int = 64
    nq_z_thresh: float = 1.0
    nq_trend_fast: int = 96
    nq_trend_slow: int = 288
    nq_trend_eps: float = 0.0005
    # NQ-only policy-head hysteresis thresholds for continuous-action TCN+MLP
    # adapters. The base FX classifier ignores these, but NQ-specific adapters
    # (e.g. TCNActionNQAdapter) can consume them.
    #
    # Defaults are deliberately very close to zero so that even weak policy
    # signals produce trades during early debugging:
    #   - flat -> long when a_t >= +0.02
    #   - flat -> short when a_t <= -0.02
    #   - long -> flat when a_t <= 0.0
    #   - short -> flat when a_t >= 0.0
    nq_enter_long: float = 0.02
    nq_exit_long: float = 0.0
    nq_enter_short: float = -0.02
    nq_exit_short: float = 0.0


class TCNActionAdapter:
    """TCN-based supervised action classifier over {-1,0,+1} positions."""

    name = "tcn-action-classifier"

    def __init__(
        self,
        *,
        instrument: str = "EUR_USD",
        grans: str = "M5,H1,D",
        base_gran: str = "M5",
        candle_cache_base: str = "",
        lookback_bars: int = 128,
        target_horizon: int = 4,
        epochs: int = 5,
        lr: float = 1e-3,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_blocks: int = 3,
        dropout: float = 0.1,
        label_long_thresh: float = 0.0005,
        label_short_thresh: float = -0.0005,
        label_mode: str = "bps",
        label_long_quantile: float = 0.7,
        label_short_quantile: float = 0.3,
        trade_cost_bps: float = 0.0,
        debug_force_all_long: bool = False,
        nq_ret_thresh: float = 0.0005,
        nq_vol_lookback: int = 64,
        nq_z_thresh: float = 1.0,
        nq_trend_fast: int = 96,
        nq_trend_slow: int = 288,
        nq_trend_eps: float = 0.0005,
        nq_enter_long: float = 0.02,
        nq_exit_long: float = 0.0,
        nq_enter_short: float = -0.02,
        nq_exit_short: float = 0.0,
        mlp_layers: Optional[List[int]] = None,
    ) -> None:
        self.cfg = TCNActionConfig(
            instrument=instrument.upper(),
            grans=grans,
            base_gran=base_gran.upper(),
            candle_cache_base=candle_cache_base
            or os.environ.get("CANDLE_CACHE_BASE", "http://127.0.0.1:9100"),
            lookback_bars=int(lookback_bars),
            target_horizon=int(target_horizon),
            epochs=int(epochs),
            lr=float(lr),
            hidden_channels=int(hidden_channels),
            kernel_size=int(kernel_size),
            num_blocks=int(num_blocks),
            dropout=float(dropout),
            label_long_thresh=float(label_long_thresh),
            label_short_thresh=float(label_short_thresh),
            label_mode=str(label_mode),
            label_long_quantile=float(label_long_quantile),
            label_short_quantile=float(label_short_quantile),
            trade_cost_bps=float(trade_cost_bps),
            debug_force_all_long=bool(debug_force_all_long),
            nq_ret_thresh=float(nq_ret_thresh),
            nq_vol_lookback=int(nq_vol_lookback),
            nq_z_thresh=float(nq_z_thresh),
            nq_trend_fast=int(nq_trend_fast),
            nq_trend_slow=int(nq_trend_slow),
            nq_trend_eps=float(nq_trend_eps),
            nq_enter_long=float(nq_enter_long),
            nq_exit_long=float(nq_exit_long),
            nq_enter_short=float(nq_enter_short),
            nq_exit_short=float(nq_exit_short),
            mlp_layers=list(mlp_layers) if mlp_layers is not None else None,
        )
        self.grans: List[str] = [g.strip().upper() for g in self.cfg.grans.split(",") if g.strip()]
        if self.cfg.base_gran not in self.grans:
            self.grans = [self.cfg.base_gran] + self.grans
        self._train_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self._model: Optional[TCNActionHead] = None

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Reuse the scalar adapter's feature construction
        scalar_cfg = TCNScalarConfig(
            instrument=self.cfg.instrument,
            grans=self.cfg.grans,
            base_gran=self.cfg.base_gran,
            candle_cache_base=self.cfg.candle_cache_base,
            lookback_bars=self.cfg.lookback_bars,
            target_horizon=self.cfg.target_horizon,
        )
        tmp = TCNScalarThresholdAdapter(
            instrument=scalar_cfg.instrument,
            grans=scalar_cfg.grans,
            base_gran=scalar_cfg.base_gran,
            candle_cache_base=scalar_cfg.candle_cache_base,
            lookback_bars=scalar_cfg.lookback_bars,
            target_horizon=scalar_cfg.target_horizon,
        )
        return tmp.load_window(start, end)

    def _build_sequences_and_labels(
        self,
        Xn: pd.DataFrame,
        closes: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert per-bar features into TCN sequences and discrete action labels.

        Semantic classes (external):
          -1 -> short
           0 -> flat
          +1 -> long

        Internally, these will later be shifted to indices {0,1,2} for use with
        cross-entropy loss, but this helper returns labels in {-1,0,+1} for
        clarity and easier debugging.
        """
        values = Xn.values  # (T, F)
        px = closes[self.cfg.instrument].astype(float).values  # (T,)
        T, F = values.shape
        L = self.cfg.lookback_bars
        H = self.cfg.target_horizon

        windows: List[np.ndarray] = []
        rets: List[float] = []
        for t in range(L, T - H):
            window = values[t - L : t, :]  # (L, F)
            ret = math.log(max(px[t + H], 1e-12) / max(px[t], 1e-12))
            windows.append(window.T)
            rets.append(ret)

        if not windows:
            return np.zeros((0, F, L), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        rets_arr = np.asarray(rets, dtype=float)
        ys: List[int] = []

        if self.cfg.label_mode == "quantile":
            # Compute per-window forward-return quantiles.
            try:
                long_q = float(np.quantile(rets_arr, self.cfg.label_long_quantile))
                short_q = float(np.quantile(rets_arr, self.cfg.label_short_quantile))
            except Exception:
                long_q = float(np.quantile(rets_arr, 0.7))
                short_q = float(np.quantile(rets_arr, 0.3))
            # Fallback if quantiles are degenerate or inverted.
            if not math.isfinite(long_q) or not math.isfinite(short_q) or short_q >= long_q:
                long_q = float(np.percentile(rets_arr, 66))
                short_q = float(np.percentile(rets_arr, 33))

            for ret in rets_arr:
                if ret >= long_q:
                    ys.append(1)   # long
                elif ret <= short_q:
                    ys.append(-1)  # short
                else:
                    ys.append(0)   # flat
        else:
            # Basis-point style absolute thresholds.
            long_t = float(self.cfg.label_long_thresh)
            short_t = float(self.cfg.label_short_thresh)
            for ret in rets_arr:
                if ret >= long_t:
                    ys.append(1)   # long
                elif ret <= short_t:
                    ys.append(-1)  # short
                else:
                    ys.append(0)   # flat

        X_arr = np.stack(windows).astype(np.float32)
        y_arr = np.asarray(ys, dtype=np.int64)
        return X_arr, y_arr

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        if X.empty or closes.empty:
            self._model = None
            return None
        Xn, stats = _standardize_fit(X)
        self._train_stats = stats
        X_arr, y_arr = self._build_sequences_and_labels(Xn, closes)
        if X_arr.shape[0] == 0:
            self._model = None
            return None

        # Map semantic labels {-1,0,+1} to class indices {0,1,2} for CE loss.
        y_idx = (y_arr + 1).astype(np.int64)

        B, C, L = X_arr.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TCNActionHead(
            in_channels=C,
            hidden_channels=self.cfg.hidden_channels,
            kernel_size=self.cfg.kernel_size,
            num_blocks=self.cfg.num_blocks,
            dropout=self.cfg.dropout,
            num_classes=3,
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=self.cfg.lr)
        X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=device)
        model.train()
        for ep in range(self.cfg.epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(X_t)
            loss = nn.functional.cross_entropy(logits, y_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        self._model = model
        return model

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        if model is None or X.empty or closes.empty or self._train_stats is None:
            return {
                "cum_return": 0.0,
                "sharpe": 0.0,
                "bar_sharpe": 0.0,
                "sortino": 0.0,
                "max_dd": 0.0,
                "trades": 0.0,
                "time_in_mkt": 0.0,
                "win_rate": 0.0,
                "win_loss": 0.0,
                "profit_factor": 0.0,
                "equity_r2": 0.0,
                "long_trades": 0.0,
                "short_trades": 0.0,
                "long_win_trades": 0.0,
                "long_loss_trades": 0.0,
                "short_win_trades": 0.0,
                "short_loss_trades": 0.0,
            }
        Xn = _standardize_apply(X, self._train_stats)
        X_arr, _ = self._build_sequences_and_labels(Xn, closes)
        idx = closes.index
        T = len(idx)
        positions = pd.Series(0.0, index=idx, name=self.cfg.instrument)
        if X_arr.shape[0] == 0:
            pos_df = pd.DataFrame({self.cfg.instrument: positions})
            m = compute_portfolio_metrics(closes, pos_df, 0.0)
            self._last_positions = pos_df
            self._last_trades = []
            return {
                "cum_return": float(m.cum_return),
                "sharpe": 0.0,
                "bar_sharpe": float(m.sharpe),
                "sortino": float(m.sortino),
                "max_dd": float(m.max_drawdown),
                "trades": 0.0,
                "time_in_mkt": float(m.time_in_market),
                "win_rate": 0.0,
                "win_loss": 0.0,
                "profit_factor": 0.0,
                "equity_r2": float(m.equity_r2),
                "long_trades": 0.0,
                "short_trades": 0.0,
                "long_win_trades": 0.0,
                "long_loss_trades": 0.0,
                "short_win_trades": 0.0,
                "short_loss_trades": 0.0,
            }

        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)
            logits = model(X_t)  # (N_seq, 3)
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # class indices 0,1,2

        # Map from sequence index to bar index; sequences end at t and action applies from t onward.
        L = self.cfg.lookback_bars
        H = self.cfg.target_horizon
        start_bar = L
        for i, cls_idx in enumerate(preds):
            t = start_bar + i
            if t >= T:
                break
            # Map class index {0,1,2} back to semantic label {-1,0,+1}.
            cls = int(cls_idx) - 1
            if cls == -1:
                new_pos = -1.0
            elif cls == 0:
                new_pos = 0.0
            else:
                new_pos = 1.0
            positions.iloc[t] = new_pos

        pos_df = pd.DataFrame({self.cfg.instrument: positions})
        self._last_positions = pos_df
        trade_cost = float(self.cfg.trade_cost_bps) / 1e4 if self.cfg.trade_cost_bps else 0.0
        m = compute_portfolio_metrics(closes, pos_df, trade_cost)
        trades = _bt_extract_trades(pos_df, closes, trade_cost)
        self._last_trades = trades
        closed_trades = len(trades)
        pf = 0.0
        wl = 0.0
        wr = 0.0
        r2 = 0.0
        trade_sharpe = 0.0
        if trades:
            profits = np.array([t.net_pnl for t in trades], dtype=float)
            gains = profits[profits > 0]
            losses = -profits[profits < 0]
            if losses.size > 0:
                pf = float(gains.sum() / max(1e-12, losses.sum()))
            elif gains.size > 0:
                pf = float("inf")
            n_win = int((profits > 0).sum())
            n_loss = int((profits < 0).sum())
            if n_loss > 0:
                wl = float(n_win / max(1, n_loss))
            elif n_win > 0:
                wl = float("inf")
            wr = float(n_win / max(1, len(profits)))

            # Trade-based Sharpe: mean(net_pnl) / std(net_pnl) over trades
            mu = float(profits.mean()) if profits.size > 0 else 0.0
            sigma = float(profits.std(ddof=1)) if profits.size > 1 else 0.0
            if sigma > 1e-12:
                trade_sharpe = mu / sigma
            else:
                trade_sharpe = 0.0

            # Equity curve and R^2 vs linear trend over time
            px = closes.to_numpy(dtype=np.float32, copy=False)
            pos_np = pos_df.to_numpy(dtype=np.float32, copy=False)
            ret = (px[1:, :] / np.clip(px[:-1, :], 1e-12, None) - 1.0).astype(np.float32)
            strat_ret = (pos_np[1:, :] * ret).mean(axis=1)
            if strat_ret.size > 0:
                eq = np.cumprod(1.0 + strat_ret)
                x = np.arange(eq.size, dtype=np.float64)
                y = eq.astype(np.float64)
                x_mean = x.mean()
                y_mean = y.mean()
                cov = float(((x - x_mean) * (y - y_mean)).sum())
                varx = float(((x - x_mean) ** 2).sum())
                if varx > 1e-12:
                    beta = cov / varx
                    alpha = y_mean - beta * x_mean
                    yhat = alpha + beta * x
                    ss_res = float(((y - yhat) ** 2).sum())
                    ss_tot = float(((y - y_mean) ** 2).sum())
                    r2 = float(1.0 - ss_res / max(1e-12, ss_tot)) if ss_tot > 1e-12 else 0.0
            longs = [t for t in trades if t.side > 0]
            shorts = [t for t in trades if t.side < 0]
            long_trades = len(longs)
            short_trades = len(shorts)
            long_win = len([t for t in longs if t.net_pnl > 0.0])
            long_loss = len([t for t in longs if t.net_pnl < 0.0])
            short_win = len([t for t in shorts if t.net_pnl > 0.0])
            short_loss = len([t for t in shorts if t.net_pnl < 0.0])
        else:
            long_trades = short_trades = long_win = long_loss = short_win = short_loss = 0

        return {
            "cum_return": float(m.cum_return),
            "sharpe": float(trade_sharpe),
            "bar_sharpe": float(m.sharpe),
            "sortino": float(m.sortino),
            "max_dd": float(m.max_drawdown),
            "trades": float(closed_trades),
            "time_in_mkt": float(m.time_in_market),
            "win_rate": float(wr),
            "win_loss": float(wl),
            "profit_factor": float(pf),
            "equity_r2": float(r2),
            "long_trades": float(long_trades),
            "short_trades": float(short_trades),
            "long_win_trades": float(long_win),
            "long_loss_trades": float(long_loss),
            "short_win_trades": float(short_win),
            "short_loss_trades": float(short_loss),
        }


