from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Protocol, Callable
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
import time
import sys
import importlib
import hashlib
import pickle
from pathlib import Path


class WFOAdapter(Protocol):
    """Strategy adapter interface for generic Walk-Forward Optimization.

    An adapter encapsulates how to:
    - build features/targets for a window
    - train a model on the train window
    - run validation to produce positions or metrics
    - compute and return standardized metrics
    """

    name: str

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (X_panel, closes) aligned to base step index for [start,end].
        X_panel can be MultiIndex columns or flat; adapter must be consistent internally.
        """
        ...

    def fit(self, X: pd.DataFrame, closes: pd.DataFrame) -> Any:
        """Train underlying model; return a model handle/state."""
        ...

    def validate(self, model: Any, X: pd.DataFrame, closes: pd.DataFrame) -> Dict[str, float]:
        """Run validation; return standardized metrics.
        Required keys (if applicable):
          - cum_return, sharpe, sortino, max_dd, profit_factor, win_rate, win_loss,
            trades, equity_r2
        Adapters may add extra keys. Values should be floats (NaN allowed).
        """
        ...


@dataclass
class WFOConfig:
    start: str
    end: str
    train_n: float
    val_n: float
    step_n: float
    unit: str = "months"  # one of: months,weeks,days,steps
    windows_limit: int = 0
    base_gran: str = "M5"
    out_dir: str = "wfo/runs"
    adapter_params: Optional[Dict[str, Any]] = None
    # Adapter instantiation for parallelism
    adapter_spec: Optional[str] = None  # e.g., "wfo.adapters.ac_multi20:ACMulti20Adapter"
    adapter_kwargs: Optional[Dict[str, Any]] = None
    # Runtime knobs
    parallel: int = 1
    no_chart: bool = False
    chart_every: int = 1
    quiet: bool = False
    preload_next: bool = False
    mode: str = "thorough"
    # When True, ignore `start` / `end` and instead construct windows by
    # walking backwards from "yesterday" using the geometry
    # (train_n/val_n/step_n) and `windows_limit`, then run them in
    # chronological order. This is useful for "most recent N windows"
    # sweeps without hard-coding dates in configs.
    rolling_from_yesterday: bool = False


def _downsample(y: List[float], target: int) -> List[float]:
    if target <= 0 or len(y) <= target:
        return list(y)
    xs = np.linspace(0, len(y) - 1, target)
    return [float(y[int(round(x))]) for x in xs]


def _ascii_chart(y: List[float], width: int = 80, height: int = 10) -> str:
    if not y:
        return ""
    y_ds = _downsample(y, max(2, int(width)))
    mn = min(y_ds); mx = max(y_ds)
    if abs(mx - mn) < 1e-12:
        mn = mx - 1.0
    # Normalize to 0..height-1 (invert for top-origin)
    scale = (height - 1) / (mx - mn)
    rows = [[" "] * len(y_ds) for _ in range(height)]
    for i, v in enumerate(y_ds):
        r = int(round((v - mn) * scale))
        r = max(0, min(height - 1, r))
        rows[height - 1 - r][i] = "▇"
    return "\n".join("".join(row) for row in rows)


def _render_chart(y: List[float], width: int = 80, height: int = 10) -> str:
    """Render a small terminal chart.
    Prefers asciichartpy if available; falls back to a simple block chart.
    """
    try:
        ac = importlib.import_module("asciichartpy")
    except Exception:
        ac = None
    # If equity is close to 1.0 with tiny range, convert to basis points for readability
    y_use = list(y)
    if y_use:
        mn = min(y_use); mx = max(y_use)
        if 0.9 <= mn <= 1.1 and (mx - mn) < 0.02:
            y_use = [ (v - 1.0) * 1e4 for v in y_use ]  # bp
    if ac is not None:
        try:
            y_ds = _downsample(y_use, max(2, int(width)))
            # Choose formatter
            rng = (max(y_ds) - min(y_ds)) if y_ds else 0.0
            mag = max(abs(max(y_ds)), abs(min(y_ds))) if y_ds else 0.0
            if rng < 1e-6 or mag < 1e-3 or mag >= 1e6:
                fmt = (lambda x, i=None: f"{x:.3e}")
            elif mag < 10:
                fmt = (lambda x, i=None: f"{x:.3f}")
            elif mag < 1000:
                fmt = (lambda x, i=None: f"{x:.1f}")
            else:
                fmt = (lambda x, i=None: f"{x:.0f}")
            return ac.plot(y_ds, {"height": int(height), "format": fmt})  # type: ignore[attr-defined]
        except Exception:
            pass
    return _ascii_chart(y_use, width=width, height=height)


# -----------------------------------------------------------------------------
# Optional on-disk window cache for adapter.load_window
# -----------------------------------------------------------------------------

_WINDOW_CACHE_ENABLED = os.environ.get("WFO_WINDOW_CACHE", "1") == "1"
_WINDOW_CACHE_DIR = Path(__file__).resolve().parent / "cache"
# Bump this token whenever the feature construction / adapter semantics
# change in a way that alters the shape or meaning of cached windows, so
# that we don't accidentally reuse stale pickles with mismatched channel
# counts (e.g. after adding new time-based features).
_WINDOW_CACHE_VERSION = "v2-timefeat"


def _load_window_cached(adapter: "WFOAdapter", start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load (X, closes) for a window, with optional on-disk caching.

    Cache key is based on adapter name, instrument/base_gran (if available),
    and the [start,end] timestamps.
    """
    if not _WINDOW_CACHE_ENABLED:
        return adapter.load_window(start, end)

    try:
        _WINDOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Build a reasonably unique key
        parts: List[str] = []
        parts.append(_WINDOW_CACHE_VERSION)
        name = getattr(adapter, "name", adapter.__class__.__name__)
        parts.append(str(name))
        cfg = getattr(adapter, "cfg", None)
        inst = getattr(cfg, "instrument", None) if cfg is not None else None
        base_gran = getattr(cfg, "base_gran", None) if cfg is not None else None
        if inst:
            parts.append(str(inst))
        if base_gran:
            parts.append(str(base_gran))
        parts.append(start.isoformat())
        parts.append(end.isoformat())
        raw_key = "|".join(parts)
        h = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
        path = _WINDOW_CACHE_DIR / f"{h}.pkl"

        if path.exists():
            with path.open("rb") as f:
                X, C = pickle.load(f)
            return X, C

        X, C = adapter.load_window(start, end)
        # Best-effort cache write
        try:
            tmp = path.with_suffix(".pkl.tmp")
            with tmp.open("wb") as f:
                pickle.dump((X, C), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            pass
        return X, C
    except Exception:
        # Fallback to direct load on any cache error
        return adapter.load_window(start, end)


def _add_offset(ts: pd.Timestamp, n: float, unit: str, base_gran: str = "M5") -> pd.Timestamp:
    if unit == "steps":
        steps = max(1, int(round(float(n))))
        if base_gran == "H1":
            delta = pd.Timedelta(hours=steps)
        elif base_gran == "D":
            delta = pd.Timedelta(days=steps)
        else:
            delta = pd.Timedelta(minutes=5 * steps)
        return (ts + delta)
    if unit == "days":
        return ts + pd.Timedelta(days=float(n))
    if unit == "weeks":
        return ts + pd.Timedelta(days=7.0 * float(n))
    # months default ~30d
    return ts + pd.Timedelta(days=30.0 * float(n))


def _instantiate_adapter(adapter_spec: str, adapter_kwargs: Optional[Dict[str, Any]] = None) -> WFOAdapter:
    mod_name, cls_name = adapter_spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**(adapter_kwargs or {}))


def _process_window(
    wi: int,
    ts_tr_s_iso: str,
    ts_tr_e_iso: str,
    ts_v_s_iso: str,
    ts_v_e_iso: str,
    adapter_spec: str,
    adapter_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ts_tr_s = pd.Timestamp(ts_tr_s_iso)
    ts_tr_e = pd.Timestamp(ts_tr_e_iso)
    ts_v_s = pd.Timestamp(ts_v_s_iso)
    ts_v_e = pd.Timestamp(ts_v_e_iso)
    adapter = _instantiate_adapter(adapter_spec, adapter_kwargs)
    t0 = time.time()
    X_tr, C_tr = _load_window_cached(adapter, ts_tr_s, ts_tr_e)
    model = adapter.fit(X_tr, C_tr)
    X_val, C_val = _load_window_cached(adapter, ts_v_s, ts_v_e)
    metrics = adapter.validate(model, X_val, C_val)
    dt = round(time.time() - t0, 3)
    return {
        "win": wi,
        "train": [ts_tr_s_iso, ts_tr_e_iso],
        "val": [ts_v_s_iso, ts_v_e_iso],
        "metrics": metrics,
        "sec": dt,
        "rows": {"train": int(len(X_tr)), "val": int(len(X_val))},
    }


def generate_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_n: float,
    val_n: float,
    step_n: float,
    unit: str = "months",
    base_gran: str = "M5",
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cur_train_start = start
    while True:
        cur_train_end = _add_offset(cur_train_start, float(train_n), unit, base_gran) - pd.Timedelta(seconds=1)
        cur_val_start = cur_train_end + pd.Timedelta(seconds=1)
        cur_val_end = _add_offset(cur_val_start, float(val_n), unit, base_gran) - pd.Timedelta(seconds=1)
        if cur_val_end > end:
            break
        windows.append((cur_train_start, cur_train_end, cur_val_start, cur_val_end))
        cur_train_start = _add_offset(cur_train_start, float(step_n), unit, base_gran)
    return windows


def run_wfo(adapter: WFOAdapter, cfg: WFOConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"{adapter.name}-wfo-{ts}")
    os.makedirs(run_dir, exist_ok=True)

    start_ts = pd.Timestamp(cfg.start, tz="UTC")
    end_ts = pd.Timestamp(cfg.end, tz="UTC")

    # Optional dynamic geometry: build windows_limit windows ending on
    # "yesterday" instead of using fixed start/end dates.
    if getattr(cfg, "rolling_from_yesterday", False):
        if cfg.windows_limit <= 0:
            raise ValueError("rolling_from_yesterday requires windows_limit > 0")
        # Anchor validation end at the end of "yesterday" in UTC.
        today_utc = pd.Timestamp.now(tz="UTC").normalize()
        anchor_val_end = today_utc - pd.Timedelta(seconds=1)  # yesterday 23:59:59
        # Total span from first train start to last val end in units.
        total_span_units = float(cfg.train_n) + float(cfg.val_n) + float(cfg.step_n) * float(
            max(0, cfg.windows_limit - 1)
        )
        # Invert _add_offset by using a negative total span.
        start_ts = _add_offset(
            anchor_val_end + pd.Timedelta(seconds=1),
            -total_span_units,
            cfg.unit,
            cfg.base_gran,
        )
        end_ts = anchor_val_end

    windows = generate_windows(
        start_ts,
        end_ts,
        cfg.train_n,
        cfg.val_n,
        cfg.step_n,
        unit=cfg.unit,
        base_gran=cfg.base_gran,
    )
    if cfg.windows_limit > 0:
        windows = windows[: cfg.windows_limit]

    meta = {
        "adapter": adapter.name,
        "cfg": {
            "start": cfg.start,
            "end": cfg.end,
            "train_n": cfg.train_n,
            "val_n": cfg.val_n,
            "step_n": cfg.step_n,
            "unit": cfg.unit,
            "base_gran": cfg.base_gran,
            "windows": len(windows),
        },
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    trades_path = os.path.join(run_dir, "trades.jsonl")

    # Console: run start
    if not cfg.quiet:
        print(json.dumps({
        "event": "wfo_start",
        "adapter": adapter.name,
        "run_dir": run_dir,
        "windows": len(windows),
        "unit": cfg.unit,
        "train_n": cfg.train_n,
        "val_n": cfg.val_n,
        "step_n": cfg.step_n,
        "base_gran": cfg.base_gran,
    }), flush=True)

    out_path = os.path.join(run_dir, "windows.jsonl")
    equity_points: List[float] = []
    with open(out_path, "w") as outf, open(trades_path, "w") as tlf:
        if int(cfg.parallel) > 1:
            # Parallel execution of windows (no carry-forward assumption)
            from concurrent.futures import ProcessPoolExecutor, as_completed
            spec = cfg.adapter_spec or f"{adapter.__module__}:{adapter.__class__.__name__}"
            adapter_kwargs = cfg.adapter_kwargs or {}
            jobs = []
            with ProcessPoolExecutor(max_workers=int(cfg.parallel)) as ex:
                for wi, (ts_tr_s, ts_tr_e, ts_v_s, ts_v_e) in enumerate(windows, start=1):
                    if not cfg.quiet:
                        print(json.dumps({"event": "win_begin", "win": wi}), flush=True)
                    jobs.append(ex.submit(
                        _process_window,
                        wi,
                        ts_tr_s.isoformat(), ts_tr_e.isoformat(), ts_v_s.isoformat(), ts_v_e.isoformat(),
                        spec,
                        adapter_kwargs,
                    ))
                for fut in as_completed(jobs):
                    res = fut.result()
                    wi = int(res["win"])
                    metrics = res["metrics"]
                    # Update equity curve
                    try:
                        v_cum = float(metrics.get("cum_return", 0.0))
                    except Exception:
                        v_cum = 0.0
                    last_eq = (equity_points[-1] if equity_points else 1.0)
                    equity_points.append(last_eq * (1.0 + v_cum))

                    rec = {"event": "window", "win": wi, "train": res["train"], "val": res["val"], **metrics}
                    for k in ["cum_return", "max_dd"]:
                        if k in metrics and isinstance(metrics[k], (int, float)) and not np.isnan(metrics[k]):
                            rec[f"{k}_bp"] = float(metrics[k]) * 1e4
                    outf.write(json.dumps(rec) + "\n"); outf.flush()

                    if not cfg.quiet:
                        print(json.dumps({
                            "event": "win_done", "win": wi, "sec": res.get("sec", None),
                            "v_cum": metrics.get("cum_return", None),
                        }), flush=True)
                        sess_eq = float(equity_points[-1]) if equity_points else 1.0
                        sess_cum = float(sess_eq - 1.0)
                        print(json.dumps({"event": "session", "win": wi, "cum_return": sess_cum, "cum_return_bp": sess_cum * 1e4}), flush=True)
                        if (not cfg.no_chart) and (wi % max(1, int(cfg.chart_every)) == 0):
                            chart = _render_chart(equity_points, width=60, height=8)
                            if chart:
                                print(chart, flush=True)
        else:
            for wi, (ts_tr_s, ts_tr_e, ts_v_s, ts_v_e) in enumerate(windows, start=1):
                if not cfg.quiet:
                    print(
                        json.dumps(
                            {
                                "event": "win_begin",
                                "win": wi,
                                "train": [ts_tr_s.isoformat(), ts_tr_e.isoformat()],
                                "val": [ts_v_s.isoformat(), ts_v_e.isoformat()],
                            }
                        ),
                        flush=True,
                    )

                t0 = time.time()
                X_tr, C_tr = _load_window_cached(adapter, ts_tr_s, ts_tr_e)
                if not cfg.quiet:
                    print(
                        json.dumps(
                            {
                                "event": "train_loaded",
                                "win": wi,
                                "rows": int(len(X_tr)),
                                "cols": int(X_tr.shape[1]) if hasattr(X_tr, "shape") else None,
                            }
                        ),
                        flush=True,
                    )

                t_fit0 = time.time()
                if not cfg.quiet:
                    print(json.dumps({"event": "fit_start", "win": wi}), flush=True)
                model = adapter.fit(X_tr, C_tr)
                if not cfg.quiet:
                    print(
                        json.dumps(
                            {
                                "event": "fit_done",
                                "win": wi,
                                "sec": round(time.time() - t_fit0, 3),
                            }
                        ),
                        flush=True,
                    )

                X_val, C_val = _load_window_cached(adapter, ts_v_s, ts_v_e)
                if not cfg.quiet:
                    print(
                        json.dumps(
                            {
                                "event": "val_loaded",
                                "win": wi,
                                "rows": int(len(X_val)),
                            }
                        ),
                        flush=True,
                    )
                metrics = adapter.validate(model, X_val, C_val)
                # Persist per-window trades for global analysis if adapter exposes them
                trades = getattr(adapter, "_last_trades", None)
                if trades:
                    trade_rows: list[dict[str, Any]] = []
                    for tr in trades:
                        try:
                            trade_rows.append(
                                {
                                    "win": wi,
                                    "instrument": getattr(tr, "instrument", None),
                                    "side": int(getattr(tr, "side", 0)),
                                    "net_pnl": float(getattr(tr, "net_pnl", 0.0)),
                                    "gross_pnl": float(getattr(tr, "gross_pnl", 0.0)),
                                    "entry_time": str(getattr(tr, "entry_time", "")),
                                    "exit_time": str(getattr(tr, "exit_time", "")),
                                    "bars": int(getattr(tr, "bars", 0)),
                                }
                            )
                        except Exception:
                            continue
                    if trade_rows:
                        # Write one JSON object per line for easy streaming/rg usage
                        tlf.write(json.dumps({"win": wi, "trades": trade_rows}) + "\n")
                        tlf.flush()

                # Update runner equity curve using validation cum_return
                try:
                    v_cum = float(metrics.get("cum_return", 0.0))
                except Exception:
                    v_cum = 0.0
                last_eq = equity_points[-1] if equity_points else 1.0
                equity_points.append(last_eq * (1.0 + v_cum))

                # add basis-point aliases if present
                rec = {
                    "event": "window",
                    "win": wi,
                    "train": [ts_tr_s.isoformat(), ts_tr_e.isoformat()],
                    "val": [ts_v_s.isoformat(), ts_v_e.isoformat()],
                    **{
                        k: (
                            None
                            if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                            else v
                        )
                        for k, v in metrics.items()
                    },
                }
                # auto-add bp fields for common keys
                for k in ["cum_return", "max_dd"]:
                    if (
                        k in metrics
                        and isinstance(metrics[k], (int, float))
                        and not np.isnan(metrics[k])
                    ):
                        rec[f"{k}_bp"] = float(metrics[k]) * 1e4

                outf.write(json.dumps(rec) + "\n")
                outf.flush()

                # Print per-window done + running session stats
                sess = {
                    "event": "win_done",
                    "win": wi,
                    "sec": round(time.time() - t0, 3),
                    "v_cum": metrics.get("cum_return", None),
                    "v_sh": metrics.get("sharpe", None),
                    "v_dd": metrics.get("max_dd", None),
                    "v_tr": metrics.get("trades", None),
                }
                try:
                    sess_eq = float(equity_points[-1]) if equity_points else 1.0
                    sess_cum = float(sess_eq - 1.0)
                    sess_ret_bp = sess_cum * 1e4
                    sess_win = int(wi)
                    sess_msg = {
                        "event": "session",
                        "win": sess_win,
                        "cum_return": sess_cum,
                        "cum_return_bp": sess_ret_bp,
                    }
                    if not cfg.quiet:
                        print(json.dumps(sess), flush=True)
                        print(json.dumps(sess_msg), flush=True)
                        if (not cfg.no_chart) and (
                            wi % max(1, int(cfg.chart_every)) == 0
                        ):
                            chart = _render_chart(equity_points, width=60, height=8)
                            if chart:
                                print(chart, flush=True)
                except Exception:
                    # Fallback to just win_done when charting fails
                    if not cfg.quiet:
                        print(json.dumps(sess), flush=True)

    return run_dir
