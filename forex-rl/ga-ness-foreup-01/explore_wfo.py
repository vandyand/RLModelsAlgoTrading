#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import warnings as _warnings
import pandas as pd
import json as _json

from datetime import datetime

try:
    # Optional: reuse project CSV loader for consistent parsing
    from multi_features import _load_csv  # type: ignore
except Exception:
    _load_csv = None  # fallback to local loader


def find_latest_run(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not entries:
        return None
    entries.sort(reverse=True)
    return os.path.join(base_dir, entries[0])


def safe_get(d: Dict[str, Any], path: List[str], default: float = float("nan")) -> float:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def corr(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 2:
        return float("nan")
    x = aa[mask]
    y = bb[mask]
    x = x - x.mean()
    y = y - y.mean()
    sx = float(np.sqrt((x * x).sum() / max(1, x.size - 1)))
    sy = float(np.sqrt((y * y).sum() / max(1, y.size - 1)))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    r = float((x @ y) / max(1e-12, (x.size - 1)) / (sx * sy))
    return r


def linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # Fit y = a + b*x, returns (a,b)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def quantile_mean(values: List[float], q_low: float, q_high: float) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    lo = np.quantile(arr, q_low)
    hi = np.quantile(arr, q_high)
    sel = arr[(arr >= lo) & (arr <= hi)]
    return float(sel.mean()) if sel.size > 0 else float("nan")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore WFO windows.jsonl for deeper statistics")
    p.add_argument("--run-dir", help="Path to a WFO run directory; default: latest under base-dir")
    p.add_argument("--base-dir", default=os.path.join(os.path.dirname(__file__), "runs", "wfo", "gp"))
    p.add_argument("--top-k", type=int, default=0, help="Use only top-K per window from final_population (0=all)")
    p.add_argument("--max-individuals", type=int, default=0, help="Limit total individuals processed for speed (0=all)")
    p.add_argument("--out", default="analysis_extended.json", help="Output file name")
    # Model training options
    p.add_argument("--train-model", action="store_true", help="Train a model to predict P(val_sharpe>0)")
    p.add_argument("--model-type", default="logreg", choices=["logreg", "mlp", "gboost"], help="Model type: logistic, small MLP, or gradient boosting")
    p.add_argument("--pred-csv", default="predictions.csv", help="Output predictions CSV filename")
    p.add_argument("--coef-csv", default="feature_importance.csv", help="Output coefficients/importance CSV filename")
    p.add_argument("--calib-csv", default="calibration.csv", help="Output calibration CSV filename")
    p.add_argument("--cv", default="loo", choices=["loo", "kfold"], help="Cross-validation mode: leave-one-window-out or group k-fold")
    p.add_argument("--kfolds", type=int, default=8, help="Number of group folds when --cv=kfold")
    p.add_argument("--add-regime", action="store_true", help="Compute cheap market regime features per window (EUR_USD)")
    p.add_argument("--calibration", default="isotonic", choices=["none", "sigmoid", "isotonic"], help="Probability calibration method if sklearn available")
    p.add_argument("--topk-select", type=int, default=5, help="Produce selected_topk.csv with top-K picks per window and compute equity")
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    _warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
    _warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
    run_dir = a.run_dir or find_latest_run(a.base_dir)
    if not run_dir:
        print(json.dumps({"event": "explore_error", "error": "no run dir"}))
        return
    win_path = os.path.join(run_dir, "windows.jsonl")
    if not os.path.exists(win_path):
        print(json.dumps({"event": "explore_error", "error": f"no windows.jsonl in {run_dir}"}))
        return
    # Try to get data_dir from run_meta for regime features
    data_dir: Optional[str] = None
    try:
        with open(os.path.join(run_dir, "run_meta.json"), "r", encoding="utf-8") as mf:
            meta = _json.load(mf)
            args_obj = meta.get("args", {})
            data_dir = args_obj.get("data_dir")
    except Exception:
        data_dir = None

    # Accumulators
    # Best-per-window
    bpw_train_sh: List[float] = []
    bpw_val_sh: List[float] = []
    bpw_train_cr: List[float] = []
    bpw_val_cr: List[float] = []

    # Final population feature→target maps
    # Features from train side
    feat_defs = {
        "train_sharpe": ["metrics_train", "sharpe"],
        "train_sortino": ["metrics_train", "sortino"],
        "train_cumret": ["metrics_train", "cum_return"],
        "train_dd": ["metrics_train", "max_drawdown"],
        "trades_train": ["trades_train"],
        "tf_train": ["tf_train"],
        "tree_size": ["tree_size"],
        "tree_depth": ["tree_depth"],
        # extras_train
        "equity_r2": ["extras_train", "equity_r2"],
        "exposure": ["extras_train", "exposure"],
        "win_rate": ["extras_train", "win_rate"],
        "profit_factor": ["extras_train", "profit_factor"],
        "avg_dd_dur": ["extras_train", "avg_dd_dur"],
        "avg_hold": ["extras_train", "avg_hold"],
        "avg_idle": ["extras_train", "avg_idle"],
        "turnover_per_bar": ["extras_train", "turnover_per_bar"],
        "flip_rate": ["extras_train", "flip_rate"],
        "ret_std": ["extras_train", "ret_std"],
        "downside_std": ["extras_train", "downside_std"],
        "ret_skew": ["extras_train", "ret_skew"],
        "ret_kurt": ["extras_train", "ret_kurt"],
    }

    feat_vals: Dict[str, List[float]] = {k: [] for k in feat_defs}
    target_val_sh: List[float] = []
    target_val_cr: List[float] = []
    # Regime features per window
    regime_by_win: Dict[int, Dict[str, float]] = {}

    def compute_regime(train_start: str, train_end: str) -> Dict[str, float]:
        # Cheap regime stats on EUR_USD close over the train window
        try:
            if data_dir and _load_csv is not None:
                df = _load_csv(os.path.join(data_dir, "EUR_USD_M5.csv"), start=pd.to_datetime(train_start, utc=True), end=pd.to_datetime(train_end, utc=True))
            else:
                # Fallback CSV read
                p = os.path.join(data_dir or "data", "EUR_USD_M5.csv")
                df = pd.read_csv(p)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()[["close"]]
                df = df[(df.index >= pd.to_datetime(train_start, utc=True)) & (df.index <= pd.to_datetime(train_end, utc=True))]
            close = df["close"].astype(float)
            if len(close) < 5:
                return {"mkt_vol": float("nan"), "mkt_trend": float("nan"), "mkt_skew": float("nan")}
            ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            mkt_vol = float(np.std(ret.values, ddof=1)) if len(ret) > 1 else float("nan")
            # Trend: slope of log(close) vs bar index, normalized by length
            x = np.arange(len(close), dtype=float)
            y = np.log(close.values + 1e-12)
            Xd = np.vstack([np.ones_like(x), x]).T
            try:
                beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
                slope = float(beta[1])
            except Exception:
                slope = float("nan")
            mkt_trend = slope * 1000.0 / max(1.0, len(close))
            # Skew of returns
            rv = ret.values[np.isfinite(ret.values)]
            if rv.size < 3:
                mkt_skew = float("nan")
            else:
                mu = rv.mean(); sd = rv.std(ddof=1)
                mkt_skew = float(((rv - mu) ** 3).mean() / (sd ** 3 + 1e-12)) if sd > 1e-12 else 0.0
            return {"mkt_vol": mkt_vol, "mkt_trend": mkt_trend, "mkt_skew": mkt_skew}
        except Exception:
            return {"mkt_vol": float("nan"), "mkt_trend": float("nan"), "mkt_skew": float("nan")}

    # Presence counters
    present_counts: Dict[str, int] = {k: 0 for k in feat_defs}
    total_individuals = 0
    windows = 0

    with open(win_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            windows += 1
            # Best per window
            tm = rec.get("train_metrics", {}) or {}
            vm = rec.get("val_metrics", {}) or {}
            bpw_train_sh.append(float(tm.get("sharpe", float("nan"))))
            bpw_val_sh.append(float(vm.get("sharpe", float("nan"))))
            bpw_train_cr.append(float(tm.get("cum_return", float("nan"))))
            bpw_val_cr.append(float(vm.get("cum_return", float("nan"))))

            # Population
            pop = rec.get("final_population", []) or []
            if a.top_k > 0:
                pop = pop[: int(a.top_k)]
            # Optional per-window regime features
            if a.add_regime and (windows - 1) not in regime_by_win:
                tr = rec.get("train", {}) or {}
                rs = str(tr.get("start", "")); re = str(tr.get("end", ""))
                regime_by_win[windows - 1] = compute_regime(rs, re)
            for indiv in pop:
                # Targets
                target_val_sh.append(safe_get(indiv, ["metrics_val", "sharpe"]))
                target_val_cr.append(safe_get(indiv, ["metrics_val", "cum_return"]))
                # Features
                for k, path in feat_defs.items():
                    val = safe_get(indiv, path)
                    if not math.isnan(val):
                        present_counts[k] += 1
                    feat_vals[k].append(val)
                # Append regime features (same for all indiv in window)
                if a.add_regime:
                    reg = regime_by_win.get(windows - 1, {})
                    for rk in ("mkt_vol", "mkt_trend", "mkt_skew"):
                        key = rk
                        if key not in feat_vals:
                            feat_vals[key] = []
                            present_counts[key] = 0
                        v = float(reg.get(rk, float("nan")))
                        if not math.isnan(v):
                            present_counts[key] += 1
                        feat_vals[key].append(v)
                total_individuals += 1
                if a.max_individuals > 0 and total_individuals >= int(a.max_individuals):
                    break
            if a.max_individuals > 0 and total_individuals >= int(a.max_individuals):
                break

    # Summaries
    best_per_window = {
        "windows": windows,
        "corr": {
            "sharpe": corr(bpw_train_sh, bpw_val_sh),
            "cumret": corr(bpw_train_cr, bpw_val_cr),
        },
        "deg_mean": {
            "sharpe": float(np.nanmean(np.array(bpw_val_sh) / np.array(bpw_train_sh))) if windows > 0 else float("nan"),
            "cumret": float(np.nanmean(np.array(bpw_val_cr) / np.array(bpw_train_cr))) if windows > 0 else float("nan"),
        },
        "means": {
            "train_sharpe": float(np.nanmean(bpw_train_sh)),
            "val_sharpe": float(np.nanmean(bpw_val_sh)),
            "train_cumret": float(np.nanmean(bpw_train_cr)),
            "val_cumret": float(np.nanmean(bpw_val_cr)),
        },
    }

    # Correlations: feature vs val_sharpe/val_cum
    feature_corr_sh: Dict[str, float] = {}
    feature_corr_cr: Dict[str, float] = {}
    for k, xs in feat_vals.items():
        feature_corr_sh[k] = corr(xs, target_val_sh)
        feature_corr_cr[k] = corr(xs, target_val_cr)

    # Simple linear fits on key features
    def fit_summary(x: List[float], y: List[float]) -> Dict[str, float]:
        xa = np.array(x, dtype=float)
        ya = np.array(y, dtype=float)
        a0, b0 = linreg(xa, ya)
        return {"a": a0, "b": b0}

    fits = {
        "val_sharpe_on_train_sharpe": fit_summary(feat_vals.get("train_sharpe", []), target_val_sh),
        "val_cum_on_train_cum": fit_summary(feat_vals.get("train_cumret", []), target_val_cr),
        "val_sharpe_on_equity_r2": fit_summary(feat_vals.get("equity_r2", []), target_val_sh),
        "val_sharpe_on_win_rate": fit_summary(feat_vals.get("win_rate", []), target_val_sh),
    }

    # Top correlated features
    def topn(d: Dict[str, float], n: int = 10) -> List[Tuple[str, float]]:
        return sorted(d.items(), key=lambda kv: (-(abs(kv[1]) if np.isfinite(kv[1]) else -1.0)), reverse=False)[:n]

    # Presence ratios
    presence = {k: int(v) for k, v in present_counts.items()}

    report = {
        "event": "wfo_explore",
        "run_dir": run_dir,
        "windows": windows,
        "individuals": total_individuals,
        "presence": presence,
        "best_per_window": best_per_window,
        "feature_corr": {
            "val_sharpe": feature_corr_sh,
            "val_cumret": feature_corr_cr,
        },
        "top_corr": {
            "val_sharpe": sorted(feature_corr_sh.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1.0, reverse=True)[:15],
            "val_cumret": sorted(feature_corr_cr.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1.0, reverse=True)[:15],
        },
        "fits": fits,
        "quantiles": {
            "val_sharpe_middle80": quantile_mean(target_val_sh, 0.1, 0.9),
            "val_cumret_middle80": quantile_mean(target_val_cr, 0.1, 0.9),
        },
    }

    out_path = os.path.join(run_dir, a.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"event": "wfo_explore_done", "file": out_path}), flush=True)

    # Optional: train a predictive model on final_population features
    if a.train_model:
        # Build row-wise dataset with window ids
        X_rows: List[List[float]] = []
        y_rows: List[int] = []
        w_rows: List[int] = []
        row_ids: List[int] = []
        val_sh_rows: List[float] = []
        val_cr_rows: List[float] = []
        keep_feature_names = list(feat_defs.keys())
        # Re-read to get all individuals with window ids
        total = 0
        rid = 0
        with open(win_path, "r", encoding="utf-8") as f2:
            wid = -1
            for line in f2:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                wid += 1
                pop = rec.get("final_population", []) or []
                if a.top_k > 0:
                    pop = pop[: int(a.top_k)]
                for indiv in pop:
                    y = 1 if safe_get(indiv, ["metrics_val", "sharpe"]) > 0.0 else 0
                    row: List[float] = []
                    ok = True
                    for k in keep_feature_names:
                        v = safe_get(indiv, feat_defs[k])
                        if math.isnan(v) or not math.isfinite(v):
                            ok = False
                            break
                        row.append(v)
                    if not ok:
                        continue
                    X_rows.append(row)
                    y_rows.append(y)
                    w_rows.append(wid)
                    row_ids.append(rid)
                    val_sh_rows.append(safe_get(indiv, ["metrics_val", "sharpe"]))
                    val_cr_rows.append(safe_get(indiv, ["metrics_val", "cum_return"]))
                    total += 1
                    rid += 1
        if total == 0:
            print(json.dumps({"event": "wfo_model_error", "error": "no data rows"}))
            return

        X = np.array(X_rows, dtype=float)
        y = np.array(y_rows, dtype=int)
        groups = np.array(w_rows, dtype=int)
        row_ids_arr = np.array(row_ids, dtype=int)
        val_sh_arr = np.array(val_sh_rows, dtype=float)
        val_cr_arr = np.array(val_cr_rows, dtype=float)

        # Cross-validated by window (grouped): leave-one-group-out
        uniq_groups = np.unique(groups)
        preds_all: List[float] = []
        y_all: List[int] = []
        win_all: List[int] = []
        row_all: List[int] = []

        coef_sum = np.zeros(X.shape[1], dtype=float)
        coef_count = 0

        # Define model builders
        def fit_logreg(Xtr: np.ndarray, ytr: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
            # Standardize
            mu = Xtr.mean(axis=0)
            sigma = Xtr.std(axis=0)
            sigma[sigma == 0.0] = 1.0
            Xtrn = (Xtr - mu) / sigma
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.calibration import CalibratedClassifierCV
                base = LogisticRegression(max_iter=1000, class_weight="balanced")
                if a.calibration != "none":
                    m = CalibratedClassifierCV(base, method=a.calibration, cv=3)
                else:
                    m = base
                m.fit(Xtrn, ytr)
                coef = getattr(m, "coef", None)
                if coef is not None:
                    c = np.array(coef).reshape(-1)
                else:
                    c = np.zeros(Xtr.shape[1], dtype=float)
                return (m, mu, sigma), c, np.array([])
            except Exception:
                # Numpy fallback: simple L2-regularized logistic via gradient descent
                w = np.zeros(Xtr.shape[1], dtype=float)
                b = 0.0
                lr = 0.1
                reg = 1e-3
                for _ in range(300):
                    z = Xtrn @ w + b
                    p = 1.0 / (1.0 + np.exp(-z))
                    g = Xtrn.T @ (p - ytr) / Xtrn.shape[0] + reg * w
                    gb = float(p.mean() - ytr.mean())
                    w -= lr * g
                    b -= lr * gb
                class Model:
                    def __init__(self, w, b):
                        self.w = w
                        self.b = b
                    def predict_proba(self, Xt):
                        z = Xt @ self.w + self.b
                        p = 1.0 / (1.0 + np.exp(-z))
                        return np.vstack([1 - p, p]).T
                m = Model(w, b)
                return (m, mu, sigma), w, np.array([])

        def fit_mlp(Xtr: np.ndarray, ytr: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
            mu = Xtr.mean(axis=0)
            sigma = Xtr.std(axis=0)
            sigma[sigma == 0.0] = 1.0
            Xtrn = (Xtr - mu) / sigma
            try:
                from sklearn.neural_network import MLPClassifier
                from sklearn.calibration import CalibratedClassifierCV
                base = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=1000)
                if a.calibration != "none":
                    m = CalibratedClassifierCV(base, method=a.calibration, cv=3)
                else:
                    m = base
                m.fit(Xtrn, ytr)
                return (m, mu, sigma), np.array([]), np.array([])
            except Exception:
                # Fallback: small numpy MLP (1 hidden layer)
                rng = np.random.default_rng(42)
                n_in = Xtrn.shape[1]
                n_h = 16
                W1 = 0.1 * rng.standard_normal((n_in, n_h))
                b1 = np.zeros((n_h,), dtype=float)
                W2 = 0.1 * rng.standard_normal((n_h, 1))
                b2 = np.zeros((1,), dtype=float)
                ytrf = ytr.reshape(-1, 1).astype(float)
                lr = 0.05
                reg = 1e-4
                epochs = 400
                for _ in range(epochs):
                    Z1 = Xtrn @ W1 + b1
                    H = np.maximum(0.0, Z1)
                    Z2 = H @ W2 + b2
                    P = 1.0 / (1.0 + np.exp(-Z2))
                    dZ2 = (P - ytrf)
                    dW2 = (H.T @ dZ2) / Xtrn.shape[0] + reg * W2
                    db2 = dZ2.mean(axis=0)
                    dH = dZ2 @ W2.T
                    dZ1 = dH * (Z1 > 0)
                    dW1 = (Xtrn.T @ dZ1) / Xtrn.shape[0] + reg * W1
                    db1 = dZ1.mean(axis=0)
                    W2 -= lr * dW2
                    b2 -= lr * db2
                    W1 -= lr * dW1
                    b1 -= lr * db1
                class M:
                    def __init__(self, W1, b1, W2, b2):
                        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
                    def predict_proba(self, Xt):
                        Z1 = Xt @ self.W1 + self.b1
                        H = np.maximum(0.0, Z1)
                        Z2 = H @ self.W2 + self.b2
                        P = 1.0 / (1.0 + np.exp(-Z2))
                        return np.hstack([1.0 - P, P])
                m = M(W1, b1, W2, b2)
                return (m, mu, sigma), np.array([]), np.array([])

        def fit_gboost(Xtr: np.ndarray, ytr: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
            mu = Xtr.mean(axis=0)
            sigma = Xtr.std(axis=0)
            sigma[sigma == 0.0] = 1.0
            Xtrn = (Xtr - mu) / sigma
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.calibration import CalibratedClassifierCV
                base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08)
                if a.calibration != "none":
                    m = CalibratedClassifierCV(base, method=a.calibration, cv=3)
                else:
                    m = base
                m.fit(Xtrn, ytr)
                return (m, mu, sigma), np.array([]), np.array([])
            except Exception:
                # Fallback: use logistic
                return fit_logreg(Xtr, ytr)

        def predict(model_pack: Tuple[object, np.ndarray, np.ndarray], Xte: np.ndarray) -> np.ndarray:
            m, mu, sigma = model_pack
            Xten = (Xte - mu) / sigma
            try:
                proba = m.predict_proba(Xten)[:, 1]
            except Exception:
                # Fallback if model returns array
                z = Xten @ m.w + m.b  # type: ignore[attr-defined]
                proba = 1.0 / (1.0 + np.exp(-z))
            return np.array(proba, dtype=float)

        if a.cv == "kfold":
            K = max(2, int(a.kfolds))
            ug = uniq_groups.tolist()
            for i in range(K):
                te_groups = set(ug[i::K])
                mask_te = np.isin(groups, list(te_groups))
                mask_tr = ~mask_te
                if mask_tr.sum() < 5 or mask_te.sum() < 1:
                    continue
                Xtr, ytr = X[mask_tr], y[mask_tr]
                Xte, yte = X[mask_te], y[mask_te]
                if a.model_type == "mlp":
                    model_pack, coef, _ = fit_mlp(Xtr, ytr)
                elif a.model_type == "gboost":
                    model_pack, coef, _ = fit_gboost(Xtr, ytr)
                else:
                    model_pack, coef, _ = fit_logreg(Xtr, ytr)
                if coef.size > 0:
                    coef_sum += coef
                    coef_count += 1
                p = predict(model_pack, Xte)
                preds_all.extend(p.tolist())
                y_all.extend(yte.tolist())
                win_all.extend(groups[mask_te].tolist())
                row_all.extend(row_ids_arr[mask_te].tolist())
        else:
            for g in uniq_groups:
                mask_te = groups == g
                mask_tr = ~mask_te
                if mask_tr.sum() < 5 or mask_te.sum() < 1:
                    continue
                Xtr, ytr = X[mask_tr], y[mask_tr]
                Xte, yte = X[mask_te], y[mask_te]

                if a.model_type == "mlp":
                    model_pack, coef, _ = fit_mlp(Xtr, ytr)
                elif a.model_type == "gboost":
                    model_pack, coef, _ = fit_gboost(Xtr, ytr)
                else:
                    model_pack, coef, _ = fit_logreg(Xtr, ytr)
                if coef.size > 0:
                    coef_sum += coef
                    coef_count += 1
                p = predict(model_pack, Xte)
                preds_all.extend(p.tolist())
                y_all.extend(yte.tolist())
                win_all.extend([int(g)] * len(yte))
                row_all.extend(row_ids_arr[mask_te].tolist())

        preds = np.array(preds_all, dtype=float)
        ytrue = np.array(y_all, dtype=int)
        row_pred = np.array(row_all, dtype=int)

        # Metrics
        try:
            from sklearn.metrics import roc_auc_score, precision_score
            auc = float(roc_auc_score(ytrue, preds)) if (np.unique(ytrue).size == 2) else float("nan")
            prec50 = float(precision_score(ytrue, preds >= 0.5))
        except Exception:
            # Manual AUC approx
            try:
                order = np.argsort(preds)
                y_sorted = ytrue[order]
                cum_pos = np.cumsum(y_sorted[::-1])
                cum_neg = np.cumsum(1 - y_sorted[::-1])
                tpr = cum_pos / max(1, int(ytrue.sum()))
                fpr = cum_neg / max(1, int((ytrue == 0).sum()))
                auc = float(np.trapz(tpr, fpr))
            except Exception:
                auc = float("nan")
            # Precision at 0.5
            pred_lbl = (preds >= 0.5).astype(int)
            tp = int(((pred_lbl == 1) & (ytrue == 1)).sum())
            fp = int(((pred_lbl == 1) & (ytrue == 0)).sum())
            prec50 = float(tp / max(1, tp + fp))

        # Precision@Top10% per window
        prec_at10_list: List[float] = []
        for g in uniq_groups:
            m = (np.array(win_all) == int(g))
            if m.sum() < 10:
                continue
            k = max(1, int(round(0.1 * m.sum())))
            idx = np.where(m)[0]
            topk = idx[np.argsort(preds[idx])[::-1][:k]]
            prec_at10_list.append(float(ytrue[topk].mean()))
        prec_at10 = float(np.nanmean(prec_at10_list)) if len(prec_at10_list) > 0 else float("nan")

        # Coefficients average (for logreg)
        coef_mean = (coef_sum / max(1, coef_count)).tolist()

        # Write predictions CSV (plot-ready)
        pred_csv = os.path.join(run_dir, a.pred_csv)
        with open(pred_csv, "w", encoding="utf-8") as pf:
            cols = ["window", "row_id", "prob", "y_true", "val_sharpe", "val_cumret"]
            pf.write(",".join(cols) + "\n")
            for i in range(len(preds)):
                rid_i = row_pred[i]
                pf.write(f"{win_all[i]},{rid_i},{preds[i]:.6f},{int(ytrue[i])},{val_sh_arr[rid_i]:.6f},{val_cr_arr[rid_i]:.6f}\n")

        # Also produce per-window selections
        select_csv = os.path.join(run_dir, "selected_per_window.csv")
        with open(select_csv, "w", encoding="utf-8") as sf:
            sf.write("window,selected@top10_frac\n")
            for g in uniq_groups:
                m = (np.array(win_all) == int(g))
                if m.sum() == 0:
                    continue
                k = max(1, int(round(0.1 * m.sum())))
                idx = np.where(m)[0]
                topk = idx[np.argsort(preds[idx])[::-1][:k]]
                sf.write(f"{int(g)},{k/float(m.sum()):.4f}\n")

        # Selected top-K per window list
        if int(a.topk_select) > 0:
            topk_list_csv = os.path.join(run_dir, "selected_topk.csv")
            with open(topk_list_csv, "w", encoding="utf-8") as tf:
                tf.write("window,row_id,prob,val_sharpe,val_cumret\n")
                for g in uniq_groups:
                    m = (np.array(win_all) == int(g))
                    if m.sum() == 0:
                        continue
                    k = max(1, int(a.topk_select))
                    idx = np.where(m)[0]
                    topk = idx[np.argsort(preds[idx])[::-1][:k]]
                    for j in topk:
                        rid_j = row_pred[j]
                        tf.write(f"{int(g)},{rid_j},{preds[j]:.6f},{val_sh_arr[rid_j]:.6f},{val_cr_arr[rid_j]:.6f}\n")

        # Write coefficients CSV
        coef_csv = os.path.join(run_dir, a.coef_csv)
        with open(coef_csv, "w", encoding="utf-8") as cf:
            cf.write("feature,coef\n")
            for name, c in zip(keep_feature_names, coef_mean):
                cf.write(f"{name},{c}\n")

        # Calibration CSV
        calib_csv = os.path.join(run_dir, a.calib_csv)
        bins = np.linspace(0.0, 1.0, 11)
        with open(calib_csv, "w", encoding="utf-8") as cb:
            cb.write("bin_left,bin_right,mean_prob,fraction_positive,count\n")
            for i in range(len(bins) - 1):
                l, r = bins[i], bins[i + 1]
                m = (preds >= l) & (preds < r)
                if m.sum() == 0:
                    cb.write(f"{l},{r},, ,0\n")
                else:
                    cb.write(f"{l},{r},{float(preds[m].mean())},{float(ytrue[m].mean())},{int(m.sum())}\n")

        # Recommend threshold/rules based on max F1 threshold search
        ths = np.linspace(0.1, 0.9, 17)
        best_f1 = -1.0
        best_th = 0.5
        for t in ths:
            pred_lbl = (preds >= t).astype(int)
            tp = int(((pred_lbl == 1) & (ytrue == 1)).sum())
            fp = int(((pred_lbl == 1) & (ytrue == 0)).sum())
            fn = int(((pred_lbl == 0) & (ytrue == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, (prec + rec))
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(t)

        model_summary = {
            "event": "wfo_model",
            "run_dir": run_dir,
            "model": a.model_type,
            "auc": auc,
            "precision@0.5": prec50,
            "precision@top10%": prec_at10,
            "threshold_f1": best_th,
            "coef_top_pos": sorted(list(zip(keep_feature_names, coef_mean)), key=lambda kv: kv[1], reverse=True)[:8],
            "coef_top_neg": sorted(list(zip(keep_feature_names, coef_mean)), key=lambda kv: kv[1])[:8],
            "pred_csv": pred_csv,
            "coef_csv": coef_csv,
            "calib_csv": calib_csv,
            "select_csv": select_csv,
        }
        with open(os.path.join(run_dir, "meta_model.json"), "w", encoding="utf-8") as mf:
            json.dump(model_summary, mf, indent=2)
        print(json.dumps(model_summary), flush=True)

        # Evaluate equity if selecting by threshold or by top-10%
        def evaluate_equity_by_threshold(th: float) -> Dict[str, float]:
            eq = 1.0
            picked = 0
            no_sel = 0
            for g in np.unique(win_all):
                m = np.array(win_all) == int(g)
                if m.sum() == 0:
                    continue
                sel = (preds[m] >= th)
                if sel.sum() == 0:
                    no_sel += 1
                    continue
                picked += int(sel.sum())
                rows = row_pred[m][sel]
                avg_cr = float(np.nanmean(val_cr_arr[rows]))
                if not np.isfinite(avg_cr):
                    avg_cr = 0.0
                eq *= (1.0 + avg_cr)
            return {"equity": eq, "windows_no_selection": float(no_sel), "avg_picks_per_window": float(picked) / max(1, len(np.unique(win_all)))}

        def evaluate_equity_top_frac(frac: float = 0.1) -> Dict[str, float]:
            eq = 1.0
            picked = 0
            for g in np.unique(win_all):
                m = np.array(win_all) == int(g)
                if m.sum() == 0:
                    continue
                k = max(1, int(round(frac * m.sum())))
                idx = np.where(m)[0]
                topk = idx[np.argsort(preds[idx])[::-1][:k]]
                picked += len(topk)
                rows = row_pred[topk]
                avg_cr = float(np.nanmean(val_cr_arr[rows]))
                if not np.isfinite(avg_cr):
                    avg_cr = 0.0
                eq *= (1.0 + avg_cr)
            return {"equity": eq, "avg_picks_per_window": float(picked) / max(1, len(np.unique(win_all)))}

        equity_th = evaluate_equity_by_threshold(best_th)
        equity_top10 = evaluate_equity_top_frac(0.10)
        # Evaluate equity for K in {1,3,5} with p >= 0.3 per window
        def evaluate_equity_topk_threshold(k: int, pmin: float) -> Dict[str, float]:
            eq = 1.0
            picked = 0
            zeros = 0
            for g in np.unique(win_all):
                m = np.array(win_all) == int(g)
                if m.sum() == 0:
                    continue
                idx = np.where(m)[0]
                # filter by threshold
                idx_th = idx[preds[idx] >= pmin]
                if idx_th.size == 0:
                    zeros += 1
                    continue
                # top-k by prob
                ksel = np.argsort(preds[idx_th])[::-1][:max(1, int(k))]
                sel_idx = idx_th[ksel]
                picked += sel_idx.size
                rows = row_pred[sel_idx]
                avg_cr = float(np.nanmean(val_cr_arr[rows]))
                if not np.isfinite(avg_cr):
                    avg_cr = 0.0
                eq *= (1.0 + avg_cr)
            return {"equity": eq, "windows_no_selection": float(zeros), "avg_picks_per_window": float(picked) / max(1, len(np.unique(win_all)))}

        equity_k1_t30 = evaluate_equity_topk_threshold(1, 0.30)
        equity_k1_t35 = evaluate_equity_topk_threshold(1, 0.35)
        equity_k1_t40 = evaluate_equity_topk_threshold(1, 0.40)
        equity_k1_t50 = evaluate_equity_topk_threshold(1, 0.50)
        equity_k3_t30 = evaluate_equity_topk_threshold(3, 0.30)
        equity_k5_t30 = evaluate_equity_topk_threshold(5, 0.30)
        with open(os.path.join(run_dir, "selection_outcome.json"), "w", encoding="utf-8") as ef:
            json.dump({
                "threshold": best_th,
                "equity_threshold": equity_th,
                "equity_top10pct": equity_top10,
                "equity_k1_t30": equity_k1_t30,
                "equity_k1_t35": equity_k1_t35,
                "equity_k1_t40": equity_k1_t40,
                "equity_k1_t50": equity_k1_t50,
                "equity_k3_t30": equity_k3_t30,
                "equity_k5_t30": equity_k5_t30,
            }, ef, indent=2)
        print(json.dumps({"event": "selection_outcome", **{
            "threshold": best_th,
            **equity_th,
            "equity_top10": equity_top10["equity"],
            "k1_t30": equity_k1_t30["equity"],
            "k1_t35": equity_k1_t35["equity"],
            "k1_t40": equity_k1_t40["equity"],
            "k1_t50": equity_k1_t50["equity"],
            "k3_t30": equity_k3_t30["equity"],
            "k5_t30": equity_k5_t30["equity"],
        }}), flush=True)


if __name__ == "__main__":
    main()
