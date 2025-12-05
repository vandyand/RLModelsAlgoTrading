#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def find_latest_run(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    try:
        entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not entries:
            return None
        entries.sort(reverse=True)
        return os.path.join(base_dir, entries[0])
    except Exception:
        return None


essential_metrics = [
    ("sharpe", "shrp"),
    ("sortino", "srtn"),
    ("cum_return", "cum"),
    ("max_drawdown", "dd"),
]


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
    if np.all(np.isnan(aa)) or np.all(np.isnan(bb)):
        return float("nan")
    mask = ~np.isnan(aa) & ~np.isnan(bb)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def lin_regress(x: List[float], y: List[float]) -> Dict[str, float]:
    # Fit y = a + b*x ignoring NaNs
    xx = np.array(x, dtype=float)
    yy = np.array(y, dtype=float)
    mask = ~np.isnan(xx) & ~np.isnan(yy)
    if mask.sum() < 2:
        return {"a": float("nan"), "b": float("nan")}
    X = np.vstack([np.ones(mask.sum()), xx[mask]]).T
    coef, *_ = np.linalg.lstsq(X, yy[mask], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return {"a": a, "b": b}


def summarize_windows(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Best-per-window summary
    train_sh, val_sh = [], []
    train_cr, val_cr = [], []
    win_count = len(windows)
    for rec in windows:
        tm = rec.get("train_metrics", {}) or {}
        vm = rec.get("val_metrics", {}) or {}
        train_sh.append(float(tm.get("sharpe", float("nan"))))
        train_cr.append(float(tm.get("cum_return", float("nan"))))
        val_sh.append(float(vm.get("sharpe", float("nan"))))
        val_cr.append(float(vm.get("cum_return", float("nan"))))
    return {
        "windows": win_count,
        "corr": {
            "sharpe": corr(train_sh, val_sh),
            "cumret": corr(train_cr, val_cr),
        },
        "deg": {
            "sharpe_mean": float(
                np.nanmean(np.divide(np.array(val_sh), np.array(train_sh), where=np.isfinite(np.array(train_sh))))
            ) if win_count > 0 else float("nan"),
            "cumret_mean": float(
                np.nanmean(np.divide(np.array(val_cr), np.array(train_cr), where=np.isfinite(np.array(train_cr))))
            ) if win_count > 0 else float("nan"),
        },
        "means": {
            "train_sharpe": float(np.nanmean(train_sh)) if win_count > 0 else float("nan"),
            "val_sharpe": float(np.nanmean(val_sh)) if win_count > 0 else float("nan"),
            "train_cumret": float(np.nanmean(train_cr)) if win_count > 0 else float("nan"),
            "val_cumret": float(np.nanmean(val_cr)) if win_count > 0 else float("nan"),
        },
        "reg": {
            "sharpe": lin_regress(train_sh, val_sh),
            "cumret": lin_regress(train_cr, val_cr),
        },
        "counts": {
            "wins_pos_val_sharpe": int(np.sum(np.array(val_sh) > 0.0)),
            "wins_pos_val_cum": int(np.sum(np.array(val_cr) > 0.0)),
        },
    }


def summarize_population(windows: List[Dict[str, Any]], top_k: int = 0) -> Dict[str, Any]:
    # Flatten final population across windows
    pop_train_sh, pop_val_sh = [], []
    pop_train_cr, pop_val_cr = [], []
    total = 0
    for rec in windows:
        pop = rec.get("final_population", []) or []
        if not pop:
            continue
        pop_sorted = pop if top_k <= 0 else pop[: int(top_k)]
        for indiv in pop_sorted:
            tm = indiv.get("metrics_train", {}) or {}
            vm = indiv.get("metrics_val", {}) or {}
            pop_train_sh.append(float(tm.get("sharpe", float("nan"))))
            pop_val_sh.append(float(vm.get("sharpe", float("nan"))))
            pop_train_cr.append(float(tm.get("cum_return", float("nan"))))
            pop_val_cr.append(float(vm.get("cum_return", float("nan"))))
            total += 1
    return {
        "individuals": total,
        "corr": {
            "sharpe": corr(pop_train_sh, pop_val_sh),
            "cumret": corr(pop_train_cr, pop_val_cr),
        },
        "deg": {
            "sharpe_mean": float(np.nanmean(np.array(pop_val_sh) / np.array(pop_train_sh))) if total > 0 else float("nan"),
            "cumret_mean": float(np.nanmean(np.array(pop_val_cr) / np.array(pop_train_cr))) if total > 0 else float("nan"),
        },
        "means": {
            "train_sharpe": float(np.nanmean(pop_train_sh)) if total > 0 else float("nan"),
            "val_sharpe": float(np.nanmean(pop_val_sh)) if total > 0 else float("nan"),
            "train_cumret": float(np.nanmean(pop_train_cr)) if total > 0 else float("nan"),
            "val_cumret": float(np.nanmean(pop_val_cr)) if total > 0 else float("nan"),
        },
        "reg": {
            "sharpe": lin_regress(pop_train_sh, pop_val_sh),
            "cumret": lin_regress(pop_train_cr, pop_val_cr),
        },
        "counts": {
            "pos_val_sharpe": int(np.sum(np.array(pop_val_sh) > 0.0)),
            "pos_val_cum": int(np.sum(np.array(pop_val_cr) > 0.0)),
        },
    }


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze WFO windows and final populations")
    p.add_argument("--run-dir", help="Path to a specific WFO run (directory with windows.jsonl)")
    p.add_argument(
        "--base-dir",
        default=os.path.join(os.path.dirname(__file__), "runs", "wfo", "gp"),
        help="Base directory to auto-pick latest run if --run-dir not provided",
    )
    p.add_argument("--population-top-k", type=int, default=0, help="Use only top-K individuals per window for population analysis (0=all)")
    p.add_argument("--out", default="analysis.json", help="Output file name (saved under run dir)")
    p.add_argument("--segments", default="half,quarter,12,8,4", help="Comma list of segments: half,quarter,N where N is integer windows")
    return p.parse_args()


def main() -> None:
    a = parse_cli()
    run_dir = a.run_dir or find_latest_run(a.base_dir)
    if not run_dir:
        print(json.dumps({"event": "analysis_error", "error": "No run directory found"}))
        return
    win_path = os.path.join(run_dir, "windows.jsonl")
    if not os.path.exists(win_path):
        print(json.dumps({"event": "analysis_error", "error": f"windows.jsonl not found in {run_dir}"}))
        return
    windows: List[Dict[str, Any]] = []
    with open(win_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                windows.append(json.loads(line))
            except Exception:
                continue

    best_summary = summarize_windows(windows)
    # Segmented summaries
    segs = [s.strip() for s in str(a.segments).split(',') if s.strip()]
    seg_report: Dict[str, Any] = {}
    n = len(windows)
    for s in segs:
        key = f"last_{s}"
        if s == "half":
            seg_report[key] = summarize_windows(windows[n//2:])
        elif s == "quarter":
            seg_report[key] = summarize_windows(windows[(3*n)//4:])
        else:
            try:
                k = int(s)
                if k > 0:
                    seg_report[key] = summarize_windows(windows[-k:])
            except Exception:
                continue
    pop_summary = summarize_population(windows, top_k=int(a.population_top_k))

    report = {
        "event": "wfo_analysis",
        "run_dir": run_dir,
        "best_per_window": best_summary,
        "final_population": pop_summary,
        "segments": seg_report,
    }

    out_path = os.path.join(run_dir, str(a.out))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
