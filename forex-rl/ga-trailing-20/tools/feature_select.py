#!/usr/bin/env python3
"""Feature ranking / dimensionality reduction utility.

Current implementation:
- Loads aligned features via DatasetLoader for a given instrument set and date range.
- Computes several per-feature scores:
  - Variance
  - Absolute Pearson correlation with next-bar return of the first instrument
- Produces multiple rankings (by variance, by abs_corr) and an ensemble rank
  (average of ranks).
- Saves results as JSON + a plain-text feature list sorted by ensemble rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: second-stage MI via scikit-learn if available
try:  # pragma: no cover - optional dependency
    from sklearn.feature_selection import mutual_info_regression
except Exception:  # pragma: no cover
    mutual_info_regression = None

# Optional: linear model stage (Ridge) for parametric ranking
try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover
    Ridge = None

# Make ga-trailing-20 importable when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from data import DatasetLoader, LoaderConfig  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank features by simple statistical importance scores")
    p.add_argument("--raw-dir", default=str(REPO_ROOT / "continuous-trader" / "data"))
    p.add_argument("--feature-dir", default=str(REPO_ROOT / "continuous-trader" / "data" / "features"))
    p.add_argument("--instruments", default="USD_PLN")
    p.add_argument("--base-gran", default="M5")
    p.add_argument("--aux", default="D", help="Aux granularities, e.g. 'D' or 'M5,D' (empty string for none)")
    # Default to recent window; if no overlap, tool will still report.
    p.add_argument("--date-range", default="2025-06-26:2025-10-01")
    p.add_argument("--max-rows", type=int, default=20000, help="Cap rows loaded per instrument")
    p.add_argument("--max-bars", type=int, default=5000, help="Limit number of bars used for ranking")
    p.add_argument("--top-k-stage1", type=int, default=150, help="Candidate pool size for stage-2 ranking")
    p.add_argument("--out-json", default=str(PKG_ROOT / "config" / "feature_ranking.json"))
    p.add_argument("--out-list", default=str(PKG_ROOT / "config" / "selected_features.txt"))
    return p.parse_args()


def _parse_range(s: str) -> Tuple[str, str]:
    start, end = [x.strip() for x in s.split(":", 1)]
    return start, end


def _build_panel(args: argparse.Namespace) -> Dict[str, Any]:
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    aux = [tok.strip().upper() for tok in args.aux.split(",") if tok.strip()]
    lcfg = LoaderConfig(
        instruments=instruments,
        raw_dir=Path(args.raw_dir),
        feature_dir=Path(args.feature_dir),
        base_granularity=str(args.base_gran).upper(),
        aux_granularities=aux,
        normalize=True,
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
    )
    loader = DatasetLoader(lcfg)
    # _build_panel is internal but fine for tooling
    panel = loader._build_panel()  # type: ignore[attr-defined]

    # Apply date-range + max-bars
    start, end = _parse_range(args.date_range)
    start_ts = pd.Timestamp(start).tz_localize("UTC")
    end_ts = pd.Timestamp(end).tz_localize("UTC")
    idx: pd.Index = panel["index"]
    mask = (idx >= start_ts) & (idx <= end_ts)
    idx_sel = idx[mask]
    if args.max_bars and len(idx_sel) > int(args.max_bars):
        idx_sel = idx_sel[-int(args.max_bars):]
    # Slice features and bars
    feats: pd.DataFrame = panel["features"].reindex(idx_sel)
    bars_map: Dict[str, pd.DataFrame] = {k: v.reindex(idx_sel) for k, v in panel["bars"].items()}
    return {"index": idx_sel, "features": feats, "bars": bars_map}


def rank_features(panel: Dict[str, Any], top_k_stage1: int) -> Dict[str, Dict[str, float]]:
    idx: pd.Index = panel["index"]
    X: pd.DataFrame = panel["features"]
    bars_map: Dict[str, pd.DataFrame] = panel["bars"]
    if X.empty:
        raise RuntimeError("No features available for ranking")

    # Stage 1: per-instrument IC aggregation
    # --------------------------------------
    # Use all aligned rows except last (for next-bar return)
    X_aligned = X.iloc[:-1]
    instruments = sorted(bars_map.keys())

    var = X_aligned.var().astype(float)

    # Per-instrument ICs
    ic_by_inst: Dict[str, pd.Series] = {}
    for inst in instruments:
        closes = bars_map[inst]["close"].astype(float)
        ret = closes.shift(-1) / closes - 1.0
        y_i = ret.iloc[:-1].fillna(0.0)
        # Corr of each feature with y_i
        ic_i = X_aligned.corrwith(y_i).fillna(0.0).astype(float)
        ic_by_inst[inst] = ic_i

    # Aggregate IC across instruments
    ic_matrix = pd.DataFrame(ic_by_inst)  # index: feature, columns: instruments
    abs_ic_matrix = ic_matrix.abs()
    mean_abs_ic = abs_ic_matrix.mean(axis=1)
    ic_std = abs_ic_matrix.std(axis=1).replace(0.0, np.nan)
    ic_ir = mean_abs_ic / ic_std.replace({0.0: np.nan})

    # Stage 1 ranks (lower is better)
    rank_var = var.rank(ascending=False, method="average")
    rank_mean_abs_ic = mean_abs_ic.rank(ascending=False, method="average")
    rank_ic_ir = ic_ir.rank(ascending=False, method="average")
    stage1_ensemble = (rank_var + rank_mean_abs_ic + rank_ic_ir) / 3.0

    # Build panel return and candidate pool once; reused by MI and linear model
    rets: List[np.ndarray] = []
    for inst in instruments:
        closes = bars_map[inst]["close"].astype(float)
        ret = closes.shift(-1) / closes - 1.0
        rets.append(ret.iloc[:-1].fillna(0.0).values)
    y_panel: np.ndarray | None = None
    if rets:
        R = np.stack(rets, axis=1)
        y_panel = R.mean(axis=1)

    k = min(int(top_k_stage1), X_aligned.shape[1])
    cand_features = list(stage1_ensemble.sort_values().index[:k])

    # Stage 2a: mutual information on candidate pool (if sklearn is available)
    # -----------------------------------------------------------------------
    mi_scores = pd.Series(0.0, index=X_aligned.columns)
    rank_mi = pd.Series(float("inf"), index=X_aligned.columns)
    if mutual_info_regression is not None and y_panel is not None:
        X_cand = X_aligned[cand_features].values
        try:  # pragma: no cover - depends on sklearn presence
            mi = mutual_info_regression(X_cand, y_panel, random_state=42)
            for name, score in zip(cand_features, mi):
                mi_scores.loc[name] = float(score)
            rank_mi = mi_scores.rank(ascending=False, method="average")
        except Exception:
            # If MI computation fails, leave mi_scores at 0 and rank_mi at inf
            pass

    # Stage 2b: linear model (Ridge) on candidate pool (if sklearn is available)
    # --------------------------------------------------------------------------
    lin_coef_abs = pd.Series(0.0, index=X_aligned.columns)
    rank_lin = pd.Series(float("inf"), index=X_aligned.columns)
    if Ridge is not None and y_panel is not None and len(cand_features) > 0:
        X_cand = X_aligned[cand_features].values
        # Simple standardization to make coefficients comparable
        mean = X_cand.mean(axis=0)
        std = X_cand.std(axis=0)
        std[std == 0.0] = 1.0
        X_std = (X_cand - mean) / std
        try:  # pragma: no cover - optional dependency
            model = Ridge(alpha=1.0)
            model.fit(X_std, y_panel)
            coefs = np.abs(model.coef_)
            for name, coef in zip(cand_features, coefs):
                lin_coef_abs.loc[name] = float(coef)
            rank_lin = lin_coef_abs.rank(ascending=False, method="average")
        except Exception:
            # If linear stage fails, leave lin_coef_abs at 0 and rank_lin at inf
            pass

    # Final ensemble rank: combine stage1 ensemble and MI rank (if available)
    # Normalize ranks by dividing by max, then average.
    max_stage1 = stage1_ensemble.max() if stage1_ensemble.size > 0 else 1.0
    max_rank_mi = rank_mi.replace(float("inf"), np.nan).max()
    norm_stage1 = stage1_ensemble / (max_stage1 if max_stage1 else 1.0)
    if np.isfinite(max_rank_mi) and max_rank_mi and not np.isnan(max_rank_mi):
        norm_rank_mi = rank_mi / max_rank_mi
        final_ensemble = (norm_stage1 + norm_rank_mi) / 2.0
    else:
        final_ensemble = norm_stage1

    out: Dict[str, Dict[str, float]] = {}
    for col in X_aligned.columns:
        out[col] = {
            "variance": float(var.get(col, 0.0)),
            "stage1_mean_abs_ic": float(mean_abs_ic.get(col, 0.0)),
            "stage1_ic_ir": float(ic_ir.get(col, 0.0)) if not np.isnan(ic_ir.get(col, np.nan)) else 0.0,
            "stage1_rank_variance": float(rank_var.get(col, np.nan)),
            "stage1_rank_mean_abs_ic": float(rank_mean_abs_ic.get(col, np.nan)),
            "stage1_rank_ic_ir": float(rank_ic_ir.get(col, np.nan)),
            "stage1_ensemble_rank": float(stage1_ensemble.get(col, np.nan)),
            "mi_score": float(mi_scores.get(col, 0.0)),
            "rank_mi": float(rank_mi.get(col, np.nan)),
            "linear_coef_abs": float(lin_coef_abs.get(col, 0.0)),
            "rank_linear_coef_abs": float(rank_lin.get(col, np.nan)),
            "final_ensemble_score": float(final_ensemble.get(col, np.nan)),
        }
    return out


def save_ranking(stats: Dict[str, Dict[str, float]], out_json: Path, out_list: Path) -> None:
    # Sort by final_ensemble_score ascending
    items = sorted(stats.items(), key=lambda kv: kv[1].get("final_ensemble_score", float("inf")))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in items}, fh, indent=2)

    with open(out_list, "w", encoding="utf-8") as fh:
        for name, meta in items:
            fh.write(f"{name}\n")


def main() -> None:
    args = parse_args()
    panel = _build_panel(args)
    stats = rank_features(panel, top_k_stage1=int(args.top_k_stage1))
    save_ranking(stats, Path(args.out_json), Path(args.out_list))
    print({
        "features": len(stats),
        "out_json": args.out_json,
        "out_list": args.out_list,
    })


if __name__ == "__main__":  # pragma: no cover
    main()
