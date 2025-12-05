#!/usr/bin/env python3
"""Ensemble feature ranking across multiple instruments.

This tool aggregates per-instrument feature ranking JSONs (produced by
`feature_select.py`) into a cross-instrument ensemble ranking suitable for
multi-asset models.

Assumptions:
- Each input JSON is a mapping: feature_name -> metrics dict, where metrics
  include at least:
    - final_ensemble_score  (lower is better)
    - linear_coef_abs       (higher is better; may be 0.0 if unavailable)
- Filenames follow a pattern like:
    feature_ranking_{INSTR}_{BASE}{AUX}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Repo roots (mirror other tools)
REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ga-trailing-20"


# Mirror DEFAULT_OANDA_20 universe used in multi20 scripts
DEFAULT_OANDA_20: Tuple[str, ...] = (
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "AUD_USD",
    "USD_CHF",
    "USD_CAD",
    "NZD_USD",
    "EUR_JPY",
    "GBP_JPY",
    "EUR_GBP",
    "EUR_CHF",
    "EUR_AUD",
    "EUR_CAD",
    "GBP_CHF",
    "AUD_JPY",
    "AUD_CHF",
    "CAD_JPY",
    "NZD_JPY",
    "GBP_AUD",
    "AUD_NZD",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble per-instrument feature rankings into a cross-instrument ranking")
    p.add_argument(
        "--instruments",
        default=",".join(DEFAULT_OANDA_20),
        help="Comma-separated list of instruments; defaults to DEFAULT_OANDA_20",
    )
    p.add_argument("--base-gran", default="M5", help="Base granularity used in filenames, e.g. M5")
    p.add_argument("--aux", default="D", help="Aux granularity code used in filenames, e.g. D or M5D")
    p.add_argument(
        "--config-dir",
        default=str(PKG_ROOT / "config"),
        help="Directory containing per-instrument feature_ranking_*.json files",
    )
    p.add_argument(
        "--pattern",
        default="feature_ranking_{inst}_{base}{aux}.json",
        help="Filename pattern relative to config-dir; tokens: {inst}, {base}, {aux}",
    )
    p.add_argument(
        "--out-json",
        default=str(PKG_ROOT / "config" / "feature_ranking_MULTI20.json"),
        help="Output JSON path for cross-instrument ranking",
    )
    p.add_argument(
        "--out-list",
        default=str(PKG_ROOT / "config" / "selected_features_MULTI20.txt"),
        help="Output plain-text feature list (sorted by ensemble score)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=150,
        help="Top-K threshold per instrument for top-k frequency statistics",
    )
    return p.parse_args()


def _load_per_instrument(
    inst: str, base: str, aux: str, cfg_dir: Path, pattern: str
) -> Dict[str, Dict[str, float]] | None:
    fname = pattern.format(inst=inst, base=base, aux=aux)
    path = cfg_dir / fname
    if not path.exists():
        # Missing file is non-fatal; caller can log/skip.
        print(f"[WARN] Missing ranking file for {inst}: {path}", flush=True)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Ranking JSON must be a dict of feature -> metrics")
        return data
    except Exception as exc:
        print(f"[WARN] Failed to load ranking for {inst} from {path}: {exc}", flush=True)
        return None


def ensemble_rankings(
    per_inst: Dict[str, Dict[str, Dict[str, float]]], total_instruments: int, top_k: int
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-instrument metrics into cross-instrument scores.

    per_inst: mapping inst -> (mapping feature -> metrics dict)
    """
    # Collect all feature names
    all_features: set[str] = set()
    for metrics in per_inst.values():
        all_features.update(metrics.keys())

    # Raw aggregates
    agg: Dict[str, Dict[str, Any]] = {}
    for feat in all_features:
        agg[feat] = {
            "final_scores": [],
            "linear_scores": [],
            "mean_abs_ic": [],
            "ic_ir": [],
            "present_in": 0,
            "topk_final_count": 0,
            "topk_linear_count": 0,
        }

    # Per-instrument top-K ranks
    for inst, metrics in per_inst.items():
        if not metrics:
            continue
        # Build vectors for this instrument
        feats = list(metrics.keys())
        final_vals = np.array(
            [float(metrics[f].get("final_ensemble_score", np.nan)) for f in feats], dtype=float
        )
        linear_vals = np.array([float(metrics[f].get("linear_coef_abs", 0.0)) for f in feats], dtype=float)

        # Presence and raw metric accumulation
        for f in feats:
            m = metrics[f]
            a = agg[f]
            a["present_in"] += 1
            fs = float(m.get("final_ensemble_score", np.nan))
            if np.isfinite(fs):
                a["final_scores"].append(fs)
            ls = float(m.get("linear_coef_abs", 0.0))
            if ls != 0.0 and np.isfinite(ls):
                a["linear_scores"].append(ls)
            ic = float(m.get("stage1_mean_abs_ic", 0.0))
            if np.isfinite(ic):
                a["mean_abs_ic"].append(ic)
            icir = float(m.get("stage1_ic_ir", 0.0))
            if np.isfinite(icir):
                a["ic_ir"].append(icir)

        # Top-K by final_ensemble_score (lower is better)
        if np.any(np.isfinite(final_vals)):
            valid_idx = np.where(np.isfinite(final_vals))[0]
            if valid_idx.size > 0:
                sorted_idx = valid_idx[np.argsort(final_vals[valid_idx])]
                k = min(int(top_k), sorted_idx.size)
                for idx in sorted_idx[:k]:
                    f = feats[int(idx)]
                    agg[f]["topk_final_count"] += 1

        # Top-K by linear_coef_abs (higher is better)
        if np.any(np.isfinite(linear_vals)):
            valid_idx = np.where(np.isfinite(linear_vals))[0]
            if valid_idx.size > 0:
                sorted_idx = valid_idx[np.argsort(-linear_vals[valid_idx])]
                k = min(int(top_k), sorted_idx.size)
                for idx in sorted_idx[:k]:
                    f = feats[int(idx)]
                    agg[f]["topk_linear_count"] += 1

    # Convert aggregates into a DataFrame for ranking
    rows: List[Dict[str, Any]] = []
    for feat, a in agg.items():
        fs = np.array(a["final_scores"], dtype=float) if a["final_scores"] else np.array([], dtype=float)
        ls = np.array(a["linear_scores"], dtype=float) if a["linear_scores"] else np.array([], dtype=float)
        ic = np.array(a["mean_abs_ic"], dtype=float) if a["mean_abs_ic"] else np.array([], dtype=float)
        icir = np.array(a["ic_ir"], dtype=float) if a["ic_ir"] else np.array([], dtype=float)
        present = int(a["present_in"])
        rows.append(
            {
                "feature": feat,
                "present_in": present,
                "presence_frac": float(present / max(1, total_instruments)),
                "mean_final_score": float(np.nanmean(fs)) if fs.size else float("nan"),
                "median_final_score": float(np.nanmedian(fs)) if fs.size else float("nan"),
                "mean_linear_coef_abs": float(np.nanmean(ls)) if ls.size else 0.0,
                "median_linear_coef_abs": float(np.nanmedian(ls)) if ls.size else 0.0,
                "mean_stage1_mean_abs_ic": float(np.nanmean(ic)) if ic.size else 0.0,
                "mean_stage1_ic_ir": float(np.nanmean(icir)) if icir.size else 0.0,
                "topk_final_frac": float(a["topk_final_count"] / max(1, total_instruments)),
                "topk_linear_frac": float(a["topk_linear_count"] / max(1, total_instruments)),
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows).set_index("feature")

    # Ranks: lower mean_final_score is better; higher mean_linear_coef_abs is better.
    df["rank_mean_final"] = df["mean_final_score"].rank(ascending=True, method="average")
    if (df["mean_linear_coef_abs"] > 0.0).any():
        df["rank_mean_linear"] = df["mean_linear_coef_abs"].rank(ascending=False, method="average")
    else:
        df["rank_mean_linear"] = np.nan

    # Normalize ranks and combine into final ensemble rank.
    max_rf = float(df["rank_mean_final"].max() or 1.0)
    norm_rf = df["rank_mean_final"] / max_rf

    if df["rank_mean_linear"].notna().any():
        max_rl = float(df["rank_mean_linear"].max() or 1.0)
        norm_rl = df["rank_mean_linear"] / max_rl
        # Where linear rank is available, blend; otherwise fall back to norm_rf only.
        df["final_ensemble_score"] = np.where(
            np.isfinite(norm_rl), (norm_rf + norm_rl) / 2.0, norm_rf
        )
    else:
        df["final_ensemble_score"] = norm_rf

    # Convert back to plain dict
    out: Dict[str, Dict[str, Any]] = {}
    for feat, row in df.sort_values("final_ensemble_score", ascending=True).iterrows():
        out[feat] = {
            "present_in": int(row["present_in"]),
            "presence_frac": float(row["presence_frac"]),
            "mean_final_score": float(row["mean_final_score"]),
            "median_final_score": float(row["median_final_score"]),
            "mean_linear_coef_abs": float(row["mean_linear_coef_abs"]),
            "median_linear_coef_abs": float(row["median_linear_coef_abs"]),
            "mean_stage1_mean_abs_ic": float(row["mean_stage1_mean_abs_ic"]),
            "mean_stage1_ic_ir": float(row["mean_stage1_ic_ir"]),
            "topk_final_frac": float(row["topk_final_frac"]),
            "topk_linear_frac": float(row["topk_linear_frac"]),
            "rank_mean_final": float(row["rank_mean_final"]),
            "rank_mean_linear": float(row["rank_mean_linear"])
            if np.isfinite(row["rank_mean_linear"])
            else float("nan"),
            "final_ensemble_score": float(row["final_ensemble_score"]),
        }
    return out


def save_ensemble(stats: Dict[str, Dict[str, Any]], out_json: Path, out_list: Path) -> None:
    # Already sorted by final_ensemble_score in ensemble_rankings
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    with open(out_list, "w", encoding="utf-8") as fh:
        for name in stats.keys():
            fh.write(f"{name}\n")


def main() -> None:
    args = parse_args()
    instruments = [tok.strip().upper() for tok in args.instruments.split(",") if tok.strip()]
    base = str(args.base_gran).upper()
    aux = str(args.aux).upper()
    cfg_dir = Path(args.config_dir)

    per_inst: Dict[str, Dict[str, Dict[str, float]]] = {}
    for inst in instruments:
        data = _load_per_instrument(inst, base, aux, cfg_dir, args.pattern)
        if data is not None:
            per_inst[inst] = data

    total = len(instruments)
    if not per_inst:
        print("[ERROR] No per-instrument ranking JSONs could be loaded; nothing to ensemble.", flush=True)
        return

    stats = ensemble_rankings(per_inst, total_instruments=total, top_k=int(args.top_k))
    save_ensemble(stats, Path(args.out_json), Path(args.out_list))
    print(
        {
            "features": len(stats),
            "instruments_used": sorted(per_inst.keys()),
            "out_json": args.out_json,
            "out_list": args.out_list,
        }
    )


if __name__ == "__main__":  # pragma: no cover
    main()

