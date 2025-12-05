#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
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
CT_ROOT = os.path.join(FX_ROOT, "continuous-trader")
if CT_ROOT not in sys.path:
    sys.path.append(CT_ROOT)

from model import SiameseMultiGranActorCritic  # type: ignore
from instruments import load_68  # type: ignore


@dataclass
class M5Config:
    instruments_csv: Optional[str]
    feature_dir: str
    feature_grans: List[str]  # e.g. ["M5","H1","D"]
    max_units: int = 100
    batch_size: int = 1  # sequential over time
    epochs: int = 1
    gamma: float = 0.99
    actor_sigma: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    reward_scale: float = 1.0
    seed: int = 42
    model_path: str = "forex-rl/continuous-trader/checkpoints/m5_siamese.pt"


# ---------- Data loading (M5-focused) ----------


def _safe_inst(inst: str) -> str:
    return inst.replace('/', '_')


def _load_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect index in first column
    if df.columns[0].lower() in ("timestamp", "date", "index"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], utc=True)
        df = df.set_index(df.columns[0]).sort_index()
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    return df


essential_cols_cache: Dict[str, List[str]] = {}
_ts_colname_cache: Dict[str, str] = {}
_ret_col_cache: Dict[str, str] = {}


def load_flat_matrix_and_index(cfg: M5Config, instruments: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[int]]]]:
    """Load features for each (gran,instrument), align by M5 timestamps, build a flat matrix and an indices mapping.
    For memory reasons, start with M5 only by default; H1/D can be added if present.
    """
    grans = [g.strip().upper() for g in cfg.feature_grans if g.strip()]
    feat_root = cfg.feature_dir

    # Load M5 timestamps from EUR_USD as base
    base = None
    m5_path = os.path.join(feat_root, "M5", f"EUR_USD_M5_features.csv")
    if not os.path.exists(m5_path):
        raise RuntimeError("Missing M5 features for EUR_USD; export features first.")
    base_df = _load_feature_csv(m5_path)
    base = base_df.index

    # Build per (gran, inst) DataFrames aligned to base M5 timestamps
    frames_by_gran_inst: Dict[str, Dict[str, pd.DataFrame]] = {g: {} for g in grans}

    for gran in grans:
        for inst in instruments:
            p = os.path.join(feat_root, gran, f"{_safe_inst(inst)}_{gran}_features.csv")
            if not os.path.exists(p):
                continue
            df = _load_feature_csv(p)
            if gran != "M5":
                # Upsample to M5 by forward-fill to latest available within the day/hour
                rule = '5T'
                # First ensure it's at least hourly or daily index
                df = df.resample(rule).ffill().reindex(base).ffill().fillna(0.0)
            else:
                df = df.reindex(base).ffill().fillna(0.0)
            # Keep only essential numeric cols
            if p not in essential_cols_cache:
                essential_cols_cache[p] = [c for c in df.columns if isinstance(df[c].iloc[0], (int, float, np.floating)) or np.issubdtype(df[c].dtype, np.number)]
            df = df[essential_cols_cache[p]].astype(np.float32)
            frames_by_gran_inst[gran][inst] = df

    # Construct flat X and index map
    blocks: List[pd.DataFrame] = []
    indices: Dict[str, Dict[str, List[int]]] = {}
    col_start = 0
    for gran in grans:
        indices[gran] = {}
        for inst in instruments:
            df = frames_by_gran_inst.get(gran, {}).get(inst)
            if df is None:
                indices[gran][inst] = []
                continue
            cols = df.columns
            blocks.append(df)
            indices[gran][inst] = list(range(col_start, col_start + len(cols)))
            col_start += len(cols)

    if not blocks:
        raise RuntimeError("No feature blocks loaded.")

    X = pd.concat(blocks, axis=1).astype(np.float32)
    return X, indices


# ---------- Memory-efficient streaming helpers ----------


def _read_header_numeric_cols(path: str) -> List[str]:
    """Read only header to get numeric columns (excluding timestamp)."""
    if not os.path.exists(path):
        return []
    df0 = pd.read_csv(path, nrows=1)
    cols = list(df0.columns)
    if not cols:
        return []
    # First col is timestamp
    return [c for c in cols[1:]]


def _load_feature_block_range(path: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, base_index: pd.DatetimeIndex, expected_cols: Optional[List[str]] = None, gran: str = "M5") -> pd.DataFrame:
    """Load a time slice [start_ts, end_ts] from a feature CSV and align to base_index with ffill.
    For H1/D, resample to 5T before aligning.
    """
    if not os.path.exists(path):
        # Missing file: return zeros with expected columns if available
        cols = expected_cols or []
        if not cols:
            return pd.DataFrame(index=base_index)
        z = pd.DataFrame(0.0, index=base_index, columns=cols, dtype=np.float32)
        return z

    # Determine timestamp column name from header (cached)
    ts_colname = _ts_colname_cache.get(path)
    if ts_colname is None:
        head = pd.read_csv(path, nrows=0)
        if head.shape[1] == 0:
            return pd.DataFrame(0.0, index=base_index, columns=expected_cols or [], dtype=np.float32)
        ts_colname = head.columns[0]
        _ts_colname_cache[path] = ts_colname
    usecols = None
    if expected_cols is not None and len(expected_cols) > 0:
        usecols = [ts_colname] + expected_cols  # include timestamp + needed cols (all strings)

    it = pd.read_csv(
        path,
        parse_dates=[ts_colname],
        dtype="float32",
        usecols=usecols,
        chunksize=200_000,
    )

    pieces: List[pd.DataFrame] = []
    for chunk in it:
        # Ensure index set to timestamp
        # Ensure index set to timestamp
        chunk[ts_colname] = pd.to_datetime(chunk[ts_colname], utc=True)
        mask = (chunk[ts_colname] >= start_ts) & (chunk[ts_colname] <= end_ts)
        sel = chunk.loc[mask]
        if not sel.empty:
            sel = sel.set_index(ts_colname).sort_index()
            pieces.append(sel)
        # Early exit if we've passed end_ts (chunks are in order)
        if chunk[ts_colname].iloc[-1] > end_ts:
            break

    if pieces:
        df = pd.concat(pieces, axis=0)
    else:
        # No data found in range; create empty frame to be aligned and ffilled (zeros)
        df = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    if gran != "M5":
        # Upsample to M5 by forward-fill to latest available within the hour/day
        df = df.resample("5min").ffill()

    # Align to base index and ffill within the chunk; fill remaining gaps with 0
    out = df.reindex(base_index).ffill()
    # Ensure expected columns exist/order without per-column insertions (avoid fragmentation warnings)
    if expected_cols is not None and len(expected_cols) > 0:
        out = out.reindex(columns=expected_cols, fill_value=0.0)
    out = out.fillna(0.0).astype(np.float32)
    return out


def build_indices_from_headers(feature_dir: str, grans: List[str], instruments: List[str]) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[Tuple[str, str], List[str]], int]:
    """Scan headers only to build a stable indices mapping and column cache without loading full data.
    Returns (indices_by_gran_inst, cols_cache, total_dim).
    """
    indices: Dict[str, Dict[str, List[int]]] = {g: {} for g in grans}
    cols_cache: Dict[Tuple[str, str], List[str]] = {}
    col_start = 0
    for gran in grans:
        for inst in instruments:
            path = os.path.join(feature_dir, gran, f"{_safe_inst(inst)}_{gran}_features.csv")
            cols = _read_header_numeric_cols(path)
            cols_cache[(gran, inst)] = cols
            if len(cols) == 0:
                indices[gran][inst] = []
            else:
                indices[gran][inst] = list(range(col_start, col_start + len(cols)))
                col_start += len(cols)
    total_dim = col_start
    return indices, cols_cache, total_dim


def extract_target_returns_m5(cfg: M5Config, base_index: pd.DatetimeIndex, target_inst: str = "EUR_USD") -> np.ndarray:
    """Extract target M5 returns aligned to base_index using streaming read.
    Prefer exact <INST>_ret_1, then <INST>_roc_1, then <INST>_roc_5.
    """
    path = os.path.join(cfg.feature_dir, "M5", f"{_safe_inst(target_inst)}_M5_features.csv")
    # Determine column once, cache it
    colname = _ret_col_cache.get(path)
    if colname is None:
        head = pd.read_csv(path, nrows=0)
        cols = list(head.columns)
        lower_to_col: Dict[str, str] = {c.lower(): c for c in cols}
        prefix = target_inst.replace('_', '/').lower() + "_"
        preferred = [prefix + "ret_1", prefix + "roc_1", prefix + "roc_5"]
        found = None
        for key in preferred:
            if key in lower_to_col:
                found = lower_to_col[key]
                break
        if found is None:
            # suffix search among prefixed
            for c in cols:
                cl = c.lower()
                if cl.startswith(prefix) and (cl.endswith("ret_1") or cl.endswith("roc_1") or cl.endswith("roc_5")):
                    found = c
                    break
        if found is None:
            _ret_col_cache[path] = "__NONE__"
        else:
            _ret_col_cache[path] = found
        colname = _ret_col_cache[path]

    if colname and colname != "__NONE__":
        df = _load_feature_block_range(
            path=path,
            start_ts=base_index[0],
            end_ts=base_index[-1],
            base_index=base_index,
            expected_cols=[colname],
            gran="M5",
        )
        s = df[colname].astype(float).values if colname in df.columns else np.zeros(len(base_index), dtype=float)
        return np.nan_to_num(s, nan=0.0)

    # Last resort: zeros
    return np.zeros(len(base_index), dtype=float)


# ---------- Composite reward (windowed episodic) ----------


def _profit_ratio(gross_profit: float, gross_loss: float) -> float:
    return gross_profit / (abs(gross_loss) + 1e-8)


def _r2_consistency(y: np.ndarray) -> float:
    n = len(y)
    if n <= 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean(); y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum() ** 2
    den = ((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum() + 1e-8
    return float(num / den)


def window_stats(pnls: List[float]) -> Dict[str, float]:
    if len(pnls) == 0:
        return {"pnl_sum": 0.0, "sharpe": 0.0, "sortino": 0.0, "dd": 0.0, "var": 0.0, "pr": 0.0, "r2": 0.0}
    arr = np.array(pnls, dtype=float)
    pnl_sum = float(arr.sum())
    mean = float(arr.mean())
    std = float(arr.std() + 1e-8)
    neg = arr[arr < 0]
    sortino_denom = float(np.sqrt(np.mean(neg * neg) + 1e-8)) if len(neg) > 0 else 1.0
    sharpe = mean / std
    sortino = mean / sortino_denom
    # Max drawdown on cumulative pnl
    cum = np.cumsum(arr)
    peak = -1e9
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    dd_frac = float(mdd)
    var = float(np.var(arr))
    gp = float(arr[arr > 0].sum())
    gl = float(arr[arr < 0].sum())
    pr = _profit_ratio(gp, gl)
    r2 = _r2_consistency(cum)
    return {"pnl_sum": pnl_sum, "sharpe": sharpe, "sortino": sortino, "dd": dd_frac, "var": var, "pr": pr, "r2": r2}


def composite_reward(pnl_step: float, win_short: List[float], win_long: List[float], w: Dict[str, float]) -> float:
    """Combine step pnl with short/long window stats into a scalar reward.
    w keys: w_step, w_pnl, w_sharpe, w_sortino, w_dd, w_consistency, w_pr, w_var
    dd is negative; we add (1 - |dd|) weight.
    """
    s = window_stats(win_short)
    l = window_stats(win_long)
    # Average stats across windows for stability
    pnl_sum = 0.5 * (s["pnl_sum"] + l["pnl_sum"])
    sharpe = 0.5 * (s["sharpe"] + l["sharpe"]) 
    sortino = 0.5 * (s["sortino"] + l["sortino"]) 
    dd_term = 0.5 * (s["dd"] + l["dd"])  # negative value (drawdown)
    r2 = 0.5 * (s["r2"] + l["r2"]) 
    pr = 0.5 * (s["pr"] + l["pr"]) 
    var = 0.5 * (s["var"] + l["var"]) 

    reward = (
        w.get("w_step", 1.0) * pnl_step +
        w.get("w_pnl", 0.1) * pnl_sum +
        w.get("w_sharpe", 0.05) * sharpe +
        w.get("w_sortino", 0.05) * sortino +
        w.get("w_dd", 0.05) * (1.0 + dd_term) +
        w.get("w_consistency", 0.05) * r2 +
        w.get("w_pr", 0.05) * pr -
        w.get("w_var", 0.02) * var
    )
    return float(reward)


# ---------- Training ----------


def main() -> None:
    p = argparse.ArgumentParser(description="M5 offline training with Siamese multi-gran trunk and composite reward")
    p.add_argument("--instruments-csv", default=None)
    p.add_argument("--feature-dir", default=os.path.join(FX_ROOT, "continuous-trader", "data", "features"))
    p.add_argument("--feature-grans", default="M5,H1,D")
    p.add_argument("--max-units", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor-sigma", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.001)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", default="forex-rl/continuous-trader/checkpoints/m5_siamese.pt")
    # Reward weights and windows
    p.add_argument("--w-step", type=float, default=1.0, help="Weight for step pnl")
    p.add_argument("--w-pnl", type=float, default=0.1, help="Weight for window pnl sum")
    p.add_argument("--w-sharpe", type=float, default=0.05)
    p.add_argument("--w-sortino", type=float, default=0.05)
    p.add_argument("--w-dd", type=float, default=0.05, help="Weight for (1+drawdown)")
    p.add_argument("--w-consistency", type=float, default=0.05, help="Weight for R^2 consistency")
    p.add_argument("--w-pr", type=float, default=0.05, help="Weight for profit ratio")
    p.add_argument("--w-var", type=float, default=0.02, help="Penalty weight for variance")
    p.add_argument("--wshort", type=int, default=12, help="Short window size (M5 bars)")
    p.add_argument("--wlong", type=int, default=288, help="Long window size (M5 bars)")
    p.add_argument("--no-attention", action="store_true", help="Disable attention aggregation (use mean)")
    p.add_argument("--only-eur-or-usd", action="store_true", help="Limit instruments to those containing EUR or USD")
    p.add_argument("--only-target", action="store_true", help="Use only target instruments as features")
    p.add_argument("--target-instruments", default="EUR_USD", help="Comma-separated list of target instruments (e.g. EUR_USD,AUD_USD,EUR_JPY)")
    p.add_argument("--week-chunks", action="store_true", help="Process data in 1-week chunks")
    p.add_argument("--reset-rewards-at-weekend", action="store_true", help="Reset episodic reward windows across weekends")
    # Streaming/Chunking options
    p.add_argument("--chunk-minutes", type=int, default=24*60, help="Minutes per training chunk (default 1 day)")
    p.add_argument("--chunk-overlap-min", type=int, default=60, help="Overlap minutes between chunks for ffill stability")
    p.add_argument("--dry-run", action="store_true", help="Load shapes only then exit")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def _parse_targets(s: str) -> List[str]:
        items = [x.strip() for x in (s or "").split(',') if x.strip()]
        norm = [x.replace('/', '_').upper() for x in items]
        return norm or ["EUR_USD"]

    targets = _parse_targets(args.target_instruments)

    instruments = load_68(args.instruments_csv)
    if args.only_target:
        instruments = list(dict.fromkeys(targets))
    elif args.only_eur_or_usd:
        instruments = [inst for inst in instruments if ("EUR" in inst or "USD" in inst)]
        # Ensure targets are included
        for t in targets:
            if t not in instruments:
                instruments.insert(0, t)
    else:
        # Ensure targets are included
        for t in reversed(targets):
            if t not in instruments:
                instruments.insert(0, t)

    cfg = M5Config(
        instruments_csv=args.instruments_csv,
        feature_dir=args.feature_dir,
        feature_grans=[s.strip().upper() for s in (args.feature_grans or "M5").split(',') if s.strip()],
        max_units=int(args.max_units),
        epochs=int(args.epochs),
        gamma=float(args.gamma),
        actor_sigma=float(args.actor_sigma),
        entropy_coef=float(args.entropy_coef),
        value_coef=float(args.value_coef),
        max_grad_norm=float(args.max_grad_norm),
        reward_scale=float(args.reward_scale),
        seed=int(args.seed),
        model_path=args.model_path,
    )

    print(json.dumps({"status": "init_stream", "grans": cfg.feature_grans, "feature_dir": cfg.feature_dir}), flush=True)

    # Build indices from headers only (no data load)
    idx_map, cols_cache, total_dim = build_indices_from_headers(cfg.feature_dir, cfg.feature_grans, instruments)
    # Load base M5 index from the first target instrument to determine time range
    base_inst = targets[0] if len(targets) > 0 else "EUR_USD"
    m5_path = os.path.join(cfg.feature_dir, "M5", f"{_safe_inst(base_inst)}_M5_features.csv")
    base_full = _load_feature_csv(m5_path).index
    print(json.dumps({"status": "header_indices", "D": int(total_dim), "T": int(len(base_full)), "instruments": len(instruments), "targets": targets}), flush=True)
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SiameseMultiGranActorCritic(
        flat_input_dim=total_dim,
        indices_by_gran_inst=idx_map,
        embed_dim=64,
        hidden_per_inst=256,
        policy_hidden=512,
        value_hidden=512,
        use_attention=(not args.no_attention),
    ).to(device)
    opt = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)

    # Reward windows (overlapping): short ~1h (12 M5 bars), long ~1 day (288 M5 bars)
    W_SHORT = int(args.wshort)
    W_LONG = int(args.wlong)
    win_short: List[float] = []
    win_long: List[float] = []

    # Training loop over time steps
    sigma = float(cfg.actor_sigma)
    log_sigma_const = float(np.log(sigma + 1e-8))

    # NAV proxy: 1.0 baseline; PnL step computed from returns times position
    nav = 1.0

    def step_epoch(train: bool) -> Dict[str, float]:
        T = len(base_full)
        total = {"loss": 0.0, "actor": 0.0, "value": 0.0, "entropy": 0.0, "reward": 0.0}
        steps = 0
        if train:
            net.train()
        else:
            net.eval()
        # Chunked iteration over time to limit memory
        step_minutes = 5
        chunk_minutes = int(args.chunk_minutes)
        if args.week_chunks:
            chunk_minutes = 7 * 24 * 60
        chunk_T = max(1, int(chunk_minutes // step_minutes))
        ovl_T = int(max(0, args.chunk_overlap_min // step_minutes))
        start_idx = 0
        while start_idx < T - 1:
            end_idx = min(T, start_idx + chunk_T)
            ext_start_idx = max(0, start_idx - ovl_T)
            cur_base = base_full[ext_start_idx:end_idx]

            # Build current X chunk from files on the fly per (gran, inst)
            blocks: List[pd.DataFrame] = []
            for gran in cfg.feature_grans:
                for inst in instruments:
                    cols = cols_cache.get((gran, inst), [])
                    if len(cols) == 0:
                        continue
                    path = os.path.join(cfg.feature_dir, gran, f"{_safe_inst(inst)}_{gran}_features.csv")
                    df_block = _load_feature_block_range(
                        path=path,
                        start_ts=cur_base[0],
                        end_ts=cur_base[-1],
                        base_index=cur_base,
                        expected_cols=cols,
                        gran=gran,
                    )
                    blocks.append(df_block)
            if not blocks:
                # Nothing to train on
                break
            X_chunk = pd.concat(blocks, axis=1).astype(np.float32)
            if not X_chunk.index.equals(cur_base):
                X_chunk = X_chunk.reindex(cur_base).ffill().fillna(0.0)
            print(json.dumps({"status": "chunk_built", "start": str(cur_base[0]), "end": str(cur_base[-1]), "rows": int(len(cur_base)), "cols": int(X_chunk.shape[1])}), flush=True)

            # Extract returns aligned to cur_base for primary target
            r_chunk = extract_target_returns_m5(cfg, cur_base, target_inst=base_inst)

            # Iterate within chunk (excluding the overlapped prefix for training updates)
            start_local = start_idx - ext_start_idx
            for t_local in range(max(0, start_local), len(cur_base) - 1):
                # Build state at time t_local
                s_t_np = X_chunk.iloc[t_local:t_local+1].values
                s_tp1_np = X_chunk.iloc[t_local+1:t_local+2].values
                s_t = torch.tensor(s_t_np, dtype=torch.float32, device=device)
                s_tp1 = torch.tensor(s_tp1_np, dtype=torch.float32, device=device)
                # Next-step EUR_USD return for pnl computation
                r_tp1 = float(r_chunk[t_local + 1])

                a_t, logit_t, v_t = net(s_t)
                with torch.no_grad():
                    _, _, v_tp1 = net(s_tp1)
                # Policy noise
                eps = torch.randn_like(logit_t)
                pre = logit_t + sigma * eps
                # Position from action
                pos = (torch.sigmoid(pre) - 0.5) * 2.0 * float(cfg.max_units)

                # Step PnL proxy
                pnl_step = float(pos.item()) * float(r_tp1)

                # Update windows
                win_short.append(pnl_step)
                if len(win_short) > W_SHORT:
                    win_short.pop(0)
                win_long.append(pnl_step)
                if len(win_long) > W_LONG:
                    win_long.pop(0)

                # Composite reward
                rw = composite_reward(pnl_step, win_short, win_long, w={
                    "w_step": args.w_step,
                    "w_pnl": args.w_pnl,
                    "w_sharpe": args.w_sharpe,
                    "w_sortino": args.w_sortino,
                    "w_dd": args.w_dd,
                    "w_consistency": args.w_consistency,
                    "w_pr": args.w_pr,
                    "w_var": args.w_var,
                }) * cfg.reward_scale

                # Advantage
                adv = (rw + cfg.gamma * v_tp1[0] - v_t[0]).detach()
                # Log prob under Normal on pre-logit
            logprob = -0.5 * (((pre - logit_t) / sigma) ** 2 + np.log(2 * np.pi) + 2 * log_sigma_const)
            # Entropy of Normal(sigma): 0.5 * (1 + log(2*pi*sigma^2))
            entropy = 0.5 * (1.0 + torch.log(2 * torch.tensor(np.pi) * (sigma ** 2)))
                actor_loss = -adv * logprob.mean()
                value_target = (rw + cfg.gamma * v_tp1[0]).detach()
                value_loss = cfg.value_coef * 0.5 * (value_target - v_t[0]) ** 2
                loss = actor_loss + value_loss - cfg.entropy_coef * entropy

                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()

                total["loss"] += float(loss.item())
                total["actor"] += float(actor_loss.item())
                total["value"] += float(value_loss.item())
                total["entropy"] += float(entropy.item()) if isinstance(entropy, torch.Tensor) else float(entropy)
                total["reward"] += float(rw)
                steps += 1
                if steps % 5000 == 0:
                    print(json.dumps({"status": "progress", "steps": int(steps), "loss": float(total["loss"]) / max(1, steps), "reward": float(total["reward"]) / max(1, steps)}), flush=True)

                # Optional weekend reset of reward windows
                if args.reset_rewards_at_weekend:
                    cur_ts = cur_base[t_local]
                    nxt_ts = cur_base[t_local + 1]
                    if (cur_ts.weekday() == 4 and nxt_ts.weekday() < 4) or (nxt_ts - cur_ts > pd.Timedelta(days=2)):
                        win_short.clear()
                        win_long.clear()

            # Cleanup to free memory between chunks
            import gc
            try:
                del blocks
            except Exception:
                pass
            try:
                del X_chunk
            except Exception:
                pass
            try:
                del r_chunk
            except Exception:
                pass
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

            start_idx = end_idx
        for k in total:
            total[k] = total[k] / max(1, steps)
        return total

    for ep in range(cfg.epochs):
        tr = step_epoch(train=True)
        print(json.dumps({"phase": "train", "epoch": ep + 1, **tr}), flush=True)

    # Save
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    torch.save({
        "cfg": asdict(cfg),
        "indices": idx_map,
        "model_state": net.state_dict(),
    }, cfg.model_path)
    print(json.dumps({"saved": cfg.model_path}), flush=True)


if __name__ == "__main__":
    main()
