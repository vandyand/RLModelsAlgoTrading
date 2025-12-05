from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features_loader import load_feature_panel


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, num_instruments: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, num_instruments)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.policy(h)
        v = self.value(h).squeeze(-1)
        return h, mu, v


def standardize_apply(X: pd.DataFrame, stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for c in X.columns:
        m, s = stats.get(c, (0.0, 1.0))
        if s == 0.0:
            s = 1.0
        Xn[c] = (X[c] - m) / s
    return Xn.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer actions from AC Multi-20 checkpoint using latest M5/H1/D features")
    p.add_argument("--model", required=True, help="Path to checkpoint saved by ac-multi20 trainer (e.g., ac-multi20/checkpoints/....pt)")
    p.add_argument("--instruments", default="", help="Comma-separated instruments; default loads from checkpoint meta")
    p.add_argument("--lookback-days", type=int, default=5)
    p.add_argument("--start", default="", help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default="", help="YYYY-MM-DD (optional)")
    p.add_argument("--max-units", type=int, default=100, help="Units for state +/-1 (after threshold mapping)")
    p.add_argument("--grans", default="", help="Override granularities to include (e.g., 'H1,D'); default from checkpoint meta")
    p.add_argument("--enter-long", type=float, default=None)
    p.add_argument("--exit-long", type=float, default=None)
    p.add_argument("--enter-short", type=float, default=None)
    p.add_argument("--exit-short", type=float, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    meta = ckpt.get("meta", {})
    col_order: List[str] = ckpt.get("col_order", [])
    stats: Dict[str, Tuple[float, float]] = ckpt.get("feature_stats", {})
    hidden = int(meta.get("hidden", 256))
    meta_grans = meta.get("grans", ["M5","H1","D"])

    # Instruments
    if args.instruments.strip():
        instruments = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]
    else:
        instruments = list(meta.get("instruments", []))
    if len(instruments) == 0:
        raise RuntimeError("No instruments provided and none in checkpoint meta")

    # Load a small panel and take last row
    lookback_days = int(args.lookback_days)
    start = args.start.strip() or None
    end = args.end.strip() or None
    grans = [g.strip().upper() for g in (args.grans or ",".join(meta_grans)).split(',') if g.strip()]
    X_panel, closes = load_feature_panel(
        type("_Tmp", (), {
            "instruments": instruments,
            "lookback_days": lookback_days,
            "start_date": start,
            "end_date": end,
            "data_root": "continuous-trader/data",
            "features_root": "continuous-trader/data/features",
            "include_grans": grans,
        })()
    )

    # Flatten deterministically to match training
    flat_cols: List[Tuple[str, str]] = []
    for inst in instruments:
        for col in X_panel[inst].columns:
            flat_cols.append((inst, col))
    X_flat = X_panel.reindex(columns=pd.MultiIndex.from_tuples(flat_cols)).copy()
    X_flat.columns = [f"{i}::{c}" for i, c in X_flat.columns]

    if X_flat.empty:
        raise RuntimeError("No features available for inference")
    x_last = X_flat.tail(1)

    # Reorder/align columns to training order
    if col_order:
        for c in col_order:
            if c not in x_last.columns:
                x_last[c] = 0.0
        x_last = x_last[col_order]
    # Standardize
    if stats:
        x_last = standardize_apply(x_last, stats)

    input_dim = int(meta.get("input_dim", x_last.shape[1]))
    num_inst = int(meta.get("num_instruments", len(instruments)))

    if input_dim != x_last.shape[1]:
        cols = list(x_last.columns)
        if x_last.shape[1] < input_dim:
            need = input_dim - x_last.shape[1]
            for i in range(need):
                x_last[f"__pad_{i}"] = 0.0
        else:
            x_last = x_last.iloc[:, :input_dim]

    model = ActorCritic(input_dim=input_dim, num_instruments=num_inst, hidden=hidden)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    xt = torch.tensor(x_last.values, dtype=torch.float32)
    with torch.no_grad():
        _, mu, _ = model(xt)
        a = torch.sigmoid(mu)[0].cpu().numpy().astype(float).tolist()

    import math
    def trunc4(x: float) -> float:
        return math.trunc(float(x) * 1e4) / 1e4

    a4 = [trunc4(v) for v in a[:len(instruments)]]
    # Threshold mapping (absolute) to state {-1,0,1}
    th = meta.get("thresholds", {})
    el = float(args.enter_long if args.enter_long is not None else th.get("enter_long", 0.8))
    xl = float(args.exit_long if args.exit_long is not None else th.get("exit_long", 0.6))
    es = float(args.enter_short if args.enter_short is not None else th.get("enter_short", 0.2))
    xs = float(args.exit_short if args.exit_short is not None else th.get("exit_short", 0.4))
    state = []
    for v in a[:len(instruments)]:
        s = 0
        if v > el:
            s = 1
        elif v < es:
            s = -1
        # If in long/short we'd check exits; here we just emit entry mapping
        state.append(s)
    units = [int(round(s * float(args.max_units))) for s in state]

    ts = None
    try:
        ts = x_last.index[-1].isoformat()
    except Exception:
        pass

    print(json.dumps({
        "ph": "infer",
        "ts": ts,
        "inst": instruments,
        "a": a4,
        "units": units,
        "state": state,
    }), flush=True)


if __name__ == "__main__":
    main()
