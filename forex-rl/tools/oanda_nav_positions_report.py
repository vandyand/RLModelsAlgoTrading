#!/usr/bin/env python3
from __future__ import annotations

"""
Terminal visualizer for OANDA NAV + positions snapshots.

Reads the JSONL file produced by `oanda_nav_positions_sampler.py` and plots:
- Account NAV and balance over time (USD)
- Unrealized PnL over time (USD)
- Margin used / available over time (USD)
- Per-instrument margin used over time for the top-N instruments

Requires `plotext` (already used in WFO reports).
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import plotext as plt  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    plt = None


@dataclass
class Snapshot:
    t: datetime
    t_label: str
    account: Dict[str, Any]
    positions: List[Dict[str, Any]]


def _parse_time(s: str) -> Optional[datetime]:
    try:
        # Handle both "....Z" and "+00:00" variants
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_snapshots(path: Path) -> List[Snapshot]:
    out: List[Snapshot] = []
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("event") != "oanda_nav_positions_snapshot":
                continue
            t_raw = str(rec.get("time_iso") or "")
            t = _parse_time(t_raw)
            if t is None:
                continue
            t_label = t.strftime("%m-%d %H:%M")
            account = rec.get("account") or {}
            positions = rec.get("positions") or []
            if not isinstance(account, dict):
                account = {}
            if not isinstance(positions, list):
                positions = []
            out.append(Snapshot(t=t, t_label=t_label, account=account, positions=positions))
    out.sort(key=lambda s: s.t)
    return out


def downsample(xs: List[str], ys: List[float], max_points: int) -> Tuple[List[str], List[float]]:
    if len(xs) <= max_points or max_points <= 0:
        return xs, ys
    step = max(1, len(xs) // max_points)
    return xs[::step], ys[::step]


def _extract_account_series(snaps: List[Snapshot]) -> Dict[str, List[Optional[float]]]:
    nav: List[Optional[float]] = []
    balance: List[Optional[float]] = []
    unreal: List[Optional[float]] = []
    margin_used: List[Optional[float]] = []
    margin_avail: List[Optional[float]] = []

    for s in snaps:
        a = s.account
        nav.append(_to_float(a.get("NAV") or a.get("trueNAV")))
        balance.append(_to_float(a.get("balance")))
        unreal.append(_to_float(a.get("unrealizedPL") or a.get("trueUnrealizedPL")))
        margin_used.append(_to_float(a.get("marginUsed")))
        margin_avail.append(_to_float(a.get("marginAvailable")))

    return {
        "nav": nav,
        "balance": balance,
        "unrealized": unreal,
        "margin_used": margin_used,
        "margin_available": margin_avail,
    }


def _extract_top_instrument_margin_series(
    snaps: List[Snapshot],
    top_n: int = 3,
) -> Tuple[List[str], Dict[str, List[float]]]:
    # First pass: compute max marginUsed per instrument across time
    max_by_inst: Dict[str, float] = {}
    for s in snaps:
        for p in s.positions:
            inst = str(p.get("instrument") or "").upper()
            if not inst:
                continue
            mu = _to_float(p.get("marginUsed"))
            if mu is None:
                continue
            prev = max_by_inst.get(inst, 0.0)
            if mu > prev:
                max_by_inst[inst] = mu

    # Pick top-N instruments by peak margin used
    ordered = sorted(max_by_inst.items(), key=lambda kv: kv[1], reverse=True)
    top_insts = [inst for inst, _ in ordered[:top_n]]
    if not top_insts:
        return [], {}

    # Second pass: build full time series per instrument (0.0 when absent)
    series: Dict[str, List[float]] = {inst: [] for inst in top_insts}
    for s in snaps:
        by_inst_mu: Dict[str, float] = {}
        for p in s.positions:
            inst = str(p.get("instrument") or "").upper()
            if inst not in top_insts:
                continue
            mu = _to_float(p.get("marginUsed")) or 0.0
            by_inst_mu[inst] = by_inst_mu.get(inst, 0.0) + mu
        for inst in top_insts:
            series[inst].append(by_inst_mu.get(inst, 0.0))

    return top_insts, series


def _sanitize(values: List[Optional[float]]) -> List[float]:
    out: List[float] = []
    last: float = 0.0
    for v in values:
        if v is None:
            out.append(last)
        else:
            last = float(v)
            out.append(last)
    return out


def _plot_series(
    title: str,
    xs: List[str],
    ys: List[float],
    xlabel: str = "time",
) -> None:
    print(f"\n== {title} ==")
    if not xs or not ys:
        print("(no data)")
        return
    if plt is None:
        print("(plotext not installed; values:) ")
        print(list(zip(xs, ys)))
        return
    # Use numeric x-coordinates and map labels manually to avoid plotext date parsing.
    idx = list(range(len(xs)))
    # Show at most ~10 x tick labels for readability.
    if len(idx) <= 10:
        xticks_pos = idx
        xticks_lab = xs
    else:
        step = max(1, len(idx) // 10)
        xticks_pos = idx[::step]
        xticks_lab = xs[::step]
    plt.clear_figure()
    plt.clear_data()
    plt.title(title)
    plt.plot(idx, ys, marker="dot")
    plt.xticks(xticks_pos, xticks_lab)
    plt.xlabel(xlabel)
    plt.show()


def _plot_multi_series(
    title: str,
    xs: List[str],
    ys_by_label: Dict[str, List[float]],
    xlabel: str = "time",
) -> None:
    print(f"\n== {title} ==")
    if not xs or not ys_by_label:
        print("(no data)")
        return
    if plt is None:
        print("(plotext not installed; series:) ")
        for label, ys in ys_by_label.items():
            print(label, list(zip(xs, ys)))
        return
    idx = list(range(len(xs)))
    if len(idx) <= 10:
        xticks_pos = idx
        xticks_lab = xs
    else:
        step = max(1, len(idx) // 10)
        xticks_pos = idx[::step]
        xticks_lab = xs[::step]
    plt.clear_figure()
    plt.clear_data()
    plt.title(title)
    for label, ys in ys_by_label.items():
        plt.plot(idx, ys, marker="dot", label=label)
    plt.xticks(xticks_pos, xticks_lab)
    plt.xlabel(xlabel)
    # Some plotext versions don't support legend; print labels instead.
    try:  # pragma: no cover - best-effort
        if hasattr(plt, "legend"):
            plt.legend(True)  # type: ignore[attr-defined]
    except Exception:
        pass
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize OANDA NAV + positions snapshots (JSONL) using plotext.",
    )
    ap.add_argument(
        "--log",
        default="logs/oanda_nav_positions.jsonl",
        help="Path to JSONL log produced by oanda_nav_positions_sampler.py (default: logs/oanda_nav_positions.jsonl)",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=400,
        help="Maximum points per chart (downsample if more; 0 = no limit).",
    )
    ap.add_argument(
        "--top-instruments",
        type=int,
        default=3,
        help="Number of instruments to plot by margin used (default: 3).",
    )
    args = ap.parse_args()

    path = Path(args.log)
    snaps = load_snapshots(path)
    if not snaps:
        print(f"No snapshots found in {path}")
        return

    print(f"Loaded {len(snaps)} snapshots from {path}")
    xs_all = [s.t_label for s in snaps]
    acct_series = _extract_account_series(snaps)

    # NAV / balance
    nav = _sanitize(acct_series["nav"])
    bal = _sanitize(acct_series["balance"])
    xs_nav, nav_ds = downsample(xs_all, nav, args.max_points)
    _, bal_ds = downsample(xs_all, bal, args.max_points)
    _plot_multi_series(
        "Account NAV and Balance (USD)",
        xs_nav,
        {"NAV": nav_ds, "Balance": bal_ds},
    )

    # Unrealized PnL
    unreal = _sanitize(acct_series["unrealized"])
    xs_unreal, unreal_ds = downsample(xs_all, unreal, args.max_points)
    _plot_series("Unrealized PnL (USD)", xs_unreal, unreal_ds)

    # Margin used / available
    mu = _sanitize(acct_series["margin_used"])
    ma = _sanitize(acct_series["margin_available"])
    xs_mu, mu_ds = downsample(xs_all, mu, args.max_points)
    _, ma_ds = downsample(xs_all, ma, args.max_points)
    _plot_multi_series(
        "Margin Used / Available (USD)",
        xs_mu,
        {"margin_used": mu_ds, "margin_available": ma_ds},
    )

    # Per-instrument margin used (top-N)
    if args.top_instruments > 0:
        top_insts, per_inst = _extract_top_instrument_margin_series(snaps, top_n=args.top_instruments)
        if top_insts:
            # Downsample each series consistently using the same xs
            xs_pi, _ = downsample(xs_all, [0.0] * len(xs_all), args.max_points)
            ys_by_label: Dict[str, List[float]] = {}
            for inst in top_insts:
                ys_full = _sanitize(per_inst[inst])
                _, ys_ds = downsample(xs_all, ys_full, args.max_points)
                ys_by_label[inst] = ys_ds
            _plot_multi_series(
                f"Per-instrument margin used (top {len(top_insts)}, USD)",
                xs_pi,
                ys_by_label,
            )


if __name__ == "__main__":  # pragma: no cover
    main()

