#!/usr/bin/env python3
"""
Build a universe of the most liquid Alpaca US equities.

This script talks directly to Alpaca using alpaca-py:

- Uses the Trading API to enumerate ACTIVE, tradable US equities.
- Uses the Market Data API to fetch recent daily bars and compute
  average daily dollar volume (ADTV) per symbol.
- Ranks symbols by ADTV and writes the top-N symbols to a JSON file.

Intended usage (from forex-rl root):

  # Build a top-100 universe over the last ~90 calendar days
  python -m tools.build_alpaca_equity_universe

  # Customize window length / universe size / output file
  python -m tools.build_alpaca_equity_universe \\
      --days 120 \\
      --top-n 150 \\
      --out equities_universe_top150.json

Environment:
- Requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in the environment.
- Uses Alpaca paper environment semantics where applicable.

The resulting JSON file looks like:

  {
    "asset_class": "US_EQUITY",
    "asof_utc": "2025-12-31T23:59:59Z",
    "window_days": 90,
    "top_n": 100,
    "symbols": ["SPY", "QQQ", "AAPL", "..."],
    "scores": {
      "SPY": 1234567890.0,
      "QQQ": 987654321.0,
      ...
    }
  }

The equities actor-critic stack can later consume this file as its
instrument universe.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple

from alpaca.common.enums import Sort
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest


def _get_alpaca_keys() -> Tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit(
            "Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in environment. "
            "Export them or add them to the appropriate .env file."
        )
    return key, secret


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fetch_candidate_symbols(trading: TradingClient) -> List[str]:
    """
    Retrieve the base US equity universe from Alpaca.

    Filters:
    - asset_class = US_EQUITY
    - status = ACTIVE
    - tradable, marginable, shortable = True
    """
    req = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE,
    )
    assets = trading.get_all_assets(req)

    symbols: List[str] = []
    for a in assets:
        # Defensive: guard for missing attributes on older SDKs
        if not getattr(a, "tradable", False):
            continue
        if not getattr(a, "marginable", False):
            continue
        if not getattr(a, "shortable", False):
            continue
        sym = getattr(a, "symbol", None)
        if not sym:
            continue
        # Stick to plain stock-like tickers (avoid weird test symbols)
        if "_" in sym or "/" in sym or " " in sym:
            continue
        symbols.append(sym.upper())

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _chunk(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _compute_liquidity_scores(
    data_client: StockHistoricalDataClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    batch_size: int = 200,
) -> Dict[str, float]:
    """
    Compute average daily dollar volume over [start, end] for each symbol.

    Returns:
        dict[symbol] -> ADTV (float)
    """
    scores: Dict[str, float] = {}

    for batch in _chunk(symbols, batch_size):
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=None,
            # Most Alpaca paper / free subscriptions only permit IEX,
            # whereas SIP often requires higher-tier data.
            feed=DataFeed.IEX,
            sort=Sort.ASC,
        )
        barset = data_client.get_stock_bars(request_params=req)

        # BarSet behaves like a mapping symbol -> List[Bar]
        # but older versions may expose .data; handle both defensively.
        mapping = getattr(barset, "data", None)
        if not isinstance(mapping, dict):
            # Fallback: assume direct mapping/iterable interface
            try:
                mapping = dict(barset.items())  # type: ignore[arg-type]
            except Exception:
                mapping = {}

        for sym in batch:
            bars = mapping.get(sym, []) if isinstance(mapping, dict) else getattr(barset, sym, [])
            if not bars:
                continue

            total_dollar = 0.0
            count = 0
            for b in bars:
                # Handle both Bar objects and dict-like bars
                close = getattr(b, "close", None)
                volume = getattr(b, "volume", None)
                if close is None and isinstance(b, dict):
                    close = b.get("c") or b.get("close")
                if volume is None and isinstance(b, dict):
                    volume = b.get("v") or b.get("volume")

                try:
                    close_f = float(close)
                    vol_f = float(volume or 0.0)
                except Exception:
                    continue

                total_dollar += close_f * vol_f
                count += 1

            if count > 0:
                scores[sym] = total_dollar / float(count)

    return scores


def _symbols_with_history(
    data_client: StockHistoricalDataClient,
    symbols_sorted: List[str],
    start: datetime,
    end: datetime,
    min_days: int,
    batch_size: int = 50,
) -> List[str]:
    """
    Filter symbols to those with sufficiently long daily history in [start, end].

    A symbol is kept if:
      - At least `min_days` daily bars exist in [start, end], and
      - The first bar is not far after `start` (<= ~1 month tolerance),
      - The last bar is not far before `end` (<= ~1 month tolerance).
    """
    from datetime import date

    keep: List[str] = []
    start_date = start.date()
    end_date = end.date()
    max_start_lag = timedelta(days=31)
    max_end_lag = timedelta(days=31)

    for batch in _chunk(symbols_sorted, batch_size):
        req = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=None,
            feed=DataFeed.IEX,
            sort=Sort.ASC,
        )
        barset = data_client.get_stock_bars(request_params=req)
        mapping = getattr(barset, "data", None)
        if not isinstance(mapping, dict):
            try:
                mapping = dict(barset.items())  # type: ignore[arg-type]
            except Exception:
                mapping = {}

        for sym in batch:
            bars = mapping.get(sym, []) if isinstance(mapping, dict) else getattr(barset, sym, [])
            if not bars:
                continue

            first = bars[0]
            last = bars[-1]
            ts_first = getattr(first, "timestamp", None)
            ts_last = getattr(last, "timestamp", None)
            if ts_first is None and isinstance(first, dict):
                ts_first = first.get("t") or first.get("timestamp")
            if ts_last is None and isinstance(last, dict):
                ts_last = last.get("t") or last.get("timestamp")

            try:
                if isinstance(ts_first, str):
                    dt_first = datetime.fromisoformat(ts_first.replace("Z", "+00:00"))
                else:
                    dt_first = ts_first
                if isinstance(ts_last, str):
                    dt_last = datetime.fromisoformat(ts_last.replace("Z", "+00:00"))
                else:
                    dt_last = ts_last
            except Exception:
                continue

            if not isinstance(dt_first, datetime) or not isinstance(dt_last, datetime):
                continue

            d_first: date = dt_first.date()
            d_last: date = dt_last.date()
            n = len(bars)

            if (
                (d_first - start_date) <= max_start_lag
                and (end_date - d_last) <= max_end_lag
                and n >= min_days
            ):
                keep.append(sym)

    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a top-N liquid Alpaca US equity universe.")
    ap.add_argument(
        "--days",
        type=int,
        default=90,
        help="Window length in calendar days for liquidity lookback (default: 90).",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of most-liquid symbols to keep (default: 100).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="equities_universe_top100.json",
        help="Output JSON file path (relative to forex-rl root by default).",
    )
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="Maximum number of candidate symbols to consider (default: 5000).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for bar requests (default: 200).",
    )
    ap.add_argument(
        "--history-start",
        type=str,
        default=None,
        help=(
            "Optional history start date (YYYY-MM-DD). When set, compute "
            "five_years_history_symbols: symbols with sufficiently long "
            "daily history in [history-start, history-end]."
        ),
    )
    ap.add_argument(
        "--history-end",
        type=str,
        default=None,
        help="Optional history end date (YYYY-MM-DD). Defaults to now when omitted.",
    )
    ap.add_argument(
        "--history-min-days",
        type=int,
        default=750,
        help=(
            "Minimum number of daily bars required in the history window when "
            "--history-end is provided (default: 750). Ignored when only "
            "--history-start is set."
        ),
    )
    ap.add_argument(
        "--history-window-days",
        type=int,
        default=30,
        help=(
            "Size of the window (in days) around --history-start to check for "
            "the existence of daily bars when --history-end is omitted "
            "(default: 30)."
        ),
    )
    args = ap.parse_args()

    key, secret = _get_alpaca_keys()

    # Trading client (paper mode for consistency with usage elsewhere)
    trading = TradingClient(key, secret, paper=True)
    data_client = StockHistoricalDataClient(key, secret)

    print("Fetching candidate US_EQUITY assets from Alpaca...")
    symbols = _fetch_candidate_symbols(trading)
    if not symbols:
        raise SystemExit("No candidate symbols returned from Alpaca.")

    if len(symbols) > args.max_candidates:
        symbols = symbols[: args.max_candidates]

    print(f"Found {len(symbols)} candidate symbols after filtering.")

    end = _now_utc()
    start = end - timedelta(days=args.days)
    print(f"Computing average daily dollar volume from {start.isoformat()} to {end.isoformat()}...")

    scores = _compute_liquidity_scores(
        data_client=data_client,
        symbols=symbols,
        start=start,
        end=end,
        batch_size=args.batch_size,
    )

    if not scores:
        raise SystemExit("No liquidity scores computed (no bar data returned).")

    sorted_syms = sorted(scores, key=scores.get, reverse=True)
    top_n = max(1, min(args.top_n, len(sorted_syms)))
    top_syms = sorted_syms[:top_n]

    print(f"Top {top_n} symbols by ADTV:")
    for i, sym in enumerate(top_syms[:20], start=1):
        print(f"{i:3d}. {sym:8s}  ADTV=${scores[sym]:.2f}")
    if len(top_syms) > 20:
        print(f"... ({len(top_syms) - 20} more)")

    # Optionally, further filter by long history window
    five_year_syms: List[str] = []
    if args.history_start:
        # Parse history-start as the anchor date.
        try:
            center = datetime.fromisoformat(args.history_start.strip() + "T00:00:00+00:00")
        except Exception:
            raise SystemExit(f"Invalid --history-start date '{args.history_start}', expected YYYY-MM-DD")

        # Two modes:
        # - If history_end is provided -> use full window [history_start, history_end]
        #   and require history_min_days bars.
        # - If history_end is omitted -> use a small symmetric window around
        #   history_start (history_window_days) and only require that at least
        #   one bar exists in that window (cheap presence test).
        if args.history_end:
            try:
                hist_start = center
                hist_end = datetime.fromisoformat(args.history_end.strip() + "T23:59:59+00:00")
            except Exception:
                raise SystemExit(f"Invalid --history-end date '{args.history_end}', expected YYYY-MM-DD")
            min_days = int(args.history_min_days)
            desc = f"at least {min_days} daily bars"
        else:
            half = max(1, int(args.history_window_days) // 2)
            hist_start = center - timedelta(days=half)
            hist_end = center + timedelta(days=half)
            min_days = 1  # just need at least one bar in the window
            desc = "at least 1 daily bar"

        print(
            f"Filtering symbols with {desc} between "
            f"{hist_start.isoformat()} and {hist_end.isoformat()}..."
        )
        five_year_syms = _symbols_with_history(
            data_client=data_client,
            symbols_sorted=sorted_syms,
            start=hist_start,
            end=hist_end,
            min_days=min_days,
            batch_size=min(args.batch_size, 100),
        )
        print(f"{len(five_year_syms)} symbols have sufficient history in the requested window.")

    out_path = args.out
    # If relative, treat as forex-rl root relative (when run as `python -m tools...`)
    if not os.path.isabs(out_path):
        here = os.path.abspath(os.path.dirname(__file__))
        root = os.path.abspath(os.path.join(here, os.pardir))
        out_path = os.path.join(root, out_path)

    payload = {
        "asset_class": "US_EQUITY",
        "asof_utc": end.isoformat().replace("+00:00", "Z"),
        "window_days": int(args.days),
        "top_n": int(top_n),
        # Top-N universe (backward-compatible fields)
        "symbols": top_syms,
        "scores": {s: float(scores[s]) for s in top_syms},
        # Full liquidity ranking
        "all_symbols_sorted": sorted_syms,
        "all_scores": {s: float(scores[s]) for s in sorted_syms},
    }

    if five_year_syms:
        payload["five_years_history_symbols"] = five_year_syms

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote universe ({top_n} symbols) to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

