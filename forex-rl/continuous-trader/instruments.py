from __future__ import annotations

import csv
import os
from typing import List, Optional


def _fx_root() -> str:
    """Return absolute path to the forex-rl directory."""
    here = os.path.abspath(os.path.dirname(__file__))
    # forex-rl/continuous-trader -> forex-rl
    return os.path.dirname(here)


def _default_csv_path() -> str:
    # Default to forex-rl/oanda-instrument-list.csv per user
    return os.path.join(_fx_root(), "oanda-instrument-list.csv")


def load_instruments(csv_path: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """Load OANDA instrument symbols from a CSV file.

    The CSV may be formatted in one of the following ways:
    - Single column with header 'instrument' (preferred)
    - Single column without header (one instrument per line)
    - Multi-column where one column is named 'instrument'

    Args:
      csv_path: Optional override path; defaults to repo-root/oanda-instrument-list.csv
      limit: If provided, return at most this many instruments from the top

    Returns:
      List of instrument symbols as strings (e.g., 'EUR_USD'). Duplicates removed, order preserved.
    """
    path = csv_path or _default_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Instrument CSV not found at: {path}")

    instruments: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        # Sniff dialect to handle commas vs. other delimiters robustly
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        return []

    # Detect header
    header = rows[0]
    has_header = False
    if any(h.lower().strip() in ("instrument", "instruments") for h in header):
        has_header = True
        # Find the instrument column
        idx = None
        for i, h in enumerate(header):
            if h.lower().strip() in ("instrument", "instruments"):
                idx = i
                break
        if idx is None:
            # Fallback to first column
            idx = 0
        data_rows = rows[1:]
        for r in data_rows:
            if not r:
                continue
            sym = (r[idx] if idx < len(r) else "").strip()
            if sym:
                instruments.append(sym)
    else:
        # No header; take first column of each row
        for r in rows:
            if not r:
                continue
            sym = (r[0] or "").strip()
            if sym and sym.lower() != "instrument":
                instruments.append(sym)

    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for s in instruments:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    if limit is not None:
        uniq = uniq[: int(limit)]
    return uniq


def load_68(csv_path: Optional[str] = None) -> List[str]:
    """Convenience loader for the full 68-instrument list (or as many as available)."""
    return load_instruments(csv_path=csv_path, limit=None)
