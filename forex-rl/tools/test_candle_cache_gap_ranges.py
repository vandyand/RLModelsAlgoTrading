#!/usr/bin/env python3
"""Validate candle_ranges coverage logic for gappy ranges.

This is a pure-SQL test that mimics the SELECT used by candle_cache_service
for range coverage, ensuring that disjoint ranges are NOT treated as a
single contiguous coverage window.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def main() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE candle_ranges (
            environment TEXT NOT NULL,
            instrument  TEXT NOT NULL,
            granularity TEXT NOT NULL,
            price       TEXT NOT NULL,
            from_time   TEXT NOT NULL,
            to_time     TEXT NOT NULL,
            PRIMARY KEY (environment, instrument, granularity, price, from_time, to_time)
        )
        """
    )
    env = "practice"
    inst = "EUR_USD"
    gran = "H1"
    price = "M"

    # First cached block: Jan 1 - Jan 31
    conn.execute(
        "INSERT INTO candle_ranges VALUES (?,?,?,?,?,?)",
        (env, inst, gran, price, "2025-01-01T00:00:00Z", "2025-01-31T23:00:00Z"),
    )
    # Second cached block: Mar 1 - Mar 31
    conn.execute(
        "INSERT INTO candle_ranges VALUES (?,?,?,?,?,?)",
        (env, inst, gran, price, "2025-03-01T00:00:00Z", "2025-03-31T23:00:00Z"),
    )
    conn.commit()

    def covered(from_t: str, to_t: str) -> bool:
        cur = conn.execute(
            """
            SELECT 1 FROM candle_ranges
            WHERE environment=? AND instrument=? AND granularity=? AND price=?
              AND from_time <= ? AND to_time >= ?
            LIMIT 1
            """,
            (env, inst, gran, price, from_t, to_t),
        )
        return cur.fetchone() is not None

    # This range is fully covered by the first block.
    assert covered("2025-01-10T00:00:00Z", "2025-01-20T00:00:00Z"), "Expected Jan sub-range to be covered"

    # This range spans the gap between blocks and should NOT be covered.
    assert not covered("2025-01-10T00:00:00Z", "2025-03-15T00:00:00Z"), "Gap-spanning range must not be treated as covered"

    print({"status": "ok", "message": "gap coverage logic behaves as expected"})


if __name__ == "__main__":  # pragma: no cover
    main()
