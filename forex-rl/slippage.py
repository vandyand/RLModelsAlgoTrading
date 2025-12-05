from __future__ import annotations

import time
from typing import Optional


def linear_ramp_bps(
    *,
    start_bps: float,
    target_bps: float,
    start_epoch_ts: Optional[float],
    ramp_days: float,
    now_ts: Optional[float] = None,
) -> float:
    """Return current slippage in basis points using a linear ramp over ramp_days.

    If start_epoch_ts is None, defaults to process start time (first call) or now.
    If ramp_days <= 0, returns target_bps immediately.
    """
    if now_ts is None:
        now_ts = time.time()
    if start_epoch_ts is None:
        start_epoch_ts = now_ts
    if ramp_days <= 0:
        return float(target_bps)
    total_seconds = float(ramp_days) * 86400.0
    progress = max(0.0, min(1.0, float(now_ts - start_epoch_ts) / total_seconds))
    return float(start_bps + progress * (target_bps - start_bps))
