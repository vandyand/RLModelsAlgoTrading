from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Prefer local continuous-trader/data/*.csv then fallback to ga-ness/data


def load_m5_csv(instrument: str, fx_root: Optional[str] = None) -> Optional[pd.DataFrame]:
    base = fx_root or os.path.dirname(os.path.dirname(__file__))  # forex-rl
    ct_dir = os.path.join(base, "continuous-trader", "data")
    ga_dir = os.path.join(base, "ga-ness", "data")
    names = [os.path.join(ct_dir, f"{instrument}_M5.csv"), os.path.join(ga_dir, f"{instrument}_M5.csv")]
    for path in names:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df = df.set_index("timestamp").sort_index()
                # Normalize column names
                cols = {c.lower(): c for c in df.columns}
                df = df[[cols.get("open", "open"), cols.get("high", "high"), cols.get("low", "low"), cols.get("close", "close"), cols.get("volume", "volume")]]
                df.columns = ["open","high","low","close","volume"]
                return df.astype({"open":"float32","high":"float32","low":"float32","close":"float32","volume":"float32"})
            except Exception:
                continue
    return None


def available_m5_instruments(fx_root: Optional[str] = None) -> List[str]:
    base = fx_root or os.path.dirname(os.path.dirname(__file__))
    ct_dir = os.path.join(base, "continuous-trader", "data")
    ga_dir = os.path.join(base, "ga-ness", "data")
    out: List[str] = []
    for d in [ct_dir, ga_dir]:
        if os.path.exists(d):
            for name in os.listdir(d):
                if name.endswith("_M5.csv"):
                    sym = name.replace("_M5.csv", "")
                    if sym not in out:
                        out.append(sym)
    return out


def summarize(fx_root: Optional[str] = None) -> Dict[str, object]:
    base = fx_root or os.path.dirname(os.path.dirname(__file__))
    ct_dir = os.path.join(base, "continuous-trader", "data")
    ga_dir = os.path.join(base, "ga-ness", "data")
    insts = available_m5_instruments(base)
    counts: Dict[str, Optional[int]] = {}
    for inst in insts:
        n = None
        for d in [ct_dir, ga_dir]:
            p = os.path.join(d, f"{inst}_M5.csv")
            if os.path.exists(p):
                try:
                    # rows minus header
                    n = sum(1 for _ in open(p, 'r', encoding='utf-8')) - 1
                    break
                except Exception:
                    n = None
        counts[inst] = n
    return {
        "continuous_trader_data": ct_dir,
        "ga_ness_data": ga_dir,
        "available_m5": insts,
        "row_counts": counts,
        "count": len(insts),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(), indent=2))
