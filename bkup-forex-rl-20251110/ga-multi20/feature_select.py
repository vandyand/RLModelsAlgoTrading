from __future__ import annotations
from typing import List, Dict

# Invariant feature sets by granularity
# 7 from M5, 7 from H1, 6 from D => total 20 per instrument
M5_FEATURES: List[str] = [
    "macdh_12_26_9",
    "rsi_15",
    "bb_percB_15",
    "adx_15",
    "atr_15",
    "obv_z15",
    "tema_15",
]

H1_FEATURES: List[str] = [
    "macdh_12_26_9",
    "rsi_45",
    "bb_percB_45",
    "adx_45",
    "atr_45",
    "kst",
    "ichimoku_kijun_26",
]

D_FEATURES: List[str] = [
    "rsi_135",
    "bb_percB_135",
    "adx_135",
    "atr_135",
    "ema_405",
    "aroon_osc_405",
]

FEATURES_BY_GRANULARITY: Dict[str, List[str]] = {
    "M5": M5_FEATURES,
    "H1": H1_FEATURES,
    "D": D_FEATURES,
}


def prefixed_feature_names(pair_with_slash: str, granularity: str) -> List[str]:
    base = FEATURES_BY_GRANULARITY[granularity]
    return [f"{pair_with_slash}_{feat}" for feat in base]
