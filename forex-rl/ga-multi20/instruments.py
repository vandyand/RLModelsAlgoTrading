from __future__ import annotations

# Default 20 most liquid OANDA FX pairs (from actor-critic multi20)
DEFAULT_OANDA_20 = [
    "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CHF",
    "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
    "EUR_CHF", "EUR_AUD", "EUR_CAD", "GBP_CHF", "AUD_JPY",
    "AUD_CHF", "CAD_JPY", "NZD_JPY", "GBP_AUD", "AUD_NZD",
]


def underscore_to_slash(instrument: str) -> str:
    return instrument.replace("_", "/")


def slash_to_underscore(instrument: str) -> str:
    return instrument.replace("/", "_")
