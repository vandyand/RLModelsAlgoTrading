import time
from typing import Iterator, Dict, Any, Optional, Iterable

import requests


class OandaRestCandlesAdapter:
    """Pulls 1m EUR_USD candles via OANDA REST (practice env) and yields unified candles.

    This is a simple polling adapter suitable for backfill or slow live consumption.
    For heavy live use prefer the streaming tick aggregator.
    """

    def __init__(self, instrument: str, granularity: str = "M1", environment: str = "practice", access_token: Optional[str] = None):
        # OANDA expects instrument names with underscore, e.g., EUR_USD
        self.instrument = (instrument or "").replace('/', '_')
        self.granularity = granularity
        self.environment = environment
        self.access_token = access_token
        self.base_url = "https://api-fxpractice.oanda.com" if environment == "practice" else "https://api-fxtrade.oanda.com"
        self.session = requests.Session()
        if access_token:
            self.session.headers.update({"Authorization": f"Bearer {access_token}"})

    def fetch(self, count: int = 500, from_time: Optional[str] = None, to_time: Optional[str] = None, include_first: bool = False) -> Iterator[Dict[str, Any]]:
        """Single request for up to `count` candles (OANDA max 5000). If both `from_time` and `to_time` are specified,
        the server determines how many candles to return and `count` may be ignored by the server.
        """
        params = {
            "price": "M",              # mid prices
            "granularity": self.granularity,
        }
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        # Only include count when not specifying both from and to
        if count and not (from_time and to_time):
            params["count"] = int(count)
        # OANDA rejects includeFirst when only count is provided.
        # Only send includeFirst when a from_time is specified.
        if from_time is not None:
            params["includeFirst"] = "true" if include_first else "false"
        url = f"{self.base_url}/v3/instruments/{self.instrument}/candles"
        r = self.session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for c in data.get("candles", []):
            # Only complete candles
            if not c.get("complete", True):
                continue
            mid = c.get("mid") or {}
            ts = c.get("time")
            yield {
                "timestamp": ts.replace(".000000000", "").replace(".000000", "").replace(".000Z", "Z").replace("+00:00", "Z"),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": float(c.get("volume", 0)),
            }

    def fetch_range(self, from_time: str, to_time: Optional[str] = None, batch: int = 5000, max_candles: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over a time range by making multiple paged requests.

        Uses `from` + `count` pagination and advances by the last candle time returned. Respects OANDA's 5000 limit by
        capping `batch` at 5000.
        """
        from datetime import datetime, timedelta
        current_from = from_time
        yielded = 0
        batch = min(batch, 5000)
        while True:
            candles = list(self.fetch(count=batch, from_time=current_from, to_time=None, include_first=False))
            if not candles:
                break
            for candle in candles:
                # Stop if we've passed to_time
                if to_time is not None:
                    if candle["timestamp"] > to_time:
                        return
                yield candle
                yielded += 1
                if max_candles is not None and yielded >= max_candles:
                    return
            # Advance: set next from to just after the last candle timestamp to avoid duplicates
            last_ts = candles[-1]["timestamp"]
            try:
                dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00')) + timedelta(seconds=1)
                current_from = dt.isoformat().replace('+00:00', 'Z')
            except Exception:
                current_from = last_ts
            # Safety: if only one repeating timestamp, break to avoid infinite loop
            if len(candles) == 1:
                # Nudge by one second
                try:
                    dt = datetime.fromisoformat(current_from.replace('Z', '+00:00'))
                    current_from = (dt.replace(microsecond=0)).isoformat().replace('+00:00', 'Z')
                except Exception:
                    pass

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Example of continuous polling (not used in training yet)
        while True:
            try:
                for candle in self.fetch(count=300):
                    yield candle
                time.sleep(5)
            except Exception:
                time.sleep(5)
