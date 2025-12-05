from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

# Add forex FRB paths
REPO_ROOT = str(Path(__file__).resolve().parents[1].parent)
FRB_DIR = str(Path(REPO_ROOT) / "flat-ring-buffer" / "forex")
FRB_SCHEMA_DIR = str(Path(REPO_ROOT) / "flat-ring-buffer" / "forex" / "schema")
for p in (FRB_DIR, FRB_SCHEMA_DIR):
    if p not in sys.path:
        sys.path.append(p)

from oanda_frb import SharedRingBuffer, RingReader  # type: ignore
from frbschema.MarketMessage import MarketMessage  # type: ignore
from frbschema.MessageType import MessageType as FBType  # type: ignore
from frbschema.Ticker import Ticker  # type: ignore
from frbschema.DepthUpdate import DepthUpdate  # type: ignore


class FRBConsumerFX:
    def __init__(self, path: str = "/dev/shm/market_data", start_at_head: bool = True) -> None:
        self.ring = SharedRingBuffer(path, create=False)
        self.reader = RingReader(self.ring, start_at_head=start_at_head)

    def stream(self, instruments: Iterable[str]) -> Generator[Tuple[str, Dict[str, float]], None, None]:
        instset = set(s.upper() for s in instruments)
        while True:
            rec = self.reader.read_next(block=True)
            if not rec:
                continue
            header, payload = rec
            msg = MarketMessage.GetRootAs(payload, 0)
            t = msg.Type()
            # Prefer Ticker; Depth can be parsed if needed later
            if t == FBType.TICKER:
                tk = Ticker(); tk.Init(msg.Data().Bytes, msg.Data().Pos)
                sym = tk.Symbol().decode() if isinstance(tk.Symbol(), (bytes, bytearray)) else tk.Symbol()
                sym = sym.upper()
                if sym not in instset:
                    continue
                yield sym, {
                    "mid": float(tk.Last()),
                    "open": float(tk.Open()),
                    "high": float(tk.High()),
                    "low": float(tk.Low()),
                }
            elif t == FBType.DEPTH:
                # Optional: could derive mid from best bid/ask; not required initially
                du = DepthUpdate(); du.Init(msg.Data().Bytes, msg.Data().Pos)
                sym = du.Symbol().decode() if isinstance(du.Symbol(), (bytes, bytearray)) else du.Symbol()
                sym = sym.upper()
                if sym not in instset:
                    continue
                try:
                    best_bid = float(du.Bids(0).Price()) if du.BidsLength() > 0 else None
                except Exception:
                    best_bid = None
                try:
                    best_ask = float(du.Asks(0).Price()) if du.AsksLength() > 0 else None
                except Exception:
                    best_ask = None
                if best_bid is None and best_ask is None:
                    continue
                mid = (best_bid + best_ask) / 2.0 if (best_bid is not None and best_ask is not None) else (best_bid or best_ask)
                yield sym, {"mid": float(mid)}
