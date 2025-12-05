"""Trailing-stop simulator engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from .config import GapFillMode, TrailingConfig, TrailingMode
from .costs import pip_size


class DatasetSplit(Protocol):
    def __iter__(self) -> Iterable[Any]:  # pragma: no cover - interface placeholder
        ...


class Strategy(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


class CostModelProtocol(Protocol):
    def entry_cost(self, instrument: str, price: float, units: float, bar_info: Dict[str, Any]) -> float:  # pragma: no cover
        ...

    def exit_cost(self, instrument: str, price: float, units: float, bar_info: Dict[str, Any]) -> float:  # pragma: no cover
        ...

    def apply_slippage(self, instrument: str, intended_price: float, direction: int, bar_info: Dict[str, Any]) -> float:  # pragma: no cover
        ...


@dataclass
class TradeRecord:
    instrument: str
    direction: int
    units: float
    entry_time: Any
    exit_time: Optional[Any]
    entry_price: float
    exit_price: Optional[float]
    stop_distance: Optional[float]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    cum_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0
    time_in_market: float = 0.0
    exposure: float = 0.0
    # Per-instrument time-in-market statistics: for each instrument we compute
    # the fraction of bars during which it had an open position, then report
    # the mean and standard deviation across the universe. This is more
    # informative for multi-asset portfolios than a single global fraction.
    per_instrument_tim_mean: float = 0.0
    per_instrument_tim_std: float = 0.0


@dataclass
class SimulatorResult:
    metrics: PortfolioMetrics
    trades: List[TradeRecord]
    equity_curve: Optional[np.ndarray] = None
    per_bar_stats: Optional[Dict[str, Any]] = None


@dataclass
class InstrumentState:
    is_open: bool = False
    direction: int = 0
    units: float = 0.0
    entry_price: float = 0.0
    entry_time: Any = None
    stop_distance: float = 0.0
    trail_price: float = 0.0
    favorable_price: float = 0.0
    bars_held: int = 0
    entry_cost: float = 0.0
    last_close: float = 0.0

    def reset(self) -> None:
        self.is_open = False
        self.direction = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_distance = 0.0
        self.trail_price = 0.0
        self.favorable_price = 0.0
        self.bars_held = 0
        self.entry_cost = 0.0
        self.last_close = 0.0


class TrailingStopSimulator:
    """High-level orchestrator for per-bar trailing-stop evaluation."""

    def __init__(self, trailing_cfg: TrailingConfig, cost_model: CostModelProtocol) -> None:
        self.cfg = trailing_cfg
        self.cost_model = cost_model

    def evaluate(
        self,
        strategy: Strategy,
        split: DatasetSplit,
        *,
        record_equity: bool = True,
        return_trades: bool = True,
    ) -> SimulatorResult:
        trades: List[TradeRecord] = []
        equity_curve: List[float] = []
        states: Dict[str, InstrumentState] = {}
        instruments: Sequence[str] = ()
        nav = 1.0
        exposure_sum = 0
        time_in_market_bars = 0
        total_bars = 0
        bar_returns: List[float] = []

        for batch in split:
            bars = batch.get("bars")
            if bars is None:
                raise ValueError("Dataset split must yield dicts containing 'bars'")
            if not instruments:
                instruments = list(bars.keys())
                states = {inst: InstrumentState() for inst in instruments}

            total_bars += 1

            # Snapshot NAV before processing this bar so that per-bar return
            # reflects the actual change in equity (including costs and
            # realized PnL). This ensures that Sharpe has the correct sign
            # relative to cumulative return.
            nav_before = nav

            features = self._prepare_features(batch.get("features"))
            preds = np.asarray(strategy.predict(features), dtype=np.float32)
            if preds.ndim > 1:
                preds = preds[0]

            timestamp = batch.get("timestamp")
            units_map = batch.get("units") or {}
            # We previously approximated bar-wise returns from per-instrument
            # price moves; instead we now derive returns directly from NAV
            # changes so that statistics like Sharpe are consistent with the
            # reported cumulative return.
            bar_ret = 0.0

            # Advance trailing stops and close positions if hit
            for inst in instruments:
                state = states[inst]
                bar = bars[inst]
                if not state.is_open:
                    continue
                state.bars_held += 1
                exit_info = self._advance_and_check_exit(inst, state, bar)
                if exit_info is not None:
                    exit_price, reason = exit_info
                    exit_price = self.cost_model.apply_slippage(inst, exit_price, -state.direction, bar)
                    exit_cost = self.cost_model.exit_cost(inst, exit_price, state.units, bar)
                    pnl = state.direction * (exit_price - state.entry_price) * state.units
                    net = pnl - state.entry_cost - exit_cost
                    nav += net
                    trades.append(
                        TradeRecord(
                            instrument=inst,
                            direction=state.direction,
                            units=state.units,
                            entry_time=state.entry_time,
                            exit_time=timestamp or bar.get("timestamp"),
                            entry_price=state.entry_price,
                            exit_price=exit_price,
                            stop_distance=state.stop_distance,
                            reason=reason,
                            metadata={"pnl": net, "bars": state.bars_held},
                        )
                    )
                    state.reset()
                else:
                    # Position remains open; we track last_close for future
                    # diagnostics but do not mark-to-market NAV here. NAV only
                    # changes on realized exits, and per-bar returns are
                    # derived from NAV deltas below.
                    state.last_close = float(bar.get("close"))

            # Decode strategy outputs into per-instrument entrance booleans,
            # direction signals, position-size fractions, and trailing-distance
            # fractions when provided by the strategy. Backwards-compatible
            # fallbacks are preserved for simpler strategies.
            expected = len(instruments)
            enter_flags = np.ones(expected, dtype=np.float32)
            dir_scores = preds
            pos_fracs = np.ones(expected, dtype=np.float32)
            trail_fracs = np.ones(expected, dtype=np.float32)

            if preds.size == expected * 4:
                enter_flags = np.clip(preds[:expected], 0.0, 1.0)
                dir_scores = preds[expected : 2 * expected]
                pos_fracs = np.clip(preds[2 * expected : 3 * expected], 0.0, 1.0)
                trail_fracs = np.clip(preds[3 * expected : 4 * expected], 0.0, 1.0)
            elif preds.size == expected * 3:
                # Legacy layout: [dir, pos_frac, trail_frac]
                dir_scores = preds[:expected]
                pos_fracs = np.clip(preds[expected : 2 * expected], 0.0, 1.0)
                trail_fracs = np.clip(preds[2 * expected : 3 * expected], 0.0, 1.0)
            elif preds.size != expected:
                dir_scores = np.resize(preds, expected)

            # Generate new signals from direction scores
            signals = self._signals_from_predictions(dir_scores, expected)

            # Optional hard cap on concurrently open instruments.
            max_open = self.cfg.max_open_instruments
            open_now = sum(1 for s in states.values() if s.is_open)

            for idx, inst in enumerate(instruments):
                state = states[inst]
                if state.is_open:
                    continue
                if max_open is not None and open_now >= max_open:
                    break
                # Optional explicit entrance boolean per instrument: if the
                # model indicates no-entry, force flat even if the direction
                # score would suggest otherwise.
                if preds.size == expected * 4 and float(enter_flags[idx]) < 0.5:
                    continue
                signal = signals[idx]
                if signal == 0:
                    continue
                bar = bars[inst]
                close_price = float(bar.get("close"))
                # Position size: base units scaled by model's pos_frac.
                base_units = float(units_map.get(inst, 1.0))
                size_factor = max(0.0, float(pos_fracs[idx]))
                units = base_units * size_factor
                entry_price = self.cost_model.apply_slippage(inst, close_price, signal, bar)
                entry_cost = self.cost_model.entry_cost(inst, entry_price, units, bar)
                # Trailing distance: if model provides a trail_frac, map it to
                # a pip range [min_pips, max_pips]; otherwise fall back to
                # standard config-based distance.
                if preds.size == expected * 3:
                    min_pips = max(10.0, float(self.cfg.min_distance_pips))
                    max_pips_cfg = float(self.cfg.max_trailing_pips) if self.cfg.max_trailing_pips is not None else 20.0
                    max_pips = max(min_pips, max_pips_cfg)
                    tfrac = float(np.clip(trail_fracs[idx], 0.0, 1.0))
                    trail_pips = min_pips + tfrac * (max_pips - min_pips)
                    pip = pip_size(inst)
                    stop_distance = max(pip, trail_pips * pip)
                else:
                    stop_distance = self._compute_stop_distance(inst, bar)
                trail_price = entry_price - stop_distance if signal > 0 else entry_price + stop_distance
                state.is_open = True
                state.direction = signal
                state.units = units
                state.entry_price = entry_price
                state.entry_time = timestamp or bar.get("timestamp")
                state.stop_distance = stop_distance
                state.trail_price = trail_price
                state.favorable_price = entry_price
                state.entry_cost = entry_cost
                state.bars_held = 0
                state.last_close = entry_price
                open_now += 1

            open_positions = sum(1 for s in states.values() if s.is_open)
            exposure_sum += open_positions
            if open_positions > 0:
                time_in_market_bars += 1
            # Per-bar portfolio return derived from NAV change for this bar.
            if nav_before > 0.0:
                bar_ret = (nav - nav_before) / nav_before
            else:
                bar_ret = 0.0
            bar_returns.append(bar_ret)
            if record_equity:
                equity_curve.append(nav)

        metrics = self._compute_metrics(
            nav,
            trades,
            bar_returns,
            exposure_sum,
            time_in_market_bars,
            total_bars,
            len(instruments) or 1,
        )
        return SimulatorResult(
            metrics=metrics,
            trades=trades if return_trades else [],
            equity_curve=np.array(equity_curve) if record_equity else None,
            per_bar_stats={"returns": bar_returns} if record_equity else None,
        )

    def _prepare_features(self, features: Any) -> np.ndarray:
        if features is None:
            raise ValueError("Dataset batch missing 'features'")
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _signals_from_predictions(self, preds: np.ndarray, expected: int) -> np.ndarray:
        if preds.size != expected:
            preds = np.resize(preds, expected)
        signals = np.zeros(expected, dtype=int)
        # Default mapping: expect discrete {-1, 0, +1} or continuous scores.
        # If strategies emit {-1, 0, +1} directly (recommended), this mapping
        # will still work: +1 -> long, -1 -> short, 0 -> flat.
        signals[preds > 0.5] = 1
        signals[preds < -0.5] = -1
        return signals

    def _advance_and_check_exit(self, instrument: str, state: InstrumentState, bar: Dict[str, Any]) -> Optional[tuple[float, str]]:
        price_high = float(bar.get("high"))
        price_low = float(bar.get("low"))
        price_open = float(bar.get("open"))
        if state.direction > 0:
            state.favorable_price = max(state.favorable_price, price_high)
            candidate = state.favorable_price - state.stop_distance
            state.trail_price = max(state.trail_price, candidate)
            if price_low <= state.trail_price:
                exit_price = self._resolve_exit_price(state.trail_price, price_open, state.direction)
                return exit_price, "TRAIL_HIT"
        else:
            state.favorable_price = min(state.favorable_price, price_low)
            candidate = state.favorable_price + state.stop_distance
            state.trail_price = min(state.trail_price, candidate)
            if price_high >= state.trail_price:
                exit_price = self._resolve_exit_price(state.trail_price, price_open, state.direction)
                return exit_price, "TRAIL_HIT"
        return None

    def _resolve_exit_price(self, trail_price: float, bar_open: float, direction: int) -> float:
        if self.cfg.gap_mode == GapFillMode.STOP_PRICE:
            return trail_price
        if direction > 0:
            return min(bar_open, trail_price)
        return max(bar_open, trail_price)

    def _compute_stop_distance(self, instrument: str, bar: Dict[str, Any]) -> float:
        mode = self.cfg.canonical_mode()
        pip = pip_size(instrument)
        distance = pip * self.cfg.min_distance_pips
        if mode == TrailingMode.PIP:
            distance = max(distance, pip * self.cfg.pip_distance)
        elif mode == TrailingMode.ATR:
            atr = float(bar.get("atr", abs(float(bar.get("high")) - float(bar.get("low")))))
            distance = max(distance, atr * float(self.cfg.atr_multiplier))
        else:
            vol = float(bar.get("tick_vol", abs(float(bar.get("close")) - float(bar.get("open")))))
            distance = max(distance, vol * float(self.cfg.tick_vol_multiplier))
        if self.cfg.max_trailing_pips is not None:
            distance = min(distance, pip * float(self.cfg.max_trailing_pips))
        if self.cfg.enforce_quantization and self.cfg.price_precision > 0:
            quantum = 10 ** (-self.cfg.price_precision)
            distance = round(distance / quantum) * quantum
        return max(distance, 1e-6)

    def _compute_metrics(
        self,
        nav: float,
        trades: List[TradeRecord],
        bar_returns: List[float],
        exposure_sum: int,
        time_in_market_bars: int,
        total_bars: int,
        num_instruments: int,
    ) -> PortfolioMetrics:
        bar_ret = np.array(bar_returns or [0.0], dtype=np.float32)
        mean_return = float(bar_ret.mean()) if bar_ret.size > 0 else 0.0
        std_return = float(bar_ret.std()) if bar_ret.size > 1 else 0.0
        sharpe = mean_return / (std_return + 1e-8) * np.sqrt(max(total_bars, 1))
        neg_returns = bar_ret[bar_ret < 0]
        sortino = (
            mean_return / (float(neg_returns.std()) + 1e-8) * np.sqrt(max(total_bars, 1))
            if neg_returns.size > 0
            else sharpe
        )

        pnl_values = [t.metadata.get("pnl", 0.0) for t in trades]
        gains = [p for p in pnl_values if p > 0]
        losses = [abs(p) for p in pnl_values if p < 0]
        profit_factor = (sum(gains) / (sum(losses) + 1e-8)) if losses else (float("inf") if gains else 0.0)
        win_rate = len(gains) / max(1, len(trades))
        avg_duration = sum(t.metadata.get("bars", 0) for t in trades) / max(1, len(trades))

        exposure = exposure_sum / max(1, total_bars * max(1, num_instruments))
        time_in_market = time_in_market_bars / max(1, total_bars)

        # Per-instrument time-in-market: approximate using trade durations. For
        # each instrument we sum the number of bars its trades were held, then
        # divide by total_bars to get a per-instrument fraction.
        per_inst_bars: Dict[str, float] = {}
        for t in trades:
            bars_held = float(t.metadata.get("bars", 0.0))
            per_inst_bars[t.instrument] = per_inst_bars.get(t.instrument, 0.0) + bars_held
        per_inst_fractions: List[float] = []
        if total_bars > 0 and num_instruments > 0:
            # Ensure every instrument is represented, including those with no trades.
            if not per_inst_bars:
                # If we have no trades at all, pretend all instruments are flat.
                per_inst_fractions = [0.0 for _ in range(num_instruments)]
            else:
                # Fractions for instruments that traded at least once.
                for inst_bars in per_inst_bars.values():
                    per_inst_fractions.append(float(inst_bars) / float(total_bars))
                # And zeros for instruments that never traded.
                missing = max(0, num_instruments - len(per_inst_bars))
                if missing > 0:
                    per_inst_fractions.extend([0.0 for _ in range(missing)])
        tim_mean = float(np.mean(per_inst_fractions)) if per_inst_fractions else 0.0
        tim_std = float(np.std(per_inst_fractions)) if per_inst_fractions else 0.0
        metrics = PortfolioMetrics(
            cum_return=nav - 1.0,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=self._max_drawdown(bar_ret),
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_trade_duration=avg_duration,
            time_in_market=time_in_market,
            exposure=exposure,
            per_instrument_tim_mean=tim_mean,
            per_instrument_tim_std=tim_std,
        )
        return metrics

    def _max_drawdown(self, bar_returns: np.ndarray) -> float:
        if bar_returns.size == 0:
            return 0.0
        equity = np.cumsum(bar_returns)
        peaks = np.maximum.accumulate(equity)
        drawdowns = peaks - equity
        return float(drawdowns.max()) if drawdowns.size > 0 else 0.0
