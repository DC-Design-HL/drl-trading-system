"""Broker — owns position lifecycle in the backtest.

Contract:
  * `open_position(intent, ctx)` → Position (or None if rejected)
  * `on_bar(ctx)` → list[Trade] (zero or more closes/partials this bar)
  * `force_close(ctx, reason)` → Trade

Exit checks executed in order each bar:
  1. SL hit (high/low penetrates SL price)
  2. Partial TP1 / TP2 (intra-bar high/low touches partial price)
  3. Trailing stop (if activated; uses bar low for LONG, high for SHORT)
  4. Stagnant exit (time-since-open ≥ STAGNANT_HOURS and pnl in band)
  5. REVERSE_CLOSE on opposite intent (handled by runner, not broker)

We use INTRA-BAR high/low to detect SL/TP touches. This is the standard
backtest convention — a more conservative variant (close-only) would
under-report SL hits since intra-bar wicks can stop us out.

Slippage / fees mirror live_constants:
  * entry: fill at `intent_price * (1 ± slippage)`, taker fee
  * partial close at SL/TP: same slippage + taker fee on closed portion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .. import live_constants as LC
from ..strategy.indicators import atr as _atr
from ..strategy.base import Intent
from .context import Ctx
from .position import Position

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """One realized fill — either a partial or a full close."""
    symbol: str
    side: str
    entry_ts_ms: int
    exit_ts_ms: int
    entry_price: float
    exit_price: float
    units: float
    pnl_usd: float
    pnl_r: float            # PnL in units of initial risk
    fees_usd: float
    reason: str             # 'sl' | 'tp_partial_1' | 'tp_partial_2' | 'tp_full'
                            #  | 'trail' | 'stagnant' | 'reverse_close' | 'eod'
    confidence: float
    mfe_pct: float
    mae_pct: float
    holding_minutes: float
    is_full_close: bool     # True if position closed completely (vs partial)
    extras: dict = field(default_factory=dict)


class Broker:
    """Per-config execution engine. Tracks positions across symbols."""

    def __init__(self, fees_taker: float = LC.TRADING_FEE_TAKER,
                 slippage_pct: float = LC.SLIPPAGE_PCT,
                 starting_balance: float = LC.INITIAL_BALANCE,
                 max_concurrent: int = 4):
        self.fees_taker = fees_taker
        self.slippage_pct = slippage_pct
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.realized_pnl = 0.0
        self.max_concurrent = max_concurrent
        self.positions: dict[str, Position] = {}

    # ── opens ──────────────────────────────────────────────────────
    def can_open(self) -> bool:
        return len(self.positions) < self.max_concurrent

    def open_position(self, intent: Intent, ctx: Ctx,
                      sizing_usd: float,
                      atr_period: int = 14) -> Optional[Position]:
        if intent.action not in ("OPEN_LONG", "OPEN_SHORT"):
            return None
        if ctx.symbol in self.positions:
            return None
        if not self.can_open():
            return None

        side = "LONG" if intent.action == "OPEN_LONG" else "SHORT"
        # Slippage on entry — adverse
        slip = self.slippage_pct
        raw_price = ctx.current_close
        entry_price = raw_price * (1 + slip) if side == "LONG" else raw_price * (1 - slip)

        # ATR-floored SL/TP
        atr_series = _atr(ctx.primary, period=atr_period).dropna()
        atr_pct = float(atr_series.iloc[-1]) / entry_price if len(atr_series) else 0.0
        sl_pct = max(LC.BASE_SL_PCT, atr_pct * LC.ATR_SL_FLOOR_MULT)
        tp_pct = max(LC.BASE_TP_PCT, atr_pct * LC.ATR_TP_FLOOR_MULT)

        if side == "LONG":
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
            partial_tp1_price = entry_price * (1 + LC.PARTIAL_TP1_R * sl_pct)
            partial_tp2_price = entry_price * (1 + LC.PARTIAL_TP2_R * sl_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
            partial_tp1_price = entry_price * (1 - LC.PARTIAL_TP1_R * sl_pct)
            partial_tp2_price = entry_price * (1 - LC.PARTIAL_TP2_R * sl_pct)

        # Sizing: notional / leverage; cap by FIXED_MAX_NOTIONAL
        notional = min(sizing_usd, LC.FIXED_MAX_NOTIONAL)
        units = notional / entry_price
        leverage = LC.LEVERAGE
        # Liq safety: actual_liq_dist = 1/leverage must be ≥ sl_pct + LIQ_BUFFER_PCT
        required_liq_dist = sl_pct + LC.LIQ_BUFFER_PCT
        if 1.0 / leverage < required_liq_dist:
            leverage = max(1, int(1.0 / required_liq_dist))

        entry_fee = entry_price * units * self.fees_taker
        self.balance -= entry_fee

        pos = Position(
            symbol=ctx.symbol, side=side,
            entry_price=entry_price, initial_units=units, units=units,
            sl_price=sl_price, tp_price=tp_price,
            partial_tp1_price=partial_tp1_price, partial_tp2_price=partial_tp2_price,
            sl_pct=sl_pct, tp_pct=tp_pct, leverage=leverage,
            confidence=intent.confidence,
            open_ts_ms=ctx.now_ms,
            open_intent_reason=intent.reason,
            fees_paid=entry_fee,
            extras={"open_intent_extras": dict(intent.extras)},
        )
        self.positions[ctx.symbol] = pos
        return pos

    # ── per-bar exit checks ────────────────────────────────────────
    def on_bar(self, ctx: Ctx) -> List[Trade]:
        """Returns trades closed (partial or full) on this bar."""
        pos = self.positions.get(ctx.symbol)
        if pos is None: return []

        bar = ctx.current_bar
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        pos.update_excursions(bar_high, bar_low)

        trades: List[Trade] = []

        # 1. SL check (priority over TP if both hit on same bar — conservative)
        if self._sl_hit(pos, bar_high, bar_low):
            trades.append(self._close_remainder(pos, ctx, pos.sl_price, "sl"))
            self.positions.pop(ctx.symbol, None)
            return trades

        # 2. Partial TP1 / TP2
        if pos.partial_tp_level == 0 and self._partial_tp1_hit(pos, bar_high, bar_low):
            trades.append(self._partial_close(pos, ctx, pos.partial_tp1_price,
                                              LC.PARTIAL_TP1_FRACTION, "tp_partial_1"))
            pos.partial_tp_level = 1
            pos.trail_check()
        if pos.partial_tp_level == 1 and self._partial_tp2_hit(pos, bar_high, bar_low):
            trades.append(self._partial_close(pos, ctx, pos.partial_tp2_price,
                                              LC.PARTIAL_TP2_FRACTION, "tp_partial_2"))
            pos.partial_tp_level = 2
            pos.trail_check()

        # 3. Trailing stop
        pos.trail_check()
        if pos.trailing_active and self._trail_hit(pos, bar_high, bar_low):
            trades.append(self._close_remainder(pos, ctx, pos.trailing_stop_price, "trail"))
            self.positions.pop(ctx.symbol, None)
            return trades

        # 4. Final TP — only relevant if partial-TP-level reached the end and
        # the position is still open after trailing kicked in. Live's full TP
        # at tp_price kicks in if trailing isn't yet active.
        if not pos.trailing_active and self._tp_hit(pos, bar_high, bar_low):
            trades.append(self._close_remainder(pos, ctx, pos.tp_price, "tp_full"))
            self.positions.pop(ctx.symbol, None)
            return trades

        # 5. Stagnant exit
        if self._stagnant_hit(pos, ctx):
            trades.append(self._close_remainder(pos, ctx, ctx.current_close, "stagnant"))
            self.positions.pop(ctx.symbol, None)
            return trades

        return trades

    # ── close helpers ──────────────────────────────────────────────
    def _partial_close(self, pos: Position, ctx: Ctx,
                       price: float, fraction: float, reason: str) -> Trade:
        units_to_close = pos.initial_units * fraction
        slip_price = price * (1 - self.slippage_pct) if pos.is_long \
                     else price * (1 + self.slippage_pct)
        gross = (slip_price - pos.entry_price) * units_to_close if pos.is_long \
                else (pos.entry_price - slip_price) * units_to_close
        fee = slip_price * units_to_close * self.fees_taker
        net = gross - fee
        pos.units -= units_to_close
        pos.realized_pnl += net
        pos.fees_paid += fee
        self.balance += net
        self.realized_pnl += net
        return self._make_trade(pos, ctx, slip_price, units_to_close, net, fee,
                                reason, is_full=False)

    def _close_remainder(self, pos: Position, ctx: Ctx,
                         price: float, reason: str) -> Trade:
        slip_price = price * (1 - self.slippage_pct) if pos.is_long \
                     else price * (1 + self.slippage_pct)
        units_to_close = pos.units
        gross = (slip_price - pos.entry_price) * units_to_close if pos.is_long \
                else (pos.entry_price - slip_price) * units_to_close
        fee = slip_price * units_to_close * self.fees_taker
        net = gross - fee
        pos.units = 0
        pos.realized_pnl += net
        pos.fees_paid += fee
        self.balance += net
        self.realized_pnl += net
        return self._make_trade(pos, ctx, slip_price, units_to_close, net, fee,
                                reason, is_full=True)

    def force_close(self, ctx: Ctx, reason: str) -> Optional[Trade]:
        """Force-close from outside (e.g., REVERSE_CLOSE_LONG on opposite signal)."""
        pos = self.positions.get(ctx.symbol)
        if pos is None: return None
        trade = self._close_remainder(pos, ctx, ctx.current_close, reason)
        self.positions.pop(ctx.symbol, None)
        return trade

    def _make_trade(self, pos: Position, ctx: Ctx, exit_price: float,
                    units: float, net: float, fee: float, reason: str,
                    is_full: bool) -> Trade:
        holding_min = (ctx.now_ms - pos.open_ts_ms) / 60_000.0
        # PnL in R units
        risk_per_unit = pos.entry_price * pos.sl_pct if pos.sl_pct > 0 else 1
        pnl_r = net / (risk_per_unit * pos.initial_units) if risk_per_unit else 0
        return Trade(
            symbol=pos.symbol, side=pos.side,
            entry_ts_ms=pos.open_ts_ms, exit_ts_ms=ctx.now_ms,
            entry_price=pos.entry_price, exit_price=exit_price,
            units=units, pnl_usd=net, pnl_r=pnl_r, fees_usd=fee,
            reason=reason, confidence=pos.confidence,
            mfe_pct=pos.mfe_pct, mae_pct=pos.mae_pct,
            holding_minutes=holding_min, is_full_close=is_full,
            extras=dict(pos.extras),
        )

    # ── price comparisons ─────────────────────────────────────────
    @staticmethod
    def _sl_hit(pos: Position, high: float, low: float) -> bool:
        return (pos.is_long and low <= pos.sl_price) or \
               (pos.is_short and high >= pos.sl_price)

    @staticmethod
    def _partial_tp1_hit(pos: Position, high: float, low: float) -> bool:
        return (pos.is_long and high >= pos.partial_tp1_price) or \
               (pos.is_short and low <= pos.partial_tp1_price)

    @staticmethod
    def _partial_tp2_hit(pos: Position, high: float, low: float) -> bool:
        return (pos.is_long and high >= pos.partial_tp2_price) or \
               (pos.is_short and low <= pos.partial_tp2_price)

    @staticmethod
    def _tp_hit(pos: Position, high: float, low: float) -> bool:
        return (pos.is_long and high >= pos.tp_price) or \
               (pos.is_short and low <= pos.tp_price)

    @staticmethod
    def _trail_hit(pos: Position, high: float, low: float) -> bool:
        if pos.trailing_stop_price <= 0: return False
        return (pos.is_long and low <= pos.trailing_stop_price) or \
               (pos.is_short and high >= pos.trailing_stop_price)

    @staticmethod
    def _stagnant_hit(pos: Position, ctx: Ctx) -> bool:
        elapsed_h = (ctx.now_ms - pos.open_ts_ms) / 3_600_000.0
        if elapsed_h < LC.STAGNANT_HOURS: return False
        cur = ctx.current_close
        pnl_pct = (cur - pos.entry_price) / pos.entry_price if pos.is_long \
                  else (pos.entry_price - cur) / pos.entry_price
        return LC.STAGNANT_PCT_MIN <= pnl_pct <= LC.STAGNANT_PCT_MAX
