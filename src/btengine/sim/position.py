"""Position — mirrors live_trading_htf.py's open/close + partial-TP state.

The live bot tracks per-symbol position attributes on the HTFLiveBot
instance: position (LONG/SHORT/FLAT), position_price, position_units,
sl_price, tp_price, partial_tp1_price, partial_tp2_price,
partial_tp_level, mfe_pct, mae_pct, peak_price, entry_time.

This dataclass is the backtest equivalent. Math must match live so that
the M5 parity test can replay the last 14 days within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .. import live_constants as LC


@dataclass
class Position:
    symbol: str
    side: str             # 'LONG' | 'SHORT'
    entry_price: float
    initial_units: float  # size at open, used as denominator for partials
    units: float          # remaining size after partial closes
    sl_price: float
    tp_price: float
    partial_tp1_price: float
    partial_tp2_price: float
    sl_pct: float         # |sl - entry| / entry
    tp_pct: float
    leverage: int
    confidence: float
    open_ts_ms: int
    open_intent_reason: str = ""

    # Mutable state
    partial_tp_level: int = 0    # 0 = no partials hit, 1 = TP1 hit, 2 = TP1+TP2 hit
    peak_price: float = 0.0      # highest (LONG) or lowest (SHORT) seen since open
    mfe_pct: float = 0.0         # max favorable excursion
    mae_pct: float = 0.0         # max adverse excursion (negative number)
    trailing_active: bool = False
    trailing_stop_price: float = 0.0
    realized_pnl: float = 0.0    # PnL from partial closes that already booked
    fees_paid: float = 0.0       # cumulative fees on this position (entry + partials)

    # Free-form snapshot for downstream analysis
    extras: dict = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────────
    @property
    def is_long(self) -> bool: return self.side == "LONG"

    @property
    def is_short(self) -> bool: return self.side == "SHORT"

    def update_excursions(self, high: float, low: float) -> None:
        """Update MFE/MAE/peak using bar high+low (path-aware)."""
        if self.is_long:
            fav = (high - self.entry_price) / self.entry_price
            adv = (low - self.entry_price) / self.entry_price
            if fav > self.mfe_pct: self.mfe_pct = fav
            if adv < self.mae_pct: self.mae_pct = adv
            if high > self.peak_price: self.peak_price = high
        else:
            fav = (self.entry_price - low) / self.entry_price
            adv = (self.entry_price - high) / self.entry_price
            if fav > self.mfe_pct: self.mfe_pct = fav
            if adv < self.mae_pct: self.mae_pct = adv
            if self.peak_price == 0 or low < self.peak_price:
                self.peak_price = low

    # ── trailing-stop management ──────────────────────────────────
    def activate_trailing_if_eligible(self) -> None:
        """Once profit ≥ TRAILING_ACTIVATE_PCT, start trailing the peak."""
        if self.trailing_active or self.peak_price == 0: return
        if self.is_long:
            move_pct = (self.peak_price - self.entry_price) / self.entry_price
        else:
            move_pct = (self.entry_price - self.peak_price) / self.entry_price
        if move_pct >= LC.TRAILING_ACTIVATE_PCT:
            self.trailing_active = True
            self._update_trailing_stop()

    def _update_trailing_stop(self) -> None:
        """Recompute the trailing stop based on peak + current distance."""
        distance = (LC.TRAILING_DISTANCE_POST_TP1
                    if self.partial_tp_level >= 1
                    else LC.TRAILING_DISTANCE_PRE_TP1)
        if self.is_long:
            new_stop = self.peak_price * (1 - distance)
            # Trailing stops only ratchet up
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
        else:
            new_stop = self.peak_price * (1 + distance)
            if self.trailing_stop_price == 0 or new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop

    def trail_check(self) -> None:
        """Update trailing stop after MFE/peak update + partial-TP changes."""
        if self.trailing_active:
            self._update_trailing_stop()
        else:
            self.activate_trailing_if_eligible()
