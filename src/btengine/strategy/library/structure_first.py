"""structure_first_v3 — port of live's STRUCTURE_FIRST_MODE entry path.

Wraps `src.signals.bos_choch.MarketStructure` as an EntryRule. Uses the
ctx's primary dataframe (typically 15m) as the structure timeframe and
`ctx.htf['1h']` / `ctx.htf['4h']` as multi-timeframe confirmation.

Phase 1 deviation from live: the live bot uses 5m for structure
detection. Here we use the primary timeframe (15m by default) for both
structure and entry decisions. This matches `backtest_realistic.py`.
The Ctx supports a 5m-structure variant if needed in Phase 2 (just pass
df_primary_5m on the ctx and read it here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from src.signals.bos_choch import MarketStructure

from ..base import EntryRule, Intent, Strategy, register_strategy
from ... import live_constants as LC

logger = logging.getLogger(__name__)


@dataclass
class StructureFirstEntry(EntryRule):
    """Entry rule that defers to MarketStructure for direction + confidence."""

    swing_lookback: int = 8
    min_confidence: float = LC.MIN_CONFIDENCE
    htf_intervals: tuple = ("1h", "4h")
    min_primary_bars: int = 30        # need this much history before signaling
    primary_window: int = 200         # only last N bars passed to MarketStructure
                                      # (huge speedup vs growing df; swing_lookback=8 only needs ~20 bars)
    htf_window: int = 200             # cap HTF context size too
    _market_structure: Optional[MarketStructure] = field(default=None, init=False)

    def __post_init__(self):
        self._market_structure = MarketStructure(swing_lookback=self.swing_lookback)

    def __call__(self, ctx) -> Intent:
        if len(ctx.primary) < self.min_primary_bars:
            return Intent(action="HOLD", reason="warmup")
        # Cap context to last N bars — MarketStructure only needs recent swings
        primary = ctx.primary.iloc[-self.primary_window:] if len(ctx.primary) > self.primary_window else ctx.primary
        df_1h = ctx.htf_up_to_now("1h") if "1h" in self.htf_intervals else None
        df_4h = ctx.htf_up_to_now("4h") if "4h" in self.htf_intervals else None
        if df_1h is not None and len(df_1h) > self.htf_window:
            df_1h = df_1h.iloc[-self.htf_window:]
        if df_4h is not None and len(df_4h) > self.htf_window:
            df_4h = df_4h.iloc[-self.htf_window:]

        try:
            sig = self._market_structure.get_signals(
                df_primary=primary, df_1h=df_1h, df_4h=df_4h,
            )
        except Exception as exc:
            logger.debug("MarketStructure failed at %s %s: %s",
                         ctx.symbol, ctx.t_now_iso, exc)
            return Intent(action="HOLD", reason="ms_error")

        # Persist for guards / exits to inspect later
        sig_d = sig if isinstance(sig, dict) else (sig.to_dict() if hasattr(sig, "to_dict") else {})
        ctx.extras["structure"] = sig_d

        confidence = float(sig_d.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return Intent(action="HOLD",
                          confidence=confidence,
                          reason=f"low_conf<{self.min_confidence:.2f}")

        # Direction: BOS bullish or CHOCH bullish → LONG; bearish → SHORT
        bos_bull = bool(sig_d.get("bos_bullish", False))
        bos_bear = bool(sig_d.get("bos_bearish", False))
        choch_bull = bool(sig_d.get("choch_bullish", False))
        choch_bear = bool(sig_d.get("choch_bearish", False))
        fake_bos = bool(sig_d.get("fake_bos", False))

        if fake_bos:
            return Intent(action="HOLD", confidence=confidence, reason="fake_bos")

        action = "HOLD"
        if bos_bull or choch_bull:
            action = "OPEN_LONG"
        elif bos_bear or choch_bear:
            action = "OPEN_SHORT"

        return Intent(
            action=action, confidence=confidence,
            reason="structure_first",
            extras={
                "bos_bull": bos_bull, "bos_bear": bos_bear,
                "choch_bull": choch_bull, "choch_bear": choch_bear,
                "trend": sig_d.get("trend"),
            },
        )


@register_strategy("structure_first_v3")
class StructureFirstV3(Strategy):
    """Phase 1 strategy: structure-only entry, no guards/exits yet.

    Guards land in M4 (wired via GuardChain on top of this entry).
    Exits land in M3 (ExitPolicy with partial TPs + trailing + stagnant).

    Today this is just an entry oracle — useful for emitting historical
    intent counts to compare against live trade history.
    """

    def __init__(self, **overrides):
        self.entry = StructureFirstEntry(
            swing_lookback=int(overrides.get("swing_lookback", 8)),
            min_confidence=float(overrides.get("min_confidence", LC.MIN_CONFIDENCE)),
        )
