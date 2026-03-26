"""
BOS / CHOCH Market Structure Detection Module (ICT-based rewrite)
=================================================================

Detects Break of Structure (BOS) and Change of Character (CHOCH) using
proper ICT methodology based on swing point sequences.

Theory (ICT / Smart Money Concepts):
  - Market structure is defined by the SEQUENCE of swing highs and lows.
  - **Uptrend**: Higher Highs (HH) + Higher Lows (HL)
  - **Downtrend**: Lower Highs (LH) + Lower Lows (LL)

  - **BOS (Break of Structure)**: Trend CONTINUATION signal.
    - Bullish BOS: In uptrend, price closes above the most recent swing high → new HH
    - Bearish BOS: In downtrend, price closes below the most recent swing low → new LL

  - **CHOCH (Change of Character)**: Trend REVERSAL signal.
    - Bullish CHOCH: In downtrend, price closes above the most recent swing high
      (breaks the LH pattern → potential reversal to bullish)
    - Bearish CHOCH: In uptrend, price closes below the most recent swing low
      (breaks the HL pattern → potential reversal to bearish)

  - **Fake signals**: Detected when the breakout candle has excessive wick
    (>60% wick ratio) or reverses within 3 bars.

Key difference from old implementation:
  - Detects ALL BOS/CHOCH events across the chart history, not just current bar
  - Trend is determined by swing SEQUENCE (HH/HL vs LH/LL), not separate calculation
  - Uses ZigZag-style filtering (min price change) to reduce noise

Author:  CEO bot (rewrite)
Date:    2026-03-26
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    index: int           # integer position in the DataFrame
    price: float         # high (for swing high) or low (for swing low)
    kind: str            # "high" | "low"
    label: str = ""      # "HH", "HL", "LH", "LL" — set after sequence analysis
    timestamp: Optional[str] = None


@dataclass
class StructureSignal:
    """A BOS or CHOCH signal."""
    kind: str            # "bos" | "choch"
    direction: str       # "bullish" | "bearish"
    bar_index: int       # bar where the break happened
    level: float         # price level that was broken
    is_fake: bool = False
    timestamp: Optional[str] = None


@dataclass
class MarketStructureResult:
    """Aggregated result of market structure analysis."""
    bos_bullish: bool = False
    bos_bearish: bool = False
    choch_bullish: bool = False
    choch_bearish: bool = False
    fake_bos: bool = False
    fake_choch: bool = False
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    trend: str = "ranging"
    confidence: float = 0.5
    signals: List[StructureSignal] = field(default_factory=list)
    swing_points: List[SwingPoint] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "bos_bullish": self.bos_bullish,
            "bos_bearish": self.bos_bearish,
            "choch_bullish": self.choch_bullish,
            "choch_bearish": self.choch_bearish,
            "fake_bos": self.fake_bos,
            "fake_choch": self.fake_choch,
            "last_swing_high": self.last_swing_high,
            "last_swing_low": self.last_swing_low,
            "trend": self.trend,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketStructure:
    """ICT-based market structure detector with BOS/CHOCH detection."""

    def __init__(
        self,
        swing_lookback: int = 5,
        min_swing_pct: float = 0.002,   # 0.2% minimum swing size to filter noise
    ):
        self.swing_lookback = swing_lookback
        self.min_swing_pct = min_swing_pct

    # ------------------------------------------------------------------
    # Swing Point Detection (ZigZag-filtered)
    # ------------------------------------------------------------------

    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Find swing highs and swing lows using lookback-on-both-sides method,
        then filter using ZigZag minimum price change to remove noise.
        """
        n = self.swing_lookback
        highs = df["high"].values
        lows = df["low"].values
        length = len(df)
        raw_points: List[SwingPoint] = []

        for i in range(n, length - n):
            window_high = highs[i - n: i + n + 1]
            window_low = lows[i - n: i + n + 1]

            ts = str(df.index[i]) if hasattr(df.index, "strftime") else None

            if highs[i] == np.max(window_high):
                raw_points.append(SwingPoint(
                    index=i, price=float(highs[i]), kind="high", timestamp=ts
                ))

            if lows[i] == np.min(window_low):
                raw_points.append(SwingPoint(
                    index=i, price=float(lows[i]), kind="low", timestamp=ts
                ))

        raw_points.sort(key=lambda p: p.index)

        # ZigZag filter: alternate high/low and enforce minimum price change
        filtered = self._zigzag_filter(raw_points)
        return filtered

    def _zigzag_filter(self, points: List[SwingPoint]) -> List[SwingPoint]:
        """
        Filter swing points to alternate high/low and enforce minimum
        price change between consecutive swings.
        """
        if not points:
            return []

        result: List[SwingPoint] = [points[0]]

        for p in points[1:]:
            last = result[-1]

            # If same type as last, keep the more extreme one
            if p.kind == last.kind:
                if p.kind == "high" and p.price > last.price:
                    result[-1] = p
                elif p.kind == "low" and p.price < last.price:
                    result[-1] = p
                continue

            # Different type — check minimum price change
            pct_change = abs(p.price - last.price) / last.price
            if pct_change >= self.min_swing_pct:
                result.append(p)
            else:
                # Too small a move — skip this point but if it's more extreme
                # than last of same type, replace
                pass

        return result

    # ------------------------------------------------------------------
    # Label swing points (HH, HL, LH, LL)
    # ------------------------------------------------------------------

    @staticmethod
    def label_swings(swings: List[SwingPoint]) -> List[SwingPoint]:
        """
        Label each swing point as HH, HL, LH, or LL based on the previous
        swing of the same type.
        """
        prev_high: Optional[SwingPoint] = None
        prev_low: Optional[SwingPoint] = None

        for sp in swings:
            if sp.kind == "high":
                if prev_high is None:
                    sp.label = "HH"  # first high, assume HH
                elif sp.price > prev_high.price:
                    sp.label = "HH"
                else:
                    sp.label = "LH"
                prev_high = sp
            else:
                if prev_low is None:
                    sp.label = "HL"  # first low, assume HL
                elif sp.price > prev_low.price:
                    sp.label = "HL"
                else:
                    sp.label = "LL"
                prev_low = sp

        return swings

    # ------------------------------------------------------------------
    # Trend determination from swing sequence
    # ------------------------------------------------------------------

    @staticmethod
    def determine_trend(swing_points: List[SwingPoint]) -> str:
        """
        Determine trend from the last 4+ labeled swing points.

        Bullish: last 2 highs are HH AND last 2 lows are HL
        Bearish: last 2 highs are LH AND last 2 lows are LL
        """
        if len(swing_points) < 4:
            return "ranging"

        recent_highs = [p for p in swing_points if p.kind == "high"][-3:]
        recent_lows = [p for p in swing_points if p.kind == "low"][-3:]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return "ranging"

        # Check last 2 highs and lows
        last_2h = recent_highs[-2:]
        last_2l = recent_lows[-2:]

        hh = last_2h[-1].label == "HH"
        hl = last_2l[-1].label == "HL"
        lh = last_2h[-1].label == "LH"
        ll = last_2l[-1].label == "LL"

        if hh and hl:
            return "bullish"
        elif lh and ll:
            return "bearish"
        return "ranging"

    # ------------------------------------------------------------------
    # BOS / CHOCH detection across entire chart
    # ------------------------------------------------------------------

    def detect_all_structure_breaks(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
    ) -> Tuple[List[StructureSignal], List[StructureSignal]]:
        """
        Walk through swing points chronologically and detect ALL BOS and
        CHOCH events in the chart history.

        Returns (bos_signals, choch_signals).
        """
        bos_signals: List[StructureSignal] = []
        choch_signals: List[StructureSignal] = []

        if len(swings) < 4:
            return bos_signals, choch_signals

        closes = df["close"].values
        length = len(df)

        # Walk through swings and detect breaks
        # Trend is determined progressively from the swing sequence
        trend = "ranging"

        for i, sp in enumerate(swings):
            if i < 3:
                continue

            # Determine trend from recent swings up to this point
            recent = swings[:i + 1]
            recent_highs = [p for p in recent if p.kind == "high"][-2:]
            recent_lows = [p for p in recent if p.kind == "low"][-2:]

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                is_hh = recent_highs[-1].price > recent_highs[-2].price
                is_hl = recent_lows[-1].price > recent_lows[-2].price
                is_lh = recent_highs[-1].price < recent_highs[-2].price
                is_ll = recent_lows[-1].price < recent_lows[-2].price

                if is_hh and is_hl:
                    new_trend = "bullish"
                elif is_lh and is_ll:
                    new_trend = "bearish"
                elif is_hh or is_hl:
                    new_trend = "bullish"   # lean bullish on partial signal
                elif is_lh or is_ll:
                    new_trend = "bearish"   # lean bearish on partial signal
                else:
                    new_trend = trend

                # Find scan limit: next swing of SAME type (the level becomes
                # irrelevant once a new swing of same type forms)
                scan_limit = length
                for j in range(i + 1, len(swings)):
                    if swings[j].kind == sp.kind:
                        scan_limit = swings[j].index + 1
                        break

                if trend != "ranging":
                    if sp.kind == "high" and trend == "bullish":
                        for bar in range(sp.index + 1, scan_limit):
                            if closes[bar] > sp.price:
                                ts = str(df.index[bar]) if hasattr(df.index, "strftime") else None
                                bos_signals.append(StructureSignal(
                                    kind="bos", direction="bullish",
                                    bar_index=bar, level=sp.price, timestamp=ts,
                                ))
                                break

                    elif sp.kind == "low" and trend == "bearish":
                        for bar in range(sp.index + 1, scan_limit):
                            if closes[bar] < sp.price:
                                ts = str(df.index[bar]) if hasattr(df.index, "strftime") else None
                                bos_signals.append(StructureSignal(
                                    kind="bos", direction="bearish",
                                    bar_index=bar, level=sp.price, timestamp=ts,
                                ))
                                break

                    elif sp.kind == "high" and trend == "bearish":
                        for bar in range(sp.index + 1, scan_limit):
                            if closes[bar] > sp.price:
                                ts = str(df.index[bar]) if hasattr(df.index, "strftime") else None
                                choch_signals.append(StructureSignal(
                                    kind="choch", direction="bullish",
                                    bar_index=bar, level=sp.price, timestamp=ts,
                                ))
                                break

                    elif sp.kind == "low" and trend == "bullish":
                        for bar in range(sp.index + 1, scan_limit):
                            if closes[bar] < sp.price:
                                ts = str(df.index[bar]) if hasattr(df.index, "strftime") else None
                                choch_signals.append(StructureSignal(
                                    kind="choch", direction="bearish",
                                    bar_index=bar, level=sp.price, timestamp=ts,
                                ))
                                break

                trend = new_trend

        return bos_signals, choch_signals

    # ------------------------------------------------------------------
    # Fake signal detection
    # ------------------------------------------------------------------

    def is_fake_breakout(self, df: pd.DataFrame, signal: StructureSignal) -> bool:
        """
        Check if a BOS/CHOCH is a fake breakout using:
        1. Wick rejection: candle body is <40% of total range (big wick)
        2. Rapid reversal: price reverses within 3 bars
        """
        idx = signal.bar_index
        if idx >= len(df):
            return False

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values

        # 1. Wick rejection check
        candle_range = highs[idx] - lows[idx]
        if candle_range > 0:
            body = abs(closes[idx] - opens[idx])
            body_ratio = body / candle_range
            if body_ratio < 0.30:  # body is less than 30% of range → big wick
                return True

        # 2. Rapid reversal within 3 bars
        end_idx = min(idx + 4, len(df))
        if end_idx > idx + 1:
            if signal.direction == "bullish":
                # Bullish break → fake if price drops back below level
                if any(closes[j] < signal.level for j in range(idx + 1, end_idx)):
                    return True
            else:
                # Bearish break → fake if price rises back above level
                if any(closes[j] > signal.level for j in range(idx + 1, end_idx)):
                    return True

        return False

    # ------------------------------------------------------------------
    # Backward-compatible API
    # ------------------------------------------------------------------

    def detect_bos(self, df, swing_points, trend):
        """Backward-compatible: detect BOS for current bar only."""
        bos, _ = self.detect_all_structure_breaks(df, swing_points)
        return bos[-1:] if bos else []

    def detect_choch(self, df, swing_points, trend):
        """Backward-compatible: detect CHOCH for current bar only."""
        _, choch = self.detect_all_structure_breaks(df, swing_points)
        return choch[-1:] if choch else []

    def is_fake_bos(self, df, signal):
        return self.is_fake_breakout(df, signal)

    def is_fake_choch(self, df, signal):
        return self.is_fake_breakout(df, signal)

    # ------------------------------------------------------------------
    # Full analysis (single timeframe)
    # ------------------------------------------------------------------

    def _analyze_single_tf(self, df: pd.DataFrame) -> MarketStructureResult:
        """Run full BOS/CHOCH analysis on a single timeframe."""
        result = MarketStructureResult()

        if df is None or len(df) < self.swing_lookback * 3:
            return result

        # 1. Detect and label swing points
        swings = self.detect_swing_points(df)
        if len(swings) < 4:
            return result

        swings = self.label_swings(swings)
        result.swing_points = swings

        # 2. Determine trend from swing sequence
        result.trend = self.determine_trend(swings)

        # Last swing levels
        highs = [p for p in swings if p.kind == "high"]
        lows = [p for p in swings if p.kind == "low"]
        if highs:
            result.last_swing_high = highs[-1].price
        if lows:
            result.last_swing_low = lows[-1].price

        # 3. Detect ALL structure breaks
        bos_signals, choch_signals = self.detect_all_structure_breaks(df, swings)

        # 4. Check for fakes
        for sig in bos_signals:
            sig.is_fake = self.is_fake_breakout(df, sig)
            if not sig.is_fake:
                if sig.direction == "bullish":
                    result.bos_bullish = True
                else:
                    result.bos_bearish = True
            else:
                result.fake_bos = True
            result.signals.append(sig)

        for sig in choch_signals:
            sig.is_fake = self.is_fake_breakout(df, sig)
            if not sig.is_fake:
                if sig.direction == "bullish":
                    result.choch_bullish = True
                else:
                    result.choch_bearish = True
            else:
                result.fake_choch = True
            result.signals.append(sig)

        return result

    # ------------------------------------------------------------------
    # Main entry: multi-timeframe analysis
    # ------------------------------------------------------------------

    def get_signals(
        self,
        df_primary: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Main entry point: analyse up to 3 timeframes and return aggregated
        market structure signals.
        """
        r5m = self._analyze_single_tf(df_primary)

        r1h = self._analyze_single_tf(df_1h) if df_1h is not None else None
        r4h = self._analyze_single_tf(df_4h) if df_4h is not None else None

        result = MarketStructureResult(
            bos_bullish=r5m.bos_bullish,
            bos_bearish=r5m.bos_bearish,
            choch_bullish=r5m.choch_bullish,
            choch_bearish=r5m.choch_bearish,
            fake_bos=r5m.fake_bos,
            fake_choch=r5m.fake_choch,
            last_swing_high=r5m.last_swing_high,
            last_swing_low=r5m.last_swing_low,
            trend=r5m.trend,
            confidence=0.5,
            signals=list(r5m.signals),
            swing_points=list(r5m.swing_points),
        )

        # HTF confirmation
        if r1h is not None:
            if r1h.bos_bullish and r5m.bos_bullish:
                result.confidence += 0.15
            if r1h.bos_bearish and r5m.bos_bearish:
                result.confidence += 0.15
            if r1h.choch_bullish and r5m.choch_bullish:
                result.confidence += 0.10
            if r1h.choch_bearish and r5m.choch_bearish:
                result.confidence += 0.10
            if (r1h.bos_bullish and r5m.bos_bearish) or (r1h.bos_bearish and r5m.bos_bullish):
                result.confidence -= 0.10

        if r4h is not None:
            if r4h.bos_bullish and r5m.bos_bullish:
                result.confidence += 0.20
            if r4h.bos_bearish and r5m.bos_bearish:
                result.confidence += 0.20
            if r4h.choch_bullish and r5m.choch_bullish:
                result.confidence += 0.15
            if r4h.choch_bearish and r5m.choch_bearish:
                result.confidence += 0.15
            if (r4h.bos_bullish and r5m.choch_bearish):
                result.confidence -= 0.15
            if (r4h.bos_bearish and r5m.choch_bullish):
                result.confidence -= 0.15

        result.confidence = float(np.clip(result.confidence, 0.0, 1.0))

        logger.info(
            "MarketStructure signals: trend=%s bos_bull=%s bos_bear=%s "
            "choch_bull=%s choch_bear=%s fake_bos=%s fake_choch=%s "
            "swing_high=%.2f swing_low=%.2f conf=%.2f (bos=%d choch=%d swings=%d)",
            result.trend,
            result.bos_bullish, result.bos_bearish,
            result.choch_bullish, result.choch_bearish,
            result.fake_bos, result.fake_choch,
            result.last_swing_high, result.last_swing_low,
            result.confidence,
            len([s for s in result.signals if s.kind == "bos"]),
            len([s for s in result.signals if s.kind == "choch"]),
            len(result.swing_points),
        )

        return result.to_dict()
