"""
BOS / CHOCH Market Structure Detection Module
==============================================

Detects Break of Structure (BOS), Change of Character (CHOCH), and their
"fake" variants from OHLCV DataFrames.  Used by the HTF trading bots to
dynamically adjust SL/TP while a position is profitable.

Theory:
  - **BOS (Break of Structure)**: Price continues the existing trend by
    breaking a prior swing high (bullish) or swing low (bearish).
  - **CHOCH (Change of Character)**: Price *reverses* the trend by breaking
    the opposite swing level — e.g. breaking above a swing high during a
    downtrend signals a bullish reversal.
  - **Fake BOS / Fake CHOCH**: Breakouts that fail, detected by wick
    rejection, volume divergence, or rapid reversal within 3 bars.

Multi-timeframe approach:
  The primary analysis runs on the 15-minute chart.  1H and 4H signals
  serve as confirmation / override layers to increase confidence.

Author:  Builder bot
Date:    2026-03-23
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
    timestamp: Optional[str] = None


@dataclass
class StructureSignal:
    """A BOS or CHOCH signal (or their fake variants)."""
    kind: str            # "bos" | "choch"
    direction: str       # "bullish" | "bearish"
    bar_index: int       # bar where the break was confirmed
    level: float         # the swing level that was broken
    is_fake: bool = False
    confidence: float = 1.0


@dataclass
class MarketStructureResult:
    """Container returned by ``MarketStructure.get_signals()``."""
    bos_bullish: bool = False
    bos_bearish: bool = False
    choch_bullish: bool = False
    choch_bearish: bool = False
    fake_bos: bool = False
    fake_choch: bool = False
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    trend: str = "ranging"         # "bullish" | "bearish" | "ranging"
    confidence: float = 0.0
    signals: List[StructureSignal] = field(default_factory=list)

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
# MarketStructure detector
# ---------------------------------------------------------------------------

class MarketStructure:
    """
    Detects BOS, CHOCH, Fake BOS, and Fake CHOCH from OHLCV DataFrames.

    Parameters
    ----------
    swing_lookback : int
        Number of bars on each side that a local extreme must dominate
        to qualify as a swing point.  Default 5 (good for 15-min charts).
    fake_reversal_bars : int
        If price returns below/above the broken level within this many
        bars, the breakout is considered fake.  Default 3.
    fake_wick_ratio : float
        A candle is considered a wick rejection if the wick beyond the
        level is > this fraction of total range.  Default 0.60.
    volume_decline_pct : float
        If breakout-bar volume is below the rolling average by this
        fraction, flag volume divergence.  Default 0.20 (20% below avg).
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        fake_reversal_bars: int = 3,
        fake_wick_ratio: float = 0.60,
        volume_decline_pct: float = 0.20,
    ):
        self.swing_lookback = swing_lookback
        self.fake_reversal_bars = fake_reversal_bars
        self.fake_wick_ratio = fake_wick_ratio
        self.volume_decline_pct = volume_decline_pct

    # ------------------------------------------------------------------
    # Swing point detection
    # ------------------------------------------------------------------

    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Find swing highs and swing lows using the *lookback-on-both-sides*
        method.

        A swing high at bar *i* requires:
            high[i] == max(high[i-N : i+N+1])

        Similarly for swing low (using min of low).

        Parameters
        ----------
        df : DataFrame with columns ``high``, ``low``, and optionally a
             DatetimeIndex.

        Returns
        -------
        List of SwingPoint sorted by index.
        """
        n = self.swing_lookback
        highs = df["high"].values
        lows = df["low"].values
        length = len(df)
        points: List[SwingPoint] = []

        for i in range(n, length - n):
            window_high = highs[i - n: i + n + 1]
            window_low = lows[i - n: i + n + 1]

            # Swing High: highest in window
            if highs[i] == np.max(window_high):
                ts = str(df.index[i]) if hasattr(df.index, "strftime") else None
                points.append(SwingPoint(index=i, price=float(highs[i]), kind="high", timestamp=ts))

            # Swing Low: lowest in window
            if lows[i] == np.min(window_low):
                ts = str(df.index[i]) if hasattr(df.index, "strftime") else None
                points.append(SwingPoint(index=i, price=float(lows[i]), kind="low", timestamp=ts))

        return sorted(points, key=lambda p: p.index)

    # ------------------------------------------------------------------
    # Trend determination
    # ------------------------------------------------------------------

    @staticmethod
    def determine_trend(swing_points: List[SwingPoint]) -> str:
        """
        Determine trend from the last 4+ swing points.

        Bullish: Higher Highs + Higher Lows
        Bearish: Lower Lows + Lower Highs
        Ranging: mixed or insufficient data
        """
        if len(swing_points) < 4:
            return "ranging"

        # Collect recent highs and lows (last 6 points max)
        recent = swing_points[-6:]
        recent_highs = [p for p in recent if p.kind == "high"]
        recent_lows = [p for p in recent if p.kind == "low"]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return "ranging"

        # Check Higher Highs / Higher Lows
        hh = all(
            recent_highs[i].price > recent_highs[i - 1].price
            for i in range(1, len(recent_highs))
        )
        hl = all(
            recent_lows[i].price > recent_lows[i - 1].price
            for i in range(1, len(recent_lows))
        )
        if hh and hl:
            return "bullish"

        # Check Lower Lows / Lower Highs
        ll = all(
            recent_lows[i].price < recent_lows[i - 1].price
            for i in range(1, len(recent_lows))
        )
        lh = all(
            recent_highs[i].price < recent_highs[i - 1].price
            for i in range(1, len(recent_highs))
        )
        if ll and lh:
            return "bearish"

        return "ranging"

    # ------------------------------------------------------------------
    # BOS detection
    # ------------------------------------------------------------------

    def detect_bos(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        trend: str,
    ) -> List[StructureSignal]:
        """
        Detect Break of Structure.

        Bullish BOS (in an uptrend):
            Current close > most recent swing high
        Bearish BOS (in a downtrend):
            Current close < most recent swing low
        """
        signals: List[StructureSignal] = []
        if not swing_points:
            return signals

        closes = df["close"].values
        last_bar = len(df) - 1

        # Most recent swing high / low
        recent_highs = [p for p in swing_points if p.kind == "high"]
        recent_lows = [p for p in swing_points if p.kind == "low"]

        if trend in ("bullish", "ranging") and recent_highs:
            sh = recent_highs[-1]
            # Check if any of the last 3 bars closed above the swing high
            for offset in range(min(3, last_bar - sh.index)):
                bar_idx = last_bar - offset
                if bar_idx > sh.index and closes[bar_idx] > sh.price:
                    signals.append(StructureSignal(
                        kind="bos",
                        direction="bullish",
                        bar_index=bar_idx,
                        level=sh.price,
                    ))
                    break

        if trend in ("bearish", "ranging") and recent_lows:
            sl_point = recent_lows[-1]
            for offset in range(min(3, last_bar - sl_point.index)):
                bar_idx = last_bar - offset
                if bar_idx > sl_point.index and closes[bar_idx] < sl_point.price:
                    signals.append(StructureSignal(
                        kind="bos",
                        direction="bearish",
                        bar_index=bar_idx,
                        level=sl_point.price,
                    ))
                    break

        return signals

    # ------------------------------------------------------------------
    # CHOCH detection
    # ------------------------------------------------------------------

    def detect_choch(
        self,
        df: pd.DataFrame,
        swing_points: List[SwingPoint],
        trend: str,
    ) -> List[StructureSignal]:
        """
        Detect Change of Character (trend reversal signal).

        Bullish CHOCH (in a downtrend):
            Close > most recent swing high  →  potential reversal to bullish
        Bearish CHOCH (in an uptrend):
            Close < most recent swing low   →  potential reversal to bearish
        """
        signals: List[StructureSignal] = []
        if not swing_points:
            return signals

        closes = df["close"].values
        last_bar = len(df) - 1

        recent_highs = [p for p in swing_points if p.kind == "high"]
        recent_lows = [p for p in swing_points if p.kind == "low"]

        # Bullish CHOCH: downtrend + close above swing high
        if trend == "bearish" and recent_highs:
            sh = recent_highs[-1]
            for offset in range(min(3, last_bar - sh.index)):
                bar_idx = last_bar - offset
                if bar_idx > sh.index and closes[bar_idx] > sh.price:
                    signals.append(StructureSignal(
                        kind="choch",
                        direction="bullish",
                        bar_index=bar_idx,
                        level=sh.price,
                    ))
                    break

        # Bearish CHOCH: uptrend + close below swing low
        if trend == "bullish" and recent_lows:
            sl_point = recent_lows[-1]
            for offset in range(min(3, last_bar - sl_point.index)):
                bar_idx = last_bar - offset
                if bar_idx > sl_point.index and closes[bar_idx] < sl_point.price:
                    signals.append(StructureSignal(
                        kind="choch",
                        direction="bearish",
                        bar_index=bar_idx,
                        level=sl_point.price,
                    ))
                    break

        return signals

    # ------------------------------------------------------------------
    # Fake BOS / Fake CHOCH validation
    # ------------------------------------------------------------------

    def is_fake_bos(
        self,
        df: pd.DataFrame,
        signal: StructureSignal,
        volume_col: str = "volume",
    ) -> bool:
        """
        Check whether a BOS signal is fake (failed breakout).

        Criteria (any one triggers fake):
          1. **Wick rejection** — more than ``fake_wick_ratio`` of the
             breakout candle's range is wick *beyond* the level.
          2. **Volume divergence** — breakout bar volume is below a 20-bar
             rolling average by ``volume_decline_pct``.
          3. **Rapid reversal** — price closes back below/above the level
             within ``fake_reversal_bars`` candles after the breakout.
        """
        idx = signal.bar_index
        level = signal.level
        direction = signal.direction

        # --- 1. Wick rejection ---
        if idx < len(df):
            row = df.iloc[idx]
            candle_range = float(row["high"]) - float(row["low"])
            if candle_range > 0:
                if direction == "bullish":
                    # Wick above close
                    wick_beyond = float(row["high"]) - float(row["close"])
                else:
                    # Wick below close
                    wick_beyond = float(row["close"]) - float(row["low"])
                wick_ratio = wick_beyond / candle_range
                if wick_ratio > self.fake_wick_ratio:
                    return True

        # --- 2. Volume divergence ---
        if volume_col in df.columns and idx >= 20:
            vol_series = df[volume_col].values
            breakout_vol = float(vol_series[idx])
            avg_vol = float(np.mean(vol_series[idx - 20: idx]))
            if avg_vol > 0 and breakout_vol < avg_vol * (1.0 - self.volume_decline_pct):
                return True

        # --- 3. Rapid reversal ---
        closes = df["close"].values
        end_check = min(idx + self.fake_reversal_bars + 1, len(df))
        for j in range(idx + 1, end_check):
            if direction == "bullish" and closes[j] < level:
                return True
            if direction == "bearish" and closes[j] > level:
                return True

        return False

    def is_fake_choch(
        self,
        df: pd.DataFrame,
        signal: StructureSignal,
        volume_col: str = "volume",
    ) -> bool:
        """
        Check whether a CHOCH signal is fake (noise in a ranging market).

        Uses the same three criteria as ``is_fake_bos()`` because the
        mechanics are identical — only the *context* differs (reversal vs
        continuation).
        """
        return self.is_fake_bos(df, signal, volume_col=volume_col)

    # ------------------------------------------------------------------
    # Multi-timeframe signal aggregation
    # ------------------------------------------------------------------

    def _analyze_single_tf(self, df: pd.DataFrame) -> MarketStructureResult:
        """Run full BOS/CHOCH analysis on a single timeframe DataFrame."""
        result = MarketStructureResult()

        if df is None or len(df) < self.swing_lookback * 3:
            return result

        # 1. Detect swing points
        swings = self.detect_swing_points(df)
        if not swings:
            return result

        # 2. Determine trend
        trend = self.determine_trend(swings)
        result.trend = trend

        # Last swing high / low
        highs = [p for p in swings if p.kind == "high"]
        lows = [p for p in swings if p.kind == "low"]
        if highs:
            result.last_swing_high = highs[-1].price
        if lows:
            result.last_swing_low = lows[-1].price

        # 3. Detect BOS
        bos_signals = self.detect_bos(df, swings, trend)
        for sig in bos_signals:
            fake = self.is_fake_bos(df, sig)
            sig.is_fake = fake
            if not fake:
                if sig.direction == "bullish":
                    result.bos_bullish = True
                else:
                    result.bos_bearish = True
            else:
                result.fake_bos = True
            result.signals.append(sig)

        # 4. Detect CHOCH
        choch_signals = self.detect_choch(df, swings, trend)
        for sig in choch_signals:
            fake = self.is_fake_choch(df, sig)
            sig.is_fake = fake
            if not fake:
                if sig.direction == "bullish":
                    result.choch_bullish = True
                else:
                    result.choch_bearish = True
            else:
                result.fake_choch = True
            result.signals.append(sig)

        return result

    def get_signals(
        self,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Main entry point: analyse up to 3 timeframes and return aggregated
        market structure signals.

        The 15-minute analysis is *primary*; 1H and 4H provide confirmation
        that raises or lowers the confidence score.

        Returns
        -------
        dict with keys:
            bos_bullish, bos_bearish, choch_bullish, choch_bearish,
            fake_bos, fake_choch, last_swing_high, last_swing_low,
            trend, confidence
        """
        # --- Primary: 15-minute ---
        r15 = self._analyze_single_tf(df_15m)

        # --- Higher timeframes (optional) ---
        r1h = self._analyze_single_tf(df_1h) if df_1h is not None else None
        r4h = self._analyze_single_tf(df_4h) if df_4h is not None else None

        # --- Aggregate ---
        result = MarketStructureResult(
            bos_bullish=r15.bos_bullish,
            bos_bearish=r15.bos_bearish,
            choch_bullish=r15.choch_bullish,
            choch_bearish=r15.choch_bearish,
            fake_bos=r15.fake_bos,
            fake_choch=r15.fake_choch,
            last_swing_high=r15.last_swing_high,
            last_swing_low=r15.last_swing_low,
            trend=r15.trend,
            confidence=0.5,          # base confidence from 15m alone
            signals=list(r15.signals),
        )

        # --- HTF confirmation boosts ---
        if r1h is not None:
            # Same-direction BOS on 1H → boost confidence
            if r1h.bos_bullish and r15.bos_bullish:
                result.confidence += 0.15
            if r1h.bos_bearish and r15.bos_bearish:
                result.confidence += 0.15
            # Same-direction CHOCH on 1H → higher confidence reversal warning
            if r1h.choch_bullish and r15.choch_bullish:
                result.confidence += 0.10
            if r1h.choch_bearish and r15.choch_bearish:
                result.confidence += 0.10
            # Conflicting signals → lower confidence
            if (r1h.bos_bullish and r15.bos_bearish) or (r1h.bos_bearish and r15.bos_bullish):
                result.confidence -= 0.10
            # Use 1H swing levels if available (more significant)
            if r1h.last_swing_high > 0:
                result.last_swing_high = r1h.last_swing_high
            if r1h.last_swing_low > 0:
                result.last_swing_low = r1h.last_swing_low

        if r4h is not None:
            # 4H is the strongest confirmation layer
            if r4h.bos_bullish and r15.bos_bullish:
                result.confidence += 0.20
            if r4h.bos_bearish and r15.bos_bearish:
                result.confidence += 0.20
            if r4h.choch_bullish and r15.choch_bullish:
                result.confidence += 0.15
            if r4h.choch_bearish and r15.choch_bearish:
                result.confidence += 0.15
            # Strong 4H contradiction → suppress 15m signal
            if (r4h.bos_bullish and r15.choch_bearish):
                result.confidence -= 0.15
            if (r4h.bos_bearish and r15.choch_bullish):
                result.confidence -= 0.15

        # Clamp confidence to [0, 1]
        result.confidence = float(np.clip(result.confidence, 0.0, 1.0))

        logger.debug(
            "MarketStructure signals: trend=%s bos_bull=%s bos_bear=%s "
            "choch_bull=%s choch_bear=%s fake_bos=%s fake_choch=%s "
            "swing_high=%.2f swing_low=%.2f conf=%.2f",
            result.trend,
            result.bos_bullish, result.bos_bearish,
            result.choch_bullish, result.choch_bearish,
            result.fake_bos, result.fake_choch,
            result.last_swing_high, result.last_swing_low,
            result.confidence,
        )

        return result.to_dict()
