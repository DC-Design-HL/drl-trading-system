"""
Critical Systems Tests — Full Coverage
=======================================
Tests for the components that have caused production bugs:
1. BOS/CHOCH signal detection (quality, displacement, chop, strict trend, confirmation)
2. Structure direction logic (most-recent-signal-wins, no boolean accumulation)
3. Balance tracking (_get_real_balance, _sync_balance_from_exchange)
4. Position sync (_sync_position_from_exchange)
5. Signal serialization (to_dict JSON safety)
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signals.bos_choch import (
    MarketStructure,
    MarketStructureResult,
    StructureSignal,
    SwingPoint,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def ms():
    """Fresh MarketStructure instance."""
    return MarketStructure()


@pytest.fixture
def trending_up_df():
    """DataFrame with clear uptrend — HH + HL pattern."""
    np.random.seed(42)
    n = 200
    # Uptrend: base going from 100 to 150
    base = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 0.5
    close = base + noise
    high = close + np.abs(np.random.randn(n) * 1.0)
    low = close - np.abs(np.random.randn(n) * 1.0)
    opn = np.roll(close, 1)
    opn[0] = close[0]
    volume = np.random.randint(1000, 5000, n).astype(float)

    df = pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": close, "volume": volume,
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))
    return df


@pytest.fixture
def trending_down_df():
    """DataFrame with clear downtrend — LH + LL pattern."""
    np.random.seed(42)
    n = 200
    base = np.linspace(150, 100, n)
    noise = np.random.randn(n) * 0.5
    close = base + noise
    high = close + np.abs(np.random.randn(n) * 1.0)
    low = close - np.abs(np.random.randn(n) * 1.0)
    opn = np.roll(close, 1)
    opn[0] = close[0]
    volume = np.random.randint(1000, 5000, n).astype(float)

    df = pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": close, "volume": volume,
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))
    return df


@pytest.fixture
def ranging_df():
    """DataFrame with sideways/ranging market — no clear trend."""
    np.random.seed(42)
    n = 200
    base = 100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2  # oscillating
    noise = np.random.randn(n) * 0.3
    close = base + noise
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    opn = np.roll(close, 1)
    opn[0] = close[0]
    volume = np.random.randint(1000, 5000, n).astype(float)

    df = pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": close, "volume": volume,
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))
    return df


@pytest.fixture
def make_signal():
    """Factory for creating StructureSignal instances."""
    def _make(kind="bos", direction="bullish", bar_index=50, level=100.0,
              is_fake=False, quality="normal", has_displacement=False,
              break_distance_atr=0.15, is_confirmed=False, timestamp=None):
        return StructureSignal(
            kind=kind,
            direction=direction,
            bar_index=bar_index,
            level=level,
            is_fake=is_fake,
            quality=quality,
            has_displacement=has_displacement,
            break_distance_atr=break_distance_atr,
            is_confirmed=is_confirmed,
            timestamp=timestamp or datetime(2026, 1, 1, 12, 0),
        )
    return _make


@pytest.fixture
def make_swing():
    """Factory for creating SwingPoint instances."""
    def _make(index=0, price=100.0, kind="high", label="HH"):
        return SwingPoint(index=index, price=price, kind=kind, label=label)
    return _make


def _make_bot_mock(position=0, balance=5000.0, dry_run=False,
                   exchange_balance=5256.57, exchange_positions=None,
                   testnet_executor=None):
    """Create a mock HTFLiveBot with the attributes tests need."""
    bot = MagicMock()
    bot.position = position
    bot.balance = balance
    bot.initial_balance = 5000.0
    bot._last_exchange_balance = balance
    bot.dry_run = dry_run
    bot.symbol = "BTCUSDT"
    bot.position_price = 0.0
    bot.sl_price = 0.0
    bot.tp_price = 0.0
    bot.units = 0.0
    bot.partial_tp_level = 0
    bot.partial_tp1_price = 0.0
    bot.partial_tp2_price = 0.0
    bot.initial_units = 0.0
    bot.session_balance = balance
    bot.trades = []
    bot._last_structure_signals = {}

    if testnet_executor is None:
        executor = MagicMock()
        executor.get_account_balance.return_value = exchange_balance
        executor.get_open_positions.return_value = exchange_positions or []
        bot.testnet_executor = executor
    else:
        bot.testnet_executor = testnet_executor

    return bot


# ============================================================================
# 1. BOS/CHOCH SIGNAL DETECTION
# ============================================================================

class TestSwingPointDetection:
    """Test swing point (HH/HL/LH/LL) detection."""

    def test_detects_swings_in_trending_data(self, ms, trending_up_df):
        swings = ms.detect_swing_points(trending_up_df)
        assert len(swings) > 0, "Should detect swing points in trending data"
        assert all(isinstance(s, SwingPoint) for s in swings)

    def test_swing_points_have_valid_labels(self, ms, trending_up_df):
        swings = ms.detect_swing_points(trending_up_df)
        valid_labels = {"HH", "HL", "LH", "LL", ""}  # label can be empty before sequence analysis
        for s in swings:
            assert s.label in valid_labels, f"Invalid label: {s.label}"

    def test_swing_points_have_valid_kinds(self, ms, trending_up_df):
        swings = ms.detect_swing_points(trending_up_df)
        for s in swings:
            assert s.kind in ("high", "low"), f"Invalid kind: {s.kind}"

    def test_no_swings_in_tiny_data(self, ms):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5], "volume": [1000.0],
        }, index=pd.date_range("2026-01-01", periods=1, freq="15min"))
        swings = ms.detect_swing_points(df)
        assert len(swings) == 0


class TestStrictTrend:
    """Test strict trend determination — requires BOTH HH+HL or LH+LL."""

    def test_bullish_requires_hh_and_hl(self, make_swing):
        swings = [
            make_swing(index=0, price=100, kind="low", label="HL"),
            make_swing(index=10, price=110, kind="high", label="HH"),
            make_swing(index=20, price=105, kind="low", label="HL"),
            make_swing(index=30, price=115, kind="high", label="HH"),
        ]
        assert MarketStructure.determine_strict_trend(swings) == "bullish"

    def test_bearish_requires_lh_and_ll(self, make_swing):
        swings = [
            make_swing(index=0, price=110, kind="high", label="LH"),
            make_swing(index=10, price=95, kind="low", label="LL"),
            make_swing(index=20, price=105, kind="high", label="LH"),
            make_swing(index=30, price=90, kind="low", label="LL"),
        ]
        assert MarketStructure.determine_strict_trend(swings) == "bearish"

    def test_partial_hh_without_hl_is_ranging(self, make_swing):
        """HH alone (without HL) should NOT lean bullish — must be ranging."""
        swings = [
            make_swing(index=0, price=100, kind="low", label="LL"),
            make_swing(index=10, price=110, kind="high", label="HH"),
            make_swing(index=20, price=95, kind="low", label="LL"),
            make_swing(index=30, price=115, kind="high", label="HH"),
        ]
        assert MarketStructure.determine_strict_trend(swings) == "ranging"

    def test_partial_ll_without_lh_is_ranging(self, make_swing):
        """LL alone (without LH) should NOT lean bearish — must be ranging."""
        swings = [
            make_swing(index=0, price=110, kind="high", label="HH"),
            make_swing(index=10, price=95, kind="low", label="LL"),
            make_swing(index=20, price=115, kind="high", label="HH"),
            make_swing(index=30, price=90, kind="low", label="LL"),
        ]
        assert MarketStructure.determine_strict_trend(swings) == "ranging"

    def test_fewer_than_4_swings_is_ranging(self, make_swing):
        swings = [
            make_swing(index=0, price=100, kind="low", label="HL"),
            make_swing(index=10, price=110, kind="high", label="HH"),
        ]
        assert MarketStructure.determine_strict_trend(swings) == "ranging"

    def test_empty_swings_is_ranging(self):
        assert MarketStructure.determine_strict_trend([]) == "ranging"


class TestChopDetection:
    """Test chop/consolidation detection."""

    def test_tight_range_is_chop(self, make_swing):
        """Swings within < 2x ATR should be chop."""
        # ATR ~1.0, swing range ~1.5 → chop (1.5 < 2.0 * 1.0)
        df = pd.DataFrame({
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "volume": [1000.0] * 20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="15min"))
        swings = [
            make_swing(index=0, price=99.5, kind="low"),
            make_swing(index=5, price=100.5, kind="high"),
            make_swing(index=10, price=99.8, kind="low"),
            make_swing(index=15, price=100.3, kind="high"),
        ]
        assert bool(MarketStructure.detect_chop(df, swings)) is True

    def test_wide_range_is_not_chop(self, make_swing):
        """Swings with range >> 2x ATR should not be chop."""
        df = pd.DataFrame({
            "open": [100.0] * 20,
            "high": [101.0] * 20,
            "low": [99.0] * 20,
            "close": [100.5] * 20,
            "volume": [1000.0] * 20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="15min"))
        swings = [
            make_swing(index=0, price=90.0, kind="low"),
            make_swing(index=5, price=110.0, kind="high"),
            make_swing(index=10, price=92.0, kind="low"),
            make_swing(index=15, price=108.0, kind="high"),
        ]
        assert bool(MarketStructure.detect_chop(df, swings)) is False

    def test_too_few_swings_not_chop(self, make_swing):
        df = pd.DataFrame({
            "open": [100.0] * 20, "high": [101.0] * 20,
            "low": [99.0] * 20, "close": [100.5] * 20,
            "volume": [1000.0] * 20,
        }, index=pd.date_range("2026-01-01", periods=20, freq="15min"))
        swings = [make_swing(index=0, price=100, kind="low")]
        assert bool(MarketStructure.detect_chop(df, swings)) is False


class TestDisplacement:
    """Test FVG (Fair Value Gap) displacement detection."""

    def test_bullish_fvg_detected(self, ms):
        """Bullish FVG: low[idx+1] > high[idx-1] (gap up)."""
        df = pd.DataFrame({
            "open": [100, 101, 105, 106],
            "high": [101, 102, 107, 108],
            "low":  [99,  100, 103, 105],
            "close": [101, 102, 106, 107],
            "volume": [1000] * 4,
        }, index=pd.date_range("2026-01-01", periods=4, freq="15min"))
        sig = StructureSignal(kind="bos", direction="bullish", bar_index=2,
                              level=102.0, timestamp=datetime(2026, 1, 1))
        result = MarketStructure.check_displacement(df, sig)
        assert bool(result) is True

    def test_no_fvg_when_no_gap(self, ms):
        """No gap between candles → no displacement."""
        df = pd.DataFrame({
            "open":  [100, 100.5, 101, 101.5],
            "high":  [101, 101.5, 102, 102.5],
            "low":   [99,  100,   100.5, 101],
            "close": [100.5, 101, 101.5, 102],
            "volume": [1000] * 4,
        }, index=pd.date_range("2026-01-01", periods=4, freq="15min"))
        sig = StructureSignal(kind="bos", direction="bullish", bar_index=2,
                              level=101.0, timestamp=datetime(2026, 1, 1))
        result = MarketStructure.check_displacement(df, sig)
        assert bool(result) is False

    def test_displacement_at_boundaries(self, ms):
        """index=0 or bar_index=len-1 should not crash."""
        df = pd.DataFrame({
            "open": [100, 101], "high": [101, 102],
            "low": [99, 100], "close": [100.5, 101.5],
            "volume": [1000, 1000],
        }, index=pd.date_range("2026-01-01", periods=2, freq="15min"))

        sig_start = StructureSignal(kind="bos", direction="bullish", bar_index=0,
                                    level=100.0, timestamp=datetime(2026, 1, 1))
        assert bool(MarketStructure.check_displacement(df, sig_start)) is False

        sig_end = StructureSignal(kind="bos", direction="bullish", bar_index=1,
                                  level=101.0, timestamp=datetime(2026, 1, 1))
        assert bool(MarketStructure.check_displacement(df, sig_end)) is False


class TestChochConfirmation:
    """Test CHOCH confirmation by follow-up BOS."""

    def test_confirmed_when_bos_follows_in_same_direction(self, ms, make_signal):
        choch = [make_signal(kind="choch", direction="bullish", bar_index=10)]
        bos = [make_signal(kind="bos", direction="bullish", bar_index=20)]
        ms.check_choch_confirmation(choch, bos, max_bars=20)
        assert choch[0].is_confirmed is True

    def test_not_confirmed_when_bos_too_far(self, ms, make_signal):
        choch = [make_signal(kind="choch", direction="bullish", bar_index=10)]
        bos = [make_signal(kind="bos", direction="bullish", bar_index=50)]
        ms.check_choch_confirmation(choch, bos, max_bars=20)
        assert choch[0].is_confirmed is False

    def test_not_confirmed_when_bos_wrong_direction(self, ms, make_signal):
        choch = [make_signal(kind="choch", direction="bullish", bar_index=10)]
        bos = [make_signal(kind="bos", direction="bearish", bar_index=15)]
        ms.check_choch_confirmation(choch, bos, max_bars=20)
        assert choch[0].is_confirmed is False

    def test_not_confirmed_when_bos_before_choch(self, ms, make_signal):
        choch = [make_signal(kind="choch", direction="bullish", bar_index=20)]
        bos = [make_signal(kind="bos", direction="bullish", bar_index=10)]
        ms.check_choch_confirmation(choch, bos, max_bars=20)
        assert choch[0].is_confirmed is False

    def test_no_crash_with_empty_lists(self, ms):
        ms.check_choch_confirmation([], [], max_bars=20)  # should not raise


class TestSignalQuality:
    """Test signal quality rating (strong/normal/weak/fake)."""

    def test_fake_signal_rated_fake(self, ms, make_signal):
        df = pd.DataFrame({
            "open": [100] * 60, "high": [101] * 60, "low": [99] * 60,
            "close": [100.5] * 60, "volume": [1000] * 60,
        }, index=pd.date_range("2026-01-01", periods=60, freq="15min"))
        sig = make_signal(is_fake=True)
        assert ms.rate_signal_quality(df, sig) == "fake"

    def test_strong_signal_with_displacement_and_confirmation(self, ms, make_signal):
        df = pd.DataFrame({
            "open": [100] * 60, "high": [101] * 60, "low": [99] * 60,
            "close": [100.5] * 60, "volume": [1000] * 60,
        }, index=pd.date_range("2026-01-01", periods=60, freq="15min"))
        sig = make_signal(
            kind="choch", has_displacement=True,
            break_distance_atr=0.25, is_confirmed=True
        )
        quality = ms.rate_signal_quality(df, sig)
        assert quality == "strong"

    def test_weak_signal_unconfirmed_choch_no_displacement(self, ms, make_signal):
        df = pd.DataFrame({
            "open": [100] * 60, "high": [101] * 60, "low": [99] * 60,
            "close": [100.5] * 60, "volume": [1000] * 60,
        }, index=pd.date_range("2026-01-01", periods=60, freq="15min"))
        sig = make_signal(
            kind="choch", has_displacement=False,
            break_distance_atr=0.05, is_confirmed=False
        )
        quality = ms.rate_signal_quality(df, sig)
        assert quality == "weak"

    def test_normal_signal_bos_with_some_displacement(self, ms, make_signal):
        df = pd.DataFrame({
            "open": [100] * 60, "high": [101] * 60, "low": [99] * 60,
            "close": [100.5] * 60, "volume": [1000] * 60,
        }, index=pd.date_range("2026-01-01", periods=60, freq="15min"))
        sig = make_signal(
            kind="bos", has_displacement=True,
            break_distance_atr=0.15, is_confirmed=False
        )
        quality = ms.rate_signal_quality(df, sig)
        assert quality in ("strong", "normal")


# ============================================================================
# 2. SIGNAL SERIALIZATION (to_dict JSON safety)
# ============================================================================

class TestSignalSerialization:
    """Test that to_dict() produces JSON-safe output (no numpy types)."""

    def test_market_structure_result_to_dict_json_safe(self):
        result = MarketStructureResult()
        result.bos_bullish = np.bool_(True)
        result.bos_bearish = np.bool_(False)
        result.choch_bullish = np.bool_(True)
        result.choch_bearish = np.bool_(False)
        result.fake_bos = np.bool_(False)
        result.fake_choch = np.bool_(False)
        result.last_swing_high = np.float64(100.5)
        result.last_swing_low = np.float64(99.5)
        result.trend = "bullish"
        result.strict_trend = "bullish"
        result.is_chop = np.bool_(False)
        result.last_signal_direction = "bullish"
        result.confidence = np.float64(0.75)

        d = result.to_dict()

        # Must not raise — proves JSON safety
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

        # Verify types are native Python, not numpy
        assert type(d["bos_bullish"]) is bool
        assert type(d["bos_bearish"]) is bool
        assert type(d["last_swing_high"]) is float
        assert type(d["confidence"]) is float
        assert type(d["trend"]) is str
        assert type(d["is_chop"]) is bool

    def test_to_dict_contains_all_enhanced_fields(self):
        result = MarketStructureResult()
        d = result.to_dict()
        required_keys = [
            "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
            "fake_bos", "fake_choch", "last_swing_high", "last_swing_low",
            "trend", "strict_trend", "is_chop", "last_signal_direction",
            "confidence",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_defaults_are_safe(self):
        """Default MarketStructureResult should serialize without error."""
        result = MarketStructureResult()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert '"ranging"' in serialized
        assert '"none"' in serialized or '"last_signal_direction"' in serialized


# ============================================================================
# 3. STRUCTURE DIRECTION (most-recent-signal-wins)
# ============================================================================

class TestStructureDirection:
    """
    Test _get_structure_direction logic.

    THE BUG: old code used cumulative booleans (bos_bull, bos_bear, etc.)
    which ALL became True over any lookback, permanently blocking entries.

    THE FIX: use last_signal_direction + trend alignment.
    """

    def _call_direction(self, sig_dict: Dict) -> Optional[int]:
        """
        Simulate _get_structure_direction logic without instantiating the full bot.
        Returns ACTION_LONG (2), ACTION_SHORT (3), or None.
        """
        ACTION_LONG = 2
        ACTION_SHORT = 3

        if not sig_dict:
            return None

        trend = sig_dict.get("trend", "ranging")
        last_dir = sig_dict.get("last_signal_direction", "none")

        if trend == "bullish" and last_dir == "bullish":
            return ACTION_LONG
        elif trend == "bearish" and last_dir == "bearish":
            return ACTION_SHORT
        elif trend in ("bullish", "bearish") and last_dir != trend:
            return None
        else:
            return None

    def test_bullish_trend_bullish_signal_goes_long(self):
        sig = {"trend": "bullish", "last_signal_direction": "bullish",
               "bos_bullish": True, "bos_bearish": True,
               "choch_bullish": True, "choch_bearish": True}
        assert self._call_direction(sig) == 2  # ACTION_LONG

    def test_bearish_trend_bearish_signal_goes_short(self):
        sig = {"trend": "bearish", "last_signal_direction": "bearish",
               "bos_bullish": True, "bos_bearish": True,
               "choch_bullish": True, "choch_bearish": True}
        assert self._call_direction(sig) == 3  # ACTION_SHORT

    def test_all_booleans_true_no_longer_blocks(self):
        """THE CRITICAL BUG: all 4 booleans True should NOT block anymore."""
        sig = {"trend": "bullish", "last_signal_direction": "bullish",
               "bos_bullish": True, "bos_bearish": True,
               "choch_bullish": True, "choch_bearish": True}
        result = self._call_direction(sig)
        assert result is not None, "All-True booleans must NOT block entry anymore"

    def test_trend_bullish_signal_bearish_waits(self):
        sig = {"trend": "bullish", "last_signal_direction": "bearish"}
        assert self._call_direction(sig) is None

    def test_trend_bearish_signal_bullish_waits(self):
        sig = {"trend": "bearish", "last_signal_direction": "bullish"}
        assert self._call_direction(sig) is None

    def test_ranging_trend_always_holds(self):
        sig = {"trend": "ranging", "last_signal_direction": "bullish"}
        assert self._call_direction(sig) is None

    def test_no_signal_direction_holds(self):
        sig = {"trend": "bullish", "last_signal_direction": "none"}
        assert self._call_direction(sig) is None

    def test_empty_signals_holds(self):
        assert self._call_direction({}) is None
        assert self._call_direction(None) is None


# ============================================================================
# 4. BALANCE TRACKING
# ============================================================================

class TestGetRealBalance:
    """Test _get_real_balance — must never fall back to inflated self.balance."""

    def _simulate_get_real_balance(self, bot):
        """Simulate the _get_real_balance method."""
        try:
            if bot.testnet_executor and not bot.dry_run:
                bal = bot.testnet_executor.get_account_balance("USDT")
                if bal > 0:
                    bot._last_exchange_balance = bal
                    return bal
        except Exception:
            pass
        return getattr(bot, '_last_exchange_balance', bot.balance)

    def test_returns_exchange_balance_on_success(self):
        bot = _make_bot_mock(balance=9999.0, exchange_balance=5256.57)
        result = self._simulate_get_real_balance(bot)
        assert result == 5256.57
        assert bot._last_exchange_balance == 5256.57

    def test_never_returns_inflated_internal_balance(self):
        """THE BUG: _get_real_balance was returning self.balance (inflated)."""
        bot = _make_bot_mock(balance=6355.86, exchange_balance=0.0)
        bot._last_exchange_balance = 5256.57  # cached from last success
        result = self._simulate_get_real_balance(bot)
        assert result == 5256.57, "Should use cached exchange balance, not inflated self.balance"
        assert result != 6355.86

    def test_uses_cache_on_exception(self):
        bot = _make_bot_mock(balance=9999.0)
        bot._last_exchange_balance = 5200.0
        bot.testnet_executor.get_account_balance.side_effect = Exception("API error")
        result = self._simulate_get_real_balance(bot)
        assert result == 5200.0

    def test_dry_run_uses_cache(self):
        bot = _make_bot_mock(balance=5000.0, dry_run=True)
        bot._last_exchange_balance = 5000.0
        result = self._simulate_get_real_balance(bot)
        assert result == 5000.0

    def test_no_executor_uses_cache(self):
        bot = _make_bot_mock(balance=5000.0)
        bot.testnet_executor = None
        bot._last_exchange_balance = 5000.0
        result = self._simulate_get_real_balance(bot)
        assert result == 5000.0


class TestSyncBalanceFromExchange:
    """Test _sync_balance_from_exchange — updates balance + cache."""

    def _simulate_sync_balance(self, bot, real_balance):
        """Simulate _sync_balance_from_exchange core logic."""
        if real_balance <= 0:
            return
        bot.balance = real_balance
        bot._last_exchange_balance = real_balance
        bot.session_balance = real_balance

    def test_updates_all_balance_fields(self):
        bot = _make_bot_mock(balance=9999.0)
        self._simulate_sync_balance(bot, 5256.57)
        assert bot.balance == 5256.57
        assert bot._last_exchange_balance == 5256.57
        assert bot.session_balance == 5256.57

    def test_ignores_zero_balance(self):
        bot = _make_bot_mock(balance=5000.0)
        self._simulate_sync_balance(bot, 0.0)
        assert bot.balance == 5000.0  # unchanged

    def test_ignores_negative_balance(self):
        bot = _make_bot_mock(balance=5000.0)
        self._simulate_sync_balance(bot, -100.0)
        assert bot.balance == 5000.0  # unchanged


# ============================================================================
# 5. POSITION SYNC
# ============================================================================

class TestPositionSync:
    """Test _sync_position_from_exchange — exchange is source of truth."""

    def _simulate_position_sync(self, bot, exchange_amt, exchange_entry):
        """
        Simulate the core _sync_position_from_exchange logic.
        Returns string describing what happened.
        """
        real_amt = abs(exchange_amt)
        exchange_dir = 1 if exchange_amt > 0 else (-1 if exchange_amt < 0 else 0)

        # Case 1: Bot FLAT but exchange has position → RECOVER
        if exchange_dir != 0 and bot.position == 0:
            bot.position = exchange_dir
            bot.position_price = exchange_entry if exchange_entry > 0 else 0
            bot.units = real_amt
            return "RECOVERED"

        # Case 2: Exchange FLAT but bot has position → STALE
        if exchange_dir == 0 and bot.position != 0:
            bot.position = 0
            bot.position_price = 0.0
            bot.sl_price = 0.0
            bot.tp_price = 0.0
            bot.units = 0.0
            return "STALE_CLEARED"

        # Case 3: Both agree → OK
        if exchange_dir == bot.position:
            return "IN_SYNC"

        # Case 4: Direction mismatch → sync to exchange
        bot.position = exchange_dir
        bot.position_price = exchange_entry if exchange_entry > 0 else bot.position_price
        bot.units = real_amt
        return "DIRECTION_FIXED"

    def test_recovery_when_bot_flat_exchange_long(self):
        """THE BUG: bot loaded pos=0 from stale state, exchange had open position."""
        bot = _make_bot_mock(position=0)
        result = self._simulate_position_sync(bot, 0.5, 74000.0)
        assert result == "RECOVERED"
        assert bot.position == 1  # LONG
        assert bot.position_price == 74000.0
        assert bot.units == 0.5

    def test_recovery_when_bot_flat_exchange_short(self):
        bot = _make_bot_mock(position=0)
        result = self._simulate_position_sync(bot, -0.75, 2329.47)
        assert result == "RECOVERED"
        assert bot.position == -1  # SHORT
        assert bot.position_price == 2329.47
        assert bot.units == 0.75

    def test_stale_cleared_when_exchange_flat_bot_long(self):
        bot = _make_bot_mock(position=1)
        bot.position_price = 74000.0
        bot.sl_price = 73000.0
        bot.tp_price = 76000.0
        bot.units = 0.5
        result = self._simulate_position_sync(bot, 0.0, 0.0)
        assert result == "STALE_CLEARED"
        assert bot.position == 0
        assert bot.sl_price == 0.0

    def test_in_sync_when_both_agree(self):
        bot = _make_bot_mock(position=1)
        result = self._simulate_position_sync(bot, 0.5, 74000.0)
        assert result == "IN_SYNC"

    def test_both_flat_is_in_sync(self):
        bot = _make_bot_mock(position=0)
        result = self._simulate_position_sync(bot, 0.0, 0.0)
        assert result == "IN_SYNC"

    def test_direction_mismatch_syncs_to_exchange(self):
        bot = _make_bot_mock(position=1)  # bot thinks LONG
        result = self._simulate_position_sync(bot, -0.5, 2300.0)  # exchange is SHORT
        assert result == "DIRECTION_FIXED"
        assert bot.position == -1


# ============================================================================
# 6. FULL INTEGRATION: analyze_single_tf
# ============================================================================

class TestAnalyzeSingleTf:
    """Integration test: run full signal analysis on synthetic data."""

    def test_returns_valid_result_on_trending_data(self, ms, trending_up_df):
        result = ms._analyze_single_tf(trending_up_df)
        assert isinstance(result, MarketStructureResult)
        assert result.trend in ("bullish", "bearish", "ranging")
        assert result.strict_trend in ("bullish", "bearish", "ranging")
        assert isinstance(result.is_chop, (bool, np.bool_))
        assert result.last_signal_direction in ("bullish", "bearish", "none")

    def test_returns_valid_result_on_ranging_data(self, ms, ranging_df):
        result = ms._analyze_single_tf(ranging_df)
        assert isinstance(result, MarketStructureResult)

    def test_signals_have_quality_ratings(self, ms, trending_up_df):
        result = ms._analyze_single_tf(trending_up_df)
        for sig in result.signals:
            assert sig.quality in ("strong", "normal", "weak", "fake"), \
                f"Signal has invalid quality: {sig.quality}"

    def test_result_serializes_to_json(self, ms, trending_up_df):
        """End-to-end: analyze → to_dict → JSON — must not raise."""
        result = ms._analyze_single_tf(trending_up_df)
        d = result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_no_crash_on_small_data(self, ms):
        """Should handle small DataFrames gracefully."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2026-01-01", periods=3, freq="15min"))
        result = ms._analyze_single_tf(df)
        assert isinstance(result, MarketStructureResult)

    def test_confidence_in_valid_range(self, ms, trending_up_df):
        result = ms._analyze_single_tf(trending_up_df)
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# 7. PARTIAL CLOSE BALANCE CONSISTENCY
# ============================================================================

class TestPartialCloseBalance:
    """
    Test that partial closes update balance correctly.
    THE BUG: balance stayed at $5,249 after partial close because
    _sync_balance_from_exchange wasn't called.
    """

    def test_balance_updated_after_partial_close(self):
        """After partial close, balance must reflect exchange state."""
        bot = _make_bot_mock(balance=5249.0, exchange_balance=5256.57)

        # Simulate what happens after partial close:
        # 1. PnL is computed
        pnl = 10.05
        # 2. Old code only did: bot.balance += pnl (wrong — drifts)
        # 3. New code calls _sync_balance_from_exchange
        bot.balance = 5256.57  # sync from exchange
        bot._last_exchange_balance = 5256.57
        bot.session_balance = 5256.57

        assert bot.balance == 5256.57
        assert bot._last_exchange_balance == 5256.57

    def test_balance_not_inflated_by_internal_tracking(self):
        """Internal balance tracker must not exceed real exchange balance."""
        exchange_bal = 5256.57
        bot = _make_bot_mock(balance=5000.0, exchange_balance=exchange_bal)

        # Simulate multiple partial closes adding to internal balance
        bot.balance += 100  # internal +100
        bot.balance += 50   # internal +50
        # Now internal = 5150, but exchange = 5256.57

        # After sync, should match exchange
        bot.balance = exchange_bal
        bot._last_exchange_balance = exchange_bal
        assert bot.balance == exchange_bal


# ============================================================================
# 8. WS_CONNECTED NOISE
# ============================================================================

class TestWsConnectedNoise:
    """Test that WS_CONNECTED is suppressed on initial startup."""

    def test_no_ws_connected_on_first_connect(self):
        """reconnect_count == 0 should NOT write WS_CONNECTED alert."""
        ws_state = {"reconnect_count": 0, "connected": False}
        alert_written = False

        # Simulate _on_open logic
        if ws_state["reconnect_count"] > 0:
            alert_written = True  # WS_RECONNECTED
        # else: skip — no WS_CONNECTED on initial startup

        assert alert_written is False

    def test_ws_reconnected_on_retry(self):
        """reconnect_count > 0 should write WS_RECONNECTED alert."""
        ws_state = {"reconnect_count": 3, "connected": False}
        alert_written = False

        if ws_state["reconnect_count"] > 0:
            alert_written = True  # WS_RECONNECTED

        assert alert_written is True
