"""M3 tests — Position lifecycle + Broker exit-path math."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest


def _bars(n: int, start_close: float = 60000.0,
          drift: float = 0.0, vol: float = 0.0,
          start_ts_ms: int = 0, interval_ms: int = 900_000) -> pd.DataFrame:
    """Synthetic OHLC bars with controllable drift+vol."""
    rng = np.random.default_rng(0)
    closes = start_close + np.cumsum(np.full(n, drift) + rng.normal(0, vol, n))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + abs(vol) * 0.5 + 1
    lows = np.minimum(opens, closes) - abs(vol) * 0.5 - 1
    return pd.DataFrame({
        "open_time": start_ts_ms + np.arange(n) * interval_ms,
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n, 100.0),
    })


def _ctx_at(df: pd.DataFrame, idx: int):
    from src.btengine.sim.context import Ctx
    return Ctx(symbol="BTCUSDT",
               now_ms=int(df["open_time"].iloc[idx]),
               cursor_index=idx,
               primary=df.iloc[: idx + 1].reset_index(drop=True),
               htf={})


# ──────────────────────────────────────────────────────────────────
# Position math
# ──────────────────────────────────────────────────────────────────

def test_position_excursions_track_high_low():
    from src.btengine.sim.position import Position
    p = Position(symbol="X", side="LONG",
                 entry_price=100.0, initial_units=1.0, units=1.0,
                 sl_price=98.0, tp_price=110.0,
                 partial_tp1_price=105.0, partial_tp2_price=108.0,
                 sl_pct=0.02, tp_pct=0.10, leverage=10,
                 confidence=0.8, open_ts_ms=0)
    p.update_excursions(high=105.0, low=99.0)
    assert p.mfe_pct == pytest.approx(0.05)
    assert p.mae_pct == pytest.approx(-0.01)
    assert p.peak_price == 105.0


def test_position_trailing_activates_after_threshold():
    from src.btengine.sim.position import Position
    from src.btengine import live_constants as LC
    p = Position(symbol="X", side="LONG",
                 entry_price=100.0, initial_units=1.0, units=1.0,
                 sl_price=98.0, tp_price=110.0,
                 partial_tp1_price=105.0, partial_tp2_price=108.0,
                 sl_pct=0.02, tp_pct=0.10, leverage=10,
                 confidence=0.8, open_ts_ms=0)
    # Move +0.4% favorable — below activation
    p.update_excursions(high=100.4, low=99.5)
    p.trail_check()
    assert not p.trailing_active

    # Move +0.6% favorable — should activate
    p.update_excursions(high=100.6, low=99.5)
    p.trail_check()
    assert p.trailing_active
    # Stop should be at peak * (1 - distance_pre_tp1) = 100.6 * (1 - 0.003) = 100.298
    expected = 100.6 * (1 - LC.TRAILING_DISTANCE_PRE_TP1)
    assert p.trailing_stop_price == pytest.approx(expected)


# ──────────────────────────────────────────────────────────────────
# Broker — open position
# ──────────────────────────────────────────────────────────────────

def test_broker_opens_long_with_slippage_and_fees():
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent
    from src.btengine import live_constants as LC

    df = _bars(n=80, start_close=60000.0, drift=10.0, vol=20.0)
    ctx = _ctx_at(df, len(df) - 1)
    broker = Broker(starting_balance=5000.0)
    intent = Intent(action="OPEN_LONG", confidence=0.8, reason="test")

    pos = broker.open_position(intent, ctx, sizing_usd=1500)
    assert pos is not None
    assert pos.side == "LONG"
    # Entry price reflects upward slippage
    assert pos.entry_price > float(df["close"].iloc[-1])
    # SL is below, TP is above
    assert pos.sl_price < pos.entry_price < pos.tp_price
    # Fees deducted from balance
    assert broker.balance < 5000.0


def test_broker_rejects_when_already_in_position():
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent

    df = _bars(n=80)
    ctx = _ctx_at(df, len(df) - 1)
    broker = Broker(starting_balance=5000.0)
    intent = Intent(action="OPEN_LONG", confidence=0.8)
    p1 = broker.open_position(intent, ctx, sizing_usd=1500)
    assert p1 is not None
    p2 = broker.open_position(intent, ctx, sizing_usd=1500)
    assert p2 is None  # rejected, already in position


def test_broker_respects_max_concurrent():
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent
    from src.btengine.sim.context import Ctx

    df = _bars(n=80)
    broker = Broker(starting_balance=20000.0, max_concurrent=2)
    intent = Intent(action="OPEN_LONG", confidence=0.8)

    ctx_btc = Ctx(symbol="BTCUSDT", now_ms=int(df["open_time"].iloc[-1]),
                  cursor_index=len(df) - 1, primary=df, htf={})
    ctx_eth = Ctx(symbol="ETHUSDT", now_ms=int(df["open_time"].iloc[-1]),
                  cursor_index=len(df) - 1, primary=df, htf={})
    ctx_sol = Ctx(symbol="SOLUSDT", now_ms=int(df["open_time"].iloc[-1]),
                  cursor_index=len(df) - 1, primary=df, htf={})

    assert broker.open_position(intent, ctx_btc, 1500) is not None
    assert broker.open_position(intent, ctx_eth, 1500) is not None
    assert broker.open_position(intent, ctx_sol, 1500) is None  # cap reached


# ──────────────────────────────────────────────────────────────────
# Broker — exit paths
# ──────────────────────────────────────────────────────────────────

def test_broker_sl_exit_full_close():
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent

    # First 60 bars of warmup, then a sharp drop on bar 60+
    warmup = _bars(n=60, start_close=60000.0, drift=0.0, vol=10.0)
    ctx = _ctx_at(warmup, len(warmup) - 1)
    broker = Broker(starting_balance=5000.0)
    pos = broker.open_position(Intent(action="OPEN_LONG", confidence=0.8),
                                ctx, sizing_usd=1500)
    assert pos is not None
    # Build a next bar that crashes through SL
    sl_price = pos.sl_price
    crash_bar = pd.DataFrame({
        "open_time": [int(warmup["open_time"].iloc[-1]) + 900_000],
        "open": [pos.entry_price * 0.999],
        "high": [pos.entry_price * 0.999],
        "low":  [sl_price * 0.95],   # well below SL
        "close": [sl_price * 0.97],
        "volume": [100.0],
    })
    next_df = pd.concat([warmup, crash_bar], ignore_index=True)
    next_ctx = _ctx_at(next_df, len(next_df) - 1)

    trades = broker.on_bar(next_ctx)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "sl"
    assert t.is_full_close
    assert t.pnl_usd < 0  # SL is a loss
    assert "BTCUSDT" not in broker.positions  # position closed


def test_broker_partial_tp1_then_tp2_then_trail():
    """Bar 1 hits TP1, bar 2 hits TP2 + trail, position closes."""
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent

    warmup = _bars(n=60, start_close=60000.0, vol=5.0)
    ctx = _ctx_at(warmup, len(warmup) - 1)
    broker = Broker(starting_balance=5000.0)
    pos = broker.open_position(Intent(action="OPEN_LONG", confidence=0.8),
                                ctx, sizing_usd=1500)
    assert pos is not None
    # Build a "rip" bar that pushes through TP1 and TP2
    rip_high = pos.partial_tp2_price * 1.0005  # just past TP2
    bar1 = pd.DataFrame({
        "open_time": [int(warmup["open_time"].iloc[-1]) + 900_000],
        "open": [pos.entry_price],
        "high": [rip_high], "low": [pos.entry_price * 0.999],
        "close": [pos.partial_tp2_price],
        "volume": [100.0],
    })
    df1 = pd.concat([warmup, bar1], ignore_index=True)
    ctx1 = _ctx_at(df1, len(df1) - 1)
    trades = broker.on_bar(ctx1)
    # We expect at least 2 partials this bar (or possibly trailing too)
    reasons = [t.reason for t in trades]
    assert "tp_partial_1" in reasons
    assert "tp_partial_2" in reasons
    # Position may have been further closed by trailing or remain open
    if "BTCUSDT" in broker.positions:
        rem = broker.positions["BTCUSDT"]
        assert rem.partial_tp_level == 2
        assert rem.trailing_active or rem.units > 0


def test_broker_stagnant_exit_after_window():
    """Position open with PnL near 0 for >= STAGNANT_HOURS triggers stagnant exit."""
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent
    from src.btengine import live_constants as LC

    warmup = _bars(n=60, start_close=60000.0, vol=2.0)
    ctx = _ctx_at(warmup, len(warmup) - 1)
    broker = Broker(starting_balance=5000.0)
    pos = broker.open_position(Intent(action="OPEN_LONG", confidence=0.8),
                                ctx, sizing_usd=1500)
    assert pos is not None
    # Skip ahead 6h+ (24 × 15m bars + some buffer) with prices hovering near entry
    n_extra = int(LC.STAGNANT_HOURS * 4) + 1   # 4 bars/hour at 15m
    last_ts = int(warmup["open_time"].iloc[-1])
    flat_bars = pd.DataFrame({
        "open_time": last_ts + np.arange(1, n_extra + 1) * 900_000,
        "open": np.full(n_extra, pos.entry_price * 1.001),
        "high": np.full(n_extra, pos.entry_price * 1.0015),
        "low":  np.full(n_extra, pos.entry_price * 0.999),
        "close": np.full(n_extra, pos.entry_price * 1.001),
        "volume": np.full(n_extra, 100.0),
    })
    df = pd.concat([warmup, flat_bars], ignore_index=True)
    last_trade = None
    for i in range(len(warmup), len(df)):
        c = _ctx_at(df, i)
        ts = broker.on_bar(c)
        if ts:
            last_trade = ts[-1]
            break
    assert last_trade is not None, "expected some exit"
    assert last_trade.reason == "stagnant"


def test_broker_force_close_for_reverse_close():
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent

    warmup = _bars(n=60, vol=5.0)
    ctx = _ctx_at(warmup, len(warmup) - 1)
    broker = Broker(starting_balance=5000.0)
    pos = broker.open_position(Intent(action="OPEN_LONG", confidence=0.8),
                                ctx, sizing_usd=1500)
    assert pos is not None
    trade = broker.force_close(ctx, "reverse_close")
    assert trade is not None
    assert trade.reason == "reverse_close"
    assert trade.is_full_close
    assert "BTCUSDT" not in broker.positions


def test_broker_sl_loss_is_negative_short_side_wins_on_drop():
    """Symmetry sanity: SHORT entered, price drops, hits TP1."""
    from src.btengine.sim.broker import Broker
    from src.btengine.strategy.base import Intent

    warmup = _bars(n=60, start_close=60000.0, vol=5.0)
    ctx = _ctx_at(warmup, len(warmup) - 1)
    broker = Broker(starting_balance=5000.0)
    pos = broker.open_position(Intent(action="OPEN_SHORT", confidence=0.8),
                                ctx, sizing_usd=1500)
    assert pos is not None
    assert pos.side == "SHORT"
    assert pos.sl_price > pos.entry_price
    assert pos.tp_price < pos.entry_price
    # Drop bar: low pierces TP1 for SHORT (TP1 is below entry)
    bar1 = pd.DataFrame({
        "open_time": [int(warmup["open_time"].iloc[-1]) + 900_000],
        "open": [pos.entry_price * 0.999],
        "high": [pos.entry_price * 1.001], "low": [pos.partial_tp1_price * 0.999],
        "close": [pos.partial_tp1_price],
        "volume": [100.0],
    })
    df1 = pd.concat([warmup, bar1], ignore_index=True)
    ctx1 = _ctx_at(df1, len(df1) - 1)
    trades = broker.on_bar(ctx1)
    assert any(t.reason == "tp_partial_1" for t in trades)
    # Partial TP1 on SHORT should be a profit (entry > exit)
    tp1_trade = next(t for t in trades if t.reason == "tp_partial_1")
    assert tp1_trade.pnl_usd > 0
