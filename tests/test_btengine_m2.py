"""M2 tests — replay, indicators, structure_first_v3 entry rule."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────
# Indicators — canonical implementations
# ──────────────────────────────────────────────────────────────────

def _build_kline_df(n=200, start_close=60000.0, drift=10.0, vol=50.0, seed=42):
    """Synthetic OHLCV with deterministic drift + volatility."""
    rng = np.random.default_rng(seed)
    closes = start_close + np.cumsum(drift + rng.normal(0, vol, n))
    opens = np.r_[closes[0], closes[:-1]]
    highs = np.maximum(opens, closes) + rng.uniform(0, vol, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, vol, n)
    return pd.DataFrame({
        "open_time": np.arange(n) * 900_000,  # 15m bars in ms
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.full(n, 100.0),
    })


def test_adx_wilder_runs_and_in_range():
    from src.btengine.strategy.indicators import adx_wilder
    df = _build_kline_df(n=200)
    adx = adx_wilder(df, period=14)
    assert len(adx) == 200
    # First 28 (2*period) values are NaN due to warmup
    assert adx.iloc[:14].isna().all()
    # Late values are sane
    tail = adx.dropna()
    assert len(tail) > 100
    assert (tail >= 0).all() and (tail <= 100).all()


def test_rsi_in_0_100_range():
    from src.btengine.strategy.indicators import rsi
    df = _build_kline_df(n=100)
    r = rsi(df["close"], period=14)
    tail = r.dropna()
    assert (tail >= 0).all() and (tail <= 100).all()


def test_atr_positive():
    from src.btengine.strategy.indicators import atr, atr_pct
    df = _build_kline_df(n=80)
    a = atr(df).dropna()
    ap = atr_pct(df).dropna()
    assert (a > 0).all()
    assert (ap > 0).all()
    assert (ap < 0.5).all()  # synthetic data is gentle; should be < 50% per bar


# ──────────────────────────────────────────────────────────────────
# Replay — multi-symbol bar interleaving + lookahead invariant
# ──────────────────────────────────────────────────────────────────

def test_replay_yields_ctx_in_chronological_order():
    from src.btengine.sim.replay import Replay
    btc = pd.DataFrame({"open_time": [100, 200, 300], "close": [1, 2, 3]})
    eth = pd.DataFrame({"open_time": [150, 250], "close": [9, 8]})
    rp = Replay(primary={"BTCUSDT": btc, "ETHUSDT": eth})
    seq = [(c.now_ms, c.symbol) for c in rp]
    assert seq == [(100, "BTCUSDT"), (150, "ETHUSDT"),
                   (200, "BTCUSDT"), (250, "ETHUSDT"),
                   (300, "BTCUSDT")]


def test_replay_ctx_primary_is_left_inclusive():
    """ctx.primary must contain rows up to AND INCLUDING the cursor — never future."""
    from src.btengine.sim.replay import Replay
    btc = pd.DataFrame({"open_time": [100, 200, 300, 400],
                        "close": [1.0, 2.0, 3.0, 4.0]})
    rp = Replay(primary={"BTCUSDT": btc})
    rows = []
    for ctx in rp:
        rows.append((ctx.now_ms, len(ctx.primary), float(ctx.primary["close"].iloc[-1])))
    assert rows == [(100, 1, 1.0), (200, 2, 2.0), (300, 3, 3.0), (400, 4, 4.0)]


def test_replay_htf_up_to_now_is_strict_lookback():
    """The HTF bar that *starts* exactly at now_ms must be invisible."""
    from src.btengine.sim.replay import Replay
    btc_15m = pd.DataFrame({"open_time": [3_600_000, 3_600_000 + 900_000],  # 1h, 1h+15m
                             "close": [1.0, 2.0]})
    btc_1h = pd.DataFrame({"open_time": [0, 3_600_000, 7_200_000],
                           "close": [10, 20, 30]})
    rp = Replay(primary={"BTCUSDT": btc_15m},
                htf={"BTCUSDT": {"1h": btc_1h}})
    ctxs = list(rp)
    # First 15m bar at t=3.6M (= start of 2nd 1h bar). HTF up_to_now should
    # show only the FIRST 1h bar (open_time=0), not the second (which is now).
    htf_1 = ctxs[0].htf_up_to_now("1h")
    assert list(htf_1["open_time"]) == [0], (
        f"expected only [0], got {list(htf_1['open_time'])}"
    )
    # Second 15m bar at t=4.5M; HTF should now include the 2nd 1h bar
    # (open_time=3.6M) which is in the past relative to 4.5M.
    htf_2 = ctxs[1].htf_up_to_now("1h")
    assert list(htf_2["open_time"]) == [0, 3_600_000]


# ──────────────────────────────────────────────────────────────────
# Structure-first entry rule
# ──────────────────────────────────────────────────────────────────

def test_structure_first_v3_registered():
    from src.btengine.strategy.base import get_strategy
    cls = get_strategy("structure_first_v3")
    assert cls.__name__ == "StructureFirstV3"


def test_structure_first_holds_during_warmup():
    """Until we have min_primary_bars, must HOLD (not signal on partial data)."""
    from src.btengine.strategy.library.structure_first import StructureFirstEntry
    from src.btengine.sim.context import Ctx

    e = StructureFirstEntry(min_primary_bars=30)
    df = _build_kline_df(n=10)
    ctx = Ctx(symbol="BTCUSDT", now_ms=int(df["open_time"].iloc[-1]),
              cursor_index=len(df) - 1, primary=df, htf={})
    intent = e(ctx)
    assert intent.action == "HOLD"
    assert intent.reason == "warmup"


def test_structure_first_holds_below_min_confidence():
    from src.btengine.strategy.library.structure_first import StructureFirstEntry
    from src.btengine.sim.context import Ctx

    # Synthetic flat data → no BOS/CHOCH → low confidence → HOLD
    e = StructureFirstEntry(min_primary_bars=30, min_confidence=0.45)
    n = 60
    flat = pd.DataFrame({
        "open_time": np.arange(n) * 900_000,
        "open": np.full(n, 100.0), "high": np.full(n, 100.5),
        "low": np.full(n, 99.5), "close": np.full(n, 100.0),
        "volume": np.full(n, 10.0),
    })
    ctx = Ctx(symbol="BTCUSDT", now_ms=int(flat["open_time"].iloc[-1]),
              cursor_index=len(flat) - 1, primary=flat, htf={})
    intent = e(ctx)
    # Either HOLD with low_conf reason or HOLD with structure_first action="HOLD"
    assert intent.action == "HOLD"


def test_structure_first_emits_intent_on_trending_data():
    """Strong uptrend with BOS should emit OPEN_LONG (or HOLD if confidence
    happens to land below threshold; we just want NO crash and consistent
    intent shape)."""
    from src.btengine.strategy.library.structure_first import StructureFirstEntry
    from src.btengine.sim.context import Ctx

    e = StructureFirstEntry(min_primary_bars=30, min_confidence=0.0)
    df = _build_kline_df(n=120, drift=50.0, vol=10.0, seed=7)  # strong drift
    ctx = Ctx(symbol="BTCUSDT", now_ms=int(df["open_time"].iloc[-1]),
              cursor_index=len(df) - 1, primary=df, htf={})
    intent = e(ctx)
    assert intent.action in ("HOLD", "OPEN_LONG", "OPEN_SHORT")
    # Confidence should be a float in [0,1]
    assert 0 <= intent.confidence <= 1


def test_structure_first_records_signal_in_extras():
    from src.btengine.strategy.library.structure_first import StructureFirstEntry
    from src.btengine.sim.context import Ctx

    e = StructureFirstEntry(min_primary_bars=30, min_confidence=0.0)
    df = _build_kline_df(n=80)
    ctx = Ctx(symbol="BTCUSDT", now_ms=int(df["open_time"].iloc[-1]),
              cursor_index=len(df) - 1, primary=df, htf={})
    e(ctx)
    assert "structure" in ctx.extras
    assert "confidence" in ctx.extras["structure"]
