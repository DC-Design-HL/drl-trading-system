"""Kline + funding cache (PROFITABILITY_PLAN.md P2.A).

Network-free: a stub exchange replays fixture rows so the resumable
refresh logic can be tested without hitting Binance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.self_improve import kline_cache as kc


# A 5m bar interval in ms (matches kline_cache._TIMEFRAME_NS["5m"]/1e6).
_5M_MS = 5 * 60 * 1000


class _StubExchange:
    """ccxt-shaped stub: returns pre-canned OHLCV / funding pages."""

    def __init__(self, klines: list[list], funding: list[dict] | None = None):
        self._klines = klines
        self._funding = funding or []
        self.kline_calls: list[tuple[str, str, int, int]] = []
        self.funding_calls: list[tuple[str, int, int]] = []

    def fetchOHLCV(self, symbol, timeframe, since=None, limit=None):
        self.kline_calls.append((symbol, timeframe, int(since or 0), int(limit or 0)))
        out = [r for r in self._klines if r[0] >= (since or 0)]
        return out[: (limit or len(out))]

    def fetchFundingRateHistory(self, symbol, since=None, limit=None):
        self.funding_calls.append((symbol, int(since or 0), int(limit or 0)))
        out = [r for r in self._funding if r["timestamp"] >= (since or 0)]
        return out[: (limit or len(out))]


def _bar(ts_ms: int, base: float = 100.0) -> list:
    return [ts_ms, base, base + 1, base - 1, base + 0.5, 1.0]


def test_first_refresh_writes_parquet(tmp_path: Path) -> None:
    rows = [_bar(1_700_000_000_000 + i * _5M_MS) for i in range(5)]
    ex = _StubExchange(klines=rows)
    end_ns = (rows[-1][0] + _5M_MS) * 1_000_000

    added = kc.refresh_klines(
        "BTCUSDT", "5m",
        exchange=ex,
        end_ts_ns=end_ns,
        base=tmp_path,
        initial_lookback_days=1,
    )
    assert added == 5
    df = kc.read_klines("BTCUSDT", "5m", base=tmp_path)
    assert len(df) == 5
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["ts"].is_monotonic_increasing


def test_refresh_is_resumable(tmp_path: Path) -> None:
    """A second refresh only fetches bars after the latest cached."""
    first_rows = [_bar(1_700_000_000_000 + i * _5M_MS) for i in range(5)]
    ex1 = _StubExchange(klines=first_rows)
    end_ns_1 = (first_rows[-1][0] + _5M_MS) * 1_000_000
    kc.refresh_klines(
        "BTCUSDT", "5m",
        exchange=ex1, end_ts_ns=end_ns_1,
        base=tmp_path, initial_lookback_days=1,
    )

    # Two more bars become available.
    all_rows = first_rows + [
        _bar(first_rows[-1][0] + _5M_MS),
        _bar(first_rows[-1][0] + 2 * _5M_MS),
    ]
    ex2 = _StubExchange(klines=all_rows)
    end_ns_2 = (all_rows[-1][0] + _5M_MS) * 1_000_000
    added = kc.refresh_klines(
        "BTCUSDT", "5m",
        exchange=ex2, end_ts_ns=end_ns_2,
        base=tmp_path, initial_lookback_days=1,
    )
    assert added == 2
    # The second exchange should have been asked only for bars AFTER the
    # first batch — never the original 5.
    earliest_since = min(c[2] for c in ex2.kline_calls)
    assert earliest_since > first_rows[-1][0]


def test_dedup_overlap(tmp_path: Path) -> None:
    """Repeated rows for the same ts are merged, not duplicated."""
    rows = [_bar(1_700_000_000_000 + i * _5M_MS) for i in range(3)]
    ex = _StubExchange(klines=rows)
    end_ns = (rows[-1][0] + _5M_MS) * 1_000_000
    kc.refresh_klines("BTCUSDT", "5m", exchange=ex, end_ts_ns=end_ns,
                      base=tmp_path, initial_lookback_days=1)

    # Re-run with the same rows — nothing should be added.
    ex2 = _StubExchange(klines=rows)
    added = kc.refresh_klines("BTCUSDT", "5m", exchange=ex2, end_ts_ns=end_ns,
                              base=tmp_path, initial_lookback_days=1)
    assert added == 0


def test_funding_refresh_writes_parquet(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 1_700_000_000_000 + i * 8 * 3600 * 1000,
         "fundingRate": 0.0001 * (i + 1)}
        for i in range(3)
    ]
    ex = _StubExchange(klines=[], funding=rows)
    end_ns = (rows[-1]["timestamp"] + 8 * 3600 * 1000) * 1_000_000
    added = kc.refresh_funding("BTCUSDT", exchange=ex,
                               end_ts_ns=end_ns, base=tmp_path,
                               initial_lookback_days=1)
    assert added == 3
    df = kc.read_funding("BTCUSDT", base=tmp_path)
    assert len(df) == 3
    assert (df["funding_rate"] > 0).all()


def test_read_range_filters(tmp_path: Path) -> None:
    rows = [_bar(1_700_000_000_000 + i * _5M_MS) for i in range(10)]
    ex = _StubExchange(klines=rows)
    end_ns = (rows[-1][0] + _5M_MS) * 1_000_000
    kc.refresh_klines("BTCUSDT", "5m", exchange=ex, end_ts_ns=end_ns,
                      base=tmp_path, initial_lookback_days=1)

    start = (rows[2][0]) * 1_000_000
    end = (rows[6][0]) * 1_000_000
    df = kc.read_klines("BTCUSDT", "5m", start=start, end=end, base=tmp_path)
    assert len(df) == 5
    assert int(df["ts"].iloc[0]) == start
    assert int(df["ts"].iloc[-1]) == end


def test_as_ohlcv_df_has_datetimeindex(tmp_path: Path) -> None:
    """Sim needs a tz-aware DatetimeIndex frame so MarketStructure works."""
    rows = [_bar(1_700_000_000_000 + i * _5M_MS) for i in range(4)]
    ex = _StubExchange(klines=rows)
    end_ns = (rows[-1][0] + _5M_MS) * 1_000_000
    kc.refresh_klines("BTCUSDT", "5m", exchange=ex, end_ts_ns=end_ns,
                      base=tmp_path, initial_lookback_days=1)

    cached = kc.read_klines("BTCUSDT", "5m", base=tmp_path)
    df = kc.as_ohlcv_df(cached)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_read_missing_cache_returns_empty(tmp_path: Path) -> None:
    df = kc.read_klines("BTCUSDT", "5m", base=tmp_path)
    assert df.empty
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]


def test_rejects_unknown_timeframe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported timeframe"):
        kc.refresh_klines("BTCUSDT", "7m",
                          exchange=_StubExchange(klines=[]),
                          end_ts_ns=0, base=tmp_path)
