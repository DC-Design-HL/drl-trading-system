"""Kline + funding-rate cache for the forward simulator (PROFITABILITY_PLAN.md P2).

The forward simulator needs deterministic, network-free access to OHLCV
history and funding rates. This module is the only place that talks to
exchanges; everything downstream reads parquet from ``data/kline_cache/``.

Layout::

    data/kline_cache/
      BTCUSDT_5m.parquet      # columns: ts (ns since epoch, UTC), open, high, low, close, volume
      BTCUSDT_15m.parquet
      BTCUSDT_1h.parquet
      BTCUSDT_4h.parquet
      ...
      funding_BTCUSDT.parquet # columns: ts, funding_rate

Mainnet rates are used as a proxy for testnet (testnet does not serve
historical funding) — same trade-off as the P0 ground-truth report.

Conventions:
  * timestamps are pandas Int64 nanoseconds UTC, sorted ascending, unique
  * parquet engine: pyarrow (committed dep in the repo)
  * fetch step is resumable: a re-run only fills in the gap from the last
    cached bar to ``now``
  * rate-limited via ccxt's ``enableRateLimit``; bounded retry on transient
    HTTP errors
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "kline_cache"
SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "4h")
SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

# How many bars one ccxt page returns. binanceusdm tops out at 1000.
_PAGE_LIMIT = 1000

# In nanoseconds, the bar interval for each timeframe.
_TIMEFRAME_NS = {
    "5m":  5 * 60 * 1_000_000_000,
    "15m": 15 * 60 * 1_000_000_000,
    "1h":  60 * 60 * 1_000_000_000,
    "4h":  4 * 60 * 60 * 1_000_000_000,
}


@dataclass(frozen=True)
class CachePath:
    """Resolved on-disk path for a (symbol, timeframe) cache file."""

    symbol: str
    timeframe: str
    path: Path

    @classmethod
    def for_klines(cls, symbol: str, timeframe: str,
                   *, base: Path = CACHE_DIR) -> "CachePath":
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            path=base / f"{symbol}_{timeframe}.parquet",
        )

    @classmethod
    def for_funding(cls, symbol: str,
                    *, base: Path = CACHE_DIR) -> "CachePath":
        return cls(
            symbol=symbol,
            timeframe="funding",
            path=base / f"funding_{symbol}.parquet",
        )


# ─── Read path ──────────────────────────────────────────────────────────


def read_klines(
    symbol: str,
    timeframe: str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    base: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Read OHLCV rows from the parquet cache.

    Returns an empty frame if the cache file does not exist. ``start`` /
    ``end`` are inclusive ns-since-epoch bounds; if omitted the full file
    is returned.
    """
    cp = CachePath.for_klines(symbol, timeframe, base=base)
    if not cp.path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.read_parquet(cp.path)
    if start is not None:
        df = df[df["ts"] >= start]
    if end is not None:
        df = df[df["ts"] <= end]
    return df.reset_index(drop=True)


def read_funding(
    symbol: str,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    base: Path = CACHE_DIR,
) -> pd.DataFrame:
    cp = CachePath.for_funding(symbol, base=base)
    if not cp.path.exists():
        return pd.DataFrame(columns=["ts", "funding_rate"])
    df = pd.read_parquet(cp.path)
    if start is not None:
        df = df[df["ts"] >= start]
    if end is not None:
        df = df[df["ts"] <= end]
    return df.reset_index(drop=True)


# ─── Write path ─────────────────────────────────────────────────────────


def _merge_and_write(existing: pd.DataFrame, new_rows: pd.DataFrame,
                     path: Path) -> int:
    """Append + dedup by ``ts``, sort, and write atomically.

    Returns the number of newly added rows after dedup. Uses tmp+rename so
    a concurrent reader never sees a half-written file.
    """
    if existing.empty:
        combined = new_rows
    elif new_rows.empty:
        return 0
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset="ts").sort_values("ts")
    combined = combined.reset_index(drop=True)
    added = len(combined) - len(existing)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(tmp, engine="pyarrow", compression="zstd")
    tmp.replace(path)
    return added


def _latest_ts(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    return int(df["ts"].iloc[-1])


# ─── Fetch path (online; rate-limited) ──────────────────────────────────


def _fetch_klines_page(exchange, symbol: str, timeframe: str,
                       since_ms: int, limit: int) -> list:
    """One ccxt page fetch with bounded retry on transient errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            return exchange.fetchOHLCV(symbol, timeframe,
                                       since=since_ms, limit=limit)
        except Exception as exc:  # noqa: BLE001 — ccxt error hierarchy is wide
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "fetchOHLCV %s %s @ %s failed (try %d/3): %s — sleeping %ss",
                symbol, timeframe, since_ms, attempt + 1, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"fetchOHLCV {symbol} {timeframe} failed after 3 retries: {last_exc}"
    )


def refresh_klines(
    symbol: str,
    timeframe: str,
    *,
    exchange=None,
    end_ts_ns: Optional[int] = None,
    base: Path = CACHE_DIR,
    initial_lookback_days: int = 90,
) -> int:
    """Top up the parquet cache for one (symbol, timeframe).

    If the cache exists, fetches only bars after the most recent cached
    bar. If empty, fetches the last ``initial_lookback_days`` days.
    Returns the number of bars added.
    """
    if exchange is None:
        import ccxt
        exchange = ccxt.binanceusdm({"enableRateLimit": True})

    if timeframe not in _TIMEFRAME_NS:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    tf_ns = _TIMEFRAME_NS[timeframe]
    now_ns = end_ts_ns or int(time.time() * 1_000_000_000)

    cp = CachePath.for_klines(symbol, timeframe, base=base)
    existing = read_klines(symbol, timeframe, base=base)

    latest = _latest_ts(existing)
    if latest is None:
        since_ns = now_ns - initial_lookback_days * 86_400 * 1_000_000_000
    else:
        since_ns = latest + tf_ns

    if since_ns >= now_ns:
        return 0  # already current

    pages: list[pd.DataFrame] = []
    cursor_ms = since_ns // 1_000_000
    # Bound the loop to keep one refresh under ~5 minutes even on long gaps.
    for _page in range(200):
        rows = _fetch_klines_page(exchange, symbol, timeframe,
                                  since_ms=cursor_ms, limit=_PAGE_LIMIT)
        if not rows:
            break
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df["ts"] = (df["ts_ms"].astype("int64") * 1_000_000)
        df = df[["ts", "open", "high", "low", "close", "volume"]]
        pages.append(df)
        last_ms = int(rows[-1][0])
        if len(rows) < _PAGE_LIMIT or last_ms >= now_ns // 1_000_000:
            break
        cursor_ms = last_ms + (tf_ns // 1_000_000)

    if not pages:
        return 0
    new_rows = pd.concat(pages, ignore_index=True)
    new_rows = new_rows[new_rows["ts"] <= now_ns]
    return _merge_and_write(existing, new_rows, cp.path)


def refresh_funding(
    symbol: str,
    *,
    exchange=None,
    end_ts_ns: Optional[int] = None,
    base: Path = CACHE_DIR,
    initial_lookback_days: int = 90,
) -> int:
    """Top up the funding-rate cache for one symbol. Returns rows added."""
    if exchange is None:
        import ccxt
        exchange = ccxt.binanceusdm({"enableRateLimit": True})

    now_ns = end_ts_ns or int(time.time() * 1_000_000_000)
    cp = CachePath.for_funding(symbol, base=base)
    existing = read_funding(symbol, base=base)

    latest = _latest_ts(existing)
    if latest is None:
        since_ns = now_ns - initial_lookback_days * 86_400 * 1_000_000_000
    else:
        # Funding settles every 8h; +1h cushion to avoid double-fetching.
        since_ns = latest + 60 * 60 * 1_000_000_000

    cursor_ms = since_ns // 1_000_000
    pages: list[pd.DataFrame] = []
    for _page in range(40):
        rows: list[Exception] = []
        try:
            rows = exchange.fetchFundingRateHistory(
                symbol, since=cursor_ms, limit=1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetchFundingRateHistory %s failed: %s", symbol, exc)
            break
        if not rows:
            break
        df = pd.DataFrame([
            {"ts": int(r["timestamp"]) * 1_000_000,
             "funding_rate": float(r["fundingRate"])}
            for r in rows
        ])
        pages.append(df)
        last_ms = int(rows[-1]["timestamp"])
        if len(rows) < 1000 or last_ms >= now_ns // 1_000_000:
            break
        cursor_ms = last_ms + 1

    if not pages:
        return 0
    new_rows = pd.concat(pages, ignore_index=True)
    new_rows = new_rows[new_rows["ts"] <= now_ns]
    return _merge_and_write(existing, new_rows, cp.path)


# ─── Convenience helper to convert to the DataFrame shape live_trading_htf uses ─────


def as_ohlcv_df(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert a cached OHLCV frame to a tz-aware DatetimeIndex DataFrame.

    Matches what ``_fetch_structure_candles`` produces in live_trading_htf
    so the same MarketStructure.get_signals call works.
    """
    if rows.empty:
        return rows
    df = rows.copy()
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]].sort_index()
