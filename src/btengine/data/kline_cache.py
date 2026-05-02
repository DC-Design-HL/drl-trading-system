"""Parquet-backed kline cache, day-keyed, multi-interval.

Layout:
    data/kline_cache/{symbol}/{interval}/{YYYY-MM-DD}.parquet

Each parquet holds the bars whose `open_time` falls within that UTC day.
Intervals supported: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 1d.

Hot path: `KlineCache.get(symbol, interval, start_dt, end_dt)` returns a
single sorted DataFrame, fetching only missing days from Binance and
writing them atomically (`*.tmp` then rename). Two runs over an
overlapping window hit zero network calls on the second run.

Phase 1 deliberately keeps this simple: full days are atomic units.
Today's partial day is fetched fresh every call (never cached) so live
backtests don't go stale.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "kline_cache"

logger = logging.getLogger(__name__)


# Binance interval → seconds
_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400,
}

# Binance public klines (futures — matches what the live bot trades on)
_BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


class KlineCache:
    """Lazy day-keyed parquet cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────
    def get(self, symbol: str, interval: str,
            start: datetime, end: datetime) -> pd.DataFrame:
        """Return klines for [start, end) (UTC), sorted by open_time."""
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(f"unsupported interval {interval!r}")
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            raise ValueError(f"end {end} must be > start {start}")

        # Days to fetch
        days = self._days_in_range(start, end)
        today_utc = datetime.now(timezone.utc).date()

        frames: List[pd.DataFrame] = []
        for d in days:
            # Today's partial: never cache, always live-fetch
            if d == today_utc:
                df = self._fetch_day(symbol, interval, d, partial=True)
            else:
                df = self._read_or_fetch_day(symbol, interval, d)
            if df is not None and len(df):
                frames.append(df)
        if not frames:
            return _empty_df()
        out = pd.concat(frames, ignore_index=True)
        # Filter to exact requested window
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        out = out[(out["open_time"] >= start_ms) & (out["open_time"] < end_ms)]
        out = out.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        return out

    def warmup(self, symbol: str, interval: str,
               start: datetime, end: datetime) -> int:
        """Pre-populate cache for the window. Returns # bars now in cache."""
        df = self.get(symbol, interval, start, end)
        return len(df)

    # ── Internals ──────────────────────────────────────────────────
    def _path(self, symbol: str, interval: str, d: date) -> Path:
        return self.cache_dir / symbol / interval / f"{d.isoformat()}.parquet"

    def _read_or_fetch_day(self, symbol: str, interval: str, d: date) -> Optional[pd.DataFrame]:
        p = self._path(symbol, interval, d)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception as exc:
                logger.warning("kline_cache read failed for %s — refetching: %s", p, exc)
        df = self._fetch_day(symbol, interval, d, partial=False)
        if df is not None and len(df):
            self._write_day_atomic(p, df)
        return df

    def _write_day_atomic(self, path: Path, df: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic on POSIX

    @staticmethod
    def _days_in_range(start: datetime, end: datetime) -> List[date]:
        """Return the UTC days [d_0, ..., d_n] whose 00:00 falls in [start, end).

        End is exclusive — if `end` is exactly midnight UTC, that day is NOT
        included. Otherwise we'd silently re-fetch one extra day every call,
        which both wastes the network and breaks cache-hit invariants.
        """
        out = []
        cur = start.date()
        while True:
            day_start = datetime(cur.year, cur.month, cur.day, tzinfo=timezone.utc)
            if day_start >= end:
                break
            out.append(cur)
            cur += timedelta(days=1)
        return out

    def _fetch_day(self, symbol: str, interval: str, d: date,
                   partial: bool) -> Optional[pd.DataFrame]:
        """Fetch klines for the UTC day `d` from Binance."""
        day_start_ms = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
        day_end_ms = day_start_ms + 86400 * 1000
        bar_ms = _INTERVAL_SECONDS[interval] * 1000
        bars_needed = (day_end_ms - day_start_ms) // bar_ms + 1

        all_rows = []
        cursor = day_start_ms
        # Walk the day in 1500-bar (Binance max for futures is 1500) chunks
        while cursor < day_end_ms:
            limit = min(1500, max(1, (day_end_ms - cursor) // bar_ms + 1))
            params = (f"symbol={symbol}&interval={interval}"
                      f"&startTime={cursor}&endTime={day_end_ms - 1}"
                      f"&limit={limit}")
            url = f"{_BINANCE_KLINES_URL}?{params}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "btengine/1"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read())
            except Exception as exc:
                logger.warning("kline fetch failed (%s %s %s): %s", symbol, interval, d, exc)
                return None
            if not data:
                break
            all_rows.extend(data)
            last_open = int(data[-1][0])
            if len(data) < limit:
                break
            cursor = last_open + bar_ms
            time.sleep(0.1)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
        for c in ("open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"):
            df[c] = df[c].astype(float)
        for c in ("open_time", "close_time", "trades"):
            df[c] = df[c].astype("int64")
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
        # If today's partial, don't bother trying to filter; caller filters anyway
        if not partial:
            # Sanity: filter to the exact day to be safe (should already be)
            df = df[(df["open_time"] >= day_start_ms) & (df["open_time"] < day_end_ms)]
            df = df.reset_index(drop=True)
        return df


def _empty_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=KLINE_COLUMNS)
    for c in ("open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    for c in ("open_time", "close_time", "trades"):
        df[c] = df[c].astype("int64")
    return df
