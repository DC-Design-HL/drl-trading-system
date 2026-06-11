#!/usr/bin/env python3
"""Refresh data/kline_cache/ for the forward simulator (PROFITABILITY_PLAN.md P2).

Resumable: each (symbol, timeframe) is fetched only from the last cached
bar onwards. Safe to cron daily.

Usage:
    python3 -m scripts.self_improve.refresh_kline_cache
    python3 -m scripts.self_improve.refresh_kline_cache --symbol BTCUSDT
    python3 -m scripts.self_improve.refresh_kline_cache --no-funding
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.kline_cache import (  # noqa: E402
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    refresh_funding,
    refresh_klines,
)

logger = logging.getLogger("refresh_kline_cache")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--symbol", "-s",
        action="append",
        choices=SUPPORTED_SYMBOLS,
        help="Restrict to these symbols (repeatable). Default: all.",
    )
    ap.add_argument(
        "--timeframe", "-t",
        action="append",
        choices=SUPPORTED_TIMEFRAMES,
        help="Restrict to these timeframes (repeatable). Default: all.",
    )
    ap.add_argument(
        "--no-funding",
        action="store_true",
        help="Skip funding-rate refresh.",
    )
    ap.add_argument(
        "--initial-lookback-days",
        type=int,
        default=90,
        help="Initial backfill horizon when a cache file does not exist.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only log warnings.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )

    symbols = tuple(args.symbol) if args.symbol else SUPPORTED_SYMBOLS
    timeframes = tuple(args.timeframe) if args.timeframe else SUPPORTED_TIMEFRAMES

    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})

    total_klines_added = 0
    total_funding_added = 0
    started = time.perf_counter()

    for symbol in symbols:
        for tf in timeframes:
            try:
                added = refresh_klines(
                    symbol, tf,
                    exchange=ex,
                    initial_lookback_days=args.initial_lookback_days,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("refresh klines %s %s failed: %s", symbol, tf, exc)
                continue
            total_klines_added += added
            logger.info("klines %s %s: +%d bars", symbol, tf, added)
        if not args.no_funding:
            try:
                added = refresh_funding(
                    symbol,
                    exchange=ex,
                    initial_lookback_days=args.initial_lookback_days,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("refresh funding %s failed: %s", symbol, exc)
                continue
            total_funding_added += added
            logger.info("funding %s: +%d rows", symbol, added)

    elapsed = time.perf_counter() - started
    logger.info(
        "done in %.1fs: +%d kline bars, +%d funding rows",
        elapsed, total_klines_added, total_funding_added,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
