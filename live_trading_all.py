#!/usr/bin/env python3
"""
live_trading_all.py — Single-process multi-symbol HTF trading bot.

Runs BTC, ETH, SOL, XRP bots as threads inside one Python process.
RAM savings vs 4 separate processes: ~1.2 GB (shared Python runtime,
shared libraries, shared PyTorch, shared feature engine imports).

Each symbol runs in its own thread with independent error isolation:
if one symbol's thread crashes, the watchdog restarts it automatically
while the other 3 continue uninterrupted.

Usage:
    python live_trading_all.py                        # dry-run, 15m interval
    python live_trading_all.py --live                 # real execution
    python live_trading_all.py --live --interval 15   # custom interval
    python live_trading_all.py --live --symbols BTCUSDT ETHUSDT  # subset
"""

import sys
import os
import time
import signal
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# Limit PyTorch thread pools — all bots in one process, share 2 CPUs
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from live_trading_htf import HTFLiveBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("trading_all")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# Watchdog restart delay — don't spin-restart a crashing symbol immediately
RESTART_DELAY_SECONDS = 30
# Max restarts per symbol before giving up (0 = unlimited)
MAX_RESTARTS = 10


def _run_symbol(
    symbol: str,
    dry_run: bool,
    initial_balance: float,
    interval_minutes: int,
    stop_event: threading.Event,
):
    """Entry point for each symbol thread. Runs the bot loop."""
    log = logging.getLogger(f"bot.{symbol}")
    log.info("Starting HTFLiveBot for %s (dry_run=%s)", symbol, dry_run)
    try:
        bot = HTFLiveBot(
            dry_run=dry_run,
            initial_balance=initial_balance,
            interval_minutes=interval_minutes,
            symbol=symbol,
        )
        bot.run_loop(stop_event=stop_event)
    except Exception as e:
        log.error("HTFLiveBot[%s] crashed: %s", symbol, e, exc_info=True)
        raise  # re-raise so watchdog can detect the failure


def _watchdog(
    symbol: str,
    dry_run: bool,
    initial_balance: float,
    interval_minutes: int,
    global_stop: threading.Event,
):
    """
    Watchdog for a single symbol.

    Starts the symbol thread and restarts it automatically on crash,
    up to MAX_RESTARTS times. Exits cleanly when global_stop is set.
    """
    log = logging.getLogger(f"watchdog.{symbol}")
    restarts = 0

    while not global_stop.is_set():
        symbol_stop = threading.Event()

        t = threading.Thread(
            target=_run_symbol,
            args=(symbol, dry_run, initial_balance, interval_minutes, symbol_stop),
            name=f"bot-{symbol}",
            daemon=True,
        )
        t.start()
        log.info("Started thread for %s (restart #%d)", symbol, restarts)

        # Wait for the thread to finish or global stop
        while t.is_alive():
            if global_stop.is_set():
                symbol_stop.set()
                t.join(timeout=10)
                return
            t.join(timeout=5)

        if global_stop.is_set():
            return

        # Thread exited unexpectedly
        restarts += 1
        if MAX_RESTARTS > 0 and restarts >= MAX_RESTARTS:
            log.error(
                "%s reached max restarts (%d) — giving up.",
                symbol, MAX_RESTARTS,
            )
            return

        log.warning(
            "%s thread exited (restart %d/%s). Waiting %ds before restart...",
            symbol, restarts, MAX_RESTARTS or "∞", RESTART_DELAY_SECONDS,
        )
        global_stop.wait(timeout=RESTART_DELAY_SECONDS)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-symbol HTF trading bot (single process)"
    )
    parser.add_argument("--live", action="store_true",
                        help="Enable real trade execution (default: dry-run)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Decision interval in minutes (default: 15)")
    parser.add_argument("--balance", type=float, default=5_000.0,
                        help="Starting balance per symbol in USDT (default: 5000)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to trade (default: BTC ETH SOL XRP)")
    args = parser.parse_args()

    dry_run = not args.live
    symbols = [s.upper() for s in args.symbols]

    logger.info(
        "live_trading_all starting | symbols=%s dry_run=%s interval=%dm balance=%.0f",
        symbols, dry_run, args.interval, args.balance,
    )

    global_stop = threading.Event()

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        logger.info("Signal %d received — stopping all bots...", signum)
        global_stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Start one watchdog thread per symbol
    watchdog_threads = []
    for sym in symbols:
        wt = threading.Thread(
            target=_watchdog,
            args=(sym, dry_run, args.balance, args.interval, global_stop),
            name=f"watchdog-{sym}",
            daemon=False,  # non-daemon so main process waits for clean shutdown
        )
        wt.start()
        watchdog_threads.append(wt)
        # Stagger starts by 3s to avoid all 4 hammering the API simultaneously
        time.sleep(3)

    logger.info("All %d symbol watchdogs running. Press Ctrl+C to stop.", len(symbols))

    # Wait for all watchdog threads to finish (they exit when global_stop is set)
    for wt in watchdog_threads:
        wt.join()

    logger.info("All bots stopped cleanly.")


if __name__ == "__main__":
    main()
