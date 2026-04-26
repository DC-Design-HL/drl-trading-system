#!/usr/bin/env python3
"""
Threshold sensitivity backtest for the USDT.D filter.

For each LONG trade in the recent history, compute the synthetic USDT.D
basket change at the moment of open, then evaluate each candidate
threshold (-0.2%, -0.3%, -0.4%, -0.5%, -0.7%, -1.0%) and report:

  * Trades blocked vs allowed at each threshold
  * Pnl saved (loser-blocked trades) + pnl lost (winner-blocked trades)
  * Net delta vs baseline
  * Win-rate impact on the surviving trade set

Source data: paired OPEN_LONG/CLOSE_* trades from data/trading.db
(already includes the 23 ghost trades merged on Apr 24).

Klines: 1m closes from Binance Futures testnet for all 4 proxy symbols.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
BASE = "https://testnet.binancefuture.com"
PROXY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
LOOKBACK_HOURS = 2

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]  # absolute %, the filter compares basket <= -threshold


def parse_ts(s: str) -> datetime:
    s = s.replace("+00:00", "")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def load_long_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, timestamp, symbol, action, price, pnl, reason "
        "FROM trades WHERE timestamp >= '2026-04-06' AND is_testnet=1 ORDER BY timestamp"
    )
    open_pos: dict[str, dict] = {}
    longs: list[dict] = []
    for tid, ts, sym, action, price, pnl, reason in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {"open_id": tid, "open_ts": ts, "side": action.replace("OPEN_", ""), "entry": price}
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            if o["side"] == "LONG":
                longs.append({**o, "symbol": sym, "close_id": tid, "close_ts": ts, "exit": price, "pnl": pnl or 0.0, "reason": reason})
    conn.close()
    return longs


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    out = []
    cur = start_ms
    while cur < end_ms:
        r = requests.get(f"{BASE}/fapi/v1/klines", params={
            "symbol": symbol, "interval": "1m",
            "startTime": cur, "endTime": end_ms, "limit": 1500,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend((int(k[0]), float(k[4])) for k in data)
        cur = int(data[-1][0]) + 60_000
        if len(data) < 1500:
            break
        time.sleep(0.05)
    return out


def price_at(klines: list[tuple[int, float]], target_ms: int) -> Optional[float]:
    last = None
    for ts, p in klines:
        if ts > target_ms:
            return last
        last = p
    return last


def main() -> int:
    longs = load_long_trades()
    if not longs:
        print("No LONG trades found.")
        return 1
    print(f"Loaded {len(longs)} LONG round-trips ({longs[0]['open_ts'][:10]} → {longs[-1]['open_ts'][:10]}).")

    # Klines for all proxy symbols spanning the trade window
    earliest = min(parse_ts(t["open_ts"]).timestamp() for t in longs) * 1000
    latest = max(parse_ts(t["close_ts"]).timestamp() for t in longs) * 1000
    pad = LOOKBACK_HOURS * 3_600_000 + 60_000
    klines: dict[str, list] = {}
    for sym in PROXY_SYMBOLS:
        print(f"  Fetching {sym} klines ...", flush=True)
        klines[sym] = fetch_klines(sym, int(earliest - pad), int(latest + 60_000))
        print(f"    {len(klines[sym])} candles")

    # For each LONG, compute basket_change_pct at open time
    for t in longs:
        open_ms = int(parse_ts(t["open_ts"]).timestamp() * 1000)
        lookback_ms = open_ms - LOOKBACK_HOURS * 3_600_000
        deltas = []
        for sym in PROXY_SYMBOLS:
            p_now = price_at(klines[sym], open_ms)
            p_back = price_at(klines[sym], lookback_ms)
            if p_now is None or p_back is None or p_back <= 0:
                continue
            deltas.append((p_now - p_back) / p_back * 100.0)
        t["basket_pct"] = sum(deltas) / len(deltas) if deltas else None

    # Drop trades we couldn't compute
    skipped = sum(1 for t in longs if t.get("basket_pct") is None)
    longs = [t for t in longs if t.get("basket_pct") is not None]
    if skipped:
        print(f"  Skipped {skipped} trades with insufficient kline history.")

    base_total = sum(t["pnl"] for t in longs)
    base_wins = sum(1 for t in longs if t["pnl"] > 0)
    print(f"\nBaseline LONG pnl: ${base_total:+.2f} over {len(longs)} trades, "
          f"WR {base_wins / len(longs) * 100:.1f}%")
    print(f"\n{'thresh':>8} {'blocked':>9} {'allowed':>9} {'block_pnl':>11} {'allowed_pnl':>13} {'delta':>9} {'allowed_WR':>11}")
    print("-" * 80)

    for thr in THRESHOLDS:
        blocked = [t for t in longs if t["basket_pct"] <= -thr]
        allowed = [t for t in longs if t["basket_pct"] > -thr]
        block_pnl = sum(t["pnl"] for t in blocked)
        allowed_pnl = sum(t["pnl"] for t in allowed)
        delta = -block_pnl  # what we save (or lose) by avoiding the blocked trades
        wins_allowed = sum(1 for t in allowed if t["pnl"] > 0)
        wr_allowed = wins_allowed / len(allowed) * 100 if allowed else 0
        print(f"  -{thr:.1f}% {len(blocked):>8} {len(allowed):>9} ${block_pnl:>+9.2f} ${allowed_pnl:>+11.2f} ${delta:>+7.2f} {wr_allowed:>10.1f}%")

    # Show the distribution of basket_pct at LONG opens
    sorted_pcts = sorted(t["basket_pct"] for t in longs)
    n = len(sorted_pcts)
    print(f"\nBasket-pct distribution at LONG opens (n={n}):")
    print(f"  min={sorted_pcts[0]:+.3f}%, p10={sorted_pcts[n//10]:+.3f}%, "
          f"p50={sorted_pcts[n//2]:+.3f}%, p90={sorted_pcts[int(n*0.9)]:+.3f}%, "
          f"max={sorted_pcts[-1]:+.3f}%")

    # Top 10 worst blockable trades (would have been blocked at -0.3%)
    print(f"\nTop 5 LONG losers with most-negative basket_pct (potential filter candidates):")
    bad = sorted([t for t in longs if t["pnl"] < 0], key=lambda t: t["basket_pct"])[:5]
    for t in bad:
        print(f"  {t['open_ts'][:19]} {t['symbol']:8} pnl=${t['pnl']:+.2f}  basket={t['basket_pct']:+.3f}%  reason={t['reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
