#!/usr/bin/env python3
"""
Backtest: would a max-hold rule and/or widened stagnant band have improved
realized PnL over the existing 20-day trade history?

Inputs: paired OPEN/CLOSE rows from data/trading.db (is_testnet=1).
For each closed trade we replay against 1m klines from Binance to compute
the price the bot WOULD have seen at each candidate force-close timestamp,
then re-derive pnl as if the rule had fired then.

Two rules under test:
  A) Max-hold T hours — force CLOSE at open_ts + T regardless of pnl.
     Tested for T ∈ {12, 18, 24, 36}.
  B) Widened stagnant band (closes after 6h if pnl% in new band).
     Compared against the current band [-0.3%, +0.5%].

Skip funding fees (the cost order-of-magnitude is < $2/symbol over a day,
small relative to typical PnL swings).
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


def parse_ts(s: str) -> datetime:
    s = s.replace("+00:00", "")
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Fetch 1m klines, return [(open_time_ms, close_price), ...]."""
    out: list[tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        url = f"{BASE}/fapi/v1/klines"
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for k in data:
            out.append((int(k[0]), float(k[4])))
        cursor = int(data[-1][0]) + 60_000
        if len(data) < 1500:
            break
        time.sleep(0.05)
    return out


def price_at(klines: list[tuple[int, float]], target_ms: int) -> Optional[float]:
    """Linear scan; klines are minute-spaced, return the closest 1m close at-or-before target."""
    last_price = None
    for ts, price in klines:
        if ts > target_ms:
            return last_price
        last_price = price
    return last_price


def load_trades() -> list[dict]:
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, timestamp, symbol, action, price, pnl, reason, data "
        "FROM trades WHERE timestamp >= '2026-04-06' AND is_testnet=1 ORDER BY timestamp"
    )
    open_pos: dict[str, dict] = {}
    trades: list[dict] = []
    for tid, ts, sym, action, price, pnl, reason, data in cur.fetchall():
        d = json.loads(data) if data else {}
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {
                "open_id": tid,
                "open_ts": ts,
                "side": action.replace("OPEN_", ""),
                "entry": price,
                "units": d.get("units", 0.0),
            }
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            trades.append(
                {
                    **o,
                    "symbol": sym,
                    "close_id": tid,
                    "close_ts": ts,
                    "exit": price,
                    "actual_pnl": pnl or 0.0,
                    "actual_reason": reason,
                }
            )
    conn.close()
    return trades


def pnl_at_price(side: str, entry: float, exit_p: float, units: float) -> float:
    direction = 1 if side == "LONG" else -1
    return (exit_p - entry) * units * direction


def main() -> int:
    trades = load_trades()
    if not trades:
        print("No closed trades found.")
        return 1
    print(f"Backtesting {len(trades)} closed trades over {trades[0]['open_ts'][:10]} → {trades[-1]['close_ts'][:10]}")

    # Fetch 1m klines per symbol covering the whole period (open of first to close of last + 48h pad)
    sym_klines: dict[str, list[tuple[int, float]]] = {}
    earliest = min(parse_ts(t["open_ts"]) for t in trades)
    latest = max(parse_ts(t["close_ts"]) for t in trades)
    end_ms = int(latest.timestamp() * 1000) + 48 * 3600 * 1000
    start_ms = int(earliest.timestamp() * 1000)
    for sym in {t["symbol"] for t in trades}:
        print(f"  Fetching klines for {sym} ...", flush=True)
        sym_klines[sym] = fetch_klines(sym, start_ms, end_ms)
        print(f"    {len(sym_klines[sym])} candles")

    actual_total = sum(t["actual_pnl"] for t in trades)
    actual_wins = sum(1 for t in trades if t["actual_pnl"] > 0)

    # ---- Rule A: max-hold ---------------------------------------------------
    for max_hours in (12, 18, 24, 36):
        new_total = 0.0
        wins = 0
        affected = 0
        for t in trades:
            o_dt = parse_ts(t["open_ts"])
            c_dt = parse_ts(t["close_ts"])
            actual_hours = (c_dt - o_dt).total_seconds() / 3600
            if actual_hours <= max_hours:
                new_total += t["actual_pnl"]
                if t["actual_pnl"] > 0:
                    wins += 1
                continue
            target_ms = int(o_dt.timestamp() * 1000) + max_hours * 3600 * 1000
            price = price_at(sym_klines[t["symbol"]], target_ms)
            if price is None or not t.get("units") or not t.get("entry"):
                new_total += t["actual_pnl"]
                if t["actual_pnl"] > 0:
                    wins += 1
                continue
            sim_pnl = pnl_at_price(t["side"], t["entry"], price, t["units"])
            new_total += sim_pnl
            if sim_pnl > 0:
                wins += 1
            affected += 1
        delta = new_total - actual_total
        print(
            f"Rule A max-hold {max_hours}h: total_pnl=${new_total:.2f} "
            f"(Δ ${delta:+.2f}) wins={wins}/{len(trades)} ({wins / len(trades) * 100:.1f}%) "
            f"affected={affected}"
        )

    # ---- Rule B: widened stagnant band -------------------------------------
    new_bands = [(-0.5, 0.5), (-0.7, 0.5), (-1.0, 0.5), (-1.5, 0.5)]
    for lo, hi in new_bands:
        new_total = 0.0
        wins = 0
        affected = 0
        for t in trades:
            o_dt = parse_ts(t["open_ts"])
            c_dt = parse_ts(t["close_ts"])
            actual_hours = (c_dt - o_dt).total_seconds() / 3600
            if actual_hours <= 6 or not t.get("units") or not t.get("entry"):
                new_total += t["actual_pnl"]
                if t["actual_pnl"] > 0:
                    wins += 1
                continue
            target_ms = int(o_dt.timestamp() * 1000) + 6 * 3600 * 1000
            price = price_at(sym_klines[t["symbol"]], target_ms)
            if price is None:
                new_total += t["actual_pnl"]
                if t["actual_pnl"] > 0:
                    wins += 1
                continue
            sign = 1 if t["side"] == "LONG" else -1
            pnl_pct_at_6h = (price / t["entry"] - 1) * 100 * sign
            current_band = (-0.3, 0.5)
            in_new = lo <= pnl_pct_at_6h <= hi
            in_current = current_band[0] <= pnl_pct_at_6h <= current_band[1]
            if in_new and not in_current:
                # New band would have closed at 6h; current didn't.
                sim_pnl = pnl_at_price(t["side"], t["entry"], price, t["units"])
                new_total += sim_pnl
                if sim_pnl > 0:
                    wins += 1
                affected += 1
            else:
                new_total += t["actual_pnl"]
                if t["actual_pnl"] > 0:
                    wins += 1
        delta = new_total - actual_total
        print(
            f"Rule B band [{lo:+.1f}%, {hi:+.1f}%]: total_pnl=${new_total:.2f} "
            f"(Δ ${delta:+.2f}) wins={wins}/{len(trades)} ({wins / len(trades) * 100:.1f}%) "
            f"affected={affected}"
        )

    print()
    print(f"Baseline: total_pnl=${actual_total:.2f} wins={actual_wins}/{len(trades)} ({actual_wins / len(trades) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
