#!/usr/bin/env python3
"""
Extended backtest: reconstruct round-trip positions from Binance Futures
userTrades fills (~30 days), then replay max-hold + widened-band rules.

Why this script exists: the local DB only goes back ~19 days; the user asked
for 60 days. Binance testnet retains ~30 days max — that's the cap. This
script gets us the longest backtest possible by reading raw fills.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import hmac
import hashlib
import urllib.parse

REPO = Path(__file__).resolve().parent.parent
BASE = "https://testnet.binancefuture.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
LOOKBACK_DAYS = 35  # ask for 35 to make sure we get all of testnet's retention


def _env() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def signed_get(env: dict[str, str], path: str, params: dict) -> list:
    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    params.setdefault("recvWindow", 20000)
    q = urllib.parse.urlencode(params)
    sig = hmac.new(env["BINANCE_FUTURES_API_SECRET"].encode(), q.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{path}?{q}&signature={sig}"
    return requests.get(url, headers={"X-MBX-APIKEY": env["BINANCE_FUTURES_API_KEY"]}, timeout=15).json()


def fetch_user_trades(env: dict, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """
    userTrades retains ~30d but the API restricts each query to a 7-day
    window. We chunk by 7-day windows and paginate within each.
    """
    out: list[dict] = []
    seen_ids: set[int] = set()
    WINDOW_MS = 7 * 24 * 3_600_000 - 60_000  # leave 1m headroom
    cursor = start_ms
    while cursor < end_ms:
        window_end = min(cursor + WINDOW_MS, end_ms)
        sub = cursor
        while sub < window_end:
            page = signed_get(env, "/fapi/v1/userTrades", {
                "symbol": symbol, "limit": 1000,
                "startTime": sub, "endTime": window_end,
            })
            if not isinstance(page, list) or not page:
                break
            new_added = 0
            for f in page:
                fid = int(f["id"])
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                out.append(f)
                new_added += 1
            last_t = int(page[-1]["time"])
            if new_added == 0 or len(page) < 1000:
                break
            sub = last_t + 1
            time.sleep(0.05)
        cursor = window_end + 1
    return out


def reconstruct_positions(fills: list[dict]) -> list[dict]:
    """
    Walk fills in time order, track signed position. When position crosses
    zero, that's a round-trip close. Yield {open_ts, close_ts, side, entry,
    exit, units} for each completed round trip.

    Reduce-only / trailing fills don't matter here — we just track size & sign.
    """
    pos_size = 0.0  # signed: positive = long, negative = short
    pos_avg = 0.0
    pos_open_ts = None
    trips: list[dict] = []
    for f in sorted(fills, key=lambda x: x["time"]):
        qty = float(f["qty"])
        px = float(f["price"])
        side = 1 if f["side"] == "BUY" else -1
        # signed delta added to position
        delta = side * qty
        new_size = pos_size + delta

        if abs(pos_size) < 1e-9 and abs(new_size) > 1e-9:
            # Opening a fresh position
            pos_size = new_size
            pos_avg = px
            pos_open_ts = f["time"]
        elif (pos_size > 0 and new_size >= 0) or (pos_size < 0 and new_size <= 0):
            # Same direction: either adding (avg shifts) or partially reducing (avg stays)
            if (pos_size > 0 and delta > 0) or (pos_size < 0 and delta < 0):
                # adding to position
                new_avg = (pos_avg * abs(pos_size) + px * qty) / abs(new_size)
                pos_avg = new_avg
            pos_size = new_size
            if abs(pos_size) < 1e-9:
                # closed cleanly to zero
                trips.append({
                    "open_ts": pos_open_ts, "close_ts": f["time"],
                    "side": "LONG" if pos_avg > 0 else "SHORT",
                    "entry": pos_avg, "exit": px,
                    "units": abs(pos_size + (-delta)),  # units we just closed
                    "_avg_units_at_open": None,
                })
                pos_size = 0.0
                pos_avg = 0.0
                pos_open_ts = None
        else:
            # Sign flip: close-to-zero + open in opposite direction
            closed_units = abs(pos_size)
            trips.append({
                "open_ts": pos_open_ts, "close_ts": f["time"],
                "side": "LONG" if pos_size > 0 else "SHORT",
                "entry": pos_avg, "exit": px,
                "units": closed_units,
            })
            # remainder opens new position in the opposite direction
            remainder = abs(new_size)
            pos_size = new_size
            pos_avg = px
            pos_open_ts = f["time"]

    return trips


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(f"{BASE}/fapi/v1/klines", params={
            "symbol": symbol, "interval": "1m",
            "startTime": cursor, "endTime": end_ms, "limit": 1500,
        }, timeout=15)
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
    last = None
    for ts, p in klines:
        if ts > target_ms:
            return last
        last = p
    return last


def pnl_at(side: str, entry: float, exit_p: float, units: float) -> float:
    return (exit_p - entry) * units * (1 if side == "LONG" else -1)


def main() -> int:
    env = _env()
    end_ms = int(time.time() * 1000)
    start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    all_trips: list[dict] = []
    sym_klines: dict[str, list[tuple[int, float]]] = {}

    for sym in SYMBOLS:
        print(f"  Fetching userTrades for {sym} ...", flush=True)
        fills = fetch_user_trades(env, sym, start_ms, end_ms)
        print(f"    {len(fills)} fills")
        trips = reconstruct_positions(fills)
        for t in trips:
            t["symbol"] = sym
        all_trips.extend(trips)
        print(f"    {len(trips)} round-trip positions reconstructed")

    if not all_trips:
        print("No round-trip positions found.")
        return 1

    earliest = min(t["open_ts"] for t in all_trips)
    latest = max(t["close_ts"] for t in all_trips)
    print(f"\n{len(all_trips)} round trips spanning "
          f"{datetime.fromtimestamp(earliest/1000,tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(latest/1000,tz=timezone.utc).date()}")

    # Klines for each symbol
    for sym in SYMBOLS:
        print(f"  Fetching klines for {sym} ...", flush=True)
        sym_klines[sym] = fetch_klines(sym, earliest, latest + 48 * 3600 * 1000)
        print(f"    {len(sym_klines[sym])} candles")

    # Baseline pnl from fills
    actual_total = sum(pnl_at(t["side"], t["entry"], t["exit"], t["units"]) for t in all_trips)
    actual_wins = sum(1 for t in all_trips if pnl_at(t["side"], t["entry"], t["exit"], t["units"]) > 0)
    print(f"\nBaseline (from reconstructed fills): pnl=${actual_total:.2f} wins={actual_wins}/{len(all_trips)} ({actual_wins/len(all_trips)*100:.1f}%)")

    # Rule A: max hold
    print("\nRule A — max-hold:")
    for max_hours in (12, 18, 24, 36):
        new_total = 0.0
        wins = 0
        affected = 0
        for t in all_trips:
            actual_hours = (t["close_ts"] - t["open_ts"]) / 3_600_000
            actual_pnl = pnl_at(t["side"], t["entry"], t["exit"], t["units"])
            if actual_hours <= max_hours:
                new_total += actual_pnl
                if actual_pnl > 0:
                    wins += 1
                continue
            target = t["open_ts"] + max_hours * 3_600_000
            px = price_at(sym_klines[t["symbol"]], target)
            if px is None:
                new_total += actual_pnl
                if actual_pnl > 0:
                    wins += 1
                continue
            sim = pnl_at(t["side"], t["entry"], px, t["units"])
            new_total += sim
            if sim > 0:
                wins += 1
            affected += 1
        delta = new_total - actual_total
        print(f"  {max_hours:>3}h: pnl=${new_total:>9.2f}  Δ {delta:>+8.2f}  wins={wins}/{len(all_trips)} ({wins/len(all_trips)*100:.1f}%)  affected={affected}")

    # Rule B: widened stagnant band — close at 6h if pnl% in new band but not old
    current_band = (-0.3, 0.5)
    print("\nRule B — widen stagnant band (from [-0.3%, +0.5%]):")
    for lo, hi in [(-0.5, 0.5), (-0.7, 0.5), (-1.0, 0.5), (-1.5, 0.5)]:
        new_total = 0.0
        wins = 0
        affected = 0
        for t in all_trips:
            actual_hours = (t["close_ts"] - t["open_ts"]) / 3_600_000
            actual_pnl = pnl_at(t["side"], t["entry"], t["exit"], t["units"])
            if actual_hours <= 6:
                new_total += actual_pnl
                if actual_pnl > 0:
                    wins += 1
                continue
            target = t["open_ts"] + 6 * 3_600_000
            px = price_at(sym_klines[t["symbol"]], target)
            if px is None:
                new_total += actual_pnl
                if actual_pnl > 0:
                    wins += 1
                continue
            sign = 1 if t["side"] == "LONG" else -1
            pct_at_6h = (px / t["entry"] - 1) * 100 * sign
            in_new = lo <= pct_at_6h <= hi
            in_curr = current_band[0] <= pct_at_6h <= current_band[1]
            if in_new and not in_curr:
                sim = pnl_at(t["side"], t["entry"], px, t["units"])
                new_total += sim
                if sim > 0:
                    wins += 1
                affected += 1
            else:
                new_total += actual_pnl
                if actual_pnl > 0:
                    wins += 1
        delta = new_total - actual_total
        print(f"  [{lo:+4.1f}%, {hi:+4.1f}%]: pnl=${new_total:>9.2f}  Δ {delta:>+8.2f}  wins={wins}/{len(all_trips)} ({wins/len(all_trips)*100:.1f}%)  affected={affected}")

    # Hold distribution summary
    holds = sorted((t["close_ts"] - t["open_ts"]) / 3_600_000 for t in all_trips)
    print(f"\nHold distribution: median={holds[len(holds)//2]:.1f}h, p90={holds[int(len(holds)*0.9)]:.1f}h, max={holds[-1]:.1f}h")
    print(f"  > 24h: {sum(1 for h in holds if h > 24)} trades")
    print(f"  > 36h: {sum(1 for h in holds if h > 36)} trades")
    return 0


if __name__ == "__main__":
    sys.exit(main())
