#!/usr/bin/env python3
"""
Backtest combinations of:
  * Stagnant band rule (deployed: [-1.0%, +0.5%])
  * Max-hold rule
  * Off-hours session filter (block 21:00-08:00 UTC trades — historically -$11 / 99 trades)
  * Position-size scaling (1x, 2x, 3x, 4x, 6x)

Goal: identify any combination that projects to ≥30%/month on $5,000 base
while keeping max single-day drawdown bounded.

Inputs: reconstructed round-trip positions from Binance Futures userTrades
fills (~35 days, max retention available on testnet).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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
LOOKBACK_DAYS = 35

OFF_HOURS_UTC = set(list(range(21, 24)) + list(range(0, 8)))  # 21:00-08:00 UTC

# Each symbol has its own $5K wallet, so account total is $20K. But returns
# should be reported on per-symbol $5K (the user's mental model of "balance").
PER_SYMBOL_BALANCE = 5_000.0


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
    out: list[dict] = []
    seen: set[int] = set()
    WIN = 7 * 24 * 3_600_000 - 60_000
    cursor = start_ms
    while cursor < end_ms:
        win_end = min(cursor + WIN, end_ms)
        sub = cursor
        while sub < win_end:
            page = signed_get(env, "/fapi/v1/userTrades", {
                "symbol": symbol, "limit": 1000, "startTime": sub, "endTime": win_end,
            })
            if not isinstance(page, list) or not page:
                break
            new_added = 0
            for f in page:
                fid = int(f["id"])
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(f)
                new_added += 1
            if new_added == 0 or len(page) < 1000:
                break
            sub = int(page[-1]["time"]) + 1
            time.sleep(0.05)
        cursor = win_end + 1
    return out


def reconstruct(fills: list[dict]) -> list[dict]:
    pos = 0.0
    avg = 0.0
    open_ts = None
    trips = []
    for f in sorted(fills, key=lambda x: x["time"]):
        qty = float(f["qty"])
        px = float(f["price"])
        side = 1 if f["side"] == "BUY" else -1
        delta = side * qty
        new = pos + delta
        if abs(pos) < 1e-9 and abs(new) > 1e-9:
            pos, avg, open_ts = new, px, f["time"]
        elif (pos > 0 and new >= 0) or (pos < 0 and new <= 0):
            if (pos > 0 and delta > 0) or (pos < 0 and delta < 0):
                avg = (avg * abs(pos) + px * qty) / abs(new)
            old_pos = pos
            pos = new
            if abs(pos) < 1e-9:
                trips.append({
                    "open_ts": open_ts, "close_ts": f["time"],
                    "side": "LONG" if old_pos > 0 else "SHORT",
                    "entry": avg, "exit": px,
                    "units": abs(old_pos),
                })
                pos, avg, open_ts = 0.0, 0.0, None
        else:
            trips.append({
                "open_ts": open_ts, "close_ts": f["time"],
                "side": "LONG" if pos > 0 else "SHORT",
                "entry": avg, "exit": px,
                "units": abs(pos),
            })
            pos, avg, open_ts = new, px, f["time"]
    return trips


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


def price_at(klines: list[tuple[int, float]], target: int) -> Optional[float]:
    last = None
    for ts, p in klines:
        if ts > target:
            return last
        last = p
    return last


def pnl_of(t: dict, exit_px: float) -> float:
    return (exit_px - t["entry"]) * t["units"] * (1 if t["side"] == "LONG" else -1)


def apply_rules(
    trips: list[dict],
    klines: dict[str, list[tuple[int, float]]],
    *,
    use_stagnant_band: tuple[float, float] | None,
    max_hold_h: float | None,
    block_off_hours: bool,
) -> list[dict]:
    """Returns mutated trades with effective exit prices/timestamps after rules."""
    out = []
    for t in trips:
        # Filter: off-hours session block — skip the trade entirely
        open_dt = datetime.fromtimestamp(t["open_ts"] / 1000, tz=timezone.utc)
        if block_off_hours and open_dt.hour in OFF_HOURS_UTC:
            continue
        new_t = dict(t)
        actual_h = (t["close_ts"] - t["open_ts"]) / 3_600_000
        new_exit = t["exit"]
        new_close = t["close_ts"]
        # Stagnant band check at 6h
        if use_stagnant_band is not None and actual_h > 6:
            target = t["open_ts"] + 6 * 3_600_000
            px = price_at(klines[t["symbol"]], target)
            if px is not None:
                sign = 1 if t["side"] == "LONG" else -1
                pct = (px / t["entry"] - 1) * 100 * sign
                lo, hi = use_stagnant_band
                # Apply rule only when in widened band but NOT in current [-0.3%, +0.5%]
                if lo <= pct <= hi and not (-0.3 <= pct <= 0.5):
                    new_exit = px
                    new_close = target
        # Max-hold check
        if max_hold_h is not None and (new_close - t["open_ts"]) / 3_600_000 > max_hold_h:
            target = t["open_ts"] + max_hold_h * 3_600_000
            px = price_at(klines[t["symbol"]], target)
            if px is not None:
                new_exit = px
                new_close = target
        new_t["exit"] = new_exit
        new_t["close_ts"] = new_close
        new_t["base_pnl"] = pnl_of(new_t, new_exit)
        out.append(new_t)
    return out


def stats(trades: list[dict], scale: float, days: float) -> dict:
    if not trades:
        return {"pnl": 0, "wins": 0, "n": 0, "wr": 0, "monthly_pct": 0, "max_dd": 0}
    total_pnl = sum(t["base_pnl"] * scale for t in trades)
    wins = sum(1 for t in trades if t["base_pnl"] > 0)
    n = len(trades)
    # Per-symbol equity curve to compute drawdown (per-symbol $5K)
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)
    max_dd_sym = 0.0
    for sym, ts in by_sym.items():
        ts_sorted = sorted(ts, key=lambda x: x["close_ts"])
        equity = PER_SYMBOL_BALANCE
        peak = equity
        dd = 0.0
        for t in ts_sorted:
            equity += t["base_pnl"] * scale
            peak = max(peak, equity)
            dd = max(dd, (peak - equity) / peak)
        max_dd_sym = max(max_dd_sym, dd)
    monthly = total_pnl / days * 30
    monthly_pct_per_symbol = monthly / (PER_SYMBOL_BALANCE * 4) * 100  # treat $20K total
    return {
        "pnl": total_pnl, "wins": wins, "n": n,
        "wr": wins / n * 100 if n else 0,
        "monthly_pct": monthly_pct_per_symbol,
        "max_dd": max_dd_sym * 100,
    }


def main() -> int:
    env = _env()
    end_ms = int(time.time() * 1000)
    start_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    all_trips = []
    klines = {}
    for sym in SYMBOLS:
        print(f"  Fetching {sym} ...", flush=True)
        fills = fetch_user_trades(env, sym, start_ms, end_ms)
        trips = reconstruct(fills)
        for t in trips:
            t["symbol"] = sym
        all_trips.extend(trips)
    print(f"\n{len(all_trips)} round trips reconstructed.\n")

    earliest = min(t["open_ts"] for t in all_trips)
    latest = max(t["close_ts"] for t in all_trips)
    days_span = (latest - earliest) / 86_400_000
    for sym in SYMBOLS:
        klines[sym] = fetch_klines(sym, earliest, latest + 48 * 3_600_000)

    print(f"Window: {datetime.fromtimestamp(earliest/1000,tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(latest/1000,tz=timezone.utc).date()} ({days_span:.1f}d)")
    print(f"Per-symbol balance: ${PER_SYMBOL_BALANCE:,.0f}, total account: ${PER_SYMBOL_BALANCE*4:,.0f}\n")

    # Configurations to test
    configs = [
        ("baseline (no rules)",          dict(use_stagnant_band=None,            max_hold_h=None, block_off_hours=False)),
        ("stagnant [-1.0,+0.5]",         dict(use_stagnant_band=(-1.0, 0.5),     max_hold_h=None, block_off_hours=False)),
        ("+ max-hold 12h",               dict(use_stagnant_band=(-1.0, 0.5),     max_hold_h=12.0, block_off_hours=False)),
        ("+ off-hours block",            dict(use_stagnant_band=(-1.0, 0.5),     max_hold_h=12.0, block_off_hours=True)),
    ]

    print(f"{'config':<32} {'sz':>4} {'n':>4} {'WR':>6} {'pnl':>10} {'mo%':>7} {'maxDD%':>7}")
    print("-" * 80)
    best = (None, None, -1e9)
    for name, opts in configs:
        sims = apply_rules(all_trips, klines, **opts)
        for scale in (1.0, 2.0, 3.0, 4.0, 6.0):
            s = stats(sims, scale, days_span)
            print(f"{name:<32} {scale:>3.0f}x {s['n']:>4} {s['wr']:>5.1f}% ${s['pnl']:>8.2f} {s['monthly_pct']:>6.2f}% {s['max_dd']:>6.2f}%")
            if s["monthly_pct"] > best[2]:
                best = (name, scale, s["monthly_pct"], s)
        print()

    print(f"Best config: '{best[0]}' at {best[1]:.0f}x size → {best[2]:.2f}%/mo, max DD {best[3]['max_dd']:.2f}%")
    print(f"Trades affected: {best[3]['n']}, WR {best[3]['wr']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
