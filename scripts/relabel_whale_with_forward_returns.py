#!/usr/bin/env python3
"""
Relabel the whale labeled_v2 data with FORWARD-RETURN direction labels
instead of the wallet's own action.

Why: the labeled_v2/*.jsonl files describe what the wallet did
(LARGE_TRANSFER_IN, ACCUMULATION, etc.). When we tried using those as
supervised labels for predicting "BULLISH/BEARISH/NEUTRAL market signal,"
the dataset collapsed to 94% BULL (because most events for the sampled
wallets are inflows). The model trained to "always predict BULL" and
test accuracy hit 1.0 trivially with confidence stdev 0.009.

Fix: for each whale event timestamp, compute the realized ETH price
return over the next 4 hours, and use the SIGN of that return as the
label.

  return > +RETURN_THRESHOLD   → BULLISH
  return < -RETURN_THRESHOLD   → BEARISH
  else                          → NEUTRAL

This produces labels that actually reflect "did the market go up or
down after this whale activity," which is the right target.

Output: data/whale_behavior/labeled_v3/<wallet>_returns.jsonl
        — same fields as labeled_v2 plus `forward_return_pct` and
        `direction_label`.

Run on either server or Mac (no GPU, just network for Binance klines):
    python3 scripts/relabel_whale_with_forward_returns.py
        [--horizon_hours 4]
        [--threshold_pct 1.0]
        [--output_dir data/whale_behavior/labeled_v3/]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "data" / "whale_behavior" / "labeled_v2"

# Mainnet spot — has the longest ETHUSDT history (back to 2017-08).
# Public endpoint, no key required.
SPOT_BASE = "https://api.binance.com"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--horizon_hours", type=int, default=4,
                   help="Forward return horizon in hours (4h is the bot's primary trade horizon).")
    p.add_argument("--threshold_pct", type=float, default=1.0,
                   help="Return magnitude threshold for non-NEUTRAL labels.")
    p.add_argument("--output_dir", default="data/whale_behavior/labeled_v3/")
    p.add_argument("--src_dir", default=str(SRC_DIR))
    p.add_argument("--klines_cache", default="data/training/eth_spot_1h_cache.jsonl",
                   help="Cache 1h ETHUSDT klines so subsequent runs are fast.")
    return p.parse_args(argv)


def fetch_klines_1h(start_ms: int, end_ms: int, cache_path: Path) -> dict[int, float]:
    """Return {open_time_ms: close_price} for 1h ETHUSDT spot klines.
    Uses cache file if present and covers the requested window.
    """
    cache: dict[int, float] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                try:
                    ts, close = json.loads(line)
                    cache[int(ts)] = float(close)
                except Exception:
                    pass
        cached_min = min(cache) if cache else end_ms
        cached_max = max(cache) if cache else start_ms
        # Need to fill gaps before cached_min and after cached_max
        ranges = []
        if start_ms < cached_min:
            ranges.append((start_ms, cached_min))
        if end_ms > cached_max + 60 * 60 * 1000:
            ranges.append((cached_max + 60 * 60 * 1000, end_ms))
    else:
        ranges = [(start_ms, end_ms)]

    if not ranges:
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_handle = open(cache_path, "a")
    for r_start, r_end in ranges:
        cur = r_start
        print(f"  fetching {datetime.fromtimestamp(r_start / 1000, tz=timezone.utc).date()} → "
              f"{datetime.fromtimestamp(r_end / 1000, tz=timezone.utc).date()}")
        while cur < r_end:
            try:
                resp = requests.get(f"{SPOT_BASE}/api/v3/klines", params={
                    "symbol": "ETHUSDT", "interval": "1h",
                    "startTime": cur, "endTime": r_end, "limit": 1000,
                }, timeout=20)
                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code}: {resp.text[:120]}")
                    time.sleep(2)
                    continue
                data = resp.json()
            except Exception as e:
                print(f"    fetch err: {e}; retrying ...")
                time.sleep(2)
                continue
            if not data:
                break
            for k in data:
                ts = int(k[0])
                close = float(k[4])
                if ts not in cache:
                    cache[ts] = close
                    out_handle.write(json.dumps([ts, close]) + "\n")
            cur = int(data[-1][0]) + 60 * 60 * 1000
            if len(data) < 1000:
                break
            time.sleep(0.05)
    out_handle.close()
    return cache


def label_for_return(ret_pct: float, threshold: float) -> str:
    if ret_pct > threshold:
        return "BULLISH"
    if ret_pct < -threshold:
        return "BEARISH"
    return "NEUTRAL"


def lookup_forward_return(event_ts: int, horizon_hours: int, klines: dict[int, float]) -> Optional[float]:
    """Forward return from event_ts to event_ts + horizon_hours, using
    the closest available 1h kline closes. Returns pct (e.g. 1.5 = 1.5%).
    """
    hour_ms = 60 * 60 * 1000
    # Snap to hour boundary (kline open_time is hour-aligned)
    event_hour_ms = (event_ts // 1000 // 3600) * 3600 * 1000
    forward_hour_ms = event_hour_ms + horizon_hours * hour_ms
    p_now = klines.get(event_hour_ms)
    p_fwd = klines.get(forward_hour_ms)
    if p_now is None or p_fwd is None or p_now <= 0:
        return None
    return (p_fwd - p_now) / p_now * 100.0


def main(argv=None) -> int:
    args = parse_args(argv)
    src_dir = Path(args.src_dir)
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = REPO / args.klines_cache

    if not src_dir.exists():
        print(f"Source dir not found: {src_dir}", file=sys.stderr)
        return 1

    src_files = sorted(src_dir.glob("*.jsonl"))
    if not src_files:
        print(f"No labeled_v2 files in {src_dir}", file=sys.stderr)
        return 1

    # Collect all timestamps to figure out kline range
    print("Scanning event timestamps ...")
    all_ts = []
    for f in src_files:
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                    all_ts.append(int(d["timestamp"]) * 1000)  # ms
                except Exception:
                    continue
    if not all_ts:
        print("No timestamps found.", file=sys.stderr)
        return 1
    start_ms = min(all_ts)
    end_ms = max(all_ts) + args.horizon_hours * 60 * 60 * 1000 + 60 * 60 * 1000
    print(f"  {len(all_ts)} events spanning "
          f"{datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()}")

    # ETHUSDT only existed on Binance from 2017-08-17 onwards. Filter older events.
    SPOT_INCEPTION_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
    if start_ms < SPOT_INCEPTION_MS:
        n_dropped = sum(1 for t in all_ts if t < SPOT_INCEPTION_MS)
        print(f"  {n_dropped} events before ETHUSDT spot inception (2017-08-17) "
              f"will be skipped (no kline coverage).")

    print("Fetching/loading 1h ETHUSDT klines ...")
    fetch_start_ms = max(start_ms, SPOT_INCEPTION_MS)
    klines = fetch_klines_1h(fetch_start_ms, end_ms, cache_path)
    print(f"  {len(klines)} 1h klines available")

    # Relabel each file
    label_counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    skipped_no_kline = 0
    skipped_old = 0
    total_written = 0
    for src in src_files:
        out_path = out_dir / (src.stem.replace("_behavioral", "") + "_returns.jsonl")
        with open(src) as fp_in, open(out_path, "w") as fp_out:
            for line in fp_in:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts_ms = int(d["timestamp"]) * 1000
                if ts_ms < SPOT_INCEPTION_MS:
                    skipped_old += 1
                    continue
                ret = lookup_forward_return(ts_ms, args.horizon_hours, klines)
                if ret is None:
                    skipped_no_kline += 1
                    continue
                label = label_for_return(ret, args.threshold_pct)
                d["forward_return_pct"] = ret
                d["forward_horizon_hours"] = args.horizon_hours
                d["direction_label"] = label
                fp_out.write(json.dumps(d) + "\n")
                label_counts[label] += 1
                total_written += 1
        print(f"  wrote {out_path.name}")
    print(f"\nWrote {total_written} relabeled events to {out_dir}/")
    print(f"  skipped (pre-2017 inception): {skipped_old}")
    print(f"  skipped (no kline coverage): {skipped_no_kline}")
    print(f"  label distribution: {label_counts}")
    print(f"  → {label_counts['BULLISH'] / max(1, total_written) * 100:.1f}% BULL, "
          f"{label_counts['NEUTRAL'] / max(1, total_written) * 100:.1f}% NEUT, "
          f"{label_counts['BEARISH'] / max(1, total_written) * 100:.1f}% BEAR")

    if min(label_counts.values()) < total_written * 0.10:
        print("\n⚠️ WARNING: most-imbalanced class < 10% — try lower --threshold_pct")
    else:
        print("\n✅ Classes are reasonably balanced — ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
