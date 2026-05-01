#!/usr/bin/env python3
"""
Test 5 named whale patterns for forward-return predictive power.

Patterns tested:
  P1 — Multi-wallet coordination: ≥3 exchange-wallet deposits/withdrawals
       within 30 minutes, totaling ≥10,000 ETH.
  P2 — Burst pattern: a single wallet's activity spikes ≥5× its trailing
       30-day median in a 1h window.
  P3 — Smart Money Index divergence: whale exchange-net-OUT (accumulation)
       while ETH price has been DOWN ≥1% over preceding 4h.
  P4 — New wallet activation: first activity from a wallet that has been
       silent ≥30 days.
  P5 — Exchange-to-exchange rotation: outflow from one exchange wallet +
       inflow to a different exchange wallet within 1h.

For each pattern:
  - Identify all events in the historical window where the pattern fired
  - Compute forward log-returns at 15m / 1h / 4h / 24h (ETH spot)
  - Subtract hour-of-day baseline (drift adjustment)
  - Apply non-overlapping subsample for autocorrelation correction
  - Report n / mean / 95% CI / p-value (corrected and uncorrected)
  - Apply Bonferroni across (5 patterns × 4 horizons = 20 tests)

Output:
  data/training/whale_named_patterns_results.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, stdev

import requests

REPO = Path(__file__).resolve().parent.parent
WHALE_DIR = REPO / "data" / "whale_behavior" / "eth"
KLINE_CACHE_15M = REPO / "data" / "training" / "eth_spot_15m_cache.jsonl"
KLINE_CACHE_1H = REPO / "data" / "training" / "eth_spot_1h_cache.jsonl"
OUT = REPO / "data" / "training" / "whale_named_patterns_results.json"

EXCHANGE_WALLETS = {
    "binance_hot_wallet", "binance_cold_wallet", "binance_cold_2",
    "binance_reserve", "coinbase_institutional", "kraken_deposit",
}

WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 4, 30, tzinfo=timezone.utc)


# ── Whale + kline loading ────────────────────────────────────────────────

def load_whale_events() -> list[dict]:
    out = []
    if not WHALE_DIR.exists():
        return out
    for f in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet = f.stem
        is_ex = wallet in EXCHANGE_WALLETS
        with open(f) as fp:
            for line in fp:
                try: d = json.loads(line)
                except: continue
                ts = d.get("timestamp")
                if ts is None: continue
                try: dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                except: continue
                if dt < WINDOW_START - timedelta(days=60) or dt > WINDOW_END:
                    continue
                value = float(d.get("value_eth") or 0)
                if value < 100: continue
                out.append({
                    "ts": dt, "wallet": wallet, "is_exchange": is_ex,
                    "value": value, "direction": (d.get("direction") or "").lower(),
                })
    out.sort(key=lambda x: x["ts"])
    return out


def load_klines(cache_path: Path, interval: str) -> dict[int, float]:
    """Load cached klines, fetch any gaps from Binance public API."""
    cache: dict[int, float] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                try:
                    ts, close = json.loads(line)
                    cache[int(ts)] = float(close)
                except: pass
    start_ms = int(WINDOW_START.timestamp() * 1000)
    end_ms = int((WINDOW_END + timedelta(days=2)).timestamp() * 1000)
    cached_min = min(cache) if cache else end_ms
    cached_max = max(cache) if cache else start_ms
    ranges = []
    if start_ms < cached_min: ranges.append((start_ms, cached_min))
    if end_ms > cached_max + 60_000: ranges.append((cached_max + 60_000, end_ms))
    if not ranges:
        return cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(cache_path, "a")
    for r_start, r_end in ranges:
        cur = r_start
        while cur < r_end:
            try:
                resp = requests.get("https://api.binance.com/api/v3/klines", params={
                    "symbol": "ETHUSDT", "interval": interval,
                    "startTime": cur, "endTime": r_end, "limit": 1000,
                }, timeout=15)
                if resp.status_code != 200:
                    time.sleep(2); continue
                data = resp.json()
            except Exception:
                time.sleep(2); continue
            if not data: break
            for k in data:
                ts = int(k[0]); close = float(k[4])
                if ts not in cache:
                    cache[ts] = close
                    fp.write(json.dumps([ts, close]) + "\n")
            cur = int(data[-1][0]) + (
                15 * 60 * 1000 if interval == "15m" else 60 * 60 * 1000
            )
            if len(data) < 1000: break
            time.sleep(0.05)
    fp.close()
    return cache


def forward_return_pct(klines: dict[int, float], event_ts: datetime, hours: int) -> float | None:
    """Forward log-return % from event_ts to event_ts + hours."""
    interval_ms = 60 * 60 * 1000
    snap = (int(event_ts.timestamp()) // 3600) * 3600 * 1000
    fwd = snap + hours * interval_ms
    p_now = klines.get(snap)
    p_fwd = klines.get(fwd)
    if p_now is None or p_fwd is None or p_now <= 0:
        return None
    return math.log(p_fwd / p_now) * 100


# ── Pattern detectors ─────────────────────────────────────────────────────

def p1_multi_wallet_coord(events: list[dict]) -> list[datetime]:
    """≥3 distinct exchange wallets fire within 30 min, total ≥10,000 ETH."""
    out = []
    by_ts = sorted([e for e in events if e["is_exchange"]], key=lambda x: x["ts"])
    win = deque()
    for e in by_ts:
        win.append(e)
        cutoff = e["ts"] - timedelta(minutes=30)
        while win and win[0]["ts"] < cutoff:
            win.popleft()
        wallets = {w["wallet"] for w in win}
        total = sum(w["value"] for w in win)
        if len(wallets) >= 3 and total >= 10_000:
            out.append(e["ts"])
    # Deduplicate close-together fires (keep one per hour)
    if not out: return []
    dedup = [out[0]]
    for t in out[1:]:
        if (t - dedup[-1]).total_seconds() > 3600:
            dedup.append(t)
    return dedup


def p2_burst(events: list[dict]) -> list[tuple[datetime, str]]:
    """Single wallet activity in 1h window ≥5× its trailing 30-day median (per wallet)."""
    by_wallet: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_wallet[e["wallet"]].append(e)
    out = []
    for wallet, lst in by_wallet.items():
        lst.sort(key=lambda x: x["ts"])
        for i, e in enumerate(lst):
            t = e["ts"]
            # 30-day baseline median per-1h volume
            baseline_start = t - timedelta(days=30)
            baseline_events = [b for b in lst[:i] if b["ts"] >= baseline_start]
            if len(baseline_events) < 30:
                continue
            # bucket into 1h volumes
            buckets: dict[int, float] = defaultdict(float)
            for b in baseline_events:
                hr = int(b["ts"].timestamp()) // 3600
                buckets[hr] += b["value"]
            base_median = median(buckets.values()) if buckets else 0
            if base_median < 100: continue
            # current 1h window
            win_start = t - timedelta(hours=1)
            cur_vol = sum(b["value"] for b in lst[:i+1] if b["ts"] >= win_start)
            if cur_vol >= 5 * base_median and cur_vol >= 1000:
                out.append((t, wallet))
    if not out: return []
    out.sort(key=lambda x: x[0])
    dedup = [out[0]]
    for t, w in out[1:]:
        if (t - dedup[-1][0]).total_seconds() > 3600:
            dedup.append((t, w))
    return dedup


def p3_smart_money_divergence(events: list[dict], klines_1h: dict[int, float]) -> list[datetime]:
    """Net OUT (accumulation) ≥5,000 ETH while ETH down ≥1% over preceding 4h."""
    out = []
    # 4h rolling exchange net flow (in - out, positive = distribution)
    exchange_evts = [e for e in events if e["is_exchange"]]
    exchange_evts.sort(key=lambda x: x["ts"])
    win = deque()
    for e in exchange_evts:
        win.append(e)
        cutoff = e["ts"] - timedelta(hours=4)
        while win and win[0]["ts"] < cutoff:
            win.popleft()
        net_in = sum(w["value"] for w in win if w["direction"] == "in") \
                 - sum(w["value"] for w in win if w["direction"] == "out")
        # accumulation = net OUT (negative net_in) ≥5000 ETH
        if net_in > -5_000:
            continue
        # check price
        snap = (int(e["ts"].timestamp()) // 3600) * 3600 * 1000
        prev = snap - 4 * 60 * 60 * 1000
        p_now = klines_1h.get(snap)
        p_prev = klines_1h.get(prev)
        if not p_now or not p_prev:
            continue
        ret_4h = math.log(p_now / p_prev) * 100
        if ret_4h <= -1.0:
            out.append(e["ts"])
    if not out: return []
    dedup = [out[0]]
    for t in out[1:]:
        if (t - dedup[-1]).total_seconds() > 3600:
            dedup.append(t)
    return dedup


def p4_new_wallet_activation(events: list[dict]) -> list[tuple[datetime, str]]:
    """First activity from a wallet after ≥30 days of silence."""
    by_wallet: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_wallet[e["wallet"]].append(e)
    out = []
    for wallet, lst in by_wallet.items():
        lst.sort(key=lambda x: x["ts"])
        prev_ts = None
        for e in lst:
            if prev_ts is None:
                prev_ts = e["ts"]
                continue
            if (e["ts"] - prev_ts).total_seconds() >= 30 * 86400:
                out.append((e["ts"], wallet))
            prev_ts = e["ts"]
    out.sort(key=lambda x: x[0])
    return out


def p5_exchange_rotation(events: list[dict]) -> list[datetime]:
    """Out from one exchange + In to a different exchange within 1h."""
    ex_evts = [e for e in events if e["is_exchange"]]
    ex_evts.sort(key=lambda x: x["ts"])
    out = []
    win = deque()
    for e in ex_evts:
        win.append(e)
        cutoff = e["ts"] - timedelta(hours=1)
        while win and win[0]["ts"] < cutoff:
            win.popleft()
        out_wallets = {w["wallet"] for w in win if w["direction"] == "out"}
        in_wallets = {w["wallet"] for w in win if w["direction"] == "in"}
        if out_wallets and in_wallets and not (out_wallets & in_wallets):
            # confirm at least 1000 ETH total moved
            if sum(w["value"] for w in win) >= 1_000:
                out.append(e["ts"])
    if not out: return []
    dedup = [out[0]]
    for t in out[1:]:
        if (t - dedup[-1]).total_seconds() > 3600:
            dedup.append(t)
    return dedup


# ── Stats ─────────────────────────────────────────────────────────────────

def hour_baseline(klines_15m: dict[int, float]) -> dict[int, dict[int, float]]:
    """Per-(hour,horizon) mean drift for hour-of-day adjustment.
    Returns {horizon_hours → {hour_utc → mean_pct}}.
    """
    horizons = (0.25, 1, 4, 24)
    out: dict = {h: defaultdict(list) for h in horizons}
    by_ts = sorted(klines_15m.items())
    for ts_ms, p_now in by_ts:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        for h in horizons:
            fwd_ms = ts_ms + int(h * 3600 * 1000)
            p_fwd = klines_15m.get(fwd_ms)
            if p_fwd is None or p_now <= 0:
                continue
            ret = math.log(p_fwd / p_now) * 100
            out[h][dt.hour].append(ret)
    return {h: {hr: mean(vals) if vals else 0.0 for hr, vals in d.items()} for h, d in out.items()}


def compute_t(values: list[float]) -> tuple[float, float] | None:
    """Two-sided t-test vs zero. Returns (mean, p-value)."""
    n = len(values)
    if n < 5: return None
    m = mean(values)
    sd = stdev(values) if n > 1 else 0
    if sd == 0: return (m, 0.0 if m != 0 else 1.0)
    se = sd / math.sqrt(n)
    if se == 0: return (m, 1.0)
    t = m / se
    # Approximate p-value (two-sided) using survival of standard normal
    # for large n, t ≈ z; for small n this is an approximation
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (m, p)


def evaluate_pattern(name: str, event_ts: list[datetime], klines_15m, klines_1h, baseline) -> dict:
    """For each event, compute forward returns at 4 horizons; correct for drift."""
    horizons = ((0.25, "15m", klines_15m, 15 * 60 * 1000),
                (1, "1h", klines_1h, 60 * 60 * 1000),
                (4, "4h", klines_1h, 60 * 60 * 1000),
                (24, "24h", klines_1h, 60 * 60 * 1000))
    out = {"name": name, "n_events": len(event_ts), "horizons": {}}
    for h_val, h_label, klines, interval_ms in horizons:
        rets = []
        for t in event_ts:
            snap = (int(t.timestamp() * 1000) // interval_ms) * interval_ms
            fwd = snap + int(h_val * 3600 * 1000)
            p_now = klines.get(snap); p_fwd = klines.get(fwd)
            if p_now is None or p_fwd is None or p_now <= 0:
                continue
            ret = math.log(p_fwd / p_now) * 100
            ret -= baseline[h_val].get(t.hour, 0.0)  # drift correction
            rets.append(ret)
        # iid stats
        iid_t = compute_t(rets)
        # Non-overlap subsample (events ≥ horizon apart)
        non_overlap = []
        last = None
        for i, t in enumerate(event_ts):
            if i >= len(rets): break
            if last is None or (t - last).total_seconds() >= h_val * 3600:
                non_overlap.append(rets[i])
                last = t
        non_t = compute_t(non_overlap) if len(non_overlap) >= 5 else None
        out["horizons"][h_label] = {
            "n_returns": len(rets),
            "mean_bps": round(iid_t[0] * 100, 2) if iid_t else None,  # bps
            "iid_p": round(iid_t[1], 4) if iid_t else None,
            "n_non_overlap": len(non_overlap),
            "non_overlap_mean_bps": round(non_t[0] * 100, 2) if non_t else None,
            "non_overlap_p": round(non_t[1], 4) if non_t else None,
        }
    return out


def main():
    print("Loading whale events ...")
    events = load_whale_events()
    print(f"  {len(events)} events Jan-Apr 2026 (filtered to ≥100 ETH)")

    print("Loading klines ...")
    klines_15m = load_klines(KLINE_CACHE_15M, "15m")
    klines_1h = load_klines(KLINE_CACHE_1H, "1h")
    print(f"  {len(klines_15m)} 15m candles, {len(klines_1h)} 1h candles")

    print("Computing hour-of-day baseline ...")
    baseline = hour_baseline(klines_15m)

    print("\nDetecting patterns ...")
    p1_events = p1_multi_wallet_coord(events)
    p2_events = [t for t, _ in p2_burst(events)]
    p3_events = p3_smart_money_divergence(events, klines_1h)
    p4_events = [t for t, _ in p4_new_wallet_activation(events)]
    p5_events = p5_exchange_rotation(events)
    for name, evts in [("P1 multi-wallet coord", p1_events),
                        ("P2 burst", p2_events),
                        ("P3 smart-money divergence", p3_events),
                        ("P4 new-wallet activation", p4_events),
                        ("P5 exchange rotation", p5_events)]:
        print(f"  {name}: {len(evts)} events")

    print("\nEvaluating patterns ...")
    results = []
    for name, evts in [("P1 multi-wallet coord", p1_events),
                        ("P2 burst", p2_events),
                        ("P3 smart-money divergence", p3_events),
                        ("P4 new-wallet activation", p4_events),
                        ("P5 exchange rotation", p5_events)]:
        if len(evts) < 5:
            print(f"  {name}: too few events ({len(evts)}), skipping")
            continue
        r = evaluate_pattern(name, evts, klines_15m, klines_1h, baseline)
        results.append(r)

    # Bonferroni alpha for 5 patterns × 4 horizons = 20 tests
    BONF = 0.05 / 20
    print(f"\nResults (Bonferroni alpha = {BONF:.4f}):\n")
    print(f"{'pattern':<28} {'horizon':>8} {'n':>5} {'mean_bps':>10} {'iid_p':>8} "
          f"{'no_n':>5} {'no_mean_bps':>12} {'no_p':>8} {'sig?':>5}")
    print("-" * 110)
    for r in results:
        for h_label, hr in r["horizons"].items():
            sig = ""
            if hr["non_overlap_p"] is not None and hr["non_overlap_p"] < BONF:
                sig = "✓BONF"
            elif hr["non_overlap_p"] is not None and hr["non_overlap_p"] < 0.05:
                sig = "p<.05"
            print(f"  {r['name']:<26} {h_label:>8} {hr['n_returns']:>5} "
                  f"{hr['mean_bps'] if hr['mean_bps'] is not None else 'n/a':>10} "
                  f"{hr['iid_p'] if hr['iid_p'] is not None else 'n/a':>8} "
                  f"{hr['n_non_overlap']:>5} "
                  f"{hr['non_overlap_mean_bps'] if hr['non_overlap_mean_bps'] is not None else 'n/a':>12} "
                  f"{hr['non_overlap_p'] if hr['non_overlap_p'] is not None else 'n/a':>8} "
                  f"{sig:>5}")

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat()},
        "n_patterns_tested": len(results),
        "bonferroni_alpha": BONF,
        "results": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
