"""
Bot-independent whale -> ETH forward return correlation test.

Tests whether whale events on-chain predict ETH price moves at horizons
15m / 1h / 4h / 24h, across multiple slicings.

Honest framing: most slicings will be noise. We apply Bonferroni and BH-FDR
correction across all (slicing, horizon) combinations and only claim signal
that survives correction.

Outputs: results table to stdout + JSON dump.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/home/claude/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/workspace/drl-trading-system")
WHALE_DIR = REPO / "data/whale_behavior/eth"
KLINE_CACHE_1H = REPO / "data/training/eth_spot_1h_cache.jsonl"
KLINE_CACHE_15M = REPO / "data/training/eth_spot_15m_cache.jsonl"  # we will populate
RESULTS_JSON = REPO / "data/training/whale_forward_return_results.json"

# Window
WINDOW_START = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
WINDOW_END = int(datetime(2026, 4, 29, tzinfo=timezone.utc).timestamp())

# Filter
MIN_VALUE_ETH = 100.0

# Horizons in seconds
HORIZONS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "24h": 24 * 60 * 60,
}

# Wallet categorization
EXCHANGE_WALLETS = {
    "binance_cold_2",
    "binance_cold_wallet",
    "binance_hot_wallet",
    "binance_reserve",
    "coinbase_institutional",
    "kraken_deposit",
    "robinhood",
}
OTHER_WALLETS = {
    "eth_2.0_deposit_contract",
    "galaxy_digital",
    "jump_trading",
    "smart_money_whale_1",
}

# Action types worth treating as "real flow" (skip CONTRACT_CALL noise)
FLOW_ACTIONS = {
    "LARGE_TRANSFER_IN",
    "LARGE_TRANSFER_OUT",
    "EXCHANGE_DEPOSIT",
    "EXCHANGE_WITHDRAWAL",
}


# -----------------------------------------------------------------------------
# Kline fetching / loading
# -----------------------------------------------------------------------------

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int):
    """Fetch all klines in [start_ms, end_ms] paginated. Returns list of [open_ms, close_price]."""
    out = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        url = "https://api.binance.com/api/v3/klines?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if not data:
            break
        for k in data:
            out.append([k[0], float(k[4])])  # open_ms, close_price
        last_open = data[-1][0]
        # interval ms
        if interval.endswith("m"):
            step = int(interval[:-1]) * 60 * 1000
        elif interval.endswith("h"):
            step = int(interval[:-1]) * 3600 * 1000
        else:
            step = 60 * 1000
        cur = last_open + step
        if len(data) < 1000:
            break
        time.sleep(0.1)  # be nice
    return out


def load_jsonl_klines(path: Path):
    klines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            klines.append(json.loads(line))
    return klines


def ensure_15m_cache():
    """Build 15m cache covering window if missing."""
    start_ms = WINDOW_START * 1000
    end_ms = (WINDOW_END + 24 * 3600) * 1000  # +1d for forward lookups
    if KLINE_CACHE_15M.exists():
        existing = load_jsonl_klines(KLINE_CACHE_15M)
        if existing and existing[0][0] <= start_ms and existing[-1][0] >= end_ms - 24 * 3600 * 1000:
            print(f"[cache] 15m cache exists with {len(existing)} bars", flush=True)
            return existing
    print(f"[fetch] fetching 15m klines {start_ms} → {end_ms}", flush=True)
    klines = fetch_binance_klines("ETHUSDT", "15m", start_ms, end_ms)
    print(f"[fetch] got {len(klines)} bars", flush=True)
    with open(KLINE_CACHE_15M, "w") as f:
        for k in klines:
            f.write(json.dumps(k) + "\n")
    return klines


def ensure_1h_extension():
    """1h cache ends ~Apr 3, but window goes to Apr 29. Extend if needed."""
    klines = load_jsonl_klines(KLINE_CACHE_1H)
    last_ms = klines[-1][0] if klines else 0
    needed_end_ms = (WINDOW_END + 24 * 3600) * 1000
    if last_ms >= needed_end_ms - 3600 * 1000:
        return klines
    print(f"[fetch] extending 1h cache from {last_ms} to {needed_end_ms}", flush=True)
    extra = fetch_binance_klines("ETHUSDT", "1h", last_ms + 3600 * 1000, needed_end_ms)
    print(f"[fetch] got {len(extra)} extra 1h bars", flush=True)
    klines = klines + extra
    return klines


# -----------------------------------------------------------------------------
# Price lookup
# -----------------------------------------------------------------------------

class PriceLookup:
    """Fast price lookup. Uses 15m for short horizons, 1h for long ones."""

    def __init__(self, klines_15m, klines_1h):
        # Sort and store as parallel arrays
        klines_15m = sorted(klines_15m, key=lambda x: x[0])
        klines_1h = sorted(klines_1h, key=lambda x: x[0])
        self.t15 = [k[0] // 1000 for k in klines_15m]  # seconds
        self.p15 = [k[1] for k in klines_15m]
        self.t1h = [k[0] // 1000 for k in klines_1h]
        self.p1h = [k[1] for k in klines_1h]
        self.step15 = 15 * 60
        self.step1h = 3600

    def _price_at_or_after(self, t_target: int, times, prices, step) -> float | None:
        """Get close price of the bar that contains/just-after t_target."""
        # Bar with open_time <= t_target < open_time + step
        idx = bisect.bisect_right(times, t_target) - 1
        if idx < 0 or idx >= len(times):
            return None
        # close of this bar
        if t_target - times[idx] < step:
            return prices[idx]
        return None

    def price_at(self, t_seconds: int, prefer_15m: bool = True) -> float | None:
        if prefer_15m and self.t15:
            p = self._price_at_or_after(t_seconds, self.t15, self.p15, self.step15)
            if p is not None:
                return p
        return self._price_at_or_after(t_seconds, self.t1h, self.p1h, self.step1h)


# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------

def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = x[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def t_test_one_sample(x: np.ndarray):
    """Two-sided one-sample t-test against 0. Returns (t, p)."""
    n = len(x)
    if n < 5:
        return (float("nan"), 1.0)
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return (float("nan"), 1.0)
    t = m / (s / math.sqrt(n))
    # p-value via survival function of t-distribution; use scipy if available else normal approx
    try:
        from scipy import stats as _stats  # type: ignore
        p = float(2 * _stats.t.sf(abs(t), df=n - 1))
    except Exception:
        # Normal approximation (n is large in practice)
        from math import erf, sqrt as _sqrt
        p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / _sqrt(2)))))
    return (float(t), p)


def bh_fdr(pvals: list[float], alpha: float = 0.05):
    """Benjamini-Hochberg. Returns boolean array of which are significant."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    sig = [False] * n
    threshold = 0.0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / n * alpha:
            threshold = pvals[i]
    if threshold > 0:
        for i in range(n):
            if pvals[i] <= threshold:
                sig[i] = True
    return sig


# -----------------------------------------------------------------------------
# Load whale events
# -----------------------------------------------------------------------------

def load_whales():
    events = []
    for f in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet = f.stem
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ts = r.get("timestamp")
                if ts is None:
                    continue
                if ts < WINDOW_START or ts > WINDOW_END:
                    continue
                action = r.get("action", "")
                if action not in FLOW_ACTIONS:
                    continue
                v = float(r.get("value_eth") or 0.0)
                if v < MIN_VALUE_ETH:
                    continue
                direction = r.get("direction", "")
                events.append({
                    "ts": int(ts),
                    "wallet": wallet,
                    "action": action,
                    "value_eth": v,
                    "direction": direction,
                })
    events.sort(key=lambda e: e["ts"])
    return events


# -----------------------------------------------------------------------------
# Slicing & analysis
# -----------------------------------------------------------------------------

def compute_hour_baselines(price_lookup: PriceLookup):
    """For each (horizon, hour-of-day), compute the average forward return across
    ALL 15m bars in the window. Used to subtract calendar/drift bias."""
    base = {h: {hh: [] for hh in range(24)} for h in HORIZONS}
    for ts, p0 in zip(price_lookup.t15, price_lookup.p15):
        # ts is already in seconds (PriceLookup divides by 1000 on init)
        if ts < WINDOW_START or ts > WINDOW_END - 86400:
            continue
        hr = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        for hname, hsec in HORIZONS.items():
            prefer15 = hname in ("15m", "1h")
            p1 = price_lookup.price_at(ts + hsec, prefer_15m=prefer15)
            if p1 is None or p1 <= 0:
                continue
            base[hname][hr].append((p1 / p0 - 1.0) * 10000.0)
    base_mean = {h: {hh: (float(np.mean(v)) if v else 0.0) for hh, v in base[h].items()} for h in HORIZONS}
    return base_mean


def compute_event_returns(events, price_lookup: PriceLookup, baseline: dict):
    """Annotate each event with both raw and drift-adjusted forward returns."""
    out = []
    for e in events:
        t = e["ts"]
        p0 = price_lookup.price_at(t, prefer_15m=True)
        if p0 is None or p0 <= 0:
            continue
        hr = datetime.fromtimestamp(t, tz=timezone.utc).hour
        rets = {}
        adj = {}
        for hname, hsec in HORIZONS.items():
            prefer15 = hname in ("15m", "1h")
            p1 = price_lookup.price_at(t + hsec, prefer_15m=prefer15)
            if p1 is None or p1 <= 0:
                rets[hname] = float("nan")
                adj[hname] = float("nan")
            else:
                raw = (p1 / p0 - 1.0) * 10000.0
                rets[hname] = raw
                adj[hname] = raw - baseline[hname][hr]
        e2 = dict(e)
        e2["entry_price"] = p0
        e2["rets_bps"] = rets
        e2["adj_bps"] = adj
        e2["hour"] = hr
        e2["category"] = "exchange" if e["wallet"] in EXCHANGE_WALLETS else "other"
        out.append(e2)
    return out


def slice_stats(name: str, values_by_horizon: dict[str, list[float]],
                ts_by_horizon: dict[str, list[int]] | None = None):
    """Run stats on each horizon. Returns list of result dicts.

    If ts_by_horizon is provided, also runs an OVERLAP-CORRECTED test on a greedy
    non-overlapping subsample (events spaced ≥ horizon apart). Forward-return
    windows of overlapping events share most of their realized return, so the iid
    t-test is grossly over-confident.
    """
    rows = []
    for h, vals in values_by_horizon.items():
        pairs = list(zip(ts_by_horizon[h], vals)) if ts_by_horizon else [(None, v) for v in vals]
        pairs = [(t, v) for t, v in pairs if not math.isnan(v)]
        arr = np.array([v for _, v in pairs], dtype=float)
        n = len(arr)
        row = {
            "slicing": name, "horizon": h, "n": n,
            "mean_bps": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            "t": float("nan"), "p": float("nan"),
            "no_n": 0, "no_mean_bps": float("nan"), "no_p": float("nan"),
        }
        if n >= 10:
            row["mean_bps"] = float(arr.mean())
            row["ci_lo"], row["ci_hi"] = bootstrap_ci_mean(arr)
            row["t"], row["p"] = t_test_one_sample(arr)
        # Non-overlapping subsample
        if ts_by_horizon and n >= 10:
            horizon_sec = HORIZONS[h]
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            no_vals = []
            last = -10**18
            for t, v in pairs_sorted:
                if t >= last + horizon_sec:
                    no_vals.append(v)
                    last = t
            no_arr = np.array(no_vals, dtype=float)
            row["no_n"] = len(no_arr)
            if len(no_arr) >= 10:
                row["no_mean_bps"] = float(no_arr.mean())
                _, row["no_p"] = t_test_one_sample(no_arr)
        rows.append(row)
    return rows


def aggregate_window_flow(events, window_sec: int, price_lookup: PriceLookup,
                          horizon_sec: int, baseline: dict, horizon_label: str):
    """Net flow per fixed window → drift-adjusted forward return at horizon."""
    if not events:
        return []
    buckets = defaultdict(float)
    for e in events:
        bkt = (e["ts"] // window_sec) * window_sec
        sign = 0
        if e["category"] == "exchange":
            if e["direction"] == "out":
                sign = +1
            elif e["direction"] == "in":
                sign = -1
        else:
            if e["direction"] == "in":
                sign = +1
            elif e["direction"] == "out":
                sign = -1
        buckets[bkt] += sign * e["value_eth"]
    rows = []
    for bkt_t, net in buckets.items():
        entry_t = bkt_t + window_sec
        p0 = price_lookup.price_at(entry_t, prefer_15m=(horizon_sec <= 3600))
        if p0 is None:
            continue
        p1 = price_lookup.price_at(entry_t + horizon_sec, prefer_15m=(horizon_sec <= 3600))
        if p1 is None:
            continue
        ret_bps = (p1 / p0 - 1.0) * 10000.0
        hr = datetime.fromtimestamp(entry_t, tz=timezone.utc).hour
        adj = ret_bps - baseline[horizon_label][hr]
        rows.append({"net_eth": net, "ret_bps": adj, "bkt_t": bkt_t})
    return rows


def analyze_window_flow(events, price_lookup, window_label, window_sec, horizon_label, horizon_sec, baseline):
    """Bucket the events, then split into pos / neg / zero net flow and test each as
    a separate slice."""
    rows = aggregate_window_flow(events, window_sec, price_lookup, horizon_sec, baseline, horizon_label)
    if not rows:
        return []
    pos = [r["ret_bps"] for r in rows if r["net_eth"] > 0]
    neg = [r["ret_bps"] for r in rows if r["net_eth"] < 0]
    out = []
    if len(pos) >= 10:
        out.extend(slice_stats(
            f"agg_{window_label}_netflow>0",
            {horizon_label: pos},
        ))
    if len(neg) >= 10:
        out.extend(slice_stats(
            f"agg_{window_label}_netflow<0",
            {horizon_label: neg},
        ))
    # Also: large positive vs large negative (top/bottom 25%)
    nets = sorted(rows, key=lambda r: r["net_eth"])
    q = len(nets) // 4
    if q >= 10:
        bot = [r["ret_bps"] for r in nets[:q]]  # most negative net flow
        top = [r["ret_bps"] for r in nets[-q:]]  # most positive net flow
        out.extend(slice_stats(
            f"agg_{window_label}_netflow_bottom25%",
            {horizon_label: bot},
        ))
        out.extend(slice_stats(
            f"agg_{window_label}_netflow_top25%",
            {horizon_label: top},
        ))
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("WHALE → ETH FORWARD-RETURN CORRELATION TEST (BOT-INDEPENDENT)")
    print("=" * 80)
    print(f"Window: {datetime.fromtimestamp(WINDOW_START, tz=timezone.utc).date()} → {datetime.fromtimestamp(WINDOW_END, tz=timezone.utc).date()}")
    print(f"Filter: value_eth >= {MIN_VALUE_ETH}, actions ∈ {sorted(FLOW_ACTIONS)}")
    print()

    # Load price data
    klines_1h = ensure_1h_extension()
    klines_15m = ensure_15m_cache()
    price_lookup = PriceLookup(klines_15m, klines_1h)
    print(f"[price] 15m bars: {len(klines_15m)}, 1h bars: {len(klines_1h)}")

    # Load whale events
    events_raw = load_whales()
    print(f"[whales] loaded {len(events_raw)} events in window after filter")

    print("[baseline] computing hour-of-day drift baselines from all 15m bars...")
    baseline = compute_hour_baselines(price_lookup)
    print("[baseline] hour-of-day mean fwd returns (bps), 24h horizon:")
    print("           " + " ".join(f"{baseline['24h'][h]:+6.1f}" for h in range(24)))

    events = compute_event_returns(events_raw, price_lookup, baseline)
    print(f"[whales] {len(events)} events with valid price lookups")
    if not events:
        print("No events to analyze. Aborting.")
        return

    # Window-wide drift for context
    p_start = price_lookup.price_at(WINDOW_START)
    p_end = price_lookup.price_at(WINDOW_END)
    if p_start and p_end:
        print(f"[context] ETH window return: {(p_end/p_start - 1)*100:+.2f}%  "
              f"(${p_start:.2f} → ${p_end:.2f})")

    # Build slicings — runs on DRIFT-ADJUSTED returns by default
    all_results = []

    def add(name, predicate, use_adj: bool = True):
        sub = [e for e in events if predicate(e)]
        if len(sub) < 10:
            return
        key = "adj_bps" if use_adj else "rets_bps"
        rets = {h: [e[key][h] for e in sub] for h in HORIZONS}
        ts_by_h = {h: [e["ts"] for e in sub] for h in HORIZONS}
        rows = slice_stats(name, rets, ts_by_h)
        for r in rows:
            r["adjusted"] = use_adj
        all_results.extend(rows)

    # Direction
    add("direction=in", lambda e: e["direction"] == "in")
    add("direction=out", lambda e: e["direction"] == "out")

    # Category
    add("category=exchange", lambda e: e["category"] == "exchange")
    add("category=other", lambda e: e["category"] == "other")

    # Category × direction
    for cat in ("exchange", "other"):
        for d in ("in", "out"):
            add(f"category={cat}&direction={d}",
                lambda e, c=cat, dd=d: e["category"] == c and e["direction"] == dd)

    # Size buckets
    add("size=100-1k", lambda e: 100 <= e["value_eth"] < 1000)
    add("size=1k-10k", lambda e: 1000 <= e["value_eth"] < 10000)
    add("size=10k+", lambda e: e["value_eth"] >= 10000)

    # Size × direction
    for lo, hi, lbl in [(100, 1000, "100-1k"), (1000, 10000, "1k-10k"), (10000, 1e12, "10k+")]:
        for d in ("in", "out"):
            add(f"size={lbl}&direction={d}",
                lambda e, lo=lo, hi=hi, dd=d: lo <= e["value_eth"] < hi and e["direction"] == dd)

    # Size × category
    for lo, hi, lbl in [(100, 1000, "100-1k"), (1000, 10000, "1k-10k"), (10000, 1e12, "10k+")]:
        for cat in ("exchange", "other"):
            add(f"size={lbl}&category={cat}",
                lambda e, lo=lo, hi=hi, c=cat: lo <= e["value_eth"] < hi and e["category"] == c)

    # Hour of day (group into quarters of day to avoid 24-way blow-up)
    add("hour=00-05", lambda e: 0 <= e["hour"] < 6)
    add("hour=06-11", lambda e: 6 <= e["hour"] < 12)
    add("hour=12-17", lambda e: 12 <= e["hour"] < 18)
    add("hour=18-23", lambda e: 18 <= e["hour"] < 24)

    # Aggregated window flows (drift-adjusted)
    for win_label, win_sec in [("1h", 3600), ("4h", 4 * 3600)]:
        for h_label, h_sec in HORIZONS.items():
            rows_agg = analyze_window_flow(events, price_lookup, win_label, win_sec, h_label, h_sec, baseline)
            for r in rows_agg:
                r["adjusted"] = True
            all_results.extend(rows_agg)

    # Filter to slicings with valid p
    valid = [r for r in all_results if not math.isnan(r["p"]) and r["n"] >= 10]
    pvals = [r["p"] for r in valid]
    n_tests = len(pvals)
    bonf_alpha = 0.05 / max(n_tests, 1)
    fdr_sig = bh_fdr(pvals, alpha=0.05) if pvals else []

    for r, fdr in zip(valid, fdr_sig):
        r["bonferroni_sig"] = r["p"] < bonf_alpha
        r["fdr_sig"] = fdr

    # Sort by p-value
    valid.sort(key=lambda r: r["p"])

    print()
    print("=" * 130)
    print("DRIFT-ADJUSTED forward returns (event return - same-hour baseline) — bps")
    print("Columns: 'iid p' assumes independent obs (over-confident due to overlapping forward windows).")
    print("         'no_n / no_mean / no_p' = greedy non-overlapping subsample (events spaced ≥ horizon apart).")
    print("=" * 130)
    print(f"{'SLICING':<40} {'HZ':<4} {'N':>5} {'MEAN':>7} {'CI95':>16} {'iid_p':>10} "
          f"{'BONF':>4} {'FDR':>4} {'no_n':>5} {'no_mean':>8} {'no_p':>10}")
    print("=" * 130)
    for r in valid[:50]:
        ci = f"[{r['ci_lo']:+5.1f},{r['ci_hi']:+5.1f}]"
        no_mean = f"{r['no_mean_bps']:+7.2f}" if not math.isnan(r['no_mean_bps']) else "    -  "
        no_p = f"{r['no_p']:.3g}" if not math.isnan(r['no_p']) else "  -"
        print(f"{r['slicing']:<40} {r['horizon']:<4} {r['n']:>5d} "
              f"{r['mean_bps']:+7.2f} {ci:>16} "
              f"{r['p']:>10.3g} "
              f"{'Y' if r['bonferroni_sig'] else '.':>4} "
              f"{'Y' if r['fdr_sig'] else '.':>4} "
              f"{r['no_n']:>5d} {no_mean:>8} {no_p:>10}")

    print()
    print(f"Total tests: {n_tests}")
    print(f"Bonferroni alpha: {bonf_alpha:.2e}")
    print(f"Bonferroni-sig (iid):       {sum(1 for r in valid if r['bonferroni_sig'])}")
    print(f"BH-FDR-sig    (iid, q=.05): {sum(1 for r in valid if r['fdr_sig'])}")
    no_p05 = sum(1 for r in valid if not math.isnan(r['no_p']) and r['no_p'] < 0.05)
    no_p01 = sum(1 for r in valid if not math.isnan(r['no_p']) and r['no_p'] < 0.01)
    no_bonf = sum(1 for r in valid if not math.isnan(r['no_p']) and r['no_p'] < bonf_alpha)
    print(f"NON-OVERLAP p<0.05:         {no_p05}")
    print(f"NON-OVERLAP p<0.01:         {no_p01}")
    print(f"NON-OVERLAP p<bonf({bonf_alpha:.1e}): {no_bonf}")

    if valid:
        best = valid[0]
        print()
        print(f"BEST iid: {best['slicing']} @ {best['horizon']}: "
              f"mean={best['mean_bps']:.2f} bps, CI=[{best['ci_lo']:.2f}, {best['ci_hi']:.2f}], "
              f"iid p={best['p']:.4g}, n={best['n']}  ||  "
              f"non-overlap n={best['no_n']}, mean={best['no_mean_bps']:+.2f} bps, p={best['no_p']:.4g}")
        # Best non-overlap survivor
        no_valid = [r for r in valid if not math.isnan(r['no_p']) and r['no_n'] >= 30]
        no_valid.sort(key=lambda r: r['no_p'])
        if no_valid:
            b2 = no_valid[0]
            print(f"BEST non-overlap (n≥30): {b2['slicing']} @ {b2['horizon']}: "
                  f"non-overlap n={b2['no_n']}, mean={b2['no_mean_bps']:+.2f} bps, p={b2['no_p']:.4g}")

    out = {
        "window_start": WINDOW_START,
        "window_end": WINDOW_END,
        "min_value_eth": MIN_VALUE_ETH,
        "n_events_loaded": len(events_raw),
        "n_events_with_returns": len(events),
        "n_tests": n_tests,
        "bonferroni_alpha": bonf_alpha,
        "results": valid,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x)
    print(f"\n[saved] {RESULTS_JSON}")


if __name__ == "__main__":
    main()
