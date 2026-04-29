"""
Bot-independent test: do news events predict short-horizon crypto returns?

Loads news_events from data/trading.db, fetches Binance public klines for
the relevant asset around each event, computes forward returns at 5m / 15m /
1h / 4h, then slices by sentiment bucket, urgency, event_type, asset-tagged,
and time-of-day. Reports mean return, bootstrap 95% CI, two-sided one-sample
t-test p-value, with Bonferroni and BH-FDR multiple-comparison correction.

Run from repo root:
    python3 scripts/test_news_forward_returns.py
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import requests
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "trading.db")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "alternative_cache", "klines_news_test")
os.makedirs(CACHE_DIR, exist_ok=True)

SUPPORTED_ASSETS = {"BTC", "ETH", "SOL", "XRP"}
DEFAULT_ASSET = "ETH"  # broad-market proxy when untagged / "ALL"

HORIZONS_MIN = [5, 15, 60, 240]  # forward-return horizons in minutes
SENT_BUCKETS = [
    ("ext_neg", -10.0, -0.5),
    ("neg",     -0.5,  -0.1),
    ("neutral", -0.1,   0.1),
    ("pos",      0.1,   0.5),
    ("ext_pos",  0.5,   10.0),
]

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class NewsEvent:
    id: int
    ts_ms: int          # event published_at in epoch ms
    sentiment: float
    confidence: float
    urgency: int
    event_type: str
    assets: list[str]
    asset_tagged: bool  # True if at least one of BTC/ETH/SOL/XRP explicitly tagged
    chosen_asset: str   # symbol used for forward return (BTCUSDT etc., minus USDT)
    title: str


def parse_iso(ts: str) -> int:
    # Handles "...+00:00" and trailing "Z"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load_events() -> list[NewsEvent]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, published_at, sentiment_score, confidence, urgency,
               event_type, assets, title
        FROM news_events
        WHERE sentiment_score IS NOT NULL
        ORDER BY published_at ASC
        """
    )
    rows = cur.fetchall()
    con.close()

    events: list[NewsEvent] = []
    for r in rows:
        try:
            assets = json.loads(r["assets"]) if r["assets"] else []
        except Exception:
            assets = []
        explicit = [a for a in assets if a in SUPPORTED_ASSETS]
        asset_tagged = len(explicit) > 0
        chosen = explicit[0] if asset_tagged else DEFAULT_ASSET
        try:
            ts_ms = parse_iso(r["published_at"])
        except Exception:
            continue
        events.append(
            NewsEvent(
                id=r["id"],
                ts_ms=ts_ms,
                sentiment=float(r["sentiment_score"]),
                confidence=float(r["confidence"] or 0.0),
                urgency=int(r["urgency"] or 1),
                event_type=r["event_type"] or "other",
                assets=assets,
                asset_tagged=asset_tagged,
                chosen_asset=chosen,
                title=r["title"],
            )
        )
    return events


# ---------------------------------------------------------------------------
# Klines fetch (cached per symbol per UTC day)
# ---------------------------------------------------------------------------
def day_bucket(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_day_klines(symbol: str, day: str) -> list[list]:
    """Fetch full UTC day of 5m klines for symbol; cache to disk."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_{day}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = start_ms + 24 * 60 * 60 * 1000 - 1
    # 24h * 12 = 288 candles. Binance default limit is 500, fine.
    params = {
        "symbol": symbol,
        "interval": "5m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 500,
    }
    for attempt in range(5):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            elif r.status_code in (418, 429):
                time.sleep(2 ** attempt)
            else:
                print(f"warn: {symbol} {day} -> {r.status_code} {r.text[:100]}",
                      file=sys.stderr)
                time.sleep(1)
        except Exception as e:
            print(f"warn: {symbol} {day} fetch error {e}", file=sys.stderr)
            time.sleep(1)
    return []


def build_kline_index(events: list[NewsEvent]) -> dict[tuple[str, str], list[list]]:
    """Pre-fetch every (symbol, day) we'll need, and the next day too (for 4h forward
    that may spill over)."""
    needed: set[tuple[str, str]] = set()
    for e in events:
        sym = e.chosen_asset + "USDT"
        d0 = day_bucket(e.ts_ms)
        # also include next day for forward windows that cross midnight
        end_ms = e.ts_ms + 4 * 60 * 60 * 1000
        d1 = day_bucket(end_ms)
        needed.add((sym, d0))
        needed.add((sym, d1))

    cache: dict[tuple[str, str], list[list]] = {}
    total = len(needed)
    for i, (sym, day) in enumerate(sorted(needed), 1):
        cache[(sym, day)] = fetch_day_klines(sym, day)
        if i % 10 == 0 or i == total:
            print(f"  klines cache {i}/{total}", file=sys.stderr)
        # Be polite to Binance even with cache hits free
        time.sleep(0.05)
    return cache


def candle_close_at(cache: dict[tuple[str, str], list[list]],
                    symbol: str, ts_ms: int) -> float | None:
    """Close of the 5m candle whose [open_time, open_time+5m) contains ts_ms.
    Falls back to the nearest preceding candle in the day."""
    day = day_bucket(ts_ms)
    klines = cache.get((symbol, day), [])
    if not klines:
        return None
    # 5m bucket open
    bucket_ms = (ts_ms // (5 * 60 * 1000)) * (5 * 60 * 1000)
    # binary-search-ish lookup
    best = None
    for k in klines:
        open_t = k[0]
        if open_t == bucket_ms:
            return float(k[4])
        if open_t < bucket_ms:
            best = k
        else:
            break
    if best is not None:
        return float(best[4])
    return None


def forward_return_bps(cache, symbol: str, ts_ms: int, horizon_min: int) -> float | None:
    """Return forward log-return in basis points from event ts to ts + horizon."""
    p0 = candle_close_at(cache, symbol, ts_ms)
    p1 = candle_close_at(cache, symbol, ts_ms + horizon_min * 60 * 1000)
    if p0 is None or p1 is None or p0 <= 0 or p1 <= 0:
        return None
    return float(np.log(p1 / p0) * 10000.0)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(42)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def slice_stats(returns: np.ndarray) -> dict:
    n = len(returns)
    if n < 5:
        return {"n": n, "mean": float("nan"), "p": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "std": float("nan")}
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std == 0:
        p = 1.0
    else:
        t, p = stats.ttest_1samp(returns, 0.0)
        p = float(p)
    lo, hi = bootstrap_ci(returns)
    return {"n": n, "mean": mean, "std": std, "p": p, "ci_lo": lo, "ci_hi": hi}


def bh_fdr(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg. Returns list of survives[i] aligned with pvals."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    passed = ranked <= thresholds
    # find largest k where ranked[k] <= thresh[k]; everything <=k passes
    if not passed.any():
        survives_sorted = np.zeros(m, dtype=bool)
    else:
        k = np.max(np.where(passed)[0])
        survives_sorted = np.zeros(m, dtype=bool)
        survives_sorted[: k + 1] = True
    survives = np.zeros(m, dtype=bool)
    survives[order] = survives_sorted
    return survives.tolist()


# ---------------------------------------------------------------------------
# Slicings
# ---------------------------------------------------------------------------
def sentiment_bucket(s: float) -> str:
    for name, lo, hi in SENT_BUCKETS:
        if lo <= s < hi:
            return name
    if s >= 0.5:
        return "ext_pos"
    return "ext_neg"


def time_of_day_bucket(ts_ms: int) -> str:
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 0 <= h < 6:
        return "tod_00_06"
    if 6 <= h < 12:
        return "tod_06_12"
    if 12 <= h < 18:
        return "tod_12_18"
    return "tod_18_24"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading news events...", file=sys.stderr)
    events = load_events()
    print(f"  loaded {len(events)} events", file=sys.stderr)

    print("Pre-fetching klines (cached after first run)...", file=sys.stderr)
    cache = build_kline_index(events)

    # Compute forward returns per event and horizon
    rows = []
    skipped = 0
    for e in events:
        sym = e.chosen_asset + "USDT"
        rec = {
            "id": e.id, "ts_ms": e.ts_ms, "sentiment": e.sentiment,
            "confidence": e.confidence, "urgency": e.urgency,
            "event_type": e.event_type, "asset_tagged": e.asset_tagged,
            "chosen_asset": e.chosen_asset,
            "sent_bucket": sentiment_bucket(e.sentiment),
            "tod": time_of_day_bucket(e.ts_ms),
        }
        any_ok = False
        for h in HORIZONS_MIN:
            rec[f"r_{h}"] = forward_return_bps(cache, sym, e.ts_ms, h)
            if rec[f"r_{h}"] is not None:
                any_ok = True
        if not any_ok:
            skipped += 1
            continue
        rows.append(rec)
    print(f"  forward returns computed for {len(rows)} events ({skipped} skipped)",
          file=sys.stderr)

    # Top-5 event types by frequency
    from collections import Counter
    et_counts = Counter([r["event_type"] for r in rows])
    top5_et = [t for t, _ in et_counts.most_common(5)]

    # Build slicing groups
    slicings: list[tuple[str, str, list]] = []  # (slicing_name, group_name, list of rows)

    # ALL
    slicings.append(("all", "all", rows))

    # Sentiment buckets
    by_sent = defaultdict(list)
    for r in rows:
        by_sent[r["sent_bucket"]].append(r)
    for name, _, _ in SENT_BUCKETS:
        slicings.append(("sentiment", name, by_sent.get(name, [])))

    # Urgency
    by_urg = defaultdict(list)
    for r in rows:
        by_urg[r["urgency"]].append(r)
    for u in [1, 2, 3]:
        slicings.append(("urgency", f"u{u}", by_urg.get(u, [])))

    # Event type (top 5)
    by_et = defaultdict(list)
    for r in rows:
        by_et[r["event_type"]].append(r)
    for et in top5_et:
        slicings.append(("event_type", et, by_et.get(et, [])))

    # Asset tagged vs untagged
    tagged = [r for r in rows if r["asset_tagged"]]
    untagged = [r for r in rows if not r["asset_tagged"]]
    slicings.append(("tagged", "asset_tagged", tagged))
    slicings.append(("tagged", "untagged", untagged))

    # Time of day
    by_tod = defaultdict(list)
    for r in rows:
        by_tod[r["tod"]].append(r)
    for tod in ["tod_00_06", "tod_06_12", "tod_12_18", "tod_18_24"]:
        slicings.append(("tod", tod, by_tod.get(tod, [])))

    # Compute stats per (slicing, horizon)
    results = []
    for slicing, group, group_rows in slicings:
        for h in HORIZONS_MIN:
            arr = np.array([r[f"r_{h}"] for r in group_rows
                            if r[f"r_{h}"] is not None], dtype=float)
            s = slice_stats(arr)
            results.append({
                "slicing": slicing, "group": group, "horizon_min": h,
                **s,
            })

    # Multiple-comparison correction over the *directional* hypotheses
    # (exclude the omnibus "all" slicing from corrections; it's a sanity column)
    test_idx = [i for i, r in enumerate(results)
                if r["slicing"] != "all" and not math.isnan(r["p"])]
    pvals = [results[i]["p"] for i in test_idx]
    n_tests = len(pvals)
    bonf_alpha = 0.05 / max(n_tests, 1)
    bh_pass = bh_fdr(pvals, alpha=0.05)
    for surv, idx in zip(bh_pass, test_idx):
        results[idx]["bh_pass"] = surv
        results[idx]["bonf_pass"] = results[idx]["p"] < bonf_alpha
    for r in results:
        r.setdefault("bh_pass", False)
        r.setdefault("bonf_pass", False)

    # Save raw
    out_json = os.path.join(REPO_ROOT, "data", "news_forward_returns_results.json")
    with open(out_json, "w") as f:
        json.dump({"n_events": len(rows), "n_tests": n_tests,
                   "bonferroni_alpha": bonf_alpha,
                   "results": results,
                   "top5_event_types": top5_et}, f, indent=2)

    # ---- Print report ----
    print()
    print("=" * 92)
    print("NEWS FORWARD-RETURN PREDICTABILITY TEST")
    print("=" * 92)
    print(f"Events used: {len(rows)} (skipped {skipped} for missing klines)")
    print(f"Horizons: {HORIZONS_MIN} min   |   Tests: {n_tests}")
    print(f"Bonferroni alpha = 0.05 / {n_tests} = {bonf_alpha:.2e}")
    print(f"BH-FDR alpha = 0.05")
    print()

    # Pretty per-slicing tables
    def print_block(slicing_name: str, header_label: str):
        rows_ = [r for r in results if r["slicing"] == slicing_name]
        if not rows_:
            return
        print(f"--- {header_label} ---")
        print(f"{'group':<14} {'h(m)':>5} {'n':>5} {'mean(bps)':>11} "
              f"{'95%CI(bps)':>20} {'p':>9} {'BH':>4} {'Bonf':>5}")
        for r in rows_:
            ci = (f"[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}]"
                  if not math.isnan(r['ci_lo']) else "n/a")
            mean = f"{r['mean']:+.1f}" if not math.isnan(r['mean']) else "n/a"
            p = f"{r['p']:.4f}" if not math.isnan(r['p']) else "n/a"
            bh = "Y" if r["bh_pass"] else "."
            bo = "Y" if r["bonf_pass"] else "."
            print(f"{r['group']:<14} {r['horizon_min']:>5} {r['n']:>5} "
                  f"{mean:>11} {ci:>20} {p:>9} {bh:>4} {bo:>5}")
        print()

    print_block("all", "ALL events (sanity / unconditional)")
    print_block("sentiment", "By sentiment bucket")
    print_block("urgency", "By urgency")
    print_block("event_type", "By event_type (top 5)")
    print_block("tagged", "Asset tagged vs untagged")
    print_block("tod", "By UTC time-of-day")

    # Highlight survivors
    print("=" * 92)
    print("MULTIPLE-COMPARISON SURVIVORS")
    print("=" * 92)
    bh_survivors = [r for r in results if r["bh_pass"]]
    bonf_survivors = [r for r in results if r["bonf_pass"]]
    if not bh_survivors:
        print("BH-FDR (alpha=0.05): no slicing-horizon combo survives.")
    else:
        print(f"BH-FDR (alpha=0.05): {len(bh_survivors)} survive")
        for r in sorted(bh_survivors, key=lambda x: x["p"]):
            print(f"  {r['slicing']}/{r['group']} @ {r['horizon_min']}m: "
                  f"n={r['n']}, mean={r['mean']:+.2f} bps, "
                  f"CI=[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}], p={r['p']:.4g}")
    print()
    if not bonf_survivors:
        print(f"Bonferroni (alpha={bonf_alpha:.2e}): no slicing-horizon combo survives.")
    else:
        print(f"Bonferroni (alpha={bonf_alpha:.2e}): {len(bonf_survivors)} survive")
        for r in sorted(bonf_survivors, key=lambda x: x["p"]):
            print(f"  {r['slicing']}/{r['group']} @ {r['horizon_min']}m: "
                  f"n={r['n']}, mean={r['mean']:+.2f} bps, p={r['p']:.4g}")
    print()

    # Specifically check NEUTRAL premise: should be indistinguishable from zero
    print("=" * 92)
    print("NEUTRAL-NEWS PREMISE (N2 filter): is mean forward return ~0?")
    print("=" * 92)
    for r in results:
        if r["slicing"] == "sentiment" and r["group"] == "neutral":
            print(f"  neutral @ {r['horizon_min']}m: n={r['n']}, "
                  f"mean={r['mean']:+.2f} bps, "
                  f"CI=[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}], "
                  f"p_vs_zero={r['p']:.3f}  -> "
                  f"{'CANNOT REJECT zero' if r['p'] >= 0.05 else 'rejects zero'}")
    print()

    # Best directional finding
    print("=" * 92)
    print("BEST DIRECTIONAL FINDING (smallest p-value among slicings)")
    print("=" * 92)
    cand = [r for r in results if r["slicing"] != "all"
            and not math.isnan(r["p"]) and r["n"] >= 10]
    cand.sort(key=lambda x: x["p"])
    for r in cand[:8]:
        bh = "BH-OK" if r["bh_pass"] else "fail-BH"
        bo = "Bonf-OK" if r["bonf_pass"] else "fail-Bonf"
        print(f"  {r['slicing']}/{r['group']} @ {r['horizon_min']}m: "
              f"n={r['n']}, mean={r['mean']:+.2f} bps, "
              f"CI=[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}], "
              f"p={r['p']:.4g}  [{bh}, {bo}]")

    print()
    print(f"Raw results saved to: {out_json}")


if __name__ == "__main__":
    main()
