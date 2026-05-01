#!/usr/bin/env python3
"""
Phase 2: OI as a TRADE FILTER backtest on actual trade outcomes.

Phase 1 found a directionally-consistent OI/forward-return pattern that
failed Bonferroni under non-overlap subsampling (autocorrelation artifact
in 30d × hourly data). This script does the more direct test:

  Question: had we skipped trades whose ENTRY had extreme OI conditions,
  would realized PnL on the actual round-trips have improved?

Method:
  1. Load round-trips from trading.db (is_testnet=1, ≥ 2026-04-06).
  2. For each entry timestamp, compute OI features at the closest-on-or-before
     1h OI bar for the trade's symbol.
  3. Define candidate filter rules informed by Phase 1 directional signal:
       F1: skip LONG when oi_roc_24h is in top decile (high OI rise → bear)
       F2: skip LONG when oi_value_roc_4h is in top decile
       F3: skip SHORT when oi_z_7d is in bottom decile (low OI → bull)
       F4: skip LONG when high OI AND price-up 4h (OI/price divergence "bull-fade")
  4. For each rule, compute:
       baseline_pnl   = sum(pnl over all trades)
       filtered_pnl   = sum(pnl over trades NOT skipped) — i.e. what we'd realize
                        had we skipped the flagged trades
       delta          = filtered_pnl - baseline_pnl  (positive = filter helps)
       n_skipped, mean pnl of skipped, mean pnl of kept
  5. Bootstrap confidence interval on delta (paired resample over trades).
  6. Decide:
       * direction-consistent + bootstrap CI[5%] > 0 + ≥ 5 trades skipped
         → candidate worth deploying as LOGGED-ONLY signal
       * else: keep logging only, do not propose any trading rule

Per Chen's directive: this is observational only. Production change is
to LOG the OI features alongside trade entries; no decision-making impact.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "trading.db"
OUT = REPO / "data" / "training" / "oi_filter_backtest.json"

WINDOW_START = "2026-04-06"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
LOOKBACK_DAYS = 30
RNG_SEED = 1337

OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
KL_URL = "https://fapi.binance.com/fapi/v1/klines"


def http_get_json(url, params, retries=3, sleep=0.5):
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    full = f"{url}?{qs}"
    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "drl-pretest/1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"GET failed: {full} :: {last_err}")


def fetch_oi_hist(symbol, days):
    out = []
    end = int(time.time() * 1000)
    target_start = end - days * 24 * 3600 * 1000
    while True:
        data = http_get_json(OI_URL, {"symbol": symbol, "period": "1h", "limit": 500, "endTime": end})
        if not data: break
        out = data + out
        oldest = data[0]["timestamp"]
        if oldest <= target_start or len(data) < 500: break
        end = oldest - 1
        time.sleep(0.2)
    seen = set(); deduped = []
    for d in out:
        ts = int(d["timestamp"])
        if ts in seen: continue
        seen.add(ts); deduped.append(d)
    deduped.sort(key=lambda x: int(x["timestamp"]))
    return deduped


def fetch_klines(symbol, days):
    out = []
    end = int(time.time() * 1000)
    target_start = end - days * 24 * 3600 * 1000
    while True:
        data = http_get_json(KL_URL, {"symbol": symbol, "interval": "1h", "limit": 1000, "endTime": end})
        if not data: break
        out = data + out
        oldest = data[0][0]
        if oldest <= target_start or len(data) < 100: break
        end = oldest - 1
        time.sleep(0.2)
    seen = set(); deduped = []
    for k in out:
        ts = int(k[0])
        if ts in seen: continue
        seen.add(ts); deduped.append(k)
    deduped.sort(key=lambda x: x[0])
    return deduped


def pct_change(arr, k):
    out = [None] * len(arr)
    for i in range(k, len(arr)):
        a, b = arr[i - k], arr[i]
        if a is None or b is None or a == 0: continue
        out[i] = (b - a) / a * 100
    return out


def rolling_zscore(arr, window):
    out = [None] * len(arr)
    for i in range(window, len(arr)):
        w = [x for x in arr[i - window:i] if x is not None]
        if len(w) < window // 2: continue
        m = mean(w); s = stdev(w) if len(w) > 1 else 0
        if s == 0: continue
        out[i] = (arr[i] - m) / s if arr[i] is not None else None
    return out


def load_round_trips():
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("""SELECT id, timestamp, symbol, action, price, pnl, reason
                   FROM trades WHERE timestamp >= ? AND is_testnet=1
                   ORDER BY timestamp""", (WINDOW_START,))
    open_pos = {}
    out = []
    for tid, ts, sym, action, price, pnl, reason in cur.fetchall():
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {"open_ts": ts, "side": "LONG" if "LONG" in action else "SHORT",
                             "entry": price, "open_id": tid}
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            out.append({**o, "symbol": sym, "close_ts": ts, "exit": price,
                        "pnl": pnl or 0.0, "reason": reason or "UNKNOWN"})
    conn.close()
    return out


def features_at_or_before(ts_ms_target: int, features_by_ts: dict, max_lag_ms: int = 3600 * 1000 * 2):
    """Find the feature row at or before target ts, within max_lag (default 2h)."""
    sorted_ts = sorted(features_by_ts)
    # Binary search
    lo, hi = 0, len(sorted_ts) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] <= ts_ms_target:
            best = sorted_ts[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None: return None
    if ts_ms_target - best > max_lag_ms: return None
    return features_by_ts[best]


def bootstrap_ci_delta(kept_pnls: list[float], skipped_pnls: list[float], iters=10_000):
    """Bootstrap CI on delta = filtered - baseline = -sum(skipped).
    (Filtering = skipping = removing those trades' pnl from the total.)"""
    if not skipped_pnls: return (0, 0, 0)
    rng = random.Random(RNG_SEED)
    deltas = []
    n = len(skipped_pnls)
    for _ in range(iters):
        sample = [skipped_pnls[rng.randint(0, n - 1)] for _ in range(n)]
        deltas.append(-sum(sample))  # delta is the *removed* pnl, negated
    deltas.sort()
    lo = deltas[int(iters * 0.05)]
    md = deltas[int(iters * 0.5)]
    hi = deltas[int(iters * 0.95)]
    return (lo, md, hi)


def main():
    print(f"Loading round-trips since {WINDOW_START}…")
    rts = load_round_trips()
    print(f"  {len(rts)} closed round-trips")

    print(f"\nFetching {LOOKBACK_DAYS}d 1h OI + klines for {SYMBOLS}…")
    feats_by_sym: dict[str, dict[int, dict]] = {s: {} for s in SYMBOLS}
    for sym in SYMBOLS:
        oi = fetch_oi_hist(sym, LOOKBACK_DAYS)
        kl = fetch_klines(sym, LOOKBACK_DAYS)
        oi_by_ts = {int(d["timestamp"]): {"oi": float(d["sumOpenInterest"]),
                                          "oi_value": float(d["sumOpenInterestValue"])} for d in oi}
        kl_by_ts = {int(k[0]): float(k[4]) for k in kl}
        ts_sorted = sorted(set(oi_by_ts) & set(kl_by_ts))
        oi_arr = [oi_by_ts[t]["oi"] for t in ts_sorted]
        oi_v_arr = [oi_by_ts[t]["oi_value"] for t in ts_sorted]
        px_arr = [kl_by_ts[t] for t in ts_sorted]

        f = {
            "oi_roc_4h":       pct_change(oi_arr, 4),
            "oi_roc_24h":      pct_change(oi_arr, 24),
            "oi_z_7d":         rolling_zscore(oi_arr, 168),
            "oi_value_roc_4h": pct_change(oi_v_arr, 4),
            "px_roc_4h":       pct_change(px_arr, 4),
        }
        for i, ts in enumerate(ts_sorted):
            feats_by_sym[sym][ts] = {k: f[k][i] for k in f}
        print(f"  {sym}: {len(ts_sorted)} 1h bars with feats")

    # ── Compute decile thresholds globally per feature ────────────────
    print("\nComputing global decile thresholds…")
    pooled_feats = defaultdict(list)
    for sym in SYMBOLS:
        for d in feats_by_sym[sym].values():
            for k, v in d.items():
                if v is not None: pooled_feats[k].append(v)
    thresholds = {}
    for k, vals in pooled_feats.items():
        vals.sort()
        n = len(vals)
        thresholds[k] = {
            "p10": vals[int(n * 0.1)],
            "p90": vals[int(n * 0.9)],
        }
        print(f"  {k:<20}: p10={thresholds[k]['p10']:+.3f}  p90={thresholds[k]['p90']:+.3f}")

    # ── Annotate each trade with its entry-time features ──────────────
    print("\nAnnotating trades with entry-time OI features…")
    annotated = []
    n_missing = 0
    for t in rts:
        ts_dt = datetime.fromisoformat(t["open_ts"].replace("Z", "+00:00") if t["open_ts"].endswith("Z")
                                       else t["open_ts"])
        if ts_dt.tzinfo is None: ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        ts_ms = int(ts_dt.timestamp() * 1000)
        feats = features_at_or_before(ts_ms, feats_by_sym[t["symbol"]])
        if feats is None:
            n_missing += 1
            continue
        annotated.append({**t, "feats": feats})
    print(f"  {len(annotated)} trades annotated, {n_missing} missing (pre-30d window)")

    # ── Define and evaluate filter rules ────────────────────────────────
    print("\n" + "=" * 95)
    print("FILTER RULES")
    print("=" * 95)

    rules = [
        ("F1_skip_long_high_oi_roc_24h",
         lambda t: t["side"] == "LONG" and (t["feats"].get("oi_roc_24h") or 0) >= thresholds["oi_roc_24h"]["p90"]),
        ("F2_skip_long_high_oi_value_roc_4h",
         lambda t: t["side"] == "LONG" and (t["feats"].get("oi_value_roc_4h") or 0) >= thresholds["oi_value_roc_4h"]["p90"]),
        ("F3_skip_short_low_oi_z_7d",
         lambda t: t["side"] == "SHORT" and (t["feats"].get("oi_z_7d") is not None
                                             and t["feats"]["oi_z_7d"] <= thresholds["oi_z_7d"]["p10"])),
        ("F4_skip_long_oi_up_price_up_4h",
         lambda t: t["side"] == "LONG"
                   and (t["feats"].get("oi_roc_4h") or 0) >= thresholds["oi_roc_4h"]["p90"]
                   and (t["feats"].get("px_roc_4h") or 0) > 0),
        ("F5_skip_short_oi_down_price_down_4h",
         lambda t: t["side"] == "SHORT"
                   and (t["feats"].get("oi_roc_4h") or 0) <= thresholds["oi_roc_4h"]["p10"]
                   and (t["feats"].get("px_roc_4h") or 0) < 0),
    ]

    baseline_pnl = sum(t["pnl"] for t in annotated)
    print(f"\nBaseline PnL across {len(annotated)} annotated trades: ${baseline_pnl:+.2f}")
    print(f"\n{'rule':<40} {'n_skip':>6} {'skip_pnl':>10} {'kept_pnl':>10} "
          f"{'delta':>10} {'CI5%':>10} {'CI95%':>10} {'verdict':>10}")
    print("-" * 110)

    results = []
    for name, predicate in rules:
        skipped = [t for t in annotated if predicate(t)]
        kept = [t for t in annotated if not predicate(t)]
        skip_pnl = sum(t["pnl"] for t in skipped)
        kept_pnl = sum(t["pnl"] for t in kept)
        delta = kept_pnl - baseline_pnl
        ci_lo, ci_md, ci_hi = bootstrap_ci_delta([t["pnl"] for t in kept],
                                                  [t["pnl"] for t in skipped]) if skipped else (0, 0, 0)
        verdict = "—"
        if len(skipped) >= 5:
            if ci_lo > 0:
                verdict = "✓PASS"
            elif delta > 0:
                verdict = "weak+"
            else:
                verdict = "weak−"
        else:
            verdict = "thin"
        # side breakdown of skipped trades
        n_long = sum(1 for t in skipped if t["side"] == "LONG")
        n_short = sum(1 for t in skipped if t["side"] == "SHORT")
        results.append({
            "rule": name, "n_skipped": len(skipped),
            "n_skipped_long": n_long, "n_skipped_short": n_short,
            "skipped_pnl": round(skip_pnl, 2),
            "kept_pnl": round(kept_pnl, 2),
            "delta_pnl": round(delta, 2),
            "ci_5_95": [round(ci_lo, 2), round(ci_md, 2), round(ci_hi, 2)],
            "verdict": verdict,
        })
        print(f"  {name:<38} {len(skipped):>6} {skip_pnl:>10.2f} {kept_pnl:>10.2f} "
              f"{delta:>+10.2f} {ci_lo:>+10.2f} {ci_hi:>+10.2f} {verdict:>10}")

    # ── Summary ────────────────────────────────────────────────────────
    passers = [r for r in results if r["verdict"] == "✓PASS"]
    weak_pos = [r for r in results if r["verdict"] == "weak+"]
    print()
    print("=" * 110)
    if passers:
        print(f"✅ {len(passers)} rule(s) pass (bootstrap 5% CI > 0).")
        for r in passers:
            print(f"   → {r['rule']}: skipped {r['n_skipped']} trades, "
                  f"PnL delta ${r['delta_pnl']:+.2f}  CI [${r['ci_5_95'][0]:+.2f}, ${r['ci_5_95'][2]:+.2f}]")
        print()
        print("   Per Chen's directive: do NOT change decision-making yet.")
        print("   Add OI feature logging at trade open. Re-run this analysis on logged-feature trades")
        print("   in 4-6 weeks for direct (no historical reconstruction) validation.")
    elif weak_pos:
        print(f"⚠ {len(weak_pos)} rule(s) directionally helpful but bootstrap CI crosses zero.")
        print("   Insufficient evidence at current trade volume. Log and revisit.")
    else:
        print("❌ No rule shows positive delta with bootstrap CI > 0.")
        print("   At current trade volume, OI-based filtering does not improve realized PnL.")
        print("   Log the features anyway — re-run with more trades in 4-6 weeks.")
    print("=" * 110)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_start": WINDOW_START,
        "n_round_trips": len(rts),
        "n_annotated": len(annotated),
        "n_missing_feats": n_missing,
        "thresholds": thresholds,
        "baseline_pnl": round(baseline_pnl, 2),
        "results": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
