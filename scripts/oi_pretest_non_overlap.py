#!/usr/bin/env python3
"""
Open-Interest pre-test, non-overlap robustness check.

Phase 1 of oi_pretest.py reported 5 (feature, horizon, tail) combos passing
Bonferroni at α=0.0017 with 24h-horizon tests. But hourly observations
within a 24h window are autocorrelated — we cannot treat 720 hours as
720 independent samples for a 24h forward-return test.

This script re-tests the 5 passers with **non-overlapping subsamples**:
  * For a 24h-horizon test: keep 1 obs every 24h per symbol
  * For a 4h-horizon test:  keep 1 obs every 4h per symbol
  * Pool across symbols, then run the same top/bottom decile vs middle test

If the signal survives at α=0.0017 (Bonferroni-corrected over 5 tests),
the OI signal is statistically real and Phase 2 (actual trade-filter
backtest) is justified.

If the signal collapses, the Phase 1 result was an autocorrelation artifact
and we stop.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "training" / "oi_pretest_non_overlap.json"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
LOOKBACK_DAYS = 30
OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
KL_URL = "https://fapi.binance.com/fapi/v1/klines"

# The 5 passers from oi_pretest.py
CANDIDATES = [
    ("oi_value_roc_4h", "24h", "top"),   # delta -103 bps, p<.0001
    ("oi_roc_24h",      "24h", "top"),   # delta -100 bps
    ("oi_z_7d",         "24h", "bot"),   # delta +91 bps
    ("oi_roc_4h",       "24h", "top"),   # delta -73 bps
    ("oi_roc_24h",      "4h",  "top"),   # delta -20 bps
]


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


def t_test_two_sample(a, b):
    na, nb = len(a), len(b)
    if na < 5 or nb < 5: return None
    ma, mb = mean(a), mean(b)
    va = stdev(a) ** 2 if na > 1 else 0
    vb = stdev(b) ** 2 if nb > 1 else 0
    se = math.sqrt(va / na + vb / nb) if (va + vb) > 0 else 0
    if se == 0: return (ma, mb, 1.0)
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (ma, mb, p)


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


def main():
    print(f"Re-pulling data for non-overlap test ({LOOKBACK_DAYS}d, {len(SYMBOLS)} symbols)…")
    all_data = {}
    for sym in SYMBOLS:
        oi = fetch_oi_hist(sym, LOOKBACK_DAYS)
        kl = fetch_klines(sym, LOOKBACK_DAYS)
        oi_by_ts = {int(d["timestamp"]): {"oi": float(d["sumOpenInterest"]),
                                          "oi_value": float(d["sumOpenInterestValue"])} for d in oi}
        kl_by_ts = {int(k[0]): float(k[4]) for k in kl}
        ts_sorted = sorted(set(oi_by_ts) & set(kl_by_ts))
        all_data[sym] = {
            "ts": ts_sorted,
            "oi": [oi_by_ts[t]["oi"] for t in ts_sorted],
            "oi_value": [oi_by_ts[t]["oi_value"] for t in ts_sorted],
            "close": [kl_by_ts[t] for t in ts_sorted],
        }

    # Compute features
    features_per_sym = {}
    for sym, d in all_data.items():
        oi = d["oi"]; oi_v = d["oi_value"]
        features_per_sym[sym] = {
            "oi_roc_4h":       pct_change(oi, 4),
            "oi_roc_24h":      pct_change(oi, 24),
            "oi_z_7d":         rolling_zscore(oi, 168),
            "oi_value_roc_4h": pct_change(oi_v, 4),
        }

    # Hour-of-day baseline (use FULL hourly sample, that's still the best estimate)
    hod_baseline = {}
    for h_steps, h_label in [(1, "1h"), (4, "4h"), (24, "24h")]:
        hod_acc = defaultdict(list)
        for sym in SYMBOLS:
            ts = all_data[sym]["ts"]; px = all_data[sym]["close"]
            for i, t in enumerate(ts):
                j = i + h_steps
                if j < len(px) and px[i] > 0:
                    r = math.log(px[j] / px[i]) * 100
                    hour = datetime.fromtimestamp(t / 1000, tz=timezone.utc).hour
                    hod_acc[hour].append(r)
        hod_baseline[h_label] = {h: mean(v) if v else 0.0 for h, v in hod_acc.items()}

    # ── Non-overlap test for each candidate ────────────────────────────
    bonferroni_alpha = 0.05 / len(CANDIDATES)
    print(f"\nNon-overlap test on {len(CANDIDATES)} candidates, "
          f"Bonferroni α = {bonferroni_alpha:.4f} (over {len(CANDIDATES)} tests)\n")
    print(f"{'feature':<20} {'horiz':<5} {'tail':<5} "
          f"{'n_no':>5} {'mean_bps':>10} {'mid_bps':>10} {'delta_bps':>10} "
          f"{'p_no':>9} {'verdict':>10}")
    print("-" * 95)

    results = []
    for feat_name, h_label, tail in CANDIDATES:
        h_steps = {"1h": 1, "4h": 4, "24h": 24}[h_label]

        # Pool with non-overlap subsample: stride = h_steps
        obs = []
        for sym in SYMBOLS:
            ts = all_data[sym]["ts"]
            feat = features_per_sym[sym][feat_name]
            px = all_data[sym]["close"]
            # Stride through, only keep observations h_steps apart
            i = 0
            while i < len(ts):
                j = i + h_steps
                if j < len(px) and feat[i] is not None and px[i] > 0:
                    fr = math.log(px[j] / px[i]) * 100
                    hour = datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc).hour
                    fr_corr = fr - hod_baseline[h_label].get(hour, 0)
                    obs.append((feat[i], fr_corr))
                i += h_steps  # non-overlap

        if len(obs) < 30:
            print(f"  {feat_name:<18} {h_label:<5} {tail:<5} "
                  f"{len(obs):>5} -- INSUFFICIENT --")
            continue

        obs.sort(key=lambda x: x[0])
        n = len(obs)
        d10 = max(int(n * 0.1), 3)
        if tail == "top":
            sample = [o[1] for o in obs[-d10:]]
        else:
            sample = [o[1] for o in obs[:d10]]
        mid = [o[1] for o in obs[d10:-d10]]

        tt = t_test_two_sample(sample, mid)
        if tt is None:
            print(f"  {feat_name:<18} {h_label:<5} {tail:<5} t-test failed")
            continue
        ms, mm, p = tt
        delta_bps = (ms - mm) * 100
        passes = p < bonferroni_alpha and abs(delta_bps) >= 15
        verdict = "✓PASS" if passes else ("p ok" if p < bonferroni_alpha else
                                          "size ok" if abs(delta_bps) >= 15 else "—")
        results.append({
            "feature": feat_name, "horizon": h_label, "tail": tail,
            "n_non_overlap": len(sample), "n_mid": len(mid),
            "mean_sample_pct": round(ms, 4), "mean_mid_pct": round(mm, 4),
            "delta_bps": round(delta_bps, 2),
            "p_value": round(p, 5),
            "passes": passes,
        })
        print(f"  {feat_name:<18} {h_label:<5} {tail:<5} "
              f"{len(sample):>5} {ms*100:>10.2f} {mm*100:>10.2f} "
              f"{delta_bps:>10.2f} {p:>9.4f} {verdict:>10}")

    passers = [r for r in results if r["passes"]]
    print()
    print("=" * 95)
    if passers:
        print(f"✅ {len(passers)} candidate(s) survive non-overlap test at α={bonferroni_alpha:.4f}.")
        print("   Real signal — proceed to Phase 2 (filter backtest on actual trades).")
    else:
        print("❌ NO candidates survive non-overlap test.")
        print("   The Phase 1 result was an autocorrelation artifact, not a real signal.")
        print("   Hourly samples within a 24h-horizon test are NOT independent.")
    print("=" * 95)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bonferroni_alpha": bonferroni_alpha,
        "results": results,
        "passers": passers,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
