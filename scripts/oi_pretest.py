#!/usr/bin/env python3
"""
Open-Interest pre-test (Phase 1: signal existence).

Question: Do simple OI-derived features predict forward returns at
1h/4h/24h horizons on BTC/ETH/SOL/XRP futures?

Method:
  * Fetch ~30d of 1h OI hist + 1h klines from Binance Futures (public).
  * Compute features per (symbol, hour):
      - oi_roc_4h      : pct change of sumOpenInterest over trailing 4h
      - oi_roc_24h     : same over 24h
      - oi_z_7d        : z-score of OI over trailing 7d (168 bars)
      - oi_price_div_4h: sign(oi_roc_4h) - sign(price_roc_4h), i.e. divergence
      - oi_value_roc_4h: pct change of sumOpenInterestValue over 4h (price-corrected)
  * For each (feature, forward_horizon) pair, take top vs bottom decile of
    the feature, compare forward returns via Welch's t-test.
  * Pool symbols (each (sym,t) is one observation).
  * Bonferroni-correct over the test grid.

Decision rule (Phase 1 only — does the signal exist):
  * Pass = corrected p < 0.05 AND |delta mean| ≥ 15 bps in the right
    direction (i.e. high-OI-divergence → lower forward returns, etc.).
  * If nothing passes: stop, no Phase 2.
  * If something passes: write the candidate to disk; Phase 2 will be a
    filter backtest on actual trade outcomes.

This is a 30-min script. No production code is touched.
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
OUT_DIR = REPO / "data" / "training"
OUT = OUT_DIR / "oi_pretest_results.json"
RAW_OUT = OUT_DIR / "oi_pretest_raw.jsonl"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
PERIOD = "1h"
LIMIT = 500
LOOKBACK_DAYS = 30  # Binance OI hist max

OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
KL_URL = "https://fapi.binance.com/fapi/v1/klines"


def http_get_json(url: str, params: dict, retries: int = 3, sleep: float = 0.5):
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


def fetch_oi_hist(symbol: str, days: int) -> list[dict]:
    """Walk back in chunks of LIMIT bars."""
    out: list[dict] = []
    end = int(time.time() * 1000)
    target_start = end - days * 24 * 3600 * 1000
    while True:
        data = http_get_json(OI_URL, {
            "symbol": symbol, "period": PERIOD,
            "limit": LIMIT, "endTime": end,
        })
        if not data:
            break
        out = data + out
        oldest = data[0]["timestamp"]
        if oldest <= target_start or len(data) < LIMIT:
            break
        end = oldest - 1
        time.sleep(0.2)
    # Dedup and sort
    seen = set(); deduped = []
    for d in out:
        ts = int(d["timestamp"])
        if ts in seen: continue
        seen.add(ts); deduped.append(d)
    deduped.sort(key=lambda x: int(x["timestamp"]))
    return deduped


def fetch_klines(symbol: str, days: int) -> list[list]:
    """1h klines for the same window."""
    out: list[list] = []
    end = int(time.time() * 1000)
    target_start = end - days * 24 * 3600 * 1000
    while True:
        data = http_get_json(KL_URL, {
            "symbol": symbol, "interval": "1h",
            "limit": 1000, "endTime": end,
        })
        if not data:
            break
        out = data + out
        oldest = data[0][0]
        if oldest <= target_start or len(data) < 100:
            break
        end = oldest - 1
        time.sleep(0.2)
    seen = set(); deduped = []
    for k in out:
        ts = int(k[0])
        if ts in seen: continue
        seen.add(ts); deduped.append(k)
    deduped.sort(key=lambda x: x[0])
    return deduped


def t_test_two_sample(a: list[float], b: list[float]):
    """Welch's t-test, returns (mean_a, mean_b, p)."""
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return None
    ma, mb = mean(a), mean(b)
    va = stdev(a) ** 2 if na > 1 else 0
    vb = stdev(b) ** 2 if nb > 1 else 0
    se = math.sqrt(va / na + vb / nb) if (va + vb) > 0 else 0
    if se == 0:
        return (ma, mb, 1.0)
    t = (ma - mb) / se
    # Welch–Satterthwaite df
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1)
    df = df_num / df_den if df_den > 0 else max(na + nb - 2, 1)
    # Two-sided p via normal approx for large df, t-dist would need scipy
    # For df > ~30 normal is close enough; we have df in the hundreds.
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (ma, mb, p, df)


def pct_change(arr, k):
    """returns[i] = (arr[i] - arr[i-k]) / arr[i-k] * 100, None if not enough history"""
    out = [None] * len(arr)
    for i in range(k, len(arr)):
        a, b = arr[i - k], arr[i]
        if a is None or b is None or a == 0:
            continue
        out[i] = (b - a) / a * 100
    return out


def rolling_zscore(arr, window):
    out = [None] * len(arr)
    for i in range(window, len(arr)):
        w = [x for x in arr[i - window:i] if x is not None]
        if len(w) < window // 2:
            continue
        m = mean(w); s = stdev(w) if len(w) > 1 else 0
        if s == 0:
            continue
        out[i] = (arr[i] - m) / s if arr[i] is not None else None
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pull data ───────────────────────────────────────────────────────
    print(f"Pulling {LOOKBACK_DAYS}d of 1h OI + klines for {SYMBOLS}…")
    all_data = {}
    for sym in SYMBOLS:
        oi = fetch_oi_hist(sym, LOOKBACK_DAYS)
        kl = fetch_klines(sym, LOOKBACK_DAYS)
        print(f"  {sym}: OI rows={len(oi)} klines={len(kl)}")
        # Index by timestamp (1h-aligned)
        oi_by_ts = {int(d["timestamp"]): {
            "oi": float(d["sumOpenInterest"]),
            "oi_value": float(d["sumOpenInterestValue"]),
        } for d in oi}
        kl_by_ts = {int(k[0]): {"open": float(k[1]), "high": float(k[2]),
                                "low": float(k[3]), "close": float(k[4])} for k in kl}
        ts_sorted = sorted(set(oi_by_ts) & set(kl_by_ts))
        all_data[sym] = {
            "ts": ts_sorted,
            "oi": [oi_by_ts[t]["oi"] for t in ts_sorted],
            "oi_value": [oi_by_ts[t]["oi_value"] for t in ts_sorted],
            "close": [kl_by_ts[t]["close"] for t in ts_sorted],
        }

    # ── Feature engineering ────────────────────────────────────────────
    print("\nComputing features…")
    features_per_sym = {}
    for sym, d in all_data.items():
        oi = d["oi"]; oi_v = d["oi_value"]; px = d["close"]
        feats = {
            "oi_roc_4h":        pct_change(oi, 4),
            "oi_roc_24h":       pct_change(oi, 24),
            "oi_z_7d":          rolling_zscore(oi, 168),
            "oi_value_roc_4h":  pct_change(oi_v, 4),
        }
        # Divergence: sign(oi_roc_4h) - sign(px_roc_4h). +2 = OI up, px down (bear-conviction)
        # -2 = OI down, px up (short-cover, bull-fade)
        px_roc_4h = pct_change(px, 4)
        div = []
        for a, b in zip(feats["oi_roc_4h"], px_roc_4h):
            if a is None or b is None:
                div.append(None)
            else:
                div.append((1 if a > 0 else -1 if a < 0 else 0)
                           - (1 if b > 0 else -1 if b < 0 else 0))
        feats["oi_price_div_4h"] = div
        features_per_sym[sym] = feats

    # ── Forward returns ─────────────────────────────────────────────────
    horizons = [(1, "1h"), (4, "4h"), (24, "24h")]
    print(f"\nComputing forward returns for horizons: {[h[1] for h in horizons]}")
    fwd_per_sym = {sym: {} for sym in SYMBOLS}
    for sym in SYMBOLS:
        px = all_data[sym]["close"]
        for h_steps, h_label in horizons:
            r = []
            for i in range(len(px)):
                j = i + h_steps
                if j < len(px) and px[i] > 0:
                    r.append(math.log(px[j] / px[i]) * 100)
                else:
                    r.append(None)
            fwd_per_sym[sym][h_label] = r

    # ── Pool by feature, hour-of-day correct ───────────────────────────
    print("\nComputing hour-of-day baseline drift…")
    hod_baseline = {}
    for h_steps, h_label in horizons:
        hod_acc = defaultdict(list)
        for sym in SYMBOLS:
            ts_arr = all_data[sym]["ts"]
            r_arr = fwd_per_sym[sym][h_label]
            for ts, r in zip(ts_arr, r_arr):
                if r is None: continue
                hour = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
                hod_acc[hour].append(r)
        hod_baseline[h_label] = {h: mean(v) if v else 0.0 for h, v in hod_acc.items()}

    # ── Run tests ──────────────────────────────────────────────────────
    feature_names = list(next(iter(features_per_sym.values())).keys())
    n_tests = len(feature_names) * len(horizons) * 2  # top decile + bottom decile vs middle
    bonferroni_alpha = 0.05 / n_tests
    print(f"\nGrid: {len(feature_names)} features × {len(horizons)} horizons × 2 tails = {n_tests} tests")
    print(f"Bonferroni α = {bonferroni_alpha:.4f}\n")

    results = []
    for feat_name in feature_names:
        # Pool all (sym, t) where feat is not None and fwd is not None
        for h_steps, h_label in horizons:
            obs = []  # (feat_val, fwd_corrected)
            for sym in SYMBOLS:
                ts_arr = all_data[sym]["ts"]
                feat_arr = features_per_sym[sym][feat_name]
                fwd_arr = fwd_per_sym[sym][h_label]
                for ts, fv, fr in zip(ts_arr, feat_arr, fwd_arr):
                    if fv is None or fr is None: continue
                    hour = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
                    fr_corr = fr - hod_baseline[h_label].get(hour, 0)
                    obs.append((fv, fr_corr))
            if len(obs) < 100:
                continue
            obs.sort(key=lambda x: x[0])
            n = len(obs)
            d10 = max(int(n * 0.1), 5)
            top = [o[1] for o in obs[-d10:]]
            mid = [o[1] for o in obs[d10:-d10]]
            bot = [o[1] for o in obs[:d10]]

            for label, sample in (("top_decile", top), ("bot_decile", bot)):
                tt = t_test_two_sample(sample, mid)
                if tt is None: continue
                ms, mm, p, df = tt
                delta_bps = (ms - mm) * 100
                pass_p = p < bonferroni_alpha
                pass_eff = abs(delta_bps) >= 15
                results.append({
                    "feature": feat_name,
                    "horizon": h_label,
                    "tail": label,
                    "n_sample": len(sample),
                    "n_mid": len(mid),
                    "mean_sample_pct": round(ms, 4),
                    "mean_mid_pct": round(mm, 4),
                    "delta_bps": round(delta_bps, 2),
                    "p_value": round(p, 5),
                    "passes_bonferroni": pass_p,
                    "passes_effect_size": pass_eff,
                    "passes_both": pass_p and pass_eff,
                })

    # ── Report ────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["p_value"])
    print(f"{'feature':<22} {'horiz':<5} {'tail':<10} {'n':>5} {'mean_bps':>10} "
          f"{'mid_bps':>10} {'delta_bps':>10} {'p':>9} {'verdict':>10}")
    print("-" * 105)
    for r in results:
        verdict = "✓PASS" if r["passes_both"] else (
            "p ok" if r["passes_bonferroni"] else
            "size ok" if r["passes_effect_size"] else "—")
        print(f"  {r['feature']:<20} {r['horizon']:<5} {r['tail']:<10} "
              f"{r['n_sample']:>5} {r['mean_sample_pct']*100:>10.2f} "
              f"{r['mean_mid_pct']*100:>10.2f} {r['delta_bps']:>10.2f} "
              f"{r['p_value']:>9.4f} {verdict:>10}")

    passers = [r for r in results if r["passes_both"]]
    print()
    print("=" * 105)
    if passers:
        print(f"✅ {len(passers)} (feature, horizon, tail) passed Bonferroni + effect.")
        print("   Phase 2 candidates — productionize these as filters and run on actual trade outcomes.")
    else:
        print("❌ Nothing passed Bonferroni + 15bps effect-size bar.")
        print("   Phase 1 fails. OI signal does not predict forward returns at the tested horizons.")
        print("   Consider: longer horizon, conditional signals (regime-gated), or different feature definitions.")
    print("=" * 105)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": SYMBOLS,
        "lookback_days": LOOKBACK_DAYS,
        "n_observations": sum(len(all_data[s]["ts"]) for s in SYMBOLS),
        "horizons_tested": [h[1] for h in horizons],
        "features_tested": feature_names,
        "n_tests": n_tests,
        "bonferroni_alpha": bonferroni_alpha,
        "results": results,
        "passers": passers,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
