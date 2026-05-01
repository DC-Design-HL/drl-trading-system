#!/usr/bin/env python3
"""
Minimum Viable Test: does whale data have ANY exploitable forward-return
signal in the strictest "should-work" regime?

Per the architecture-review agent's spec: if the cleanest, most-tagged
subset of the signal on the chain with the most data fails this test,
no amount of cross-chain breadth or paid data will fix it.

Subset criteria (all must hold):
  * action ∈ {EXCHANGE_DEPOSIT, EXCHANGE_WITHDRAWAL,
              TOKEN_TO_EXCHANGE, TOKEN_FROM_EXCHANGE}
  * value_eth ≥ 100  (~$300k+)
  * Source wallet ∈ {galaxy_digital, jump_trading, smart_money_whale_1}
    OR destination exchange-cold (counterparty in {binance, coinbase, kraken})
  * 2024-01-01 → 2026-04-30 window

Forward returns at 15m / 1h / 4h horizons (3 tests).
Stats: hour-of-day baseline drift correction, non-overlap subsample
correction (events ≥ horizon apart), one-sample t-test, Bonferroni
α = 0.05/3 = 0.0167.

Decision rule:
  * REJECT H0 (signal exists): any horizon with p < 0.0167 AND |mean| ≥ 15 bps
  * FAIL TO REJECT (no signal): all horizons fail at least one criterion

Output:
  data/training/whale_mvt_results.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
WHALE_DIR = REPO / "data" / "whale_behavior" / "eth"
KLINE_15M = REPO / "data" / "training" / "eth_spot_15m_cache.jsonl"
KLINE_1H = REPO / "data" / "training" / "eth_spot_1h_cache.jsonl"
OUT = REPO / "data" / "training" / "whale_mvt_results.json"

WINDOW_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 30, tzinfo=timezone.utc)

EXCHANGE_ACTIONS = {
    "EXCHANGE_DEPOSIT", "EXCHANGE_WITHDRAWAL",
    "TOKEN_TO_EXCHANGE", "TOKEN_FROM_EXCHANGE",
}
SMART_MONEY_WALLETS = {"galaxy_digital", "jump_trading", "smart_money_whale_1"}
EXCHANGE_COUNTERPARTIES = {"binance", "coinbase", "kraken", "okx", "bitfinex"}
MIN_ETH = 100.0


def load_klines(path: Path) -> dict[int, float]:
    cache = {}
    if not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            try:
                ts, close = json.loads(line)
                cache[int(ts)] = float(close)
            except Exception:
                pass
    return cache


def load_whale_events_strict():
    """Apply the strictest subset criteria from the MVT spec."""
    out = []
    if not WHALE_DIR.exists():
        return out
    for f in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet_name = f.stem
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts is None:
                    continue
                try:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                except Exception:
                    continue
                if dt < WINDOW_START or dt > WINDOW_END:
                    continue
                action = (d.get("action") or "").upper()
                if action not in EXCHANGE_ACTIONS:
                    continue
                value = float(d.get("value_eth") or 0)
                if value < MIN_ETH:
                    continue
                # Pass either smart-money source OR exchange counterparty
                to_type = (d.get("to_type") or "").lower()
                is_smart_money = wallet_name in SMART_MONEY_WALLETS
                is_exchange_dest = any(ex in to_type for ex in EXCHANGE_COUNTERPARTIES)
                if not (is_smart_money or is_exchange_dest):
                    continue
                out.append({
                    "ts": dt, "wallet": wallet_name,
                    "value": value,
                    "action": action,
                    "to_type": to_type,
                    "direction": (d.get("direction") or "").lower(),
                    "is_smart_money": is_smart_money,
                    "is_exchange_dest": is_exchange_dest,
                })
    out.sort(key=lambda x: x["ts"])
    return out


def hour_baseline(klines_15m: dict[int, float]) -> dict[float, dict[int, float]]:
    """Per-(horizon, hour-of-day) mean drift in pct."""
    horizons = (0.25, 1, 4)
    out = {h: defaultdict(list) for h in horizons}
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
    return {h: {hr: (mean(vals) if vals else 0.0) for hr, vals in d.items()} for h, d in out.items()}


def fwd_ret(event_ts: datetime, hours: float, klines: dict[int, float]) -> float | None:
    interval_ms = 60 * 60 * 1000 if hours >= 1 else 15 * 60 * 1000
    snap = (int(event_ts.timestamp() * 1000) // interval_ms) * interval_ms
    fwd = snap + int(hours * 3600 * 1000)
    p_now = klines.get(snap)
    p_fwd = klines.get(fwd)
    if p_now is None or p_fwd is None or p_now <= 0:
        return None
    return math.log(p_fwd / p_now) * 100


def t_test(values: list[float]):
    n = len(values)
    if n < 5:
        return None
    m = mean(values)
    sd = stdev(values) if n > 1 else 0
    if sd == 0:
        return (m, 0.0 if m != 0 else 1.0)
    se = sd / math.sqrt(n)
    if se == 0:
        return (m, 1.0)
    t = m / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (m, p)


def main():
    print(f"Window: {WINDOW_START.date()} → {WINDOW_END.date()}")
    print(f"Subset: action∈{EXCHANGE_ACTIONS}, ≥{MIN_ETH} ETH, "
          f"smart-money OR exchange-counterparty\n")

    print("Loading whale events with strict filter ...")
    events = load_whale_events_strict()
    print(f"  {len(events)} events match the strict subset")
    if events:
        print(f"  smart-money sourced: {sum(1 for e in events if e['is_smart_money'])}")
        print(f"  exchange counterparty: {sum(1 for e in events if e['is_exchange_dest'])}")

    print("\nLoading klines ...")
    klines_15m = load_klines(KLINE_15M)
    klines_1h = load_klines(KLINE_1H)
    print(f"  {len(klines_15m)} 15m candles; {len(klines_1h)} 1h candles")

    if len(events) < 30:
        print(f"\n❌ INSUFFICIENT EVENTS ({len(events)}) — cannot run statistical test.")
        print("   The strictest subset is too sparse. Lower thresholds or different definition needed.")
        OUT.write_text(json.dumps({"verdict": "insufficient_events", "n": len(events)}, indent=2))
        return 0

    print("\nComputing hour-of-day baseline ...")
    baseline = hour_baseline(klines_15m)

    horizons = [(0.25, "15m", klines_15m), (1, "1h", klines_1h), (4, "4h", klines_1h)]
    BONFERRONI_ALPHA = 0.05 / len(horizons)
    print(f"\nBonferroni α (3 tests): {BONFERRONI_ALPHA:.4f}\n")

    results = {}
    print(f"{'horizon':<8} {'n':>5} {'iid_mean_bps':>13} {'iid_p':>10} "
          f"{'no_n':>5} {'no_mean_bps':>13} {'no_p':>10} {'verdict':>15}")
    print("-" * 90)
    for h_val, h_label, klines in horizons:
        rets_iid = []
        rets_no = []
        last_kept = None
        for e in events:
            r = fwd_ret(e["ts"], h_val, klines)
            if r is None:
                continue
            r_corr = r - baseline[h_val].get(e["ts"].hour, 0.0)
            rets_iid.append(r_corr)
            if last_kept is None or (e["ts"] - last_kept).total_seconds() >= h_val * 3600:
                rets_no.append(r_corr)
                last_kept = e["ts"]

        iid = t_test(rets_iid)
        no = t_test(rets_no)

        # Decision per horizon
        verdict = "—"
        if no is not None:
            no_mean_bps = no[0] * 100
            if no[1] < BONFERRONI_ALPHA and abs(no_mean_bps) >= 15:
                verdict = "✓SIG+EFFECT"
            elif no[1] < BONFERRONI_ALPHA:
                verdict = "✓sig small"
            elif no[1] < 0.05:
                verdict = "p<.05 only"
            else:
                verdict = "null"

        results[h_label] = {
            "n_iid": len(rets_iid),
            "iid_mean_bps": round(iid[0] * 100, 2) if iid else None,
            "iid_p": round(iid[1], 4) if iid else None,
            "n_non_overlap": len(rets_no),
            "non_overlap_mean_bps": round(no[0] * 100, 2) if no else None,
            "non_overlap_p": round(no[1], 4) if no else None,
            "verdict": verdict,
        }
        print(f"  {h_label:<8} {len(rets_iid):>5} "
              f"{(iid[0]*100 if iid else 0):>13.2f} "
              f"{(iid[1] if iid else 0):>10.4f} "
              f"{len(rets_no):>5} "
              f"{(no[0]*100 if no else 0):>13.2f} "
              f"{(no[1] if no else 0):>10.4f} "
              f"{verdict:>15}")

    # Global decision
    any_significant = any(
        r["non_overlap_p"] is not None
        and r["non_overlap_p"] < BONFERRONI_ALPHA
        and abs(r["non_overlap_mean_bps"] or 0) >= 15
        for r in results.values()
    )

    print()
    print("=" * 90)
    if any_significant:
        print("✅ REJECT H0 — Whale signal exists in the strictest subset.")
        print("   Action: invest in CryptoQuant Advanced ($39/mo) targeted at this regime.")
        print("   Identify the surviving (action × horizon) combo and build the live filter around it.")
    else:
        print("❌ FAIL TO REJECT — No whale signal in the strictest subset.")
        print("   Action: pause whale work permanently. Empirical case closed.")
        print("   Re-allocate effort to: SL/exit leakage fixes (-$391 per memo),")
        print("   structure-detector cleanup, SGFilter retraining once trade data accumulates.")
    print("=" * 90)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subset_criteria": {
            "actions": list(EXCHANGE_ACTIONS),
            "min_eth": MIN_ETH,
            "smart_money_wallets": list(SMART_MONEY_WALLETS),
            "exchange_counterparties": list(EXCHANGE_COUNTERPARTIES),
            "window_start": WINDOW_START.isoformat(),
            "window_end": WINDOW_END.isoformat(),
        },
        "n_events": len(events),
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "horizons": results,
        "verdict": "REJECT_H0_SIGNAL_EXISTS" if any_significant else "FAIL_TO_REJECT_NO_SIGNAL",
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
