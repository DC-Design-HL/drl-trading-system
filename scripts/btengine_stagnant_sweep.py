#!/usr/bin/env python3
"""STAGNANT_EXIT band sweep — find a high-confidence band/timeout combo.

Live observation (May 3): 3 of 4 trades since reset hit STAGNANT_EXIT.
Pre-reset, STAGNANT_EXIT was -$197/51 trades = -$3.86/trade. Worst
exit reason among non-canary'd reasons.

Tests on 60 days × 4 symbols × structure_first_v3 + symbol_blocklist
+ adx + usdtd + ext_pos_news (the live filter chain that has data).

Variant grid:
  band × timeout. Current production: [-1.0%, +0.5%] × 6h.

Variants:
  1. baseline_current     ([-1.0, +0.5], 6h)
  2. tighter_loss         ([-0.5, +0.5], 6h)   — exit losers earlier
  3. tighter_both         ([-0.5, +0.3], 6h)   — exit any plateau
  4. shorter_window       ([-1.0, +0.5], 4h)   — give less time
  5. longer_window        ([-1.0, +0.5], 8h)   — more time for recovery
  6. tighter_loss_short   ([-0.5, +0.5], 4h)
  7. disabled             (effectively never)

Output: data/training/btengine_stagnant_sweep.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btengine.config import BacktestConfig
from src.btengine.runner import BacktestRunner

OUT = REPO / "data" / "training" / "btengine_stagnant_sweep.json"

logging.basicConfig(level=logging.WARNING)


def _base(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "stag_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "guards": {
            "enabled": ["symbol_blocklist", "adx", "usdtd", "ext_pos_news"],
            "params": {"adx": {"min_adx": 25, "max_adx": 50}},
        },
        "sizing": {"type": "fixed_notional", "usd": 1500, "max_concurrent": 4},
        "exits": {
            "partial_tp": [[1.0, 0.40], [2.0, 0.35]],
            "stagnant_hours": 6,
            "stagnant_pct_min": -0.010,
            "stagnant_pct_max": 0.005,
        },
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": "runs/${run_id}"},
    }


VARIANTS = [
    {"name": "baseline_current", "stagnant": {"hours": 6, "min": -0.010, "max": 0.005}},
    {"name": "tighter_loss",     "stagnant": {"hours": 6, "min": -0.005, "max": 0.005}},
    {"name": "tighter_both",     "stagnant": {"hours": 6, "min": -0.005, "max": 0.003}},
    {"name": "shorter_window",   "stagnant": {"hours": 4, "min": -0.010, "max": 0.005}},
    {"name": "longer_window",    "stagnant": {"hours": 8, "min": -0.010, "max": 0.005}},
    {"name": "tighter_loss_short", "stagnant": {"hours": 4, "min": -0.005, "max": 0.005}},
    {"name": "disabled",         "stagnant": {"hours": 999, "min": -0.001, "max": 0.001}},
]


def run_variant(base, v):
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"stag_{v['name']}"
    cfg_dict["exits"]["stagnant_hours"] = v["stagnant"]["hours"]
    cfg_dict["exits"]["stagnant_pct_min"] = v["stagnant"]["min"]
    cfg_dict["exits"]["stagnant_pct_max"] = v["stagnant"]["max"]
    cfg_dict["output"]["dir"] = f"runs/{cfg_dict['run_id']}"
    cfg = BacktestConfig.from_dict(cfg_dict)
    t0 = time.time()
    s = BacktestRunner(cfg).run()
    return {
        "variant": v["name"], "stagnant": v["stagnant"],
        "n_full_closes": s["n_full_closes"], "wins": s["wins"],
        "win_rate_pct": s["win_rate_pct"],
        "total_pnl_usd": s["total_pnl_usd"],
        "max_dd_pct": s["max_dd_pct"],
        "by_reason": {k: v["n"] for k, v in s["by_reason"].items()},
        "blocked_total": s["blocked_total"],
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    base = _base(days)
    print(f"\nSTAGNANT band sweep — {days} days, {len(VARIANTS)} variants")
    print(f"Window: {base['window']['start']} → {base['window']['end']}\n" + "="*100)

    results = []
    for v in VARIANTS:
        print(f"\n[{v['name']}] band=[{v['stagnant']['min']*100:+.1f}%, {v['stagnant']['max']*100:+.1f}%] @ {v['stagnant']['hours']}h", flush=True)
        try:
            r = run_variant(base, v)
            print(f"  -> n={r['n_full_closes']} WR={r['win_rate_pct']:.1f}% "
                  f"pnl=${r['total_pnl_usd']:+.2f} stagnant={r['by_reason'].get('stagnant',0)} "
                  f"({r['elapsed_sec']:.0f}s)", flush=True)
            results.append(r)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            results.append({"variant": v["name"], "error": str(exc)})

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    valid = [r for r in results if "error" not in r]
    base_pnl = next((r["total_pnl_usd"] for r in valid if r["variant"] == "baseline_current"), 0)
    print(f"{'variant':<24} {'n':>4} {'WR%':>6} {'pnl':>10} {'stag':>5} {'maxDD%':>7} {'Δ vs base':>10}")
    print("-" * 100)
    for r in valid:
        delta = r["total_pnl_usd"] - base_pnl
        stag = r["by_reason"].get("stagnant", 0)
        print(f"  {r['variant']:<22} {r['n_full_closes']:>4} {r['win_rate_pct']:>5.1f}% "
              f"${r['total_pnl_usd']:>+9.2f} {stag:>5} {r['max_dd_pct']:>6.1f}% "
              f"${delta:>+9.2f}")

    if valid:
        best = max(valid, key=lambda x: x["total_pnl_usd"])
        print(f"\nBest: {best['variant']} pnl ${best['total_pnl_usd']:+.2f} "
              f"(Δ ${best['total_pnl_usd']-base_pnl:+.2f}) "
              f"band=[{best['stagnant']['min']*100:+.1f}%, {best['stagnant']['max']*100:+.1f}%] @ {best['stagnant']['hours']}h")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window_days": days, "window_start": base["window"]["start"],
        "window_end": base["window"]["end"],
        "variants": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
