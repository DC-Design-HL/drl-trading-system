#!/usr/bin/env python3
"""TP-variant sweep on the new btengine framework.

Runs the structure_first_v3 strategy with the current production guard
chain across multiple TP configurations on a longer historical window
than the historical-trade backtest could reach.

Variants tested:
  * baseline                     (no TP override, ATR-floor 3× as in live)
  * tp_multiplier 0.5            (half of ATR-floor TP)
  * tp_multiplier 0.7
  * tp_pct=0.015                 (1.5% absolute)
  * tp_pct=0.020                 (2.0% absolute)
  * tp_pct=0.025                 (2.5% absolute)
  * SHORT-only tp_pct=0.020      (asymmetric variant)
  * conditional tp_pct=0.020 when conf<0.75

Output: prints comparison table, writes JSON to data/training/btengine_tp_sweep.json.

Usage:
    python3 scripts/btengine_tp_sweep.py [days]   (default: 60)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btengine.config import BacktestConfig
from src.btengine.runner import BacktestRunner

OUT = REPO / "data" / "training" / "btengine_tp_sweep.json"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")


def _base_config(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "tp_sweep_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "guards": {
            "enabled": ["symbol_blocklist", "adx"],
            "params": {"adx": {"min_adx": 20, "max_adx": 60}},
        },
        "sizing": {"type": "fixed_notional", "usd": 1500, "max_concurrent": 4},
        "exits": {"partial_tp": [[1.0, 0.40], [2.0, 0.35]], "extras": {}},
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": "runs/${run_id}"},
    }


VARIANTS = [
    {"name": "baseline",         "extras": {}},
    {"name": "tp_mult_0.5",      "extras": {"tp_multiplier": 0.5}},
    {"name": "tp_mult_0.7",      "extras": {"tp_multiplier": 0.7}},
    {"name": "tp_pct_1.5",       "extras": {"tp_pct_override": 0.015}},
    {"name": "tp_pct_2.0",       "extras": {"tp_pct_override": 0.020}},
    {"name": "tp_pct_2.5",       "extras": {"tp_pct_override": 0.025}},
    {"name": "tp_pct_2.0_short", "extras": {"tp_pct_override": 0.020, "short_only_tp_override": True}},
    {"name": "tp_pct_2.0_lowconf", "extras": {"tp_pct_override": 0.020, "conditional_tp_max_confidence": 0.75}},
]


def run_variant(base: dict, variant: dict) -> dict:
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"tp_sweep_{variant['name']}"
    cfg_dict["exits"]["extras"] = variant["extras"]
    # Resolve to concrete dir
    cfg_dict["output"]["dir"] = f"runs/{cfg_dict['run_id']}"
    cfg = BacktestConfig.from_dict(cfg_dict)
    t0 = time.time()
    summary = BacktestRunner(cfg).run()
    elapsed = time.time() - t0
    return {
        "variant": variant["name"],
        "extras": variant["extras"],
        "n_full_closes": summary["n_full_closes"],
        "wins": summary["wins"],
        "win_rate_pct": summary["win_rate_pct"],
        "total_pnl_usd": summary["total_pnl_usd"],
        "max_dd_pct": summary["max_dd_pct"],
        "by_reason": summary["by_reason"],
        "blocks_by_guard": summary["blocks_by_guard"],
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    base = _base_config(days)
    print(f"\nbtengine TP sweep — last {days} days, {len(VARIANTS)} variants")
    print(f"Window: {base['window']['start']} → {base['window']['end']}")
    print("=" * 100)

    results = []
    for v in VARIANTS:
        print(f"\n[{v['name']}] running…", flush=True)
        try:
            r = run_variant(base, v)
            print(f"  -> n={r['n_full_closes']} WR={r['win_rate_pct']:.1f}% "
                  f"pnl=${r['total_pnl_usd']:+.2f} ({r['elapsed_sec']}s)")
            results.append(r)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback; traceback.print_exc()
            results.append({"variant": v["name"], "error": str(exc)})

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    print(f"{'variant':<24} {'n':>5} {'WR%':>6} {'pnl':>10} {'avg':>8} "
          f"{'maxDD%':>7} {'Δ vs base':>10}")
    print("-" * 100)
    base_pnl = next((r.get("total_pnl_usd", 0) for r in results
                     if r["variant"] == "baseline"), 0)
    for r in results:
        if "error" in r:
            print(f"  {r['variant']:<22} ERROR: {r['error']}")
            continue
        avg = r["total_pnl_usd"] / max(r["n_full_closes"], 1)
        delta = r["total_pnl_usd"] - base_pnl
        print(f"  {r['variant']:<22} {r['n_full_closes']:>5} "
              f"{r['win_rate_pct']:>5.1f}% ${r['total_pnl_usd']:>+9.2f} "
              f"${avg:>+7.2f} {r['max_dd_pct']:>6.1f}% ${delta:>+9.2f}")

    # ── Best variant ──────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda x: x["total_pnl_usd"])
        print(f"\nBest: {best['variant']} — pnl ${best['total_pnl_usd']:+.2f}, "
              f"WR {best['win_rate_pct']:.1f}%, Δ vs baseline ${best['total_pnl_usd']-base_pnl:+.2f}")

    # Persist
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window_days": days,
        "window_start": base["window"]["start"],
        "window_end": base["window"]["end"],
        "variants": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
