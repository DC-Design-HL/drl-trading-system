#!/usr/bin/env python3
"""90-day signal-ablation sweep on btengine.

Tests how the structure_first_v3 strategy performs under different guard
combinations on the longest window we can run (90 days of klines), and
compares against the current production guard chain as baseline.

Variant set:

A. Guard ablations (what each guard contributes):
   1. baseline_full  — all 6 production guards on (= current live config)
   2. structure_only — no guards (pure entry signal)
   3. drop_blocklist — every guard except symbol_blocklist
   4. drop_adx       — every guard except adx
   5. drop_usdtd     — every guard except usdtd

B. Parameter sensitivity:
   6. adx_strict     — ADX [25, 50] (tighter trend requirement)
   7. adx_lax        — ADX [15, 70]
   8. confidence_high — min_confidence 0.75
   9. confidence_low  — min_confidence 0.55

C. Stripped combinations:
  10. blocklist_only — only symbol_blocklist

Note: whale_neutral, funding_long, ext_pos_news have no historical
data feed in the backtest (fail-open), so they have zero effect on
90d runs. They're included for completeness but won't move numbers.

Output:
  data/training/btengine_signal_ablation_90d.json
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

OUT = REPO / "data" / "training" / "btengine_signal_ablation_90d.json"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(message)s")


def _base_config(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "ablation_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "strategy_overrides": {},
        "guards": {
            "enabled": [
                "symbol_blocklist", "adx", "usdtd",
                "funding_long", "whale_neutral", "ext_pos_news",
            ],
            "params": {"adx": {"min_adx": 20, "max_adx": 60}},
        },
        "sizing": {"type": "fixed_notional", "usd": 1500, "max_concurrent": 4},
        "exits": {"partial_tp": [[1.0, 0.40], [2.0, 0.35]]},
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": "runs/${run_id}"},
    }


VARIANTS = [
    # A. Guard ablations
    {"name": "baseline_full",
     "patch": lambda c: c},  # current production
    {"name": "structure_only",
     "patch": lambda c: _set(c, ["guards", "enabled"], [])},
    {"name": "drop_blocklist",
     "patch": lambda c: _set(c, ["guards", "enabled"],
                              ["adx", "usdtd", "funding_long", "whale_neutral", "ext_pos_news"])},
    {"name": "drop_adx",
     "patch": lambda c: _set(c, ["guards", "enabled"],
                              ["symbol_blocklist", "usdtd", "funding_long", "whale_neutral", "ext_pos_news"])},
    {"name": "drop_usdtd",
     "patch": lambda c: _set(c, ["guards", "enabled"],
                              ["symbol_blocklist", "adx", "funding_long", "whale_neutral", "ext_pos_news"])},
    # B. Parameter sensitivity
    {"name": "adx_strict",
     "patch": lambda c: _set(c, ["guards", "params", "adx"], {"min_adx": 25, "max_adx": 50})},
    {"name": "adx_lax",
     "patch": lambda c: _set(c, ["guards", "params", "adx"], {"min_adx": 15, "max_adx": 70})},
    {"name": "confidence_high",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.75)},
    {"name": "confidence_low",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.55)},
    # C. Stripped
    {"name": "blocklist_only",
     "patch": lambda c: _set(c, ["guards", "enabled"], ["symbol_blocklist"])},
]


def _set(d: dict, path: list, value) -> dict:
    """In-place nested set; returns the dict for chaining."""
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
    return d


def run_variant(base: dict, variant: dict) -> dict:
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"ablation_{variant['name']}"
    cfg_dict = variant["patch"](cfg_dict)
    cfg_dict["output"]["dir"] = f"runs/{cfg_dict['run_id']}"
    cfg = BacktestConfig.from_dict(cfg_dict)
    t0 = time.time()
    summary = BacktestRunner(cfg).run()
    elapsed = time.time() - t0
    return {
        "variant": variant["name"],
        "guards_enabled": cfg_dict["guards"]["enabled"],
        "adx_params": cfg_dict["guards"].get("params", {}).get("adx"),
        "min_confidence": cfg_dict.get("strategy_overrides", {}).get("min_confidence"),
        "n_full_closes": summary["n_full_closes"],
        "wins": summary["wins"],
        "win_rate_pct": summary["win_rate_pct"],
        "total_pnl_usd": summary["total_pnl_usd"],
        "max_dd_pct": summary["max_dd_pct"],
        "by_reason": {k: v["n"] for k, v in summary["by_reason"].items()},
        "by_symbol_side": {k: {"n": v["n"], "pnl": v["pnl"]}
                           for k, v in summary["by_symbol_side"].items()},
        "blocks_by_guard": summary["blocks_by_guard"],
        "blocked_total": summary["blocked_total"],
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    base = _base_config(days)
    print(f"\nbtengine signal ablation — last {days} days, {len(VARIANTS)} variants")
    print(f"Window: {base['window']['start']} → {base['window']['end']}")
    print("=" * 110)

    results = []
    for v in VARIANTS:
        print(f"\n[{v['name']}] running…", flush=True)
        try:
            r = run_variant(base, v)
            print(f"  -> n={r['n_full_closes']} WR={r['win_rate_pct']:.1f}% "
                  f"pnl=${r['total_pnl_usd']:+.2f} blocked={r['blocked_total']} "
                  f"({r['elapsed_sec']:.0f}s)", flush=True)
            results.append(r)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"variant": v["name"], "error": str(exc)})

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS TABLE")
    print("=" * 110)
    valid = [r for r in results if "error" not in r]
    base_pnl = next((r["total_pnl_usd"] for r in valid if r["variant"] == "baseline_full"), 0)
    print(f"{'variant':<22} {'n':>5} {'WR%':>6} {'pnl':>10} {'avg':>8} "
          f"{'maxDD%':>7} {'blocked':>8} {'Δ vs base':>10}")
    print("-" * 110)
    for r in valid:
        avg = r["total_pnl_usd"] / max(r["n_full_closes"], 1)
        delta = r["total_pnl_usd"] - base_pnl
        print(f"  {r['variant']:<20} {r['n_full_closes']:>5} "
              f"{r['win_rate_pct']:>5.1f}% ${r['total_pnl_usd']:>+9.2f} "
              f"${avg:>+7.2f} {r['max_dd_pct']:>6.1f}% "
              f"{r['blocked_total']:>8} ${delta:>+9.2f}")

    # ── Best variants ──────────────────────────────────────────────
    if valid:
        best = max(valid, key=lambda x: x["total_pnl_usd"])
        worst = min(valid, key=lambda x: x["total_pnl_usd"])
        print()
        print(f"Best:  {best['variant']:<25} pnl ${best['total_pnl_usd']:+.2f} "
              f"(Δ ${best['total_pnl_usd']-base_pnl:+.2f})")
        print(f"Worst: {worst['variant']:<25} pnl ${worst['total_pnl_usd']:+.2f} "
              f"(Δ ${worst['total_pnl_usd']-base_pnl:+.2f})")

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
