#!/usr/bin/env python3
"""90d permutation sweep — search for profitable config.

Per Chen 2026-05-04: try every existing signal/parameter combination,
no model (structure-first only), find a config that's profitable in
90d backtest.

Variant grid (14 variants):

A. Confidence threshold sensitivity (fill gaps from prior sweep):
   1. baseline_current     conf=0.55, ADX[25,50], full chain (current live)
   2. conf_0.65
   3. conf_0.75            (best from prior sweep, -$284)
   4. conf_0.85
   5. conf_0.95            extreme — only top-percentile signals

B. ADX bound exploration:
   6. adx_30_45            very tight bounds
   7. adx_30_50
   8. adx_25_45

C. Combined high-confidence + tight ADX:
   9. conf_0.75_adx_30_45  best of A × best of B
  10. conf_0.85_adx_30_45

D. Symbol restriction (data-driven from prior per-symbol findings):
  11. xrp_sol_only         the best historical symbols (drop BTC and ETH)
  12. xrp_only             single highest-edge symbol
  13. drop_blocklist       what if removing the blocklist actually helps?
                            (90d ablation showed -$736 vs base, but with
                             new conf gates it might differ)

E. Long/short asymmetric:
  14. only_short           SHORT-only (some symbols' SHORT side is +EV)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btengine.config import BacktestConfig
from src.btengine.runner import BacktestRunner

logging.basicConfig(level=logging.WARNING)
OUT = REPO / "data" / "training" / "btengine_profitability_search_90d.json"


def _base(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "psearch_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "strategy_overrides": {"min_confidence": 0.55},
        "guards": {
            "enabled": ["symbol_blocklist", "adx", "usdtd",
                        "funding_long", "whale_neutral", "ext_pos_news"],
            "params": {"adx": {"min_adx": 25, "max_adx": 50}},
        },
        "sizing": {"type": "fixed_notional", "usd": 1500, "max_concurrent": 4},
        "exits": {
            "partial_tp": [[1.0, 0.40], [2.0, 0.35]],
            "stagnant_hours": 6, "stagnant_pct_min": -0.010, "stagnant_pct_max": 0.005,
        },
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": "runs/${run_id}"},
    }


def _set(d, path, value):
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
    return d


VARIANTS = [
    # A. Confidence sensitivity
    {"name": "baseline_current",
     "patch": lambda c: c},
    {"name": "conf_0.65",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.65)},
    {"name": "conf_0.75",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.75)},
    {"name": "conf_0.85",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.85)},
    {"name": "conf_0.95",
     "patch": lambda c: _set(c, ["strategy_overrides", "min_confidence"], 0.95)},

    # B. ADX bounds
    {"name": "adx_30_45",
     "patch": lambda c: _set(c, ["guards", "params", "adx"], {"min_adx": 30, "max_adx": 45})},
    {"name": "adx_30_50",
     "patch": lambda c: _set(c, ["guards", "params", "adx"], {"min_adx": 30, "max_adx": 50})},
    {"name": "adx_25_45",
     "patch": lambda c: _set(c, ["guards", "params", "adx"], {"min_adx": 25, "max_adx": 45})},

    # C. Combined high-conf + tight ADX
    {"name": "conf_0.75_adx_30_45",
     "patch": lambda c: (
         _set(c, ["strategy_overrides", "min_confidence"], 0.75),
         _set(c, ["guards", "params", "adx"], {"min_adx": 30, "max_adx": 45}),
     )[-1]},
    {"name": "conf_0.85_adx_30_45",
     "patch": lambda c: (
         _set(c, ["strategy_overrides", "min_confidence"], 0.85),
         _set(c, ["guards", "params", "adx"], {"min_adx": 30, "max_adx": 45}),
     )[-1]},

    # D. Symbol restriction
    {"name": "xrp_sol_only",
     "patch": lambda c: _set(c, ["symbols"], ["XRPUSDT", "SOLUSDT"])},
    {"name": "xrp_only",
     "patch": lambda c: _set(c, ["symbols"], ["XRPUSDT"])},

    # E. Drop blocklist (over 90d, blocklist costs $736 — but maybe with new
    # conf/ADX gates it acts differently)
    {"name": "drop_blocklist_conf_075",
     "patch": lambda c: (
         _set(c, ["strategy_overrides", "min_confidence"], 0.75),
         _set(c, ["guards", "enabled"],
              ["adx", "usdtd", "funding_long", "whale_neutral", "ext_pos_news"]),
     )[-1]},
]


def per_month(out_dir: Path):
    p = out_dir / "trades.parquet"
    if not p.exists(): return []
    trades = pd.read_parquet(p)
    if not len(trades): return []
    full = trades[trades["is_full_close"]]
    if not len(full): return []
    full = full.copy()
    full["month"] = pd.to_datetime(full["exit_ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
    out = []
    for m, g in full.groupby("month"):
        wins = int((g["pnl_usd"] > 0).sum())
        out.append({"month": m, "n": int(len(g)), "wins": wins,
                    "wr": round(wins / len(g) * 100, 1),
                    "pnl": round(float(g["pnl_usd"].sum()), 2)})
    return out


def per_sym_side(out_dir: Path):
    p = out_dir / "trades.parquet"
    if not p.exists(): return []
    trades = pd.read_parquet(p)
    if not len(trades): return []
    full = trades[trades["is_full_close"]]
    if not len(full): return []
    out = []
    for (sym, side), g in full.groupby(["symbol", "side"]):
        wins = int((g["pnl_usd"] > 0).sum())
        out.append({"symbol": sym, "side": side, "n": int(len(g)), "wins": wins,
                    "wr": round(wins / len(g) * 100, 1),
                    "pnl": round(float(g["pnl_usd"].sum()), 2)})
    return out


def run_variant(base, v):
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"psearch_{v['name'].replace('.','_')}"
    cfg_dict = v["patch"](cfg_dict)
    cfg_dict["output"]["dir"] = f"runs/{cfg_dict['run_id']}"
    cfg = BacktestConfig.from_dict(cfg_dict)
    t0 = time.time()
    s = BacktestRunner(cfg).run()
    elapsed = time.time() - t0

    out_dir = Path(f"runs/{cfg_dict['run_id']}")
    if not out_dir.is_absolute(): out_dir = REPO / out_dir

    return {
        "variant": v["name"],
        "min_confidence": cfg_dict.get("strategy_overrides", {}).get("min_confidence"),
        "guards_enabled": cfg_dict["guards"]["enabled"],
        "adx_params": cfg_dict["guards"].get("params", {}).get("adx"),
        "symbols": cfg_dict["symbols"],
        "n_full_closes": s["n_full_closes"], "wins": s["wins"],
        "wr_pct": s["win_rate_pct"],
        "total_pnl": s["total_pnl_usd"],
        "max_dd_pct": s["max_dd_pct"],
        "blocked_total": s["blocked_total"],
        "blocks_by_guard": s["blocks_by_guard"],
        "by_reason": {k: v["n"] for k, v in s["by_reason"].items()},
        "by_month": per_month(out_dir),
        "by_sym_side": per_sym_side(out_dir),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    base = _base(days)
    print(f"\nProfitability search — {days}d, {len(VARIANTS)} variants")
    print(f"Window: {base['window']['start']} → {base['window']['end']}")
    print("=" * 110)

    results = []
    for v in VARIANTS:
        print(f"\n[{v['name']}] running…", flush=True)
        try:
            r = run_variant(base, v)
            print(f"  -> n={r['n_full_closes']} WR={r['wr_pct']:.1f}% "
                  f"pnl=${r['total_pnl']:+.2f} blocked={r['blocked_total']} "
                  f"({r['elapsed_sec']:.0f}s)", flush=True)
            results.append(r)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"variant": v["name"], "error": str(exc)})

    # Final table
    print("\n" + "=" * 110)
    print("RESULTS — sorted by total pnl")
    print("=" * 110)
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda x: -x["total_pnl"])  # best first
    print(f"{'variant':<28} {'n':>4} {'WR%':>6} {'pnl':>10} {'maxDD%':>7} {'profitable':>11}")
    print("-" * 110)
    for r in valid:
        prof = "✓" if r["total_pnl"] > 0 else ""
        print(f"  {r['variant']:<26} {r['n_full_closes']:>4} {r['wr_pct']:>5.1f}% "
              f"${r['total_pnl']:>+9.2f} {r['max_dd_pct']:>6.1f}% {prof:>11}")

    profitable = [r for r in valid if r["total_pnl"] > 0]
    if profitable:
        best = profitable[0]
        print(f"\n✅ {len(profitable)} variant(s) PROFITABLE on 90d.")
        print(f"   Best: {best['variant']} pnl ${best['total_pnl']:+.2f} WR {best['wr_pct']:.1f}%")
    else:
        print(f"\n❌ NO variants profitable on 90d. Best: {valid[0]['variant']} "
              f"pnl ${valid[0]['total_pnl']:+.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window_days": days, "window_start": base["window"]["start"],
        "window_end": base["window"]["end"],
        "variants": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
