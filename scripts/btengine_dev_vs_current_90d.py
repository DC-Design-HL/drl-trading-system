#!/usr/bin/env python3
"""90d ablation: dev (model-first) vs current production (structure-first).

Now that btengine has model_first_v1, we can run the actual dev branch's
entry path through the same simulator. Variants:

1. dev_equivalent       — model_first + NO production guards (only ADX min 20).
                          Closest reproduction of dev's pre-Apr-27 logic.
2. model_first_with_guards — model_first + symbol_blocklist + ADX [25,50]
                              + usdtd + funding/whale/news (current guard chain).
                              Tests whether adding the post-Apr-27 guards
                              would have helped the dev model.
3. structure_first_current — current live config (deployed 2026-05-03).

Output: data/training/btengine_dev_vs_current_90d.json
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
OUT = REPO / "data" / "training" / "btengine_dev_vs_current_90d.json"


def _base(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "dvc_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "guards": {"enabled": [], "params": {}},
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
    {
        "name": "dev_equivalent",
        "patch": lambda c: (
            _set(c, ["strategy"], "model_first_v1"),
            _set(c, ["strategy_overrides"], {"min_confidence": 0.45}),
            _set(c, ["guards", "enabled"], ["adx"]),  # dev had only ADX min
            _set(c, ["guards", "params", "adx"], {"min_adx": 20, "max_adx": 100}),
        )[-1],
    },
    {
        "name": "model_first_with_current_guards",
        "patch": lambda c: (
            _set(c, ["strategy"], "model_first_v1"),
            _set(c, ["strategy_overrides"], {"min_confidence": 0.55}),
            _set(c, ["guards", "enabled"],
                 ["symbol_blocklist", "adx", "usdtd",
                  "funding_long", "whale_neutral", "ext_pos_news"]),
            _set(c, ["guards", "params", "adx"], {"min_adx": 25, "max_adx": 50}),
        )[-1],
    },
    {
        "name": "structure_first_current",
        "patch": lambda c: (
            _set(c, ["strategy"], "structure_first_v3"),
            _set(c, ["strategy_overrides"], {"min_confidence": 0.55}),
            _set(c, ["guards", "enabled"],
                 ["symbol_blocklist", "adx", "usdtd",
                  "funding_long", "whale_neutral", "ext_pos_news"]),
            _set(c, ["guards", "params", "adx"], {"min_adx": 25, "max_adx": 50}),
        )[-1],
    },
]


def _set(d, path, value):
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value
    return d


def per_month(out_dir):
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


def run_variant(base, v):
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"dvc_{v['name']}"
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
        "strategy": cfg_dict["strategy"],
        "min_confidence": cfg_dict["strategy_overrides"].get("min_confidence"),
        "guards_enabled": cfg_dict["guards"]["enabled"],
        "n_full_closes": s["n_full_closes"], "wins": s["wins"],
        "wr_pct": s["win_rate_pct"],
        "total_pnl": s["total_pnl_usd"],
        "max_dd_pct": s["max_dd_pct"],
        "blocked_total": s["blocked_total"],
        "blocks_by_guard": s["blocks_by_guard"],
        "by_reason": {k: v["n"] for k, v in s["by_reason"].items()},
        "by_month": per_month(out_dir),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    base = _base(days)
    print(f"\nDev vs Current — {days}d, {len(VARIANTS)} variants")
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
    print("RESULTS")
    print("=" * 110)
    valid = [r for r in results if "error" not in r]
    print(f"{'variant':<32} {'strat':<22} {'n':>5} {'WR%':>6} {'pnl':>10} {'maxDD%':>7}")
    print("-" * 110)
    for r in valid:
        print(f"  {r['variant']:<30} {r['strategy']:<22} {r['n_full_closes']:>5} "
              f"{r['wr_pct']:>5.1f}% ${r['total_pnl']:>+9.2f} {r['max_dd_pct']:>6.1f}%")

    # Per-month
    print("\nPER-MONTH:")
    for r in valid:
        print(f"\n{r['variant']} ({r['strategy']}):")
        for m in r.get("by_month", []):
            print(f"  {m['month']}  n={m['n']:>3} WR={m['wr']:>4.1f}% pnl=${m['pnl']:>+9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window_days": days,
        "window_start": base["window"]["start"],
        "window_end": base["window"]["end"],
        "variants": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
