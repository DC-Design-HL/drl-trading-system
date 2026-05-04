#!/usr/bin/env python3
"""Backtest the EXACT current production config.

After 2026-05-03 deploys (MIN_CONFIDENCE 0.45→0.55, ADX [20,60]→[25,50])
the live config is:
  * Strategy: structure_first_v3, min_confidence=0.55
  * Guards: symbol_blocklist, adx[25,50], usdtd, funding_long,
            whale_neutral, ext_pos_news
  * Sizing: $1500 fixed notional, max 4 concurrent
  * REVERSE_CLOSE_LONG canary: all 4 symbols
  * Partial TP 40%@1R + 35%@2R + 25% trail

Runs the same config on two windows:
  90d (Feb 4 → May 4)  — broad regime, news/funding data fail-open
  25d (Apr 8 → May 4)  — news data available, smaller sample
"""

from __future__ import annotations

import json
import sys
import time
import logging
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btengine.config import BacktestConfig
from src.btengine.runner import BacktestRunner

logging.basicConfig(level=logging.WARNING)
OUT = REPO / "data" / "training" / "btengine_current_production_check.json"


def cfg(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": f"current_prod_{days}d",
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
            "stagnant_hours": 6,
            "stagnant_pct_min": -0.010,
            "stagnant_pct_max": 0.005,
        },
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": f"runs/current_prod_{days}d"},
    }


def run(days: int):
    print(f"\n=== {days}d backtest of current production config ===", flush=True)
    c = BacktestConfig.from_dict(cfg(days))
    t0 = time.time()
    s = BacktestRunner(c).run()
    elapsed = time.time() - t0

    out_dir = Path(f"runs/current_prod_{days}d")
    if out_dir.exists() and (out_dir / "trades.parquet").exists():
        trades = pd.read_parquet(out_dir / "trades.parquet")
        full = trades[trades["is_full_close"]]
        if len(full):
            full["exit_dt"] = pd.to_datetime(full["exit_ts_ms"], unit="ms", utc=True)
            full["month"] = full["exit_dt"].dt.strftime("%Y-%m")
            by_month = []
            for m, g in full.groupby("month"):
                wins = (g["pnl_usd"] > 0).sum()
                by_month.append({
                    "month": m, "n": int(len(g)), "wins": int(wins),
                    "wr": round(wins / len(g) * 100, 1),
                    "pnl": round(float(g["pnl_usd"].sum()), 2),
                })
        else:
            by_month = []
    else:
        by_month = []

    print(f"\n{days}d window {c.window.start} → {c.window.end}")
    print(f"  closes:    {s['n_full_closes']}")
    print(f"  WR:        {s['win_rate_pct']:.1f}%")
    print(f"  total pnl: ${s['total_pnl_usd']:+.2f}")
    print(f"  max DD:    {s['max_dd_pct']:.1f}%")
    print(f"  blocked:   {s['blocked_total']} (blocklist+adx+usdtd+news+canary)")
    print(f"  elapsed:   {elapsed:.0f}s")
    print(f"  by_reason: {dict(sorted(((k, v['n']) for k,v in s['by_reason'].items()), key=lambda x: -x[1]))}")
    if by_month:
        print(f"  per-month:")
        for m in by_month:
            print(f"    {m['month']}  n={m['n']:>3}  WR={m['wr']:>4.1f}%  pnl=${m['pnl']:>+9.2f}")

    return {
        "days": days,
        "window": [c.window.start.isoformat(), c.window.end.isoformat()],
        "n_full_closes": s["n_full_closes"],
        "wins": s["wins"],
        "wr_pct": s["win_rate_pct"],
        "total_pnl": s["total_pnl_usd"],
        "max_dd_pct": s["max_dd_pct"],
        "blocked": s["blocked_total"],
        "blocks_by_guard": s["blocks_by_guard"],
        "by_reason": {k: v["n"] for k, v in s["by_reason"].items()},
        "by_month": by_month,
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days_list = [int(d) for d in (sys.argv[1:] or [25, 90])]
    out = []
    for d in days_list:
        try:
            out.append(run(d))
        except Exception as exc:
            import traceback; traceback.print_exc()
            out.append({"days": d, "error": str(exc)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"results": out}, indent=2, default=str))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
