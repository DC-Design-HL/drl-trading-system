#!/usr/bin/env python3
"""Direct parity check: btengine vs live on dev's last 24h.

Replays May 5 05:12 → May 6 now through btengine with dev's actual config
and compares to the live SQLite trade log. This is the cleanest validation
of whether the backtest is correctly modeling the system.

Live last 24h:
  16 opens, 13 closes, 62% WR, net +$5.51 realized
  Balance: $4999.11 (down $0.89 from $5000 reset point)

If btengine produces dramatically different numbers, something is wrong
with the framework. If it produces similar numbers (within 50% on PnL,
within 30% on trade count), the framework is reliable.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.btengine.config import BacktestConfig
from src.btengine.runner import BacktestRunner

OUT = REPO / "data" / "training" / "btengine_dev_24h_parity.json"


def main():
    cfg_dict = {
        "run_id": "dev_24h_parity",
        # Window covering the actual live deploy: pad start by a couple
        # days so warmup completes before the deploy timestamp
        "window": {"start": "2026-05-03", "end": "2026-05-06"},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["5m", "1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "strategy_overrides": {"min_confidence": 0.0},  # dev's gate is skipped
        "guards": {
            "enabled": ["adx"],
            "params": {"adx": {"min_adx": 20, "max_adx": 999}},
        },
        "sizing": {"type": "fixed_notional", "usd": 1500, "max_concurrent": 4},
        "exits": {
            "partial_tp": [[1.0, 0.40], [2.0, 0.35]],
            "stagnant_hours": 6, "stagnant_pct_min": -0.010, "stagnant_pct_max": 0.005,
        },
        "fees": {"taker": 0.0004, "slippage_pct": 0.0005},
        "output": {"dir": "runs/dev_24h_parity"},
    }

    cfg = BacktestConfig.from_dict(cfg_dict)
    t0 = time.time()
    summary = BacktestRunner(cfg).run()
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("BTENGINE RESULT (dev config, 3-day window with warmup)")
    print("=" * 80)
    print(f"  closed trades:  {summary['n_full_closes']}")
    print(f"  win rate:       {summary['win_rate_pct']:.1f}%")
    print(f"  total pnl:      ${summary['total_pnl_usd']:+.2f}")
    print(f"  by_reason:      {[(k,v['n']) for k,v in summary['by_reason'].items()]}")
    print(f"  elapsed:        {elapsed:.0f}s")

    # Filter trades to ONLY May 5 05:12 onward (dev deploy time)
    trades = pd.read_parquet("runs/dev_24h_parity/trades.parquet")
    if len(trades):
        trades_dt = pd.to_datetime(trades["exit_ts_ms"], unit="ms", utc=True)
        deploy_ms = int(pd.Timestamp("2026-05-05 05:12:00", tz="UTC").timestamp() * 1000)
        post = trades[trades["exit_ts_ms"] >= deploy_ms]
        full_post = post[post["is_full_close"]]
        print(f"\nPost-deploy slice (May 5 05:12 onwards):")
        print(f"  closes:  {len(full_post)}")
        print(f"  wins:    {(full_post['pnl_usd']>0).sum()} ({(full_post['pnl_usd']>0).sum()/max(len(full_post),1)*100:.1f}%)")
        print(f"  pnl:     ${full_post['pnl_usd'].sum():+.2f}")
        print(f"  by reason: {full_post['reason'].value_counts().to_dict()}")

    print("\n" + "=" * 80)
    print("LIVE (from SQLite trade log)")
    print("=" * 80)
    print("  closed trades:  13")
    print("  win rate:       61.5% (8 wins / 5 losses)")
    print("  total pnl:      $+5.51")
    print("  by_reason:      STAGNANT 6, SL 4, REVERSE_CLOSE 3")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "btengine_full_window": {
            "n_full_closes": summary["n_full_closes"],
            "wr": summary["win_rate_pct"],
            "pnl": summary["total_pnl_usd"],
            "by_reason": {k: v["n"] for k, v in summary["by_reason"].items()},
        },
        "live_post_deploy": {
            "n_full_closes": 13, "wr": 61.5, "pnl": 5.51,
            "by_reason": {"STAGNANT_EXIT": 6, "SL": 4, "REVERSE_CLOSE_LONG": 1, "REVERSE_CLOSE_SHORT": 2},
        },
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
