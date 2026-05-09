#!/usr/bin/env python3
"""Backtest dev's ACTUAL current config — not the model_first proxy I ran before.

Chen 2026-05-05: 'I think our backtesting mechanism isn't good since dev
looks much better than the previous branch — can you make sure to test
the backtesting on the current logic again?'

Earlier I ran 'dev_equivalent' with model_first_v1 + only ADX. That was
WRONG. After Chen pulled origin/dev (22 commits merged), dev's config
is actually:

  STRUCTURE_FIRST_MODE = True            (not model-first!)
  MIN_CONFIDENCE       = 0.45            (but SKIPPED in structure-first
                                          mode on dev; effective floor is
                                          BOS confidence base 0.5)
  ADX_GUARD_MIN        = 20              (no MAX)
  REVERSAL_BLOCK_LONG_CANARY = {XRPUSDT}
  RISK_POOL_PCT        = 0.10
  FIXED_MAX_NOTIONAL   = 3000
  Guards present:      ADX min, REVERSE_CLOSE_LONG canary
  Guards ABSENT:       symbol_blocklist, ADX max, USDT.D, funding_long,
                       whale_neutral, ext_pos_news

This sweep tests three configs on 90d:

1. dev_actual            — what's actually live now
2. dev_actual_with_news  — same + ext_pos_news (only post-Apr-30 data
                            applies; harmless on rest of window)
3. feature_consolidation — my prior production config for comparison
                            (MIN_CONFIDENCE=0.55 + ADX[25,50] + all guards)

Output: data/training/btengine_dev_actual_config.json
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
OUT = REPO / "data" / "training" / "btengine_dev_actual_config.json"


def _base(days: int):
    end = date.today()
    start = end - timedelta(days=days)
    return {
        "run_id": "dac_base",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "intervals": {"primary": "15m", "htf": ["1h", "4h"]},
        "seed": 42,
        "strategy": "structure_first_v3",
        "strategy_overrides": {},
        "guards": {"enabled": [], "params": {}},
        "sizing": {"type": "fixed_notional", "usd": 3000, "max_concurrent": 4},
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
    {
        "name": "dev_actual",
        "patch": lambda c: (
            # Structure-first with the SKIP wrapper means MIN_CONFIDENCE is
            # a no-op. To match dev's behaviour we set it to 0 here.
            _set(c, ["strategy_overrides", "min_confidence"], 0.0),
            # Only ADX min, no max — represent by min=20, max=999
            _set(c, ["guards", "enabled"], ["adx"]),
            _set(c, ["guards", "params", "adx"], {"min_adx": 20, "max_adx": 999}),
            _set(c, ["sizing", "usd"], 1500),  # FIXED_MAX_NOTIONAL=3000 → notional 1500
        )[-1],
    },
    {
        "name": "dev_actual_plus_news",
        "patch": lambda c: (
            _set(c, ["strategy_overrides", "min_confidence"], 0.0),
            _set(c, ["guards", "enabled"], ["adx", "ext_pos_news"]),
            _set(c, ["guards", "params", "adx"], {"min_adx": 20, "max_adx": 999}),
            _set(c, ["sizing", "usd"], 1500),
        )[-1],
    },
    {
        "name": "feature_consolidation_prior",
        "patch": lambda c: (
            _set(c, ["strategy_overrides", "min_confidence"], 0.55),
            _set(c, ["guards", "enabled"],
                 ["symbol_blocklist", "adx", "usdtd",
                  "funding_long", "whale_neutral", "ext_pos_news"]),
            _set(c, ["guards", "params", "adx"], {"min_adx": 25, "max_adx": 50}),
            _set(c, ["sizing", "usd"], 1500),
        )[-1],
    },
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


def run_variant(base, v):
    cfg_dict = deepcopy(base)
    cfg_dict["run_id"] = f"dac_{v['name']}"
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
        "guards_enabled": cfg_dict["guards"]["enabled"],
        "min_confidence": cfg_dict["strategy_overrides"].get("min_confidence"),
        "adx_params": cfg_dict["guards"].get("params", {}).get("adx"),
        "sizing_usd": cfg_dict["sizing"]["usd"],
        "n_full_closes": s["n_full_closes"], "wins": s["wins"],
        "wr_pct": s["win_rate_pct"],
        "total_pnl": s["total_pnl_usd"],
        "max_dd_pct": s["max_dd_pct"],
        "blocked_total": s["blocked_total"],
        "by_reason": {k: v["n"] for k, v in s["by_reason"].items()},
        "by_month": per_month(out_dir),
        "elapsed_sec": round(elapsed, 1),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    base = _base(days)
    print(f"\nDev's ACTUAL config — {days}d, {len(VARIANTS)} variants")
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

    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)
    valid = [r for r in results if "error" not in r]
    print(f"{'variant':<32} {'n':>5} {'WR%':>6} {'pnl':>10} {'maxDD%':>7}")
    print("-" * 110)
    for r in valid:
        print(f"  {r['variant']:<30} {r['n_full_closes']:>5} {r['wr_pct']:>5.1f}% "
              f"${r['total_pnl']:>+9.2f} {r['max_dd_pct']:>6.1f}%")

    print("\nPER-MONTH:")
    for r in valid:
        if not r.get("by_month"): continue
        print(f"\n{r['variant']}:")
        for m in r["by_month"]:
            print(f"  {m['month']}  n={m['n']:>3} WR={m['wr']:>4.1f}% pnl=${m['pnl']:>+9.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "window_days": days, "window_start": base["window"]["start"],
        "window_end": base["window"]["end"],
        "variants": results,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
