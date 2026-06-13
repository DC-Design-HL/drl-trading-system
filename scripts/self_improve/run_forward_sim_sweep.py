#!/usr/bin/env python3
"""Forward-sim parameter sweep — the P2 acceptance demonstration.

Runs the forward simulator across several values of ONE config knob over a
fixed window and produces a ranked report. This proves the capability the
whole forward sim exists for: telling the autonomous loop which config
variant would have done better. It does NOT deploy anything — it ranks.

Ranking follows PROFITABILITY_PLAN.md §7 optimisation order: 30d net PnL
first, then max drawdown (gate ≤ 8%). Sharpe/Sortino are noted as not yet
computed (trade-level return series — future P5 work).

Examples:
    python3 -m scripts.self_improve.run_forward_sim_sweep \\
        --knob trailing_distance_pct --values 0.003,0.005,0.008 --weeks 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.forward_sim import (  # noqa: E402
    ForwardSimConfig,
    run_forward_sim,
)

MAX_DD_GATE_PCT = 8.0  # PROFITABILITY_PLAN.md §7


def _metrics(result) -> dict:
    """Portfolio metrics from the trade log: net PnL, trade count, win
    rate, and max drawdown on a constant-capital equity curve."""
    trades = [t for sr in result.per_symbol.values() for t in sr.trades]
    trades.sort(key=lambda t: t.exit_ts)
    n = len(trades)
    wins = sum(1 for t in trades if t.realized_pnl_usd > 0)
    net = float(sum(t.realized_pnl_usd for t in trades))
    # Max drawdown on capital_base + cumulative realised PnL.
    cap = result.config.capital_base
    equity = cap
    peak = cap
    max_dd = 0.0
    for t in trades:
        equity += t.realized_pnl_usd
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return {
        "net_pnl": net,
        "n_trades": n,
        "win_rate": (wins / n) if n else 0.0,
        "max_dd_pct": max_dd * 100.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--knob", default="trailing_distance_pct",
                    help="ForwardSimConfig field to sweep.")
    ap.add_argument("--values", default="0.003,0.005,0.008",
                    help="Comma-separated values for the knob.")
    ap.add_argument("--weeks", type=int, default=2)
    ap.add_argument("--output", "-o", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    valid = {f.name for f in fields(ForwardSimConfig)}
    if args.knob not in valid:
        print(f"unknown knob '{args.knob}'. Valid: {sorted(valid)}")
        return 2
    values = [float(v) for v in args.values.split(",")]

    end = datetime.now(timezone.utc)
    start = end - timedelta(weeks=args.weeks)

    base = ForwardSimConfig()
    runs = []
    for v in values:
        cfg = replace(base, **{args.knob: v})
        print(f"running {args.knob}={v} over {start.date()}→{end.date()}…")
        result = run_forward_sim(
            start=start, end=end, config=cfg, label=f"{args.knob}={v}")
        m = _metrics(result)
        m["value"] = v
        m["runtime_s"] = result.runtime_seconds
        runs.append(m)
        print(f"  net=${m['net_pnl']:+.2f} trades={m['n_trades']} "
              f"WR={m['win_rate']*100:.1f}% maxDD={m['max_dd_pct']:.1f}%")

    # Rank: highest net PnL first; a config breaching the DD gate is flagged.
    ranked = sorted(runs, key=lambda m: m["net_pnl"], reverse=True)

    lines = [
        f"# Forward-Sim Sweep — `{args.knob}`",
        "",
        f"**Window:** {start.date()} → {end.date()} ({args.weeks}w)  ",
        f"**Baseline config**, varying only `{args.knob}`.  ",
        "**Ranking:** net PnL (PROFITABILITY_PLAN.md §7); "
        f"max DD gate ≤ {MAX_DD_GATE_PCT:.0f}%.  ",
        "**Not deployed — demonstration of the sim's ranking capability.**",
        "",
        f"| Rank | {args.knob} | Net PnL | Trades | Win rate | Max DD | "
        "DD gate |",
        "|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for i, m in enumerate(ranked, 1):
        dd_ok = "✅" if m["max_dd_pct"] <= MAX_DD_GATE_PCT else "❌"
        lines.append(
            f"| {i} | {m['value']} | ${m['net_pnl']:+.2f} | {m['n_trades']} "
            f"| {m['win_rate']*100:.1f}% | {m['max_dd_pct']:.1f}% | {dd_ok} |"
        )
    best = ranked[0]
    lines += [
        "",
        f"**Winner: `{args.knob}={best['value']}`** — net "
        f"${best['net_pnl']:+.2f}, max DD {best['max_dd_pct']:.1f}%.",
        "",
        "_Sharpe / Sortino not yet computed (need a trade-level return "
        "series — future P5). Sim PnL runs optimistic vs live (it can't "
        "replay orderbook/whale/news/USDT.D guards), so treat these as "
        "relative rankings, not absolute forecasts._",
    ]
    report = "\n".join(lines) + "\n"
    out = args.output or (
        _REPO_ROOT / "docs" / f"forward_sim_sweep_{args.knob}.md")
    out.write_text(report)
    print(f"\nreport → {out}")
    print(f"WINNER: {args.knob}={best['value']} "
          f"(net ${best['net_pnl']:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
