#!/usr/bin/env python3
"""STAGNANT_EXIT counterfactual (2026-07-03).

Live data (since May 1) shows STAGNANT_EXIT is the #1 leak: -$595 over 126
trades, while the trailing-SL engine makes +$941. The stagnant band
[-1.0%, +0.5%] sits entirely below the +0.8% trailing-breakeven, so every
stagnant close is a trade the trailing stop never engaged — bailed at a small
loss + fees.

This runs the forward simulator (which reproduces the live entry path AND the
full exit stack incl. funding+fees) with the live baseline vs. several stagnant
variants, over one window, and prints net PnL + per-reason attribution so we can
pick the fix that recovers the most without touching the trailing-SL engine.

Usage:
    python3 -m scripts.self_improve.stagnant_counterfactual \
        --start 2026-06-08 --end 2026-06-22
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.self_improve.forward_sim import ForwardSimConfig, run_forward_sim  # noqa: E402


def _parse(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


# The variants. Each mutates ONLY the stagnant knobs off the live baseline;
# the entry path and trailing-SL engine are identical across all of them.
def _variants(base: ForwardSimConfig) -> dict[str, ForwardSimConfig]:
    return {
        "baseline (6h, [-1.0%,+0.5%])": base,
        # > max_hold_hours (168) → MAX_HOLD force-closes first, so stagnant
        # never fires (and the run still models the funding of the longer hold).
        "disable stagnant":             replace(base, stagnant_hours=500.0),
        "loss-only ([-1.0%,0.0%])":     replace(base, stagnant_pct_max=0.0),
        "loss-only tight ([-0.6%,0%])": replace(base, stagnant_pct_min=-0.006,
                                                stagnant_pct_max=0.0),
        "lengthen 12h":                 replace(base, stagnant_hours=12.0),
        "lengthen 9h + loss-only":      replace(base, stagnant_hours=9.0,
                                                stagnant_pct_max=0.0),
    }


def _summarize(result) -> dict:
    net = 0.0
    by_reason: dict[str, list[float]] = defaultdict(list)
    n_trades = 0
    for sym_res in result.per_symbol.values():
        for t in sym_res.trades:
            net += t.realized_pnl_usd
            by_reason[t.close_reason].append(t.realized_pnl_usd)
            n_trades += 1
    return {"net": net, "n": n_trades, "by_reason": by_reason}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbol", "-s", action="append", default=None)
    args = ap.parse_args()

    start, end = _parse(args.start), _parse(args.end)
    symbols = tuple(args.symbol) if args.symbol else (
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    base = ForwardSimConfig()

    print(f"Window {start.date()} → {end.date()}  symbols={symbols}\n")
    header = f"{'variant':<32}{'net$':>9}{'trades':>7}   reason breakdown (net$/n)"
    print(header)
    print("-" * len(header))

    baseline_net = None
    for name, cfg in _variants(base).items():
        res = run_forward_sim(symbols=symbols, start=start, end=end, config=cfg)
        s = _summarize(res)
        if baseline_net is None:
            baseline_net = s["net"]
        delta = s["net"] - baseline_net
        reasons = "  ".join(
            f"{r}:{sum(v):+.0f}/{len(v)}"
            for r, v in sorted(s["by_reason"].items(), key=lambda kv: sum(kv[1]))
        )
        tag = "" if name.startswith("baseline") else f"  (Δ{delta:+.0f})"
        print(f"{name:<32}{s['net']:>9.0f}{s['n']:>7}{tag}")
        print(f"{'':<32}{'':>9}{'':>7}   {reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
