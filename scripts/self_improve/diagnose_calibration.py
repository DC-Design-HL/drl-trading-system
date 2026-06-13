#!/usr/bin/env python3
"""Per-decision sim-vs-live divergence diagnostic (PROFITABILITY_PLAN.md P2).

The calibration gate reports ~40% directional agreement but not WHY. This
script localises the divergence: for every live entry it inspects what the
forward sim decided at that timestamp (via the sim's decision trace) and
classifies the miss — was the entry blocked by a specific gate, did the
structure signal point the other way / not fire, or was the sim busy in a
different position? It also reports the inverse (sim entries live never
took).

Read-only. Does not touch the gate report or any live state.

Usage:
    python3 -m scripts.self_improve.diagnose_calibration --weeks 2
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.backtest_harness import pair_open_close  # noqa: E402
from src.self_improve.forward_sim import (  # noqa: E402
    ForwardSimConfig,
    run_forward_sim,
)

# Same match window as the calibration gate.
MATCH_WINDOW_MIN = 30


def _load_live_pairs(db_path: Path, start: datetime, end: datetime) -> list:
    conn = sqlite3.connect(str(db_path))
    try:
        return pair_open_close(
            conn, start_date=start.isoformat(), end_date=end.isoformat(),
        )
    finally:
        conn.close()


def _parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _diagnose_live_entry(
    live_ts: datetime,
    live_side: str,
    symbol: str,
    sym_trace: list,
    sim_trades: list,
    window: timedelta,
) -> str:
    """Classify why the sim did or didn't match this live entry."""
    near = [
        t for t in sym_trace
        if abs(_parse_ts(t["ts"]) - live_ts) <= window
    ]
    if near:
        same_side = [t for t in near if t["side"] == live_side]
        if any(t["outcome"] == "entry" for t in same_side):
            return "match"
        if same_side:
            # Closest same-side decision's blocking gate.
            closest = min(
                same_side, key=lambda t: abs(_parse_ts(t["ts"]) - live_ts),
            )
            return f"blocked:{closest['outcome']}"
        opp = [t for t in near if t["side"] and t["side"] != live_side]
        if opp:
            return "sim_dir_opposite"
        if any(t["outcome"] == "trend" for t in near):
            return "sim_no_signal"
        return "sim_other"
    # No decision bar in window — was the sim holding a different position?
    for tr in sim_trades:
        if _parse_ts(tr["entry_ts"]) <= live_ts < _parse_ts(tr["exit_ts"]):
            return "sim_in_position"
    return "no_decision_bar"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", type=int, default=2)
    ap.add_argument("--db", default="data/trading.db")
    ap.add_argument("--output", "-o", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    window = timedelta(minutes=MATCH_WINDOW_MIN)

    end = datetime.now(timezone.utc)
    start = end - timedelta(weeks=args.weeks)

    print(f"running traced forward sim {start.date()} → {end.date()}…")
    trace: list = []
    sim = run_forward_sim(start=start, end=end, config=ForwardSimConfig(),
                          trace=trace)
    print(f"sim runtime: {sim.runtime_seconds:.1f}s, {len(trace)} decisions")

    sim_json = sim.to_json()["per_symbol"]
    trace_by_sym: dict[str, list] = {}
    for rec in trace:
        trace_by_sym.setdefault(rec["symbol"], []).append(rec)
    trades_by_sym = {
        sym: sr.get("trades", []) for sym, sr in sim_json.items()
    }

    live_pairs = _load_live_pairs(Path(args.db), start, end)

    # ── Live-entry diagnosis ────────────────────────────────────────────
    per_combo: dict[tuple[str, str], Counter] = {}
    overall = Counter()
    for p in live_pairs:
        try:
            live_ts = _parse_ts(p.open_ts)
        except ValueError:
            continue
        verdict = _diagnose_live_entry(
            live_ts, p.side, p.symbol,
            trace_by_sym.get(p.symbol, []),
            trades_by_sym.get(p.symbol, []),
            window,
        )
        per_combo.setdefault((p.symbol, p.side), Counter())[verdict] += 1
        overall[verdict] += 1

    # ── Sim-only entries (entered in sim, no live entry within window) ───
    live_by_sym: dict[str, list] = {}
    for p in live_pairs:
        try:
            live_by_sym.setdefault(p.symbol, []).append(
                (_parse_ts(p.open_ts), p.side))
        except ValueError:
            continue
    sim_only = Counter()
    for sym, sr in sim_json.items():
        for e in sr["entries"]:
            e_ts = _parse_ts(e["ts"])
            matched = any(
                lside == e["side"] and abs(lts - e_ts) <= window
                for lts, lside in live_by_sym.get(sym, [])
            )
            if not matched:
                sim_only[(sym, e["side"])] += 1

    # ── Report ──────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Forward-Sim Calibration Diagnosis")
    lines.append("")
    lines.append(f"**Window:** {start.date()} → {end.date()} "
                 f"({args.weeks}w)  ")
    lines.append(f"**Live entries analysed:** {sum(overall.values())}  ")
    lines.append(f"**Match window:** ±{MATCH_WINDOW_MIN} min  ")
    lines.append("")
    lines.append("## Why each LIVE entry did / didn't match the sim")
    lines.append("")
    total = sum(overall.values()) or 1
    lines.append("| Verdict | Count | % | Meaning |")
    lines.append("|---|---:|---:|---|")
    meanings = {
        "match": "sim entered same side within window ✅",
        "sim_dir_opposite": "sim's structure signal pointed the OTHER way",
        "sim_no_signal": "sim saw no tradable structure (trend skip)",
        "sim_in_position": "sim was holding a different trade at that time",
        "no_decision_bar": "no sim decision near that time (cadence/data gap)",
        "sim_other": "sim decision present but unclassified",
    }
    for verdict, count in overall.most_common():
        m = meanings.get(
            verdict,
            "blocked by the %s gate" % verdict.split(":", 1)[-1]
            if verdict.startswith("blocked:") else "",
        )
        lines.append(
            f"| {verdict} | {count} | {100.0*count/total:.1f}% | {m} |")
    lines.append("")
    lines.append("## Per (symbol, side)")
    lines.append("")
    for combo in sorted(per_combo):
        sym, side = combo
        c = per_combo[combo]
        tot = sum(c.values())
        parts = ", ".join(
            f"{v}={n}" for v, n in c.most_common())
        lines.append(f"- **{sym} {side}** (n={tot}): {parts}")
    lines.append("")
    lines.append("## Sim-only entries (sim traded, live did not)")
    lines.append("")
    lines.append(f"Total sim-only: {sum(sim_only.values())}")
    lines.append("")
    for combo, n in sorted(sim_only.items()):
        lines.append(f"- {combo[0]} {combo[1]}: {n}")
    lines.append("")
    lines.append("_Note: live HOLD/skip reasons are not in trading.db, so "
                 "sim-only entries can only be counted, not attributed to a "
                 "specific live guard (orderbook / whale / news / USDT.D) "
                 "without parsing live bot logs._")
    report = "\n".join(lines) + "\n"

    out = args.output or (
        _REPO_ROOT / "docs" / "forward_sim_calibration_diagnosis.md")
    out.write_text(report)
    print(f"\nreport → {out}\n")
    # Console summary
    print("LIVE-ENTRY DIAGNOSIS:")
    for verdict, count in overall.most_common():
        print(f"  {verdict:24s} {count:4d}  {100.0*count/total:5.1f}%")
    print(f"\nSIM-ONLY ENTRIES: {sum(sim_only.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
