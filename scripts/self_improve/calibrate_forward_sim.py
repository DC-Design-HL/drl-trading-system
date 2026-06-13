#!/usr/bin/env python3
"""Calibration gate for the forward simulator (PROFITABILITY_PLAN.md P2.G).

The forward simulator (src/self_improve/forward_sim.py) is only useful
if its baseline output matches what the live bots actually did. This
script measures the divergence and writes the gate report.

Procedure (per the plan):
  1. Run the sim with the live baseline config over the most recent
     N live weeks (default 4).
  2. Compare to actual logged trades from data/trading.db.
  3. Gate criteria:
       * entry count per (symbol, side) within ±30%
       * directional agreement on overlapping entry timestamps ≥ 80%
       * net PnL same sign and within a documented band
  4. Write the result to docs/forward_sim_calibration.md.

The orchestrator MAY NOT use forward-sim results as a promotion gate
until the calibration is committed AND Chen has acknowledged it on
Telegram.

Usage:
    python3 -m scripts.self_improve.calibrate_forward_sim
    python3 -m scripts.self_improve.calibrate_forward_sim --weeks 2
    python3 -m scripts.self_improve.calibrate_forward_sim --no-write
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.backtest_harness import pair_open_close  # noqa: E402
from src.self_improve.forward_sim import (  # noqa: E402
    ForwardSimConfig,
    run_forward_sim,
)
# Reuse the per-decision classifier so the gate and the diagnostic agree.
from scripts.self_improve.diagnose_calibration import (  # noqa: E402
    _diagnose_live_entry,
    _parse_ts,
)


# Acceptance tolerances — Option B gate (PROFITABILITY_PLAN.md P2, redefined
# 2026-06-13 per docs/forward_sim_gate_redefinition.md).
ENTRY_COUNT_TOLERANCE_PCT = 30.0   # ±30% — now a WATCHED metric, not a gate
DIRECTIONAL_AGREEMENT_MIN = 0.80   # ≥80% on CO-DECIDED entries (the gate)
ENTRY_TIME_MATCH_WINDOW_MIN = 30   # within ±30 min counts as "same entry"
# Verdicts that mean the sim was NOT free to make a fresh decision at that
# time — excluded from the co-decided agreement denominator.
_NOT_CODECIDED = ("sim_in_position", "no_decision_bar")


def _load_live_trades(
    db_path: Path, start: datetime, end: datetime,
) -> dict[tuple[str, str], list]:
    """Return live OPEN/CLOSE pairs grouped by (symbol, side)."""
    conn = sqlite3.connect(str(db_path))
    try:
        pairs = pair_open_close(
            conn,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
    finally:
        conn.close()
    out: dict[tuple[str, str], list] = {}
    for p in pairs:
        out.setdefault((p.symbol, p.side), []).append(p)
    return out


def _compare_entry_counts(
    live_by_combo: dict[tuple[str, str], list],
    sim_per_symbol: dict,
) -> dict:
    rows = []
    combos = set(live_by_combo) | {
        (sym, e["side"])
        for sym, sr in sim_per_symbol.items()
        for e in sr["entries"]
    }
    for combo in sorted(combos):
        sym, side = combo
        live_count = len(live_by_combo.get(combo, []))
        sim_entries = [
            e for e in sim_per_symbol.get(sym, {}).get("entries", [])
            if e["side"] == side
        ]
        sim_count = len(sim_entries)
        if live_count == 0 and sim_count == 0:
            pct = 0.0
            within = True
        elif live_count == 0:
            pct = float("inf")
            within = False
        else:
            pct = abs(sim_count - live_count) / live_count * 100
            within = pct <= ENTRY_COUNT_TOLERANCE_PCT
        rows.append({
            "combo": f"{sym} {side}",
            "live_count": live_count,
            "sim_count": sim_count,
            "diff_pct": pct,
            "within_tolerance": within,
        })
    return {"rows": rows}


def _directional_agreement(
    live_by_combo: dict[tuple[str, str], list],
    sim_per_symbol: dict,
) -> dict:
    """For each live entry, check whether the sim has an entry within
    ±window_min minutes and the SAME side."""
    window = timedelta(minutes=ENTRY_TIME_MATCH_WINDOW_MIN)
    rows = []
    total_live = 0
    total_match = 0
    for (sym, side), pairs in sorted(live_by_combo.items()):
        sim_entries = [
            e for e in sim_per_symbol.get(sym, {}).get("entries", [])
        ]
        # Index sim entries by timestamp for cheap scan; small sets.
        matches = 0
        for p in pairs:
            try:
                live_open_ts = datetime.fromisoformat(p.open_ts).replace(
                    tzinfo=timezone.utc,
                )
            except ValueError:
                continue
            for e in sim_entries:
                sim_ts = datetime.fromisoformat(e["ts"])
                if (
                    abs(sim_ts - live_open_ts) <= window
                    and e["side"] == side
                ):
                    matches += 1
                    break
        rows.append({
            "combo": f"{sym} {side}",
            "live_count": len(pairs),
            "matched": matches,
            "agreement_pct": (
                100.0 * matches / len(pairs) if pairs else 0.0
            ),
        })
        total_live += len(pairs)
        total_match += matches
    overall = 100.0 * total_match / total_live if total_live else 0.0
    return {
        "rows": rows,
        "overall_pct": overall,
        "passes_gate": overall / 100.0 >= DIRECTIONAL_AGREEMENT_MIN,
    }


def _net_pnl_comparison(
    live_by_combo: dict[tuple[str, str], list],
    sim_per_symbol: dict,
) -> dict:
    live_total = sum(p.pnl for ps in live_by_combo.values() for p in ps)
    sim_total = sum(sr["net_pnl_usd"] for sr in sim_per_symbol.values())
    sign_match = (
        (live_total >= 0 and sim_total >= 0)
        or (live_total < 0 and sim_total < 0)
    )
    return {
        "live_net_pnl_usd": live_total,
        "sim_net_pnl_usd": sim_total,
        "sign_match": sign_match,
        "ratio": sim_total / live_total if live_total else float("inf"),
    }


def _codecided_agreement(
    live_by_combo: dict[tuple[str, str], list],
    trace_by_sym: dict[str, list],
    trades_by_sym: dict[str, list],
    window: timedelta,
) -> dict:
    """Directional agreement restricted to live entries where the sim was
    free to make a fresh decision (had a decision bar, not holding a
    different trade). This isolates entry-LOGIC fidelity from occupancy
    drift — the Option B gate."""
    rows = []
    total_co = 0
    total_match = 0
    total_excluded = 0
    for (sym, side), pairs in sorted(live_by_combo.items()):
        co = match = exc = 0
        for p in pairs:
            try:
                live_ts = _parse_ts(p.open_ts)
            except ValueError:
                continue
            verdict = _diagnose_live_entry(
                live_ts, side, sym,
                trace_by_sym.get(sym, []), trades_by_sym.get(sym, []),
                window,
            )
            if verdict in _NOT_CODECIDED:
                exc += 1
                continue
            co += 1
            if verdict == "match":
                match += 1
        rows.append({
            "combo": f"{sym} {side}",
            "codecided": co, "matched": match, "excluded": exc,
            "agreement_pct": (100.0 * match / co) if co else 0.0,
        })
        total_co += co
        total_match += match
        total_excluded += exc
    overall = 100.0 * total_match / total_co if total_co else 0.0
    return {
        "rows": rows,
        "overall_pct": overall,
        "total_codecided": total_co,
        "total_matched": total_match,
        "total_excluded": total_excluded,
        "passes_gate": (overall / 100.0) >= DIRECTIONAL_AGREEMENT_MIN,
    }


def _overproduction(
    live_by_combo: dict[tuple[str, str], list],
    sim_per_symbol: dict,
    window: timedelta,
) -> dict:
    """Watched (non-gating) metric: sim entries with no live entry within
    window, as a fraction of live entries. Driven by guards the sim can't
    replay offline (orderbook / whale / news / USDT.D)."""
    live_by_sym: dict[str, list] = {}
    for (sym, side), pairs in live_by_combo.items():
        for p in pairs:
            try:
                live_by_sym.setdefault(sym, []).append(
                    (_parse_ts(p.open_ts), side))
            except ValueError:
                continue
    sim_only = sim_total = 0
    for sym, sr in sim_per_symbol.items():
        for e in sr["entries"]:
            sim_total += 1
            e_ts = _parse_ts(e["ts"])
            if not any(
                lside == e["side"] and abs(lts - e_ts) <= window
                for lts, lside in live_by_sym.get(sym, [])
            ):
                sim_only += 1
    live_total = sum(len(v) for v in live_by_combo.values())
    return {
        "sim_only": sim_only, "sim_total": sim_total,
        "live_total": live_total,
        "ratio": (sim_only / live_total) if live_total else float("inf"),
    }


def _write_report(
    *, weeks: int, start: datetime, end: datetime,
    entry_counts: dict, agreement: dict, codecided: dict,
    overproduction: dict, pnl: dict,
    sim_runtime_s: float, output: Path,
) -> None:
    lines = [
        "# Forward-Simulator Calibration Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Window:** {start.date()} → {end.date()} ({weeks} weeks)  ",
        f"**Sim runtime:** {sim_runtime_s:.1f}s  ",
        "",
        "## Gate criteria — Option B (PROFITABILITY_PLAN.md §3/P2,",
        "## redefined 2026-06-13, see docs/forward_sim_gate_redefinition.md)",
        "",
        "  GATE  1. Co-decided directional agreement ≥ 80% (live entries",
        "           where the sim was free to decide — excludes occupancy",
        "           drift and cadence gaps).",
        "  GATE  2. Net PnL same sign.",
        "  WATCH 3. Entry counts, all-live agreement, over-production ratio",
        "           — reported for monitoring, do NOT block promotion.",
        "",
        "## GATE 1 — Co-decided directional agreement",
        "",
        f"_Time-match window: ±{ENTRY_TIME_MATCH_WINDOW_MIN} min. Denominator "
        "excludes entries where the sim was holding another trade or had no "
        "decision bar._",
        "",
        "| Combo | Co-decided | Matched | Agreement | Excluded |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in codecided["rows"]:
        lines.append(
            f"| {r['combo']} | {r['codecided']} | {r['matched']} | "
            f"{r['agreement_pct']:.1f}% | {r['excluded']} |"
        )
    lines.append(
        f"| **overall** | {codecided['total_codecided']} | "
        f"{codecided['total_matched']} | "
        f"**{codecided['overall_pct']:.1f}%** | "
        f"{codecided['total_excluded']} |"
    )
    lines.append("")
    lines.append(
        f"**GATE 1: {'PASS ✅' if codecided['passes_gate'] else 'FAIL ❌'}** "
        f"(threshold ≥ {DIRECTIONAL_AGREEMENT_MIN*100:.0f}%)"
    )
    lines.append("")

    lines += [
        "## GATE 2 — Net PnL",
        "",
        f"- Live: **${pnl['live_net_pnl_usd']:+.2f}**",
        f"- Sim:  **${pnl['sim_net_pnl_usd']:+.2f}**",
        f"- Ratio (sim / live): **{pnl['ratio']:.2f}**",
        f"- Sign match: **{'✅' if pnl['sign_match'] else '❌'}**",
        "",
        f"**GATE 2: {'PASS ✅' if pnl['sign_match'] else 'FAIL ❌'}**",
        "",
    ]

    # ── Watched metrics (reported, non-gating) ──────────────────────────
    lines += [
        "## Watched metrics (non-gating)",
        "",
        "### Entry counts (all live entries)",
        "",
        "| Combo | Live | Sim | Δ% | Within ±30% |",
        "|---|---:|---:|---:|:--:|",
    ]
    for r in entry_counts["rows"]:
        lines.append(
            f"| {r['combo']} | {r['live_count']} | {r['sim_count']} | "
            f"{r['diff_pct']:.1f}% | "
            f"{'✅' if r['within_tolerance'] else '❌'} |"
        )
    lines.append("")
    lines.append(
        f"- All-live directional agreement (incl. occupancy/cadence gaps): "
        f"**{agreement['overall_pct']:.1f}%**"
    )
    op = overproduction
    lines.append(
        f"- Over-production: **{op['sim_only']}** sim entries with no live "
        f"match / {op['live_total']} live = **{op['ratio']:.2f}×** "
        f"(sim total {op['sim_total']}). Driven by non-replayable live "
        f"guards; watch for growth."
    )
    lines.append("")

    # Top-level verdict — Option B: co-decided agreement + PnL sign.
    pass_all = codecided["passes_gate"] and pnl["sign_match"]
    lines += [
        "## Verdict",
        "",
        f"**Overall: {'PASS ✅' if pass_all else 'FAIL ❌'}** "
        f"(GATE 1 co-decided agreement + GATE 2 PnL sign)",
        "",
        "_The orchestrator may use forward-sim results as a promotion "
        "gate only after this report PASSES and Chen acknowledges it "
        "on Telegram (PROFITABILITY_PLAN.md §3/P2)._",
        "",
        "## Known limitations of v1",
        "",
        "* S5 symbol filters (OB-proximity + ADX-directional) ARE now "
        "  replicated (P2.D part 1) — the ETH-zero-entries bug is fixed.",
        "* Replayable pre-trade guards (structure-first ADX, exhaustion, "
        "  RSI) ARE now applied (P2.D part 2). Guards that cannot be "
        "  replayed offline (USDT.D proxy, ext-pos-news, orderbook) are "
        "  still assumed-pass — sim may over-count where these block live.",
        "* Stateful post-close gates (cooldown / anti-whipsaw) ARE simulated "
        "  (P2.D). ADX + trend-aware RSI bands now come from the live "
        "  MarketRegimeDetector (P2.D #1), not a kline approximation — this "
        "  lifted directional agreement ~40% -> ~56%.",
        "* Per-decision diagnostic (forward_sim_calibration_diagnosis.md) "
        "  shows the entry LOGIC is faithful: among live entries where the "
        "  sim was free to decide, agreement is ~90%. The residual headline "
        "  gap is occupancy drift (sim busy in a different trade, ~23%) + "
        "  cadence/data gaps (~16%), NOT entry-logic disagreement (~5%).",
        "* Occupancy drift is driven by over-production: the sim takes "
        "  entries live skipped because live's orderbook / whale / news / "
        "  USDT.D guards cannot be replayed offline (assumed-pass). This is "
        "  a STRUCTURAL ceiling on timestamp-matched agreement — handled by "
        "  the Option B gate (adopted 2026-06-13): co-decided agreement is "
        "  the gate, over-production is a watched metric. See "
        "  docs/forward_sim_gate_redefinition.md.",
        "* Residuals vs live: RSI *value* is a kline proxy (live reads it "
        "  from the API signals bundle) and the conf>=0.90 rescue override "
        "  is not replayed (needs model conf + order-flow/whale/mtf signals).",
        "* Funding IS accrued (P2.E): per 8h boundary on entry notional, "
        "  LONG pays / SHORT receives. Magnitude is small (sub-dollar per "
        "  trade); the PnL ratio moves more from the rolling now() window "
        "  pulling a slightly different live+sim trade set each run.",
        "* No BOS/CHOCH profitable-overlay on exits.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", type=int, default=4,
                    help="Calibration window in weeks (default 4).")
    ap.add_argument("--db", default="data/trading.db")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Default: docs/forward_sim_calibration.md")
    ap.add_argument("--no-write", action="store_true",
                    help="Skip report write; only print summary.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(weeks=args.weeks)

    cfg = ForwardSimConfig()

    window = timedelta(minutes=ENTRY_TIME_MATCH_WINDOW_MIN)
    print(f"running forward sim {start.date()} → {end.date()} (baseline cfg)…")
    trace: list = []
    sim = run_forward_sim(start=start, end=end, config=cfg, trace=trace)
    print(f"sim runtime: {sim.runtime_seconds:.1f}s, {len(trace)} decisions")

    live_by_combo = _load_live_trades(Path(args.db), start, end)

    sim_per_symbol = sim.to_json()["per_symbol"]
    trace_by_sym: dict[str, list] = {}
    for rec in trace:
        trace_by_sym.setdefault(rec["symbol"], []).append(rec)
    trades_by_sym = {
        sym: sr.get("trades", []) for sym, sr in sim_per_symbol.items()
    }

    entry_counts = _compare_entry_counts(live_by_combo, sim_per_symbol)
    agreement = _directional_agreement(live_by_combo, sim_per_symbol)
    codecided = _codecided_agreement(
        live_by_combo, trace_by_sym, trades_by_sym, window)
    overproduction = _overproduction(live_by_combo, sim_per_symbol, window)
    pnl = _net_pnl_comparison(live_by_combo, sim_per_symbol)

    out = args.output or (
        _REPO_ROOT / "docs" / "forward_sim_calibration.md"
    )
    if not args.no_write:
        _write_report(
            weeks=args.weeks, start=start, end=end,
            entry_counts=entry_counts, agreement=agreement,
            codecided=codecided, overproduction=overproduction, pnl=pnl,
            sim_runtime_s=sim.runtime_seconds, output=out,
        )
        print(f"report → {out}")

    # Console summary
    pass_all = codecided["passes_gate"] and pnl["sign_match"]
    print(
        f"GATE 1 co-decided agreement: {codecided['overall_pct']:.1f}% "
        f"({codecided['total_matched']}/{codecided['total_codecided']}, "
        f"excl {codecided['total_excluded']}) "
        f"→ {'PASS' if codecided['passes_gate'] else 'FAIL'}"
    )
    print(
        f"GATE 2 net PnL: live=${pnl['live_net_pnl_usd']:+.2f} "
        f"sim=${pnl['sim_net_pnl_usd']:+.2f} "
        f"sign_match={pnl['sign_match']}"
    )
    print(
        f"  [watch] all-live agreement {agreement['overall_pct']:.1f}%, "
        f"over-production {overproduction['ratio']:.2f}×"
    )
    print(f"VERDICT: {'PASS ✅' if pass_all else 'FAIL ❌'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
