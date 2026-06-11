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


# Acceptance tolerances — match PROFITABILITY_PLAN.md P2 spec.
ENTRY_COUNT_TOLERANCE_PCT = 30.0   # ±30%
DIRECTIONAL_AGREEMENT_MIN = 0.80   # ≥80%
ENTRY_TIME_MATCH_WINDOW_MIN = 30   # within ±30 min counts as "same entry"


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


def _write_report(
    *, weeks: int, start: datetime, end: datetime,
    entry_counts: dict, agreement: dict, pnl: dict,
    sim_runtime_s: float, output: Path,
) -> None:
    lines = [
        "# Forward-Simulator Calibration Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Window:** {start.date()} → {end.date()} ({weeks} weeks)  ",
        f"**Sim runtime:** {sim_runtime_s:.1f}s  ",
        "",
        "## Gate criteria (PROFITABILITY_PLAN.md §3/P2)",
        "",
        "  1. Entry count per (symbol, side) within ±30%",
        "  2. Directional agreement on overlapping entries ≥ 80%",
        "  3. Net PnL same sign",
        "",
        "## 1. Entry counts",
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
    all_within = all(r["within_tolerance"] for r in entry_counts["rows"])
    lines.append("")
    lines.append(f"**Entry-count gate: {'PASS' if all_within else 'FAIL'}**")
    lines.append("")

    lines += [
        "## 2. Directional agreement",
        "",
        f"_Time-match window: ±{ENTRY_TIME_MATCH_WINDOW_MIN} minutes_",
        "",
        "| Combo | Live entries | Matched in sim | Agreement |",
        "|---|---:|---:|---:|",
    ]
    for r in agreement["rows"]:
        lines.append(
            f"| {r['combo']} | {r['live_count']} | {r['matched']} | "
            f"{r['agreement_pct']:.1f}% |"
        )
    lines.append(f"| **overall** | — | — | **{agreement['overall_pct']:.1f}%** |")
    lines.append("")
    lines.append(
        f"**Directional-agreement gate: "
        f"{'PASS' if agreement['passes_gate'] else 'FAIL'}** "
        f"(threshold ≥ {DIRECTIONAL_AGREEMENT_MIN*100:.0f}%)"
    )
    lines.append("")

    lines += [
        "## 3. Net PnL",
        "",
        f"- Live: **${pnl['live_net_pnl_usd']:+.2f}**",
        f"- Sim:  **${pnl['sim_net_pnl_usd']:+.2f}**",
        f"- Ratio (sim / live): **{pnl['ratio']:.2f}**",
        f"- Sign match: **{'✅' if pnl['sign_match'] else '❌'}**",
        "",
    ]

    # Top-level verdict
    pass_all = all_within and agreement["passes_gate"] and pnl["sign_match"]
    lines += [
        "## Verdict",
        "",
        f"**Overall: {'PASS ✅' if pass_all else 'FAIL ❌'}**",
        "",
        "_The orchestrator may use forward-sim results as a promotion "
        "gate only after this report PASSES and Chen acknowledges it "
        "on Telegram (PROFITABILITY_PLAN.md §3/P2)._",
        "",
        "## Known limitations of v1",
        "",
        "* S5 symbol filters (OB-proximity + ADX-directional) are NOT "
        "  replicated yet — ETH entries will be undercounted.",
        "* Pre-trade guards (RSI, ADX, exhaustion, USDT.D, ext-pos-news, "
        "  anti-whipsaw, cooldown, min-hold) NOT applied — sim entries "
        "  may be over-counted vs live where these guards block.",
        "* Funding accrual not yet wired (P2.E); fees + slippage only.",
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

    print(f"running forward sim {start.date()} → {end.date()} (baseline cfg)…")
    sim = run_forward_sim(start=start, end=end, config=cfg)
    print(f"sim runtime: {sim.runtime_seconds:.1f}s")

    live_by_combo = _load_live_trades(Path(args.db), start, end)

    sim_per_symbol = sim.to_json()["per_symbol"]
    entry_counts = _compare_entry_counts(live_by_combo, sim_per_symbol)
    agreement = _directional_agreement(live_by_combo, sim_per_symbol)
    pnl = _net_pnl_comparison(live_by_combo, sim_per_symbol)

    out = args.output or (
        _REPO_ROOT / "docs" / "forward_sim_calibration.md"
    )
    if not args.no_write:
        _write_report(
            weeks=args.weeks, start=start, end=end,
            entry_counts=entry_counts, agreement=agreement, pnl=pnl,
            sim_runtime_s=sim.runtime_seconds, output=out,
        )
        print(f"report → {out}")

    # Console summary
    for r in entry_counts["rows"]:
        flag = "OK" if r["within_tolerance"] else "FAIL"
        print(
            f"  {r['combo']:<14} live={r['live_count']:>3} "
            f"sim={r['sim_count']:>3} Δ={r['diff_pct']:>5.1f}%  {flag}"
        )
    print(
        f"overall directional agreement: {agreement['overall_pct']:.1f}% "
        f"({'PASS' if agreement['passes_gate'] else 'FAIL'})"
    )
    print(
        f"net PnL: live=${pnl['live_net_pnl_usd']:+.2f} "
        f"sim=${pnl['sim_net_pnl_usd']:+.2f} "
        f"sign_match={pnl['sign_match']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
