#!/usr/bin/env python3
"""Ground-truth report (PROFITABILITY_PLAN.md P0).

Builds a markdown report from the live SQLite trade log + bot logs.
No behavior changes — pure read.

Usage:
    python3 -m scripts.self_improve.ground_truth_report
    python3 -m scripts.self_improve.ground_truth_report --no-funding
    python3 -m scripts.self_improve.ground_truth_report --output path.md

Sections (per PROFITABILITY_PLAN.md §3/P0):
    1. Headline metrics — 7d / 30d / since 2026-05-01, portfolio + per (symbol, side)
    2. Exit-reason breakdown — PnL & count by close reason
    3. Confidence calibration — structure-confidence deciles vs WR and expectancy
    4. MFE/MAE analysis — losers' favorable excursion, TP-hit winners' overshoot
    5. Guard counterfactuals — block counts per guard from logs/bots_live.log
    6. Self-improve audit — experiments / decisions / agent_runs cost totals
    7. Funding estimate — historical funding rates × held notional, per trade
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.backtest_harness import pair_open_close  # noqa: E402
from src.self_improve.metrics import (  # noqa: E402
    TradeClose,
    max_drawdown_pct,
    net_pnl,
    parse_ts,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

DB_PATH = _REPO_ROOT / "data" / "trading.db"
LOG_PATH = _REPO_ROOT / "logs" / "bots_live.log"
DEFAULT_CAPITAL_BASE = 5000.0
RESET_DATE = "2026-05-01T12:29:00"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


# ── Section 1: headline ──────────────────────────────────────────────────


def _trades_close_view(
    conn: sqlite3.Connection, *, start_iso: str, end_iso: str
) -> list[TradeClose]:
    """Pair OPENs/CLOSEs and return a TradeClose list, in close-time order."""
    pairs = pair_open_close(conn, start_date=start_iso, end_date=end_iso)
    out: list[TradeClose] = []
    for p in pairs:
        out.append(
            TradeClose(
                ts=parse_ts(p.close_ts),
                symbol=p.symbol,
                side=p.side,
                pnl=p.pnl,
            )
        )
    return out


def _fmt_metrics(label: str, closes: list[TradeClose]) -> dict[str, Any]:
    return {
        "window": label,
        "n": len(closes),
        "net_pnl": net_pnl(closes),
        "win_rate": win_rate(closes),
        "profit_factor": profit_factor(closes),
        "sharpe": sharpe_ratio(closes, capital_base=DEFAULT_CAPITAL_BASE),
        "sortino": sortino_ratio(closes, capital_base=DEFAULT_CAPITAL_BASE),
        "max_dd_pct": max_drawdown_pct(closes, capital_base=DEFAULT_CAPITAL_BASE),
    }


def section_headline(conn: sqlite3.Connection, now: datetime) -> str:
    windows = [
        ("7d", now - timedelta(days=7)),
        ("30d", now - timedelta(days=30)),
        ("since-reset", parse_ts(RESET_DATE)),
    ]
    lines = ["## 1. Headline metrics", ""]
    end_iso = now.isoformat()

    # Portfolio
    lines.append("### Portfolio")
    lines.append("")
    lines.append("| Window | n | Net PnL ($) | WR | PF | Sharpe | Sortino | MaxDD% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    portfolio_closes: dict[str, list[TradeClose]] = {}
    for label, start in windows:
        closes = _trades_close_view(
            conn, start_iso=start.isoformat(), end_iso=end_iso
        )
        portfolio_closes[label] = closes
        m = _fmt_metrics(label, closes)
        lines.append(
            f"| {label} | {m['n']} | {m['net_pnl']:+.2f} | "
            f"{m['win_rate']*100:.1f}% | {m['profit_factor']:.2f} | "
            f"{m['sharpe']:.2f} | {m['sortino']:.2f} | {m['max_dd_pct']:.2f} |"
        )
    lines.append("")

    # Per (symbol, side) — over since-reset window
    lines.append("### Per (symbol, side) — since 2026-05-01 reset")
    lines.append("")
    lines.append("| Symbol/Side | n | Net PnL ($) | WR | PF |")
    lines.append("|---|---:|---:|---:|---:|")
    since_closes = portfolio_closes["since-reset"]
    by_combo: dict[tuple[str, str], list[TradeClose]] = defaultdict(list)
    for t in since_closes:
        by_combo[(t.symbol, t.side)].append(t)
    for key in sorted(by_combo.keys()):
        subset = by_combo[key]
        lines.append(
            f"| {key[0]} {key[1]} | {len(subset)} | {net_pnl(subset):+.2f} | "
            f"{win_rate(subset)*100:.1f}% | {profit_factor(subset):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Section 2: exit-reason breakdown ──────────────────────────────────────


def section_exit_reasons(conn: sqlite3.Connection, since_iso: str) -> str:
    """Group CLOSE rows by `reason` field — PnL & count."""
    rows = conn.execute(
        """
        SELECT reason, pnl
        FROM trades
        WHERE is_testnet = 1
          AND timestamp >= ?
          AND action LIKE 'CLOSE_%'
        """,
        (since_iso,),
    ).fetchall()

    bucket: dict[str, list[float]] = defaultdict(list)
    for reason, pnl in rows:
        if pnl is None:
            continue
        bucket[reason or "(none)"].append(float(pnl))

    lines = [
        "## 2. Exit-reason breakdown (since 2026-05-01)",
        "",
        "| Reason | n | Net PnL ($) | Avg ($) | WR |",
        "|---|---:|---:|---:|---:|",
    ]
    total_n = sum(len(v) for v in bucket.values())
    for reason in sorted(bucket.keys(), key=lambda r: -sum(bucket[r])):
        pnls = bucket[reason]
        n = len(pnls)
        net = sum(pnls)
        avg = net / n if n else 0.0
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n if n else 0.0
        lines.append(
            f"| {reason} | {n} ({n*100/total_n:.0f}%) | {net:+.2f} | "
            f"{avg:+.2f} | {wr*100:.0f}% |"
        )
    lines.append("")
    lines.append(f"_Total closes: {total_n}_")
    lines.append("")
    return "\n".join(lines)


# ── Section 3: confidence calibration ─────────────────────────────────────


def section_confidence_calibration(
    conn: sqlite3.Connection, since_iso: str
) -> str:
    """Bucket OPEN confidence into deciles, look at paired PnL outcomes."""
    end_iso = datetime.now(timezone.utc).isoformat()
    pairs = pair_open_close(conn, start_date=since_iso, end_date=end_iso)
    by_decile: dict[int, list[float]] = defaultdict(list)
    for p in pairs:
        c = p.confidence
        # Decile 0..9; values >=1.0 fold into 9
        d = min(int(c * 10), 9)
        by_decile[d].append(p.pnl)

    lines = [
        "## 3. Confidence calibration (structure-confidence deciles)",
        "",
        "_Note: in structure-first mode `confidence` on the OPEN row is the "
        "BOS/CHOCH signal confidence, **not** PPO model confidence._",
        "",
        "| Decile | Range | n | Net PnL ($) | Avg ($) | WR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for d in range(10):
        pnls = by_decile.get(d, [])
        n = len(pnls)
        if n == 0:
            lines.append(f"| D{d} | [{d/10:.1f},{(d+1)/10:.1f}) | 0 | — | — | — |")
            continue
        net = sum(pnls)
        avg = net / n
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / n
        lines.append(
            f"| D{d} | [{d/10:.1f},{(d+1)/10:.1f}) | {n} | {net:+.2f} | "
            f"{avg:+.2f} | {wr*100:.0f}% |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Section 4: MFE/MAE analysis ───────────────────────────────────────────


def section_mfe_mae(conn: sqlite3.Connection, since_iso: str) -> str:
    """For losers — how far did MFE go before reversal. For TP_HIT winners — what MFE."""
    rows = conn.execute(
        """
        SELECT data, pnl, reason
        FROM trades
        WHERE is_testnet = 1
          AND timestamp >= ?
          AND action LIKE 'CLOSE_%'
        """,
        (since_iso,),
    ).fetchall()

    losers_mfe: list[float] = []
    losers_mae: list[float] = []
    losers_with_pos_mfe = 0
    winners_tp_mfe: list[float] = []
    parsed = 0
    missing = 0
    for data, pnl, reason in rows:
        if pnl is None or not data:
            continue
        try:
            d = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            continue
        mfe = d.get("mfe_pct")
        mae = d.get("mae_pct")
        if mfe is None or mae is None:
            missing += 1
            continue
        parsed += 1
        mfe = float(mfe)
        mae = float(mae)
        if pnl < 0:
            losers_mfe.append(mfe)
            losers_mae.append(mae)
            if mfe > 0.005:  # >0.5% favorable excursion before reversing
                losers_with_pos_mfe += 1
        elif reason in ("TP", "TP_HIT") or (reason and "TP" in reason and "PARTIAL" not in reason):
            winners_tp_mfe.append(mfe)

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    lines = [
        "## 4. MFE/MAE analysis (since 2026-05-01)",
        "",
        f"_Parsed {parsed} closes with MFE/MAE; {missing} closes had no MFE/MAE field (pre-Phase 1)._",
        "",
        "### Losers' favorable excursion (was there a better exit?)",
        "",
        f"- Losing trades with MFE/MAE: **{len(losers_mfe)}**",
        f"- Avg MFE (peak favorable %): **{_avg(losers_mfe)*100:+.2f}%**",
        f"- Avg MAE (peak adverse %):   **{_avg(losers_mae)*100:+.2f}%**",
        f"- Losers that went **>0.5% in our favor** before reversing: "
        f"**{losers_with_pos_mfe}** "
        f"({losers_with_pos_mfe*100/max(len(losers_mfe),1):.0f}% of losers)",
        "",
        "### TP-hit winners' overshoot (are TPs too tight?)",
        "",
        f"- Full-TP winners with MFE/MAE: **{len(winners_tp_mfe)}**",
        f"- Avg MFE (peak): **{_avg(winners_tp_mfe)*100:+.2f}%**",
        "",
    ]
    return "\n".join(lines)


# ── Section 5: guard counterfactuals from logs ───────────────────────────


_GUARD_PATTERNS = [
    ("blocklist", "🚫 Structure-first: skipping"),
    ("rsi_adx_guard", "🚫 RSI/ADX GUARD BLOCK"),
    ("regime_veto", "🚫 Regime HARD-VETO"),
    ("signal_gate", "🚫 Signal gate BLOCK"),
    ("anti_whipsaw", "🚫 Anti-whipsaw"),
    ("directional_floor", "🚫 Directional floor"),
    ("structure_first_adx", "🚫 Structure-first ADX block"),
    ("ranging_regime", "🚫 Ranging regime filter"),
    ("exhaustion", "🚫 Exhaustion filter"),
    ("usdt_d", "🛡️ USDT.D FILTER BLOCK"),
    ("ext_pos_news", "🛡️ EXT_POS_NEWS BLOCK"),
    ("orderbook", "🛡️ ORDERBOOK GUARD BLOCK"),
    ("s5_ob_proximity", "Structure-first S5: blocked by OB proximity"),
    ("s5_adx_dir", "Structure-first S5: ADX directional block"),
]


def section_guard_counterfactuals(since_iso: str) -> str:
    """Grep logs/bots_live.log for guard-block markers, count per guard."""
    if not LOG_PATH.exists():
        return "## 5. Guard counterfactuals\n\n_logs/bots_live.log not found — skipped._\n\n"

    counts: dict[str, int] = defaultdict(int)
    since_prefix = since_iso[:19]  # log ts format: '2026-05-22 16:18:40'

    with open(LOG_PATH, "r", errors="ignore") as f:
        for line in f:
            # log line starts with 'YYYY-MM-DD HH:MM:SS'
            if len(line) < 19:
                continue
            ts = line[:19].replace(" ", "T")
            if ts < since_prefix:
                continue
            for name, marker in _GUARD_PATTERNS:
                if marker in line:
                    counts[name] += 1
                    break

    total = sum(counts.values())
    lines = [
        "## 5. Guard counterfactuals (since 2026-05-01, from logs/bots_live.log)",
        "",
        "| Guard | Block count | % of all blocks |",
        "|---|---:|---:|",
    ]
    for name, _ in _GUARD_PATTERNS:
        n = counts.get(name, 0)
        pct = n * 100 / total if total else 0
        lines.append(f"| {name} | {n} | {pct:.1f}% |")
    lines.append(f"| **total** | **{total}** | 100% |")
    lines.append("")
    return "\n".join(lines)


# ── Section 6: self-improve audit ────────────────────────────────────────


def section_self_improve_audit(conn: sqlite3.Connection) -> str:
    lines = ["## 6. Self-improve audit", ""]

    # Experiments by stage
    lines.append("### Experiments by stage")
    lines.append("")
    lines.append("| Stage | n |")
    lines.append("|---|---:|")
    rows = conn.execute(
        "SELECT stage, COUNT(*) FROM experiments GROUP BY stage ORDER BY stage"
    ).fetchall()
    for stage, n in rows:
        lines.append(f"| {stage or '(none)'} | {n} |")
    lines.append("")

    # Recent rolled-back experiments — rejection reasons
    lines.append("### Rolled-back experiments — reasons")
    lines.append("")
    rows = conn.execute(
        """
        SELECT id, ts_rolled_back, rollback_reason
        FROM experiments
        WHERE rollback_reason IS NOT NULL
        ORDER BY ts_rolled_back DESC
        LIMIT 10
        """
    ).fetchall()
    if rows:
        lines.append("| Exp # | Rolled back | Reason |")
        lines.append("|---|---|---|")
        for exp_id, ts, reason in rows:
            short = (reason or "")[:120].replace("|", "/").replace("\n", " ")
            lines.append(f"| {exp_id} | {ts or ''} | {short} |")
    else:
        lines.append("_No rolled-back experiments._")
    lines.append("")

    # Decisions outcomes
    lines.append("### Decisions outcomes")
    lines.append("")
    lines.append("| Outcome | n |")
    lines.append("|---|---:|")
    rows = conn.execute(
        "SELECT outcome, COUNT(*) FROM decisions GROUP BY outcome"
    ).fetchall()
    for outcome, n in rows:
        lines.append(f"| {outcome or '(none)'} | {n} |")
    lines.append("")

    # Agent run costs
    lines.append("### Agent run costs (tokens)")
    lines.append("")
    rows = conn.execute(
        """
        SELECT agent,
               COUNT(*),
               COALESCE(SUM(input_tokens), 0),
               COALESCE(SUM(output_tokens), 0),
               COALESCE(SUM(duration_s), 0)
        FROM agent_runs
        GROUP BY agent
        ORDER BY agent
        """
    ).fetchall()
    if rows:
        lines.append("| Agent | runs | in tokens | out tokens | total seconds |")
        lines.append("|---|---:|---:|---:|---:|")
        for agent, n, i, o, dur in rows:
            lines.append(f"| {agent} | {n} | {int(i):,} | {int(o):,} | {dur:.0f} |")
    else:
        lines.append("_No agent runs recorded._")
    lines.append("")
    return "\n".join(lines)


# ── Section 7: funding estimate ──────────────────────────────────────────


def section_funding_estimate(
    conn: sqlite3.Connection, *, since_iso: str, enable: bool
) -> str:
    """Estimate funding paid per trade via mainnet historical funding rates.

    For each closed pair (open_ts, close_ts, symbol, side, notional):
        for each 8h funding boundary inside [open_ts, close_ts]:
            funding = notional * rate * sign(side)   # LONG pays positive rate
        sum across all trades.

    Per P0 spec: testnet doesn't reliably serve funding history — we use
    mainnet rates as proxy and flag this.
    """
    if not enable:
        return ("## 7. Funding estimate\n\n"
                "_Skipped (--no-funding). Use without the flag to populate._\n\n")

    # Gather closed pairs in window WITH notional via OPEN data.
    end_iso = datetime.now(timezone.utc).isoformat()
    pairs = pair_open_close(conn, start_date=since_iso, end_date=end_iso)
    # Map open_id -> notional from data field
    open_data: dict[int, dict[str, Any]] = {}
    rows = conn.execute(
        """
        SELECT id, data FROM trades
        WHERE is_testnet=1 AND action LIKE 'OPEN_%' AND timestamp >= ?
        """,
        (since_iso,),
    ).fetchall()
    for oid, data in rows:
        if not data:
            continue
        try:
            open_data[oid] = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            continue

    # Fetch mainnet funding-rate history once per symbol over window.
    try:
        import ccxt  # noqa: F401
    except ImportError:
        return ("## 7. Funding estimate\n\n"
                "_ccxt not available — skipped._\n\n")

    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    since_ms = int(parse_ts(since_iso).timestamp() * 1000)

    funding_by_symbol: dict[str, list[tuple[int, float]]] = {}
    fetch_errors: list[str] = []
    for sym in SYMBOLS:
        try:
            cursor = since_ms
            all_rows: list[tuple[int, float]] = []
            for _ in range(20):  # bounded — ~1000 funding events per page
                page = ex.fetchFundingRateHistory(sym, since=cursor, limit=1000)
                if not page:
                    break
                for r in page:
                    all_rows.append((int(r["timestamp"]), float(r["fundingRate"])))
                cursor = page[-1]["timestamp"] + 1
                if len(page) < 1000:
                    break
            funding_by_symbol[sym] = all_rows
        except Exception as exc:  # noqa: BLE001
            fetch_errors.append(f"{sym}: {exc}")
            funding_by_symbol[sym] = []

    # Per-symbol accrual
    total_funding = 0.0
    per_symbol: dict[str, float] = {s: 0.0 for s in SYMBOLS}
    paired_trades = 0
    for p in pairs:
        d = open_data.get(p.open_id)
        if not d:
            continue
        notional = d.get("trade_value") or d.get("units", 0) * d.get("price", 0)
        if not notional:
            continue
        open_ms = int(parse_ts(p.open_ts).timestamp() * 1000)
        close_ms = int(parse_ts(p.close_ts).timestamp() * 1000)
        sign = 1 if p.side == "LONG" else -1
        for ts_ms, rate in funding_by_symbol.get(p.symbol, []):
            if open_ms <= ts_ms <= close_ms:
                # LONG pays positive funding to SHORT, so LONG funding = -notional*rate
                pay = -sign * notional * rate
                per_symbol[p.symbol] += pay
                total_funding += pay
                paired_trades += 1

    lines = [
        "## 7. Funding estimate (mainnet rates as proxy for testnet)",
        "",
        "_Testnet does not reliably serve funding history; this section uses "
        "**mainnet** funding rates × position notional summed across each 8h "
        "boundary the trade spanned. Sign: positive = received, negative = paid._",
        "",
    ]
    if fetch_errors:
        lines.append("**Fetch errors:**")
        for e in fetch_errors:
            lines.append(f"- {e}")
        lines.append("")
    lines.append("| Symbol | Funding net ($) |")
    lines.append("|---|---:|")
    for s in SYMBOLS:
        lines.append(f"| {s} | {per_symbol[s]:+.2f} |")
    lines.append(f"| **total** | **{total_funding:+.2f}** |")
    lines.append("")
    lines.append(f"_Trade-funding boundary crossings recorded: {paired_trades}._")
    lines.append("")
    return "\n".join(lines)


# ── Top-level: assemble report ───────────────────────────────────────────


def build_report(
    *, enable_funding: bool = True, now: datetime | None = None
) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        sections: list[str] = []
        sections.append(
            "# Ground-Truth Report — DRL Trading System\n\n"
            f"**Generated:** {now.isoformat()}  \n"
            f"**Repo HEAD:** see git log on `feature/profitability-p0`  \n"
            f"**Data source:** `data/trading.db` (is_testnet=1 only) + "
            f"`logs/bots_live.log`  \n"
            f"**Capital base for return metrics:** "
            f"${DEFAULT_CAPITAL_BASE:.0f} (May-1 reset)\n"
        )
        sections.append(section_headline(conn, now))
        sections.append(section_exit_reasons(conn, RESET_DATE))
        sections.append(section_confidence_calibration(conn, RESET_DATE))
        sections.append(section_mfe_mae(conn, RESET_DATE))
        sections.append(section_guard_counterfactuals(RESET_DATE))
        sections.append(section_self_improve_audit(conn))
        sections.append(section_funding_estimate(
            conn, since_iso=RESET_DATE, enable=enable_funding
        ))
        return "\n".join(sections)
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path. Default: docs/ground_truth/<UTC-date>-report.md",
    )
    ap.add_argument(
        "--no-funding",
        action="store_true",
        help="Skip funding estimate (no network calls).",
    )
    args = ap.parse_args()

    out_dir = _REPO_ROOT / "docs" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = args.output or (
        out_dir / f"{datetime.now(timezone.utc).date().isoformat()}-report.md"
    )

    report = build_report(enable_funding=not args.no_funding)
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
