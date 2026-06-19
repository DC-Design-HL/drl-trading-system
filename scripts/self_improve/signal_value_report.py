#!/usr/bin/env python3
"""Signal-value report (PROFITABILITY_PLAN.md P5 §2).

Correlates the entry-time signal snapshots (P5 `entry_signals`, type='entry')
with the realized outcome of the trade they opened, and reports each signal's
CONDITIONAL EXPECTANCY — data, not opinion, on which signals deserve to gate
or drive entries:

  * model agreement — did the PPO model's action agree with the structure-
    first side we actually took? avg PnL + win rate when it agreed vs not.
  * whale alignment — did the whale-behaviour signal point the same way?
  * (extensible) any other snapshotted signal.

Only runs the analysis once at least ``min_closes`` (default 100) entry
snapshots have a paired outcome — below that the conditional buckets are
statistical noise. Writes markdown to docs/ground_truth/ and returns a
one-line Telegram digest of the strongest finding.

Usage:
  python3 -m scripts.self_improve.signal_value_report            # write report
  python3 -m scripts.self_improve.signal_value_report --dry-run  # print only
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[2]
_DB = _REPO / "data" / "trading.db"
_OUT_DIR = _REPO / "docs" / "ground_truth"
MIN_CLOSES = 100

_OPEN = ("OPEN_LONG", "OPEN_SHORT")
_CLOSE = ("CLOSE_LONG", "CLOSE_SHORT", "REVERSE_CLOSE_LONG",
          "REVERSE_CLOSE_SHORT", "SL_HIT", "TP_HIT")


def _pnl_by_open(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """FIFO-pair OPEN→CLOSE per symbol; return {(symbol, open_ts): close_pnl}.

    The OPEN row's `timestamp` is the exact key the entry_signals snapshot was
    stamped with, so the snapshot joins to its outcome on (symbol, ts)."""
    rows = conn.execute(
        "SELECT id, timestamp, symbol, action, pnl FROM trades "
        "WHERE is_testnet=1 ORDER BY id"
    ).fetchall()
    open_stack: dict[str, list[str]] = {}
    out: dict[tuple[str, str], float] = {}
    for _id, ts, symbol, action, pnl in rows:
        if action in _OPEN:
            open_stack.setdefault(symbol, []).append(ts)
        elif action in _CLOSE and pnl is not None:
            stack = open_stack.get(symbol)
            if stack:
                open_ts = stack.pop(0)
                out[(symbol, open_ts)] = float(pnl)
    return out


def load_entry_outcomes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Entry snapshots joined to their realized PnL. Each dict has side,
    structure_conf, model_action, signals, and pnl (the outcome)."""
    try:
        rows = conn.execute(
            "SELECT ts, symbol, side, structure_conf, model_action, "
            "model_confidence, signals_json FROM entry_signals "
            "WHERE snapshot_type='entry'"
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # P5 schema not migrated yet
    pnl_map = _pnl_by_open(conn)
    out: list[dict[str, Any]] = []
    for ts, symbol, side, sconf, m_action, m_conf, sigs in rows:
        key = (symbol, ts)
        if key not in pnl_map:
            continue  # not yet closed (or unpaired)
        try:
            signals = json.loads(sigs) if sigs else {}
        except json.JSONDecodeError:
            signals = {}
        out.append({
            "symbol": symbol, "side": side, "structure_conf": sconf,
            "model_action": m_action, "model_confidence": m_conf,
            "signals": signals, "pnl": pnl_map[key],
        })
    return out


def _bucket(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if not n:
        return {"n": 0, "avg_pnl": 0.0, "win_rate": 0.0, "total_pnl": 0.0}
    total = sum(r["pnl"] for r in rows)
    wins = sum(1 for r in rows if r["pnl"] > 0)
    return {"n": n, "avg_pnl": round(total / n, 2),
            "win_rate": round(wins / n, 3), "total_pnl": round(total, 2)}


def _whale_supports(signals: dict, side: str) -> Optional[bool]:
    """True if the whale signal points the same way as the trade side.
    direction > 0.5 = buy bias. None if no usable whale signal."""
    whale = signals.get("whale") or {}
    direction = whale.get("direction")
    if direction is None or whale.get("intent") == "unavailable":
        return None
    buy_bias = float(direction) > 0.5
    return buy_bias == (str(side).upper() == "LONG")


def compute_signal_value(
    conn: sqlite3.Connection, *, min_closes: int = MIN_CLOSES,
) -> dict[str, Any]:
    rows = load_entry_outcomes(conn)
    n = len(rows)
    if n < min_closes:
        return {"ready": False, "n": n, "min_closes": min_closes}

    # Model agreement: did PPO action match the side we took?
    agree = [r for r in rows if r.get("model_action")
             and str(r["model_action"]).upper() == str(r["side"]).upper()]
    disagree = [r for r in rows if r.get("model_action")
                and str(r["model_action"]).upper() != str(r["side"]).upper()]

    # Whale alignment.
    aligned = [r for r in rows if _whale_supports(r["signals"], r["side"]) is True]
    opposed = [r for r in rows if _whale_supports(r["signals"], r["side"]) is False]

    return {
        "ready": True, "n": n,
        "overall": _bucket(rows),
        "model_agree": _bucket(agree),
        "model_disagree": _bucket(disagree),
        "whale_aligned": _bucket(aligned),
        "whale_opposed": _bucket(opposed),
    }


def render_markdown(result: dict[str, Any], *, now_iso: str) -> str:
    if not result.get("ready"):
        return (f"# Signal-Value Report\n\n_Not enough data yet: "
                f"{result['n']}/{result['min_closes']} paired entry snapshots._\n")
    L = [f"# Signal-Value Report · {now_iso[:10]}", "",
         f"Paired entry snapshots: **{result['n']}**", "",
         "| cut | n | avg PnL | win rate | total PnL |",
         "|---|---|---|---|---|"]
    for label, key in [
        ("overall", "overall"),
        ("model agreed with side", "model_agree"),
        ("model disagreed", "model_disagree"),
        ("whale aligned", "whale_aligned"),
        ("whale opposed", "whale_opposed"),
    ]:
        b = result[key]
        L.append(f"| {label} | {b['n']} | ${b['avg_pnl']:+.2f} | "
                 f"{b['win_rate'] * 100:.0f}% | ${b['total_pnl']:+.2f} |")
    L.append("")
    return "\n".join(L)


def top_finding(result: dict[str, Any]) -> str:
    if not result.get("ready"):
        return (f"📊 Signal-value: {result['n']}/{result['min_closes']} "
                f"paired snapshots — not enough to analyse yet.")
    ma, md = result["model_agree"], result["model_disagree"]
    edge = ma["avg_pnl"] - md["avg_pnl"]
    return (f"📊 Signal-value ({result['n']} trades): model-agree avg "
            f"${ma['avg_pnl']:+.2f} vs disagree ${md['avg_pnl']:+.2f} "
            f"(edge ${edge:+.2f}); whale-aligned ${result['whale_aligned']['avg_pnl']:+.2f} "
            f"vs opposed ${result['whale_opposed']['avg_pnl']:+.2f}.")


def main() -> int:
    dry = "--dry-run" in sys.argv
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(f"file:{_DB}?mode=ro", uri=True) as conn:
        result = compute_signal_value(conn)
    md = render_markdown(result, now_iso=now_iso)
    digest = top_finding(result)
    if dry:
        print(md)
        print("\nDIGEST:", digest)
        return 0
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / f"{now_iso[:10]}-signal-value.md").write_text(md, encoding="utf-8")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
