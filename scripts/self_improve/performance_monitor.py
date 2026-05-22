#!/usr/bin/env python3
"""Performance monitor — compute rolling metrics, write snapshots, evaluate triggers.

Designed for cron invocation:

    */5 * * * * cd /home/claude/.../drl-trading-system && python3 -m scripts.self_improve.performance_monitor

Each invocation:
  1. Reads closed trades from `trades` (is_testnet=1) since the May-1
     reset (configurable via --since).
  2. Computes metrics over 24h / 7d / 30d / per-symbol windows.
  3. Writes one row per window/symbol into `metrics_snapshots`.
  4. Evaluates triggers and prints fired triggers as JSON to stdout.
  5. Writes a 'heartbeat' snapshot row (window='heartbeat') so the
     watchdog (M3+) can detect a stalled monitor.

No agents are spawned in M1 — the orchestrator that consumes triggers
arrives in M4.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.self_improve.metrics import (  # noqa: E402
    TradeClose,
    parse_ts,
    summarize,
)
from src.self_improve.triggers import evaluate, hit_to_row  # noqa: E402

# Exit-action set considered a "close" for performance accounting.
# Matches the analysis pattern used in memory project files.
CLOSE_ACTIONS = (
    "CLOSE_LONG",
    "CLOSE_SHORT",
    "REVERSE_CLOSE_LONG",
    "REVERSE_CLOSE_SHORT",
    "SL_HIT",
    "TP_HIT",
)

# Default starting boundary for the rolling-window queries — May 1 reset.
DEFAULT_SINCE = "2026-05-01T12:29:00"

# Default capital base used for return-% calculations. Should be the
# wallet-balance baseline at the snapshot epoch.
DEFAULT_CAPITAL_BASE = 5000.0


def load_closes(
    conn: sqlite3.Connection, *, since: str
) -> list[TradeClose]:
    """Pull testnet close rows from the trades table since `since`."""
    sql = f"""
        SELECT timestamp, symbol, action, pnl
        FROM trades
        WHERE is_testnet = 1
          AND timestamp >= ?
          AND action IN ({",".join("?" for _ in CLOSE_ACTIONS)})
        ORDER BY timestamp
    """
    rows = conn.execute(sql, (since, *CLOSE_ACTIONS)).fetchall()
    out: list[TradeClose] = []
    for ts_str, symbol, action, pnl in rows:
        if pnl is None:
            continue
        side = "LONG" if "LONG" in action else "SHORT"
        out.append(
            TradeClose(
                ts=parse_ts(ts_str),
                symbol=symbol,
                side=side,
                pnl=float(pnl),
            )
        )
    return out


def write_snapshot(
    conn: sqlite3.Connection,
    *,
    ts: datetime,
    window: str,
    symbol: str | None,
    metrics: dict[str, float],
    metadata: dict[str, object] | None = None,
) -> int:
    """Insert one row into metrics_snapshots, return its id."""
    cur = conn.execute(
        """
        INSERT INTO metrics_snapshots
            (ts, window, symbol, net_pnl_usd, num_closes, win_rate,
             profit_factor, sharpe, sortino, max_drawdown_pct,
             metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts.isoformat(),
            window,
            symbol,
            float(metrics["net_pnl_usd"]),
            int(metrics["num_closes"]),
            float(metrics["win_rate"]),
            float(metrics["profit_factor"]),
            float(metrics["sharpe"]),
            float(metrics["sortino"]),
            float(metrics["max_drawdown_pct"]),
            json.dumps(metadata or {}),
        ),
    )
    return int(cur.lastrowid or 0)


def _safe_pf(v: float) -> float:
    """SQLite doesn't have infinity; clamp PF for storage."""
    import math
    if math.isinf(v):
        return 9999.0
    if math.isnan(v):
        return 0.0
    return v


def _serialize(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize metrics for SQLite (no inf/nan)."""
    out = dict(metrics)
    out["profit_factor"] = _safe_pf(out["profit_factor"])
    out["sharpe"] = _safe_pf(out["sharpe"])
    out["sortino"] = _safe_pf(out["sortino"])
    return out


def run_monitor(
    db_path: Path,
    *,
    since: str = DEFAULT_SINCE,
    capital_base: float = DEFAULT_CAPITAL_BASE,
    now: datetime | None = None,
) -> dict[str, object]:
    """Run one monitor tick. Returns a summary dict including fired triggers."""
    now = now or datetime.now(timezone.utc)
    summary: dict[str, object] = {
        "ts": now.isoformat(),
        "snapshots_written": 0,
        "triggers_fired": [],
    }

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        closes = load_closes(conn, since=since)

        # Portfolio-wide windows
        windows = [
            ("24h", _filter_days(closes, now, 1)),
            ("7d",  _filter_days(closes, now, 7)),
            ("30d", _filter_days(closes, now, 30)),
        ]
        for label, slice_ in windows:
            metrics = _serialize(
                summarize(slice_, capital_base=capital_base)
            )
            write_snapshot(
                conn, ts=now, window=label, symbol=None, metrics=metrics
            )
            summary["snapshots_written"] = (
                int(summary["snapshots_written"]) + 1
            )

        # Per-symbol 30d slice for the dashboard
        by_symbol: dict[str, list[TradeClose]] = {}
        last_30 = _filter_days(closes, now, 30)
        for t in last_30:
            by_symbol.setdefault(t.symbol, []).append(t)
        for sym, sym_trades in by_symbol.items():
            metrics = _serialize(
                summarize(sym_trades, capital_base=capital_base)
            )
            write_snapshot(
                conn,
                ts=now,
                window="30d",
                symbol=sym,
                metrics=metrics,
            )
            summary["snapshots_written"] = (
                int(summary["snapshots_written"]) + 1
            )

        # Heartbeat row so an external watchdog can detect stalls.
        write_snapshot(
            conn,
            ts=now,
            window="heartbeat",
            symbol=None,
            metrics={
                "net_pnl_usd": 0.0,
                "num_closes": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown_pct": 0.0,
            },
            metadata={"closes_loaded": len(closes)},
        )

        # Trigger evaluation — does NOT spawn agents in M1.
        hits = evaluate(closes, now=now, capital_base=capital_base)
        summary["triggers_fired"] = [hit_to_row(h) for h in hits]
        conn.commit()

    return summary


def _filter_days(trades, now, days):
    from datetime import timedelta
    cutoff = now - timedelta(days=days)
    return [t for t in trades if t.ts >= cutoff]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help="ISO timestamp; trades before this are ignored (default: May-1 reset)",
    )
    parser.add_argument(
        "--capital-base",
        type=float,
        default=DEFAULT_CAPITAL_BASE,
        help="Wallet baseline used for return-% calculations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full summary as JSON to stdout (default: pretty print)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ DB not found at {db_path}", file=sys.stderr)
        return 2

    summary = run_monitor(
        db_path, since=args.since, capital_base=args.capital_base
    )

    if args.json:
        print(json.dumps(summary, default=str, indent=2))
    else:
        ts = summary["ts"]
        written = summary["snapshots_written"]
        fired = summary["triggers_fired"]
        print(f"✅ Monitor tick @ {ts}  snapshots={written}  triggers={len(fired)}")
        for hit in fired:
            print(
                f"  🚨 {hit['id']} {hit['metric']} = {hit['value']:.3f} "
                f"vs threshold {hit['threshold']:.3f} ({hit['window']}) "
                f"— {hit['rationale']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
