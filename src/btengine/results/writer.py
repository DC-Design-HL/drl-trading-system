"""Result writers — trades.parquet, blocked.parquet, summary.json, equity.csv.

Output schema (committed surface — downstream notebooks rely on it):

trades.parquet — one row per CLOSE event (partial or full):
  symbol, side, entry_ts_ms, exit_ts_ms, entry_price, exit_price,
  units, pnl_usd, pnl_r, fees_usd, reason, confidence,
  mfe_pct, mae_pct, holding_minutes, is_full_close

blocked.parquet — one row per attempted-but-blocked entry:
  ts_ms, symbol, side_intent, blocking_guard, reason, confidence

summary.json — aggregate stats with per-symbol / per-side / per-reason rollups

equity.csv — bar-resolution equity curve
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def write_trades_parquet(trades: List, out_path: Path) -> int:
    if not trades:
        df = pd.DataFrame(columns=[
            "symbol", "side", "entry_ts_ms", "exit_ts_ms", "entry_price",
            "exit_price", "units", "pnl_usd", "pnl_r", "fees_usd", "reason",
            "confidence", "mfe_pct", "mae_pct", "holding_minutes", "is_full_close",
        ])
    else:
        rows = []
        for t in trades:
            d = asdict(t) if is_dataclass(t) else dict(t)
            # Drop or stringify dict fields (parquet doesn't love freeform dicts)
            d.pop("extras", None)
            rows.append(d)
        df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def write_blocked_parquet(blocked: List[Dict[str, Any]], out_path: Path) -> int:
    if not blocked:
        df = pd.DataFrame(columns=[
            "ts_ms", "symbol", "side_intent", "blocking_guard",
            "reason", "confidence",
        ])
    else:
        df = pd.DataFrame(blocked)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def write_equity_csv(equity_curve: List[Dict[str, Any]], out_path: Path) -> int:
    df = pd.DataFrame(equity_curve)
    if df.empty:
        df = pd.DataFrame(columns=["ts_ms", "balance", "open_positions"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return len(df)


def compute_summary(trades: List, blocked: List, equity_curve: List,
                    starting_balance: float, run_id: str,
                    config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate rollups for summary.json."""
    full_closes = [t for t in trades if getattr(t, "is_full_close", False)]
    n_full = len(full_closes)
    wins = sum(1 for t in full_closes if t.pnl_usd > 0)
    total_pnl = sum(t.pnl_usd for t in trades)
    fees = sum(t.fees_usd for t in trades)
    final_balance = starting_balance + total_pnl

    # Max drawdown from equity curve
    max_dd_pct = 0.0
    if equity_curve:
        peak = equity_curve[0]["balance"]
        for row in equity_curve:
            peak = max(peak, row["balance"])
            dd = (peak - row["balance"]) / peak if peak > 0 else 0
            max_dd_pct = max(max_dd_pct, dd)

    # By reason
    by_reason = {}
    for t in trades:
        r = by_reason.setdefault(t.reason, {"n": 0, "pnl": 0.0, "wins": 0})
        r["n"] += 1
        r["pnl"] += t.pnl_usd
        if t.pnl_usd > 0: r["wins"] += 1

    # By (symbol, side)
    by_sym_side = {}
    for t in full_closes:
        k = f"{t.symbol}:{t.side}"
        d = by_sym_side.setdefault(k, {"n": 0, "pnl": 0.0, "wins": 0})
        d["n"] += 1
        d["pnl"] += t.pnl_usd
        if t.pnl_usd > 0: d["wins"] += 1

    # Blocked rollup
    blocks_by_guard = {}
    for b in blocked:
        g = b.get("blocking_guard", "unknown")
        blocks_by_guard[g] = blocks_by_guard.get(g, 0) + 1

    return {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_full_closes": n_full,
        "n_partial_closes": len(trades) - n_full,
        "wins": wins,
        "win_rate_pct": (wins / n_full * 100) if n_full else 0,
        "total_pnl_usd": round(total_pnl, 2),
        "total_fees_usd": round(fees, 2),
        "starting_balance": starting_balance,
        "final_balance": round(final_balance, 2),
        "max_dd_pct": round(max_dd_pct * 100, 2),
        "by_reason": {k: {**v, "pnl": round(v["pnl"], 2)}
                      for k, v in by_reason.items()},
        "by_symbol_side": {k: {**v, "pnl": round(v["pnl"], 2)}
                            for k, v in by_sym_side.items()},
        "blocked_total": len(blocked),
        "blocks_by_guard": blocks_by_guard,
        "config_run_id": run_id,
        "strategy": config_dict.get("strategy"),
        "guards_enabled": list((config_dict.get("guards") or {}).get("enabled", [])),
    }


def write_summary_json(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
