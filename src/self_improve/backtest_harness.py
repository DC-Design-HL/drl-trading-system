"""Backtest harness — unified replay/ablation tool for the self-improvement loop.

The harness answers questions of the form:

    "If config override X had been in place during date range D, what would
    the portfolio metrics have been?"

It does so by **replaying historical trades from `data/trading.db`** against a
candidate config blob: every OPEN row in the period is checked against the
candidate's gates; OPENs that the candidate would have blocked are dropped
(along with their matched CLOSE rows); the residual CLOSEs are aggregated
via `src.self_improve.metrics`.

## What this version supports (M2)

* `MIN_CONFIDENCE` floor (portfolio-wide and per-symbol)
* `SYMBOL_SIDE_BLOCKLIST_ADD` — additional (symbol, side) blocks
* Date-range filtering (start/end ISO timestamps)
* Symbol filtering
* Multiple candidate configs in one run (for sweeps)

## What's deferred to M3 / forward simulation

* Loosening filters (would need to invent trades the bot didn't take)
* ADX / RSI / USDT.D / news / whale gate changes (no per-bar signal state
  in the DB; needs forward simulation against `data/kline_cache/`)
* Changes to SL / TP / partial-TP / trailing logic (would need bar-by-bar
  replay against klines)
* PPO model retraining impact (Mac-side, separate workflow)

## Determinism

Pure function. Same DB + same request → same result. No randomization,
no live API calls, no time-of-day dependencies. The git HEAD is recorded
in the result for full reproducibility.
"""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .metrics import TradeClose, parse_ts, summarize

# Match the Performance Monitor convention so the harness and live metrics
# layer agree on what "a close" is.
CLOSE_ACTIONS = (
    "CLOSE_LONG",
    "CLOSE_SHORT",
    "REVERSE_CLOSE_LONG",
    "REVERSE_CLOSE_SHORT",
    "SL_HIT",
    "TP_HIT",
)


# ─────────────────────────────────────────────────────────────────────────
# Request / Result dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestRequest:
    """Inputs to a single harness run.

    config_overrides is a dict with the following recognized keys (each
    optional):

      MIN_CONFIDENCE: float — portfolio-wide confidence floor; OPENs with
        confidence below this are blocked.

      SYMBOL_MIN_CONFIDENCE: dict[str, float] — per-symbol confidence
        floors. Takes precedence over MIN_CONFIDENCE for the symbols
        listed.

      SYMBOL_SIDE_BLOCKLIST_ADD: iterable of [symbol, side] pairs — extra
        (symbol, side) combos to block beyond whatever was historically
        blocked when the OPEN was logged. Sides are 'LONG' / 'SHORT'.

    Any keys not recognized are stored and echoed back in the result for
    audit, but have no behavioral effect in M2.
    """

    start_date: str          # ISO8601 — only closes since this date are considered
    end_date: str            # ISO8601 — only closes before this date are considered
    config_overrides: dict[str, Any] = field(default_factory=dict)
    symbols: Optional[tuple[str, ...]] = None   # None = all symbols
    capital_base: float = 5000.0
    mode: str = "replay"     # 'replay' (M2) | 'forward' (deferred to M3)
    label: str = ""          # human label, echoed in the result
    db_path: str = "data/trading.db"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["symbols"] = list(self.symbols) if self.symbols else None
        return d


@dataclass
class TradePair:
    """An OPEN row paired with its matching CLOSE row."""

    open_id: int
    close_id: int
    symbol: str
    side: str                # 'LONG' | 'SHORT'
    open_ts: str
    close_ts: str
    confidence: float
    pnl: float
    close_reason: str


@dataclass
class BlockedTrade:
    """An OPEN whose matched CLOSE was filtered out by the candidate config."""

    open_id: int
    close_id: int
    symbol: str
    side: str
    open_ts: str
    confidence: float
    pnl_avoided: float
    reason: str              # which override blocked it


@dataclass
class BacktestResult:
    request: BacktestRequest
    portfolio_metrics: dict[str, float]
    per_symbol_metrics: dict[str, dict[str, float]]
    trade_log: list[dict[str, Any]]
    blocked_trades: list[dict[str, Any]]
    n_input_pairs: int
    n_kept_pairs: int
    n_blocked_pairs: int
    runtime_seconds: float
    git_head: str
    mode: str
    label: str
    config_used: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = {
            "request": self.request.to_json(),
            "portfolio_metrics": _sanitize_metrics(self.portfolio_metrics),
            "per_symbol_metrics": {
                s: _sanitize_metrics(m)
                for s, m in self.per_symbol_metrics.items()
            },
            "trade_log": self.trade_log,
            "blocked_trades": self.blocked_trades,
            "n_input_pairs": self.n_input_pairs,
            "n_kept_pairs": self.n_kept_pairs,
            "n_blocked_pairs": self.n_blocked_pairs,
            "runtime_seconds": self.runtime_seconds,
            "git_head": self.git_head,
            "mode": self.mode,
            "label": self.label,
            "config_used": self.config_used,
            "warnings": self.warnings,
        }
        return d


# ─────────────────────────────────────────────────────────────────────────
# Trade pairing — the core of replay mode
# ─────────────────────────────────────────────────────────────────────────


def pair_open_close(
    conn: sqlite3.Connection,
    *,
    start_date: str,
    end_date: str,
    symbols: Optional[tuple[str, ...]] = None,
) -> list[TradePair]:
    """Pair OPEN_* rows with their matching CLOSE_* / SL_HIT / TP_HIT rows.

    Pairing is by symbol-FIFO and chronological order — the same pattern
    used in the live bot's trade reconstructor and in the May-20 XRP
    deep-dive analysis. A symbol can only have one open position at a
    time, so the first unmatched OPEN is paired with the first subsequent
    CLOSE on the same symbol.

    Returns only pairs whose CLOSE timestamp falls within
    [start_date, end_date]. OPENs that opened before the window but
    closed inside it are included (their PnL realized in-window).
    """
    sym_clause = ""
    params: list[Any] = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        sym_clause = f"AND symbol IN ({placeholders})"
        params.extend(symbols)

    rows = conn.execute(
        f"""
        SELECT id, timestamp, symbol, action, price, pnl, confidence, reason
        FROM trades
        WHERE is_testnet = 1
          AND timestamp <= ?
          {sym_clause}
        ORDER BY timestamp, id
        """,
        (end_date, *params),
    ).fetchall()

    open_stacks: dict[str, list[tuple]] = {}
    pairs: list[TradePair] = []

    for row in rows:
        _id, ts, symbol, action, price, pnl, conf, reason = row
        if not action:
            continue
        if action.startswith("OPEN_"):
            open_stacks.setdefault(symbol, []).append(row)
        elif action in CLOSE_ACTIONS or action.startswith("CLOSE_"):
            stack = open_stacks.get(symbol)
            if not stack:
                continue
            open_row = stack.pop(0)
            close_ts = ts
            if close_ts < start_date or close_ts > end_date:
                continue
            o_id, o_ts, _, o_action, _, _, o_conf, _ = open_row
            side = "LONG" if "LONG" in o_action else "SHORT"
            pairs.append(
                TradePair(
                    open_id=o_id,
                    close_id=_id,
                    symbol=symbol,
                    side=side,
                    open_ts=o_ts,
                    close_ts=close_ts,
                    confidence=float(o_conf or 0.0),
                    pnl=float(pnl or 0.0),
                    close_reason=reason or "",
                )
            )
    return pairs


# ─────────────────────────────────────────────────────────────────────────
# Override evaluation — does the candidate config block this OPEN?
# ─────────────────────────────────────────────────────────────────────────


def _block_reason(
    pair: TradePair, overrides: dict[str, Any]
) -> Optional[str]:
    """Return a human-readable block reason if the pair would be blocked,
    else None. Returns the FIRST matching reason — deterministic order:
    blocklist → per-symbol-confidence → global-confidence."""

    blocklist = overrides.get("SYMBOL_SIDE_BLOCKLIST_ADD")
    if blocklist:
        as_set = {(s.upper(), side.upper()) for s, side in blocklist}
        if (pair.symbol.upper(), pair.side.upper()) in as_set:
            return f"blocklist:{pair.symbol}/{pair.side}"

    per_sym = overrides.get("SYMBOL_MIN_CONFIDENCE") or {}
    if pair.symbol in per_sym:
        threshold = float(per_sym[pair.symbol])
        if pair.confidence < threshold:
            return (
                f"per_symbol_min_conf:{pair.symbol} "
                f"conf={pair.confidence:.3f}<{threshold:.3f}"
            )

    global_min = overrides.get("MIN_CONFIDENCE")
    if global_min is not None:
        threshold = float(global_min)
        if pair.confidence < threshold:
            return (
                f"global_min_conf conf={pair.confidence:.3f}<{threshold:.3f}"
            )

    return None


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────


def run_backtest(req: BacktestRequest) -> BacktestResult:
    """Execute one backtest. See module docstring for semantics."""
    if req.mode != "replay":
        raise NotImplementedError(
            f"Mode {req.mode!r} is not implemented in M2 — only 'replay'. "
            f"Forward simulation against the kline cache is M3 work."
        )

    started = time.perf_counter()
    warnings: list[str] = []
    db_path = Path(req.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"trading DB not found at {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        pairs = pair_open_close(
            conn,
            start_date=req.start_date,
            end_date=req.end_date,
            symbols=req.symbols,
        )

    kept_pairs: list[TradePair] = []
    blocked: list[BlockedTrade] = []
    for p in pairs:
        reason = _block_reason(p, req.config_overrides)
        if reason is None:
            kept_pairs.append(p)
        else:
            blocked.append(
                BlockedTrade(
                    open_id=p.open_id,
                    close_id=p.close_id,
                    symbol=p.symbol,
                    side=p.side,
                    open_ts=p.open_ts,
                    confidence=p.confidence,
                    pnl_avoided=p.pnl,
                    reason=reason,
                )
            )

    # Convert kept pairs into the TradeClose tuples the metrics layer
    # expects.
    closes = [
        TradeClose(
            ts=parse_ts(p.close_ts),
            symbol=p.symbol,
            side=p.side,
            pnl=p.pnl,
        )
        for p in kept_pairs
    ]

    portfolio = summarize(closes, capital_base=req.capital_base)

    by_symbol: dict[str, list[TradeClose]] = {}
    for c in closes:
        by_symbol.setdefault(c.symbol, []).append(c)
    per_symbol_metrics = {
        s: summarize(rows, capital_base=req.capital_base)
        for s, rows in by_symbol.items()
    }

    trade_log = [
        {
            "open_id": p.open_id,
            "close_id": p.close_id,
            "symbol": p.symbol,
            "side": p.side,
            "open_ts": p.open_ts,
            "close_ts": p.close_ts,
            "confidence": p.confidence,
            "pnl": p.pnl,
            "close_reason": p.close_reason,
        }
        for p in kept_pairs
    ]

    blocked_log = [
        {
            "open_id": b.open_id,
            "close_id": b.close_id,
            "symbol": b.symbol,
            "side": b.side,
            "open_ts": b.open_ts,
            "confidence": b.confidence,
            "pnl_avoided": b.pnl_avoided,
            "reason": b.reason,
        }
        for b in blocked
    ]

    if not pairs:
        warnings.append(
            "no OPEN/CLOSE pairs found in the requested window — check "
            "start_date / end_date / symbols"
        )
    if pairs and not kept_pairs:
        warnings.append(
            "every input pair was blocked by the candidate config; "
            "metrics describe an empty portfolio"
        )

    result = BacktestResult(
        request=req,
        portfolio_metrics=portfolio,
        per_symbol_metrics=per_symbol_metrics,
        trade_log=trade_log,
        blocked_trades=blocked_log,
        n_input_pairs=len(pairs),
        n_kept_pairs=len(kept_pairs),
        n_blocked_pairs=len(blocked),
        runtime_seconds=time.perf_counter() - started,
        git_head=_git_head(),
        mode=req.mode,
        label=req.label,
        config_used=dict(req.config_overrides),
        warnings=warnings,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────
# Sweep helper — run multiple candidate configs in one pass
# ─────────────────────────────────────────────────────────────────────────


def run_sweep(
    requests: Iterable[BacktestRequest],
) -> list[BacktestResult]:
    """Run every request in order, return results. Caller is responsible
    for choosing label values so the sweep is readable in JSON output."""
    return [run_backtest(r) for r in requests]


# ─────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _sanitize_metrics(d: dict[str, float]) -> dict[str, float]:
    """Strip inf/nan so the result is JSON-serializable."""
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, float):
            if math.isinf(v):
                out[k] = 9999.0 if v > 0 else -9999.0
            elif math.isnan(v):
                out[k] = 0.0
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def serialize(result: BacktestResult) -> str:
    """Stable JSON encoding for storage in experiments.backtest_result_json."""
    return json.dumps(result.to_json(), indent=2, default=str)
