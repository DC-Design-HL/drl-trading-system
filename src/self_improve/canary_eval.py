"""Canary evaluation v2 — evidence-based promotion (PROFITABILITY_PLAN.md P4).

The ambient circuit breaker (``live_apply.measure_since``) stays exactly as
it is — a safety net that reverts on a realized-loss / drawdown breach. This
module adds the *measurement* the breaker can't: did the applied change
actually do what it claimed, judged on counterfactuals instead of ambient
portfolio PnL (which during a chop regime tells you nothing about the change)?

Two change types, two counterfactuals:

  * SUPPRESSION change (struct-conf floor raise / blocklist add): forward-sim
    the entries the change BLOCKED (logged in ``suppressed_entries``, stamped
    with this experiment). If those would-have-been trades lost money on net
    (avoided PnL ≤ 0) over a meaningful sample (n ≥ min_samples), the
    suppression is earning its keep → PROMOTE. Too few simulatable samples →
    EXTEND (up to 7d), then REJECT. The blocked trades would have WON
    (avoided PnL > 0) → the change is costing us → REJECT.

  * ENVELOPE change (exit/timing knob): realized PnL of trades stamped with
    this experiment vs the forward-sim counterfactual of the BASELINE config
    over the same window. Beat-or-match baseline → PROMOTE, else REJECT;
    too few stamped closes → EXTEND then REJECT.

Everything here is best-effort: a sim/data failure degrades to EXTEND (never
a crash, never a silent promote). The orchestrator keeps the breaker primary.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .safety_envelopes import is_envelope_key, validation_engine

logger = logging.getLogger(__name__)
UTC = timezone.utc

# Decision sample floor + how long we'll keep a canary open waiting for it.
MIN_SAMPLES = 5
MAX_CANARY_DAYS = 7.0

# Entry-suppression keys (floors + blocklist add). A change touching only
# these is judged by the avoided-PnL counterfactual.
_SUPPRESSION_KEYS = frozenset({
    "MIN_CONFIDENCE", "SYMBOL_MIN_CONFIDENCE", "SYMBOL_DIRECTIONAL_CONF",
    "STRUCT_MIN_CONFIDENCE", "STRUCT_SYMBOL_MIN_CONFIDENCE",
    "STRUCT_SYMBOL_DIRECTIONAL_CONF",
    "SYMBOL_SIDE_BLOCKLIST_ADD", "SYMBOL_SIDE_BLOCKLIST",
})

_CLOSE_ACTIONS = (
    "CLOSE_LONG", "CLOSE_SHORT", "REVERSE_CLOSE_LONG",
    "REVERSE_CLOSE_SHORT", "SL_HIT", "TP_HIT",
)


def classify_change(config_changes: dict[str, Any]) -> str:
    """'envelope' if any forward-validated exit/timing knob is touched,
    else 'suppression' if only floors/blocklist, else 'unknown'."""
    keys = set(config_changes or {})
    if any(is_envelope_key(k) and validation_engine(k) == "forward" for k in keys):
        return "envelope"
    if keys and keys <= _SUPPRESSION_KEYS:
        return "suppression"
    if keys & _SUPPRESSION_KEYS:
        return "suppression"
    return "unknown"


@dataclass
class CanaryVerdict:
    decision: str                       # "promote" | "extend" | "reject"
    change_type: str
    n_samples: int = 0
    avoided_pnl: Optional[float] = None     # suppression: sum of sim'd blocked trades
    realized_pnl: Optional[float] = None    # envelope: stamped-trade realized PnL
    baseline_pnl: Optional[float] = None    # envelope: forward-sim baseline
    rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "decision": self.decision, "change_type": self.change_type,
            "n_samples": self.n_samples, "avoided_pnl": self.avoided_pnl,
            "realized_pnl": self.realized_pnl, "baseline_pnl": self.baseline_pnl,
            "rationale": self.rationale,
        }


def _parse_ts(s: str) -> datetime:
    s = str(s).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _simulate_suppressed(
    rows: list[tuple], *, capital: float, cache_base: Optional[Path],
) -> tuple[float, int]:
    """Forward-sim each blocked entry and return (sum_realized_pnl, n_simulated).

    entry_price is taken as the first 5m close at/after the suppression ts —
    what the bot would have entered at. Entries with no cached klines are
    skipped (and excluded from n_simulated)."""
    import numpy as np  # local: heavy deps stay out of import path until used
    import pandas as pd
    from .forward_sim import ForwardSimConfig, simulate_position
    from .kline_cache import as_ohlcv_df, read_funding, read_klines

    cfg = ForwardSimConfig(capital_base=capital)
    total = 0.0
    n = 0
    for ts_s, symbol, side, conf in rows:
        try:
            ts = _parse_ts(ts_s)
            start_ns = int(ts.timestamp() * 1e9) - int(2 * 3600 * 1e9)
            end_ns = int(ts.timestamp() * 1e9) + int(
                (cfg.max_hold_hours + 6) * 3600 * 1e9)
            df5 = as_ohlcv_df(read_klines(
                symbol, "5m", start=start_ns, end=end_ns, base=cache_base))
            if df5 is None or df5.empty:
                continue
            entry_ts = pd.Timestamp(ts)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            at_or_after = df5[df5.index >= entry_ts]
            if at_or_after.empty:
                continue
            entry_price = float(at_or_after.iloc[0]["close"])
            fund = read_funding(symbol, start=start_ns, end=end_ns, base=cache_base)
            f_ts = fund["ts"].to_numpy("int64") if fund is not None and not fund.empty else None
            f_rt = fund["funding_rate"].to_numpy("float64") if fund is not None and not fund.empty else None
            tr = simulate_position(
                symbol=symbol, side=str(side).upper(), entry_ts=entry_ts,
                entry_price=entry_price, confidence=float(conf or 0.0),
                df_5m=df5, cfg=cfg, funding_ts=f_ts, funding_rates=f_rt,
            )
            if tr is None:
                continue
            total += float(tr.realized_pnl_usd)
            n += 1
        except Exception as exc:  # noqa: BLE001 — one bad entry must not abort
            logger.debug("suppressed-entry sim failed (%s %s): %s", symbol, side, exc)
            continue
    return total, n


def _forward_baseline_pnl(
    *, start: datetime, end: datetime, capital: float,
    cache_base: Optional[Path],
) -> Optional[float]:
    """Net forward-sim PnL of the committed BASELINE config over [start, end]."""
    try:
        from .forward_sim import ForwardSimConfig, run_forward_sim
        res = run_forward_sim(
            start=start, end=end,
            config=ForwardSimConfig(capital_base=capital),
            cache_base=cache_base, label="canary-baseline-cf",
        )
        return float(sum(r.net_pnl_usd for r in res.per_symbol.values()))
    except Exception as exc:  # noqa: BLE001
        logger.debug("forward baseline sim failed: %s", exc)
        return None


def evaluate_canary(
    conn: sqlite3.Connection,
    *,
    experiment_id: int,
    config_changes: dict[str, Any],
    since_iso: str,
    capital: float,
    now: Optional[datetime] = None,
    cache_base: Optional[Path] = None,
    min_samples: int = MIN_SAMPLES,
) -> CanaryVerdict:
    """Decide promote / extend / reject for a canary experiment on
    counterfactual evidence. ``since_iso`` is the canary start."""
    now = now or datetime.now(UTC)
    change_type = classify_change(config_changes)
    elapsed_days = (now - _parse_ts(since_iso)).total_seconds() / 86400.0
    out_of_time = elapsed_days >= MAX_CANARY_DAYS

    if change_type == "suppression":
        try:
            rows = conn.execute(
                "SELECT ts, symbol, side, confidence FROM suppressed_entries "
                "WHERE experiment_id=? AND ts>=?",
                (experiment_id, since_iso),
            ).fetchall()
        except sqlite3.OperationalError:
            # P4 schema not migrated yet (table created on next bot restart) —
            # fail safe: no evidence → extend (or reject at the window cap),
            # never crash the orchestrator tick.
            rows = []
        avoided, n = _simulate_suppressed(
            rows, capital=capital, cache_base=cache_base)
        if n < min_samples:
            decision = "reject" if out_of_time else "extend"
            return CanaryVerdict(
                decision, change_type, n_samples=n, avoided_pnl=round(avoided, 2),
                rationale=(
                    f"only {n}/{min_samples} blocked entries simulatable after "
                    f"{elapsed_days:.1f}d — "
                    + ("max canary window reached, rejecting" if out_of_time
                       else "extending canary for more evidence")
                ),
            )
        decision = "promote" if avoided <= 0 else "reject"
        return CanaryVerdict(
            decision, change_type, n_samples=n, avoided_pnl=round(avoided, 2),
            rationale=(
                f"forward-sim of {n} blocked entries → avoided PnL "
                f"${avoided:+.2f}; " + (
                    "blocked trades would have lost → suppression earns its keep"
                    if avoided <= 0 else
                    "blocked trades would have won → suppression is costing money")
            ),
        )

    if change_type == "envelope":
        try:
            row = conn.execute(
                f"SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM trades "
                f"WHERE is_testnet=1 AND pnl IS NOT NULL AND experiment_id=? "
                f"AND action IN ({','.join('?' * len(_CLOSE_ACTIONS))}) "
                f"AND COALESCE(created_at,timestamp)>=?",
                (experiment_id, *_CLOSE_ACTIONS, since_iso),
            ).fetchone()
            realized, n = float(row[0]), int(row[1])
        except sqlite3.OperationalError:
            # trades.experiment_id not migrated yet — fail safe to no evidence.
            realized, n = 0.0, 0
        if n < min_samples:
            decision = "reject" if out_of_time else "extend"
            return CanaryVerdict(
                decision, change_type, n_samples=n, realized_pnl=round(realized, 2),
                rationale=(
                    f"only {n}/{min_samples} stamped closes after {elapsed_days:.1f}d — "
                    + ("max canary window reached, rejecting" if out_of_time
                       else "extending canary for more evidence")
                ),
            )
        baseline = _forward_baseline_pnl(
            start=_parse_ts(since_iso), end=now, capital=capital,
            cache_base=cache_base)
        if baseline is None:
            decision = "reject" if out_of_time else "extend"
            return CanaryVerdict(
                decision, change_type, n_samples=n, realized_pnl=round(realized, 2),
                rationale="baseline counterfactual unavailable — "
                          + ("rejecting at window cap" if out_of_time else "extending"),
            )
        decision = "promote" if realized >= baseline else "reject"
        return CanaryVerdict(
            decision, change_type, n_samples=n, realized_pnl=round(realized, 2),
            baseline_pnl=round(baseline, 2),
            rationale=(
                f"stamped realized ${realized:+.2f} vs forward-sim baseline "
                f"${baseline:+.2f} over {elapsed_days:.1f}d → "
                + ("beats/matches baseline" if realized >= baseline
                   else "underperforms baseline")
            ),
        )

    # Unknown change shape — no counterfactual we trust. Fail closed at the cap.
    return CanaryVerdict(
        "reject" if out_of_time else "extend", change_type,
        rationale=f"unclassified change {sorted(config_changes or {})} — "
                  + ("rejecting at window cap" if out_of_time else "holding"),
    )
