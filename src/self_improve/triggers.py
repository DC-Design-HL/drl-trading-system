"""Trigger evaluator for the self-improvement loop (PLAN.md §6).

A trigger is an "alarm" condition over recent performance metrics. When
any trigger fires, the orchestrator (M4+) spawns the Researcher to
hypothesize a remedy.

For M1 we only evaluate and *log* triggers — no agent is spawned yet.
The output of ``evaluate()`` is the structured "what fired" list that
the orchestrator will consume in M4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .metrics import (
    TradeClose,
    filter_window,
    max_drawdown_pct,
    net_pnl,
    profit_factor,
    sharpe_ratio,
    trailing_consecutive_losses,
)


@dataclass(frozen=True)
class TriggerHit:
    id: str        # e.g. 'T1'
    metric: str    # human-readable metric name
    value: float
    threshold: float
    window: str    # e.g. '7d', '20-closes'
    symbol: str | None  # NULL = portfolio-wide
    rationale: str


# Default thresholds — match PLAN.md §6 exactly.
THRESHOLDS = {
    "T1_sharpe_min": 0.5,
    "T2_pnl_pct_min": -0.03,   # -3% of capital_base
    "T3_symbol_pf_min": 0.7,
    "T3_symbol_min_closes": 10,
    "T4_consec_losses": 3,
    "T5_drawdown_pct_max": 5.0,
    # T6 (24h quiet) is handled by the orchestrator scheduler, not the
    # evaluator — it fires on a timer, not on a metric.
    # T7 (Reviewer-surfaced pattern) is qualitative — Reviewer writes
    # to decisions table directly, evaluator just propagates.
}


def evaluate_portfolio(
    trades: Sequence[TradeClose],
    *,
    now: datetime | None = None,
    capital_base: float = 5000.0,
    thresholds: dict[str, float] | None = None,
) -> list[TriggerHit]:
    """Evaluate the portfolio-wide triggers (T1, T2, T5)."""
    now = now or datetime.now(timezone.utc)
    th = {**THRESHOLDS, **(thresholds or {})}
    hits: list[TriggerHit] = []

    trades_7d = filter_window(trades, now=now, days=7)
    trades_30d = filter_window(trades, now=now, days=30)

    # T1: rolling Sharpe < 0.5 (7d window)
    sh7 = sharpe_ratio(trades_7d, capital_base=capital_base)
    if len(trades_7d) >= 3 and sh7 < th["T1_sharpe_min"]:
        hits.append(
            TriggerHit(
                id="T1",
                metric="sharpe_7d",
                value=sh7,
                threshold=th["T1_sharpe_min"],
                window="7d",
                symbol=None,
                rationale=(
                    f"7d Sharpe={sh7:.2f} below floor {th['T1_sharpe_min']:.2f} "
                    f"(n={len(trades_7d)} closes)"
                ),
            )
        )

    # T2: 7d net PnL < -3% of capital
    pnl7 = net_pnl(trades_7d)
    pnl_pct = pnl7 / capital_base if capital_base else 0.0
    if pnl_pct < th["T2_pnl_pct_min"]:
        hits.append(
            TriggerHit(
                id="T2",
                metric="net_pnl_pct_7d",
                value=pnl_pct,
                threshold=th["T2_pnl_pct_min"],
                window="7d",
                symbol=None,
                rationale=(
                    f"7d net PnL ${pnl7:+.2f} = {pnl_pct * 100:+.2f}% "
                    f"of ${capital_base:.0f}, below floor "
                    f"{th['T2_pnl_pct_min'] * 100:+.2f}%"
                ),
            )
        )

    # T5: drawdown from peak > 5% (rolling — 30d window is a defensible
    # finite proxy for "from peak"; M1 will use 30d, we can lengthen later)
    dd = max_drawdown_pct(trades_30d, capital_base=capital_base)
    if dd > th["T5_drawdown_pct_max"]:
        hits.append(
            TriggerHit(
                id="T5",
                metric="max_drawdown_pct_30d",
                value=dd,
                threshold=th["T5_drawdown_pct_max"],
                window="30d",
                symbol=None,
                rationale=(
                    f"30d max DD {dd:.2f}% exceeds ceiling "
                    f"{th['T5_drawdown_pct_max']:.2f}%"
                ),
            )
        )

    return hits


def evaluate_per_symbol(
    trades: Sequence[TradeClose],
    *,
    thresholds: dict[str, float] | None = None,
) -> list[TriggerHit]:
    """Evaluate per-symbol triggers (T3 PF and T4 consecutive losses).

    T3 evaluates over the last 20 closes per symbol (with n≥10 closes
    required); T4 evaluates the trailing run regardless of window.
    """
    th = {**THRESHOLDS, **(thresholds or {})}
    hits: list[TriggerHit] = []

    by_symbol: dict[str, list[TradeClose]] = {}
    for t in trades:
        by_symbol.setdefault(t.symbol, []).append(t)

    for symbol, sym_trades in by_symbol.items():
        sym_trades.sort(key=lambda x: x.ts)
        last_20 = sym_trades[-20:]
        if len(last_20) >= th["T3_symbol_min_closes"]:
            pf = profit_factor(last_20)
            if pf < th["T3_symbol_pf_min"]:
                hits.append(
                    TriggerHit(
                        id="T3",
                        metric=f"profit_factor_last_20_{symbol}",
                        value=pf,
                        threshold=th["T3_symbol_pf_min"],
                        window=f"last-{len(last_20)}-closes",
                        symbol=symbol,
                        rationale=(
                            f"{symbol} PF={pf:.2f} over last "
                            f"{len(last_20)} closes, below floor "
                            f"{th['T3_symbol_pf_min']:.2f}"
                        ),
                    )
                )

        # T4 fires per-symbol AND per-side because the bot can be on
        # different streaks on each side simultaneously.
        for side in ("LONG", "SHORT"):
            side_trades = [t for t in sym_trades if t.side == side]
            streak = trailing_consecutive_losses(side_trades)
            if streak >= th["T4_consec_losses"]:
                hits.append(
                    TriggerHit(
                        id="T4",
                        metric=f"consec_losses_{symbol}_{side}",
                        value=float(streak),
                        threshold=th["T4_consec_losses"],
                        window="trailing",
                        symbol=symbol,
                        rationale=(
                            f"{symbol} {side} on a {streak}-trade losing "
                            f"streak (threshold {th['T4_consec_losses']})"
                        ),
                    )
                )

    return hits


def evaluate(
    trades: Sequence[TradeClose],
    *,
    now: datetime | None = None,
    capital_base: float = 5000.0,
    thresholds: dict[str, float] | None = None,
) -> list[TriggerHit]:
    """Run all triggers (T1-T5). Returns a deduped list of hits.

    T6 (24h quiet) and T7 (Reviewer-surfaced) are not evaluated here —
    they're handled by the orchestrator scheduler and the Reviewer
    agent respectively in M3/M4.
    """
    return [
        *evaluate_portfolio(
            trades,
            now=now,
            capital_base=capital_base,
            thresholds=thresholds,
        ),
        *evaluate_per_symbol(trades, thresholds=thresholds),
    ]


def hit_to_row(hit: TriggerHit) -> dict[str, Any]:
    """Render a TriggerHit as a dict suitable for inclusion in a
    metrics_snapshots metadata_json or a decisions.trigger_* pair."""
    return {
        "id": hit.id,
        "metric": hit.metric,
        "value": hit.value,
        "threshold": hit.threshold,
        "window": hit.window,
        "symbol": hit.symbol,
        "rationale": hit.rationale,
    }
