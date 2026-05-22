"""Paper Trader — replay-validate a proposed config against forward
live data without sending any orders.

M4 mode (the only mode for now): when an experiment enters stage='paper'
at time T0, the Paper Trader does nothing immediately. The orchestrator
re-checks daily; once `now - T0 >= paper_days`, the harness's replay
mode runs against the period [T0, now] **with** the candidate config
override applied, and compares the residual portfolio to the baseline
portfolio (no override).

Pass criteria (PLAN.md §6 — "Paper gate"):

  * ≥ 15 closes in the window (statistical signal)
  * Sharpe is within ±25% of the original backtest's Sharpe
  * Zero daily-loss-limit breaches (max DD < 5% on any single day)

We can't run a true shadow-execution loop in M4 — that's M5 work — but
for tightening-only proposals this replay-against-future mode answers
exactly the right question: "would the new filter have improved things
over the last 7 days of real trading?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .backtest_harness import BacktestRequest, run_backtest

UTC = timezone.utc

# PLAN.md §6 — "Paper gate"
MIN_PAPER_CLOSES = 15
SHARPE_TOLERANCE = 0.25      # ±25% from backtest Sharpe
DD_DAILY_LIMIT_PCT = 5.0     # halt threshold from §8


@dataclass
class PaperResult:
    pass_gate: bool
    n_closes_kept: int
    n_blocked: int
    baseline_pnl: float
    candidate_pnl: float
    delta_pnl: float
    candidate_sharpe: float
    baseline_sharpe: float
    candidate_max_dd_pct: float
    reasons: list[str] = field(default_factory=list)
    candidate_result_json: Optional[dict[str, Any]] = None
    baseline_result_json: Optional[dict[str, Any]] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "pass_gate": self.pass_gate,
            "n_closes_kept": self.n_closes_kept,
            "n_blocked": self.n_blocked,
            "baseline_pnl": self.baseline_pnl,
            "candidate_pnl": self.candidate_pnl,
            "delta_pnl": self.delta_pnl,
            "candidate_sharpe": self.candidate_sharpe,
            "baseline_sharpe": self.baseline_sharpe,
            "candidate_max_dd_pct": self.candidate_max_dd_pct,
            "reasons": self.reasons,
        }


def evaluate_paper_period(
    *,
    paper_start: datetime,
    paper_end: datetime,
    config_overrides: dict[str, Any],
    backtest_sharpe_reference: float,
    db_path: str = "data/trading.db",
    capital_base: float = 5000.0,
) -> PaperResult:
    """Run the paper-period evaluation.

    Returns a PaperResult with pass_gate set based on the §6 paper-gate
    criteria.
    """
    baseline_req = BacktestRequest(
        start_date=paper_start.isoformat(),
        end_date=paper_end.isoformat(),
        config_overrides={},
        capital_base=capital_base,
        label="paper-baseline",
        db_path=db_path,
    )
    candidate_req = BacktestRequest(
        start_date=paper_start.isoformat(),
        end_date=paper_end.isoformat(),
        config_overrides=config_overrides,
        capital_base=capital_base,
        label="paper-candidate",
        db_path=db_path,
    )
    baseline = run_backtest(baseline_req)
    candidate = run_backtest(candidate_req)

    baseline_pnl = baseline.portfolio_metrics["net_pnl_usd"]
    candidate_pnl = candidate.portfolio_metrics["net_pnl_usd"]
    delta = candidate_pnl - baseline_pnl
    candidate_sharpe = candidate.portfolio_metrics["sharpe"]
    candidate_dd = candidate.portfolio_metrics["max_drawdown_pct"]

    reasons: list[str] = []
    if candidate.n_kept_pairs < MIN_PAPER_CLOSES:
        reasons.append(
            f"insufficient closes: {candidate.n_kept_pairs} < "
            f"{MIN_PAPER_CLOSES} required for statistical signal"
        )

    if candidate_dd > DD_DAILY_LIMIT_PCT:
        reasons.append(
            f"max DD {candidate_dd:.2f}% exceeds daily-loss-limit "
            f"{DD_DAILY_LIMIT_PCT:.2f}%"
        )

    # Sharpe tolerance vs the original backtest. The "backtest Sharpe"
    # is the candidate's Sharpe in a PAST period; the "paper Sharpe" is
    # its Sharpe in the FORWARD period. We accept the gate if the
    # forward Sharpe is within ±25% — i.e. the strategy didn't fall off
    # a cliff vs the past.
    if backtest_sharpe_reference != 0:
        delta_pct = (
            (candidate_sharpe - backtest_sharpe_reference)
            / abs(backtest_sharpe_reference)
        )
        if abs(delta_pct) > SHARPE_TOLERANCE:
            reasons.append(
                f"Sharpe drift {delta_pct * 100:+.1f}% exceeds ±"
                f"{SHARPE_TOLERANCE * 100:.0f}% tolerance "
                f"(backtest={backtest_sharpe_reference:.2f}, "
                f"paper={candidate_sharpe:.2f})"
            )

    # If candidate's PnL is significantly WORSE than baseline, fail
    if delta < -50:  # more than $50 worse over the paper period
        reasons.append(
            f"candidate PnL ${candidate_pnl:+.2f} is ${-delta:.2f} worse "
            f"than baseline ${baseline_pnl:+.2f}"
        )

    return PaperResult(
        pass_gate=not reasons,
        n_closes_kept=candidate.n_kept_pairs,
        n_blocked=candidate.n_blocked_pairs,
        baseline_pnl=baseline_pnl,
        candidate_pnl=candidate_pnl,
        delta_pnl=delta,
        candidate_sharpe=candidate_sharpe,
        baseline_sharpe=baseline.portfolio_metrics["sharpe"],
        candidate_max_dd_pct=candidate_dd,
        reasons=reasons,
        candidate_result_json=candidate.to_json(),
        baseline_result_json=baseline.to_json(),
    )
