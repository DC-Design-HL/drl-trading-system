"""Researcher — proposes config changes based on recent performance.

The Researcher is **LLM-only**. Without ANTHROPIC_API_KEY it cannot
function — there is no useful "deterministic fallback" for "form a
hypothesis and propose a remedy". Callers (orchestrator) detect the
degraded case and skip the pipeline gracefully.

Input context (assembled by the orchestrator from data the system
already has):

  * the trigger(s) that fired (T1-T5 from triggers.py)
  * the trailing-30d portfolio metrics snapshot
  * per-symbol breakdown from the latest metrics_snapshots row
  * the last ~50 closes (id, ts, symbol, side, pnl, reason, conf)
  * last 5 decisions (so we don't propose the same thing twice)
  * current config fingerprint (so we know what knobs are tunable)
  * the proposal must respect FORBIDDEN_AREAS — capital allocation,
    new instruments, mainnet exposure, risk-logic — and ESCALATE to
    Chen via Telegram if a hypothesis demands changes in those areas

Output: a Proposal dataclass identical in shape to the one Risk Officer
consumes, plus a `verdict` field ('propose' | 'no_change') the
orchestrator inspects.

The Researcher uses Opus (PLAN.md §3.2) — proposals are rare (max ~5/day
under the budget cap) and the reasoning quality matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .llm_client import (
    MODEL_OPUS,
    CallContext,
    LLMClient,
    LLMResponse,
    default_client,
    load_persona,
)
from .risk_officer import Proposal


# Areas the Researcher MAY NOT directly modify — anything touching these
# constants triggers an escalation instead of a normal proposal.
FORBIDDEN_AREAS = {
    "FIXED_MAX_NOTIONAL",       # capital allocation
    "STOP_LOSS_PCT",            # risk logic
    "TAKE_PROFIT_PCT",          # risk logic
    "MAX_LEVERAGE",             # risk logic
    "DAILY_LOSS_HALT_PCT",      # risk logic
    "MAX_DRAWDOWN_CEILING_PCT", # risk logic
    "LIVE_ORDER_CAP_USD",       # capital allocation
}

ALLOWED_AREAS_HINT = """\
You may propose changes in these areas:

  * Per-symbol confidence floors (SYMBOL_MIN_CONFIDENCE dict)
  * Symbol-side blocklist additions (SYMBOL_SIDE_BLOCKLIST)
  * Per-symbol directional confidence floors (SYMBOL_DIRECTIONAL_CONF)
  * Stagnant exit band (STAGNANT_HOURS, STAGNANT_PCT_MIN, _PCT_MAX)
  * Whipsaw cooldown duration (WHIPSAW_COOLDOWN_HOURS)
  * Anti-overtrading cooldowns (COOLDOWN_SECONDS, MIN_HOLD_SECONDS)
  * Trailing distance (TRAILING_DISTANCE_PCT, TRAILING_DISTANCE_POST_TP1)
  * USDT.D guard thresholds (USDT_D_THRESHOLD_PCT, USDT_D_LOOKBACK_HOURS)
  * Extreme news guard thresholds
  * Ranging-regime guards (RANGING_MIN_CONFIDENCE, RANGING_ADX_THRESHOLD)
  * REVERSE_CLOSE_LONG canary parameters
  * Per-symbol size scaling thresholds

You MAY NOT modify:

  * FIXED_MAX_NOTIONAL — position size limits (Chen-only)
  * STOP_LOSS_PCT, TAKE_PROFIT_PCT — risk logic (Chen-only)
  * MAX_LEVERAGE — risk logic (Chen-only)
  * Daily-loss / max-drawdown halt thresholds (Chen-only)
  * Adding a new symbol or venue (escalate to Chen)

If your hypothesis requires changing a forbidden area, set verdict to
"escalate" and explain WHY in `escalation_reason`.
"""


# ─────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResearcherContext:
    """Inputs the orchestrator passes to the Researcher."""

    triggers_fired: list[dict[str, Any]]
    portfolio_metrics: dict[str, Any]
    per_symbol_metrics: dict[str, dict[str, Any]]
    recent_closes: list[dict[str, Any]]
    recent_decisions: list[dict[str, Any]]
    config_fingerprint: dict[str, Any]


@dataclass
class ResearcherOutput:
    verdict: str                            # 'propose' | 'no_change' | 'escalate'
    hypothesis: str = ""
    proposal: Optional[Proposal] = None     # set if verdict == 'propose'
    expected_impact: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    alternatives_considered: list[str] = field(default_factory=list)
    escalation_reason: str = ""
    llm_response: Optional[LLMResponse] = None
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "verdict": self.verdict,
            "hypothesis": self.hypothesis,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "alternatives_considered": self.alternatives_considered,
            "escalation_reason": self.escalation_reason,
            "error": self.error,
        }
        if self.proposal:
            d["proposal"] = {
                "description": self.proposal.description,
                "config_changes": self.proposal.config_changes,
                "category": self.proposal.category,
                "rationale": self.proposal.rationale,
                "expected_impact": self.proposal.expected_impact,
            }
        return d


# ─────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────


_RESEARCHER_SYSTEM = """\
You are the Researcher for an autonomous self-improving crypto trading
system. Your job is to look at recent performance + the triggers that
fired + the current config + recent decisions, and propose ONE concrete,
single-touchpoint change (or explicitly recommend no change) that you
expect to improve the trailing-30d net PnL after fees without making
drawdown worse.

Output strict JSON of the following shape (no commentary, no code
fence):

  {
    "verdict": "propose" | "no_change" | "escalate",
    "hypothesis": "1-3 sentence statement of what's wrong and why",
    "proposal": {
      "description": "1-line summary of the proposed change",
      "config_changes": {
        "CONSTANT_NAME": <new_value>,
        ...
      },
      "category": "config_tune" | "guard_change" | "blocklist_change",
      "rationale": "Why this change should help",
      "expected_impact": "Concrete: which metric, by how much, why"
    },
    "expected_impact": {
      "metric": "net_pnl_usd | sharpe | win_rate | profit_factor",
      "delta_estimate": <number>,
      "confidence_band": "low | medium | high"
    },
    "confidence": <0.0-1.0>,
    "alternatives_considered": [
      "Short string per alternative — at least 2"
    ],
    "escalation_reason": "Set only if verdict == 'escalate'"
  }

Rules:

  * ONE touchpoint per proposal. No "kitchen-sink" PRs. If you can't
    explain in one sentence which lever moved which metric by how much,
    you don't have a proposal — return "no_change".

  * Mention at least 2 alternatives you considered and rejected. If
    you can't generate 2, return "no_change".

  * The proposal must be RUNNABLE without writing code — it must be
    expressible purely as a config_changes dict touching tunable
    constants. Constants are listed below.

  * Don't propose changes already attempted in `recent_decisions`
    unless you have new evidence the situation has changed.

  * Don't propose changes in FORBIDDEN AREAS (capital allocation /
    risk logic / new instruments). If your best hypothesis requires
    one of those, set verdict="escalate" and explain in
    `escalation_reason`.

  * Be opinionated. The point of you existing is to act. If the
    triggers fired, something is wrong — find the most likely cause.
"""


def _build_user_prompt(ctx: ResearcherContext) -> str:
    parts = ["# Recent state"]

    parts.append("\n## Triggers fired (most recent monitor tick)")
    if ctx.triggers_fired:
        parts.append(json.dumps(ctx.triggers_fired, indent=2, default=str))
    else:
        parts.append("(none — this is a quiet-tick check; propose only if you see strong evidence)")

    parts.append("\n## Portfolio metrics (30d window)")
    parts.append(json.dumps(ctx.portfolio_metrics, indent=2, default=str))

    parts.append("\n## Per-symbol metrics (30d window)")
    parts.append(json.dumps(ctx.per_symbol_metrics, indent=2, default=str))

    parts.append("\n## Recent closes (last 50)")
    for c in ctx.recent_closes[-50:]:
        parts.append(
            f"- ts={c.get('ts')} sym={c.get('symbol')} "
            f"side={c.get('side')} pnl={c.get('pnl')} "
            f"reason={c.get('reason')} conf={c.get('confidence')}"
        )

    parts.append("\n## Last 5 decisions")
    if ctx.recent_decisions:
        for d in ctx.recent_decisions[:5]:
            parts.append(
                f"- ts={d.get('ts')} agent={d.get('agent')} "
                f"type={d.get('decision_type')} "
                f"outcome={d.get('outcome')} : {d.get('summary')}"
            )
    else:
        parts.append("(no prior decisions — first proposal cycle)")

    parts.append("\n## Current config (live values for tunable constants)")
    parts.append(json.dumps(ctx.config_fingerprint, indent=2, default=str))

    parts.append("\n\n# Allowed and forbidden areas")
    parts.append(ALLOWED_AREAS_HINT)

    parts.append("\n# Output now (strict JSON only):")
    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    return text.strip()


def _parse_output(text: str) -> dict[str, Any]:
    text = _strip_code_fence(text)
    return json.loads(text)


def _touches_forbidden(config_changes: dict[str, Any]) -> set[str]:
    return set(config_changes.keys()) & FORBIDDEN_AREAS


# ─────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────


def propose(
    ctx: ResearcherContext,
    *,
    client: Optional[LLMClient] = None,
    db_path: str = "data/trading.db",
) -> ResearcherOutput:
    """Run the Researcher. Returns a ResearcherOutput. Always returns —
    no exceptions on LLM failure (the output's `error` field carries it).
    """
    cli = client or default_client(db_path)
    persona = load_persona("quantitative-researcher")
    system = _RESEARCHER_SYSTEM + (
        ("\n\n--- PERSONA ---\n\n" + persona) if persona else ""
    )

    resp = cli.call(
        ctx=CallContext(
            agent="researcher",
            context_summary=(
                f"triggers={[t.get('id') for t in ctx.triggers_fired]} "
                f"n_closes={len(ctx.recent_closes)}"
            ),
        ),
        model=MODEL_OPUS,
        system=system,
        user=_build_user_prompt(ctx),
        max_tokens=2048,
    )

    if resp.degraded:
        return ResearcherOutput(
            verdict="no_change",
            llm_response=resp,
            error=f"LLM unavailable: {resp.error}",
        )

    try:
        parsed = _parse_output(resp.text)
    except json.JSONDecodeError as exc:
        return ResearcherOutput(
            verdict="no_change",
            llm_response=resp,
            error=f"LLM output was not valid JSON: {exc}; text={resp.text[:300]!r}",
        )

    verdict = parsed.get("verdict")
    if verdict not in ("propose", "no_change", "escalate"):
        return ResearcherOutput(
            verdict="no_change",
            llm_response=resp,
            error=f"unknown verdict {verdict!r}",
        )

    if verdict == "no_change":
        return ResearcherOutput(
            verdict="no_change",
            hypothesis=parsed.get("hypothesis", ""),
            confidence=float(parsed.get("confidence") or 0.0),
            alternatives_considered=parsed.get("alternatives_considered") or [],
            llm_response=resp,
        )

    if verdict == "escalate":
        return ResearcherOutput(
            verdict="escalate",
            hypothesis=parsed.get("hypothesis", ""),
            escalation_reason=parsed.get("escalation_reason", "(none provided)"),
            confidence=float(parsed.get("confidence") or 0.0),
            alternatives_considered=parsed.get("alternatives_considered") or [],
            llm_response=resp,
        )

    # verdict == "propose" — build a Proposal
    proposal_dict = parsed.get("proposal") or {}
    config_changes = proposal_dict.get("config_changes") or {}
    forbidden_hits = _touches_forbidden(config_changes)
    if forbidden_hits:
        # Auto-escalate: Researcher tried to touch a forbidden area
        return ResearcherOutput(
            verdict="escalate",
            hypothesis=parsed.get("hypothesis", ""),
            escalation_reason=(
                f"auto-escalation: proposed change touches forbidden "
                f"areas {sorted(forbidden_hits)}; orchestrator will not "
                f"proceed without Chen's review"
            ),
            confidence=float(parsed.get("confidence") or 0.0),
            llm_response=resp,
        )

    proposal = Proposal(
        description=proposal_dict.get("description", ""),
        config_changes=config_changes,
        category=proposal_dict.get("category", "config_tune"),
        rationale=proposal_dict.get("rationale", ""),
        expected_impact=proposal_dict.get("expected_impact", ""),
    )

    expected_impact = parsed.get("expected_impact") or {}
    return ResearcherOutput(
        verdict="propose",
        hypothesis=parsed.get("hypothesis", ""),
        proposal=proposal,
        expected_impact=expected_impact,
        confidence=float(parsed.get("confidence") or 0.0),
        alternatives_considered=parsed.get("alternatives_considered") or [],
        llm_response=resp,
    )
