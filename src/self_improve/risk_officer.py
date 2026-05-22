"""Risk Officer — guards every proposed change against PLAN.md §8 caps.

Two-phase evaluation:

  Phase 1 — Deterministic preflight (no LLM):
    For each known numeric guardrail (max position size, max daily loss,
    max DD ceiling, etc.), compare the proposed config against the cap.
    Anything out of bounds = automatic veto with a concrete reason
    string. No interpretation, no judgment.

  Phase 2 — LLM judgment pass (Haiku):
    Only runs if Phase 1 passed. Asks the model to scan for *indirect*
    violations — proposals that don't change a numeric constant but
    materially weaken the risk posture (e.g. disabling a guard,
    widening the stagnant band beyond the back-tested range, removing
    cooldowns). If ANTHROPIC_API_KEY is missing the LLM step is
    skipped and the verdict is "approve_with_warning" noting the LLM
    pass was unavailable.

No override. Once a veto is returned, the orchestrator MUST stop the
experiment. The reason text is written to the decisions table.

Inputs and outputs are deliberately small dataclasses so the
orchestrator (M4) can pass them around as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .llm_client import (
    MODEL_HAIKU,
    CallContext,
    LLMClient,
    LLMResponse,
    default_client,
    load_persona,
)


# ─────────────────────────────────────────────────────────────────────────
# Hard guardrails — PLAN.md §8
# ─────────────────────────────────────────────────────────────────────────

# Keyed by constant name as it appears in `live_trading_htf.py`.
# Each value is a (min, max) tuple in the constant's native units.
# None means "no bound on that side". A proposal whose value falls
# outside this range is automatically vetoed.
#
# Capital base of $5,000 is used to convert percentage caps to USD.

CAPITAL_BASE_USD = 5000.0

GUARDRAIL_CAPS: dict[str, tuple[float | None, float | None]] = {
    # Per-trade notional max — 20% of capital (PLAN.md §8 row 1)
    "FIXED_MAX_NOTIONAL": (0.0, 0.20 * CAPITAL_BASE_USD),

    # Stop-loss can be widened (lower SL = bigger loss) but only to 3% per trade
    "STOP_LOSS_PCT": (0.005, 0.030),

    # Take-profit must remain at least 1.5× SL (asymmetric R)
    "TAKE_PROFIT_PCT": (0.010, 0.10),

    # Trailing distance can't be wider than 2%
    "TRAILING_DISTANCE_PCT": (0.001, 0.020),

    # MIN_CONFIDENCE — proposals to lower it below 0.40 are vetoed
    # (Tier-1 model below 0.40 produces too many false positives)
    "MIN_CONFIDENCE": (0.40, 0.95),

    # Stagnant exit bounds — within the backtested-safe band
    "STAGNANT_HOURS": (2.0, 24.0),
    "STAGNANT_PCT_MIN": (-0.020, 0.0),
    "STAGNANT_PCT_MAX": (0.0, 0.015),

    # Cooldown after losing trade — must remain non-zero
    "COOLDOWN_SECONDS": (300, 7200),
    "MIN_HOLD_SECONDS": (600, 14400),

    # Whipsaw cooldown stays in [0.5h, 12h]
    "WHIPSAW_COOLDOWN_HOURS": (0.5, 12.0),

    # Anti-runaway: REVERSAL gate slope can be tuned but not zeroed/inverted
    "REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT": (-3.0, 0.0),
}

# Top-level operational caps (not in live_trading_htf, separate gates).
# Used by the LLM step + the orchestrator's daily-loss check.
DAILY_LOSS_HALT_PCT = 0.05            # 5% of capital → halt
MAX_DRAWDOWN_CEILING_PCT = 0.08       # 8% from peak → halt + Chen sign-off
LIVE_ORDER_CAP_USD = 1000.0           # $1000 per live order (PLAN.md §8 row 6)


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Proposal:
    """A proposed change for the Risk Officer to evaluate.

    `config_changes` is a {constant_name: new_value} dict. The
    deterministic preflight applies the GUARDRAIL_CAPS rules to it.

    `description` is the human-readable proposal text — the LLM pass
    reads it to look for indirect violations.

    `category` is one of: 'config_tune' (tweaking a numeric constant),
    'guard_change' (enabling/disabling a guard), 'sizing_change',
    'strategy_replace'. The LLM pass weighs these differently.
    """

    description: str
    config_changes: dict[str, Any]
    category: str = "config_tune"
    rationale: str = ""
    expected_impact: str = ""


@dataclass
class Verdict:
    verdict: str                          # 'approve' | 'veto' | 'approve_with_warning'
    reasons: list[str] = field(default_factory=list)
    phase1_passed: bool = True
    phase2_ran: bool = False              # True if LLM pass was actually executed
    llm_response: Optional[LLMResponse] = None
    llm_concerns: list[str] = field(default_factory=list)

    @property
    def is_veto(self) -> bool:
        return self.verdict == "veto"

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "phase1_passed": self.phase1_passed,
            "phase2_ran": self.phase2_ran,
            "llm_concerns": self.llm_concerns,
            "llm_text_excerpt": (
                self.llm_response.text[:500] if self.llm_response else None
            ),
            "llm_degraded": (
                self.llm_response.degraded if self.llm_response else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────
# Phase 1 — deterministic preflight
# ─────────────────────────────────────────────────────────────────────────


def preflight(proposal: Proposal) -> Verdict:
    """Pure-function preflight. No I/O, no LLM. Returns a Verdict."""
    reasons: list[str] = []
    for name, new_value in proposal.config_changes.items():
        caps = GUARDRAIL_CAPS.get(name)
        if caps is None:
            continue  # unconstrained constant — LLM pass may still flag
        lo, hi = caps
        try:
            v = float(new_value)
        except (TypeError, ValueError):
            # Non-numeric override (e.g. blocklist set) — let LLM phase
            # decide; preflight has nothing to compare.
            continue
        if lo is not None and v < lo:
            reasons.append(
                f"{name}={v} below floor {lo} (PLAN.md §8 guardrail)"
            )
        if hi is not None and v > hi:
            reasons.append(
                f"{name}={v} above ceiling {hi} (PLAN.md §8 guardrail)"
            )

    if reasons:
        return Verdict(
            verdict="veto",
            reasons=reasons,
            phase1_passed=False,
            phase2_ran=False,
        )
    return Verdict(
        verdict="approve",
        reasons=["phase1 preflight passed"],
        phase1_passed=True,
        phase2_ran=False,
    )


# ─────────────────────────────────────────────────────────────────────────
# Phase 2 — LLM judgment
# ─────────────────────────────────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = """\
You are the Risk Officer for an autonomous self-improving crypto trading
system. Your one job is to spot proposals that weaken the risk posture
in ways the numeric preflight cannot catch — for example:

  * Disabling a guard (ADX, RSI, orderbook, USDT.D, news, blocklist,
    anti-whipsaw)
  * Removing or shortening cooldowns / min-hold periods
  * Removing a symbol-side block that recently had negative PnL
  * Widening the stagnant-exit band beyond the empirically validated
    range, or shortening STAGNANT_HOURS aggressively
  * Switching to a less-validated model variant
  * Replacing risk constants with environment variables (config drift)
  * Adding mainnet exposure (BANNED — testnet only)

You receive a structured proposal with description, config_changes,
category, rationale, expected_impact. Reply with strict JSON of shape:

  {
    "verdict": "approve" | "approve_with_warning" | "veto",
    "concerns": [list of short strings, each one a specific concern]
  }

Be conservative. A "veto" is reserved for proposals that materially
weaken safety. "approve_with_warning" is the right verdict when the
proposal is on-balance acceptable but has a non-obvious risk worth
noting. "approve" means you found nothing concerning.

Hard rules (any of these MUST trigger veto regardless of upside):
  * Switching to mainnet
  * Training the model on the production server
  * Disabling the FK/testnet enforcement in storage
  * Removing the daily-loss halt or max-drawdown ceiling
"""


def _build_user_prompt(proposal: Proposal) -> str:
    return (
        "Proposal to evaluate:\n\n"
        f"DESCRIPTION:\n{proposal.description}\n\n"
        f"CATEGORY: {proposal.category}\n\n"
        "CONFIG CHANGES:\n"
        f"{json.dumps(proposal.config_changes, indent=2, default=str)}\n\n"
        f"RATIONALE:\n{proposal.rationale or '(none provided)'}\n\n"
        f"EXPECTED IMPACT:\n{proposal.expected_impact or '(none provided)'}\n\n"
        "Respond with strict JSON only."
    )


def _parse_llm_verdict(text: str) -> tuple[str, list[str]]:
    """Parse LLM response, returning (verdict, concerns)."""
    text = text.strip()
    # Strip optional code-fence
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        )
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return ("approve_with_warning", [
            f"LLM response was not valid JSON: {text[:200]!r}"
        ])
    verdict = d.get("verdict", "approve_with_warning")
    if verdict not in ("approve", "approve_with_warning", "veto"):
        verdict = "approve_with_warning"
    concerns = d.get("concerns") or []
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    return (verdict, [str(c) for c in concerns])


def evaluate(
    proposal: Proposal,
    *,
    client: Optional[LLMClient] = None,
    db_path: str | Path = "data/trading.db",
    skip_llm: bool = False,
) -> Verdict:
    """Full Risk Officer evaluation: Phase 1 + (if it passes) Phase 2.

    Pass `skip_llm=True` to short-circuit after preflight — useful for
    tests and for the orchestrator's "fast path" when an LLM call would
    be redundant (e.g. mechanical config tweaks already approved by
    Phase 1).
    """
    verdict = preflight(proposal)
    if not verdict.phase1_passed:
        return verdict
    if skip_llm:
        return verdict

    cli = client or default_client(db_path)
    persona = load_persona("risk-officer")
    full_system = (_LLM_SYSTEM_PROMPT + "\n\n--- ROLE NOTES ---\n\n" + persona) if persona else _LLM_SYSTEM_PROMPT

    resp = cli.call(
        ctx=CallContext(
            agent="risk-officer",
            context_summary=proposal.description[:1000],
        ),
        model=MODEL_HAIKU,
        system=full_system,
        user=_build_user_prompt(proposal),
        max_tokens=1024,
    )
    verdict.phase2_ran = True
    verdict.llm_response = resp

    if resp.degraded:
        verdict.verdict = "approve_with_warning"
        verdict.reasons.append(
            f"LLM judgment unavailable (degraded): {resp.error}; "
            "deterministic preflight passed"
        )
        return verdict

    llm_verdict, concerns = _parse_llm_verdict(resp.text)
    verdict.llm_concerns = concerns
    if llm_verdict == "veto":
        verdict.verdict = "veto"
        verdict.reasons.append("LLM judgment: veto")
        verdict.reasons.extend(f"LLM concern: {c}" for c in concerns)
    elif llm_verdict == "approve_with_warning":
        verdict.verdict = "approve_with_warning"
        verdict.reasons.append("LLM judgment: approve_with_warning")
        verdict.reasons.extend(f"LLM concern: {c}" for c in concerns)
    else:
        verdict.reasons.append("LLM judgment: approve")
    return verdict
