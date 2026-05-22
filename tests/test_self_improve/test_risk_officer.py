"""Risk Officer tests — deterministic preflight + LLM-mocked judgment."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts.self_improve.migrate import migrate
from src.self_improve import risk_officer
from src.self_improve.llm_client import LLMResponse
from src.self_improve.risk_officer import (
    CAPITAL_BASE_USD,
    GUARDRAIL_CAPS,
    Proposal,
    Verdict,
    evaluate,
    preflight,
)


# ─────────────────────────────────────────────────────────────────────────
# Phase 1 — deterministic preflight
# ─────────────────────────────────────────────────────────────────────────


def test_preflight_passes_within_caps() -> None:
    """A boring config tweak in-bounds should preflight-approve."""
    p = Proposal(
        description="bump XRP confidence floor to 0.65",
        config_changes={"MIN_CONFIDENCE": 0.65},
        category="config_tune",
    )
    v = preflight(p)
    assert v.verdict == "approve"
    assert v.phase1_passed is True
    assert v.phase2_ran is False


def test_preflight_vetos_max_notional_over_cap() -> None:
    """20% of capital = $1000 is the cap. $2000 must veto."""
    p = Proposal(
        description="raise max position to $2k",
        config_changes={"FIXED_MAX_NOTIONAL": 2000.0},
    )
    v = preflight(p)
    assert v.verdict == "veto"
    assert any("FIXED_MAX_NOTIONAL" in r for r in v.reasons)


def test_preflight_vetos_zero_cooldown() -> None:
    """Removing cooldown entirely = veto."""
    p = Proposal(
        description="remove post-loss cooldown",
        config_changes={"COOLDOWN_SECONDS": 0},
    )
    v = preflight(p)
    assert v.verdict == "veto"


def test_preflight_vetos_widening_sl_too_far() -> None:
    """STOP_LOSS_PCT > 3% is over the cap → veto."""
    p = Proposal(
        description="widen stop loss to 5%",
        config_changes={"STOP_LOSS_PCT": 0.05},
    )
    v = preflight(p)
    assert v.verdict == "veto"


def test_preflight_passes_non_numeric_overrides() -> None:
    """Blocklist additions aren't numeric — preflight should not veto."""
    p = Proposal(
        description="add XRP to blocklist",
        config_changes={
            "SYMBOL_SIDE_BLOCKLIST_ADD": [("XRPUSDT", "LONG")],
        },
        category="config_tune",
    )
    v = preflight(p)
    assert v.verdict == "approve"


def test_preflight_aggregates_multiple_violations() -> None:
    p = Proposal(
        description="dangerous combo",
        config_changes={
            "FIXED_MAX_NOTIONAL": 5000.0,
            "STOP_LOSS_PCT": 0.10,
            "COOLDOWN_SECONDS": 0,
        },
    )
    v = preflight(p)
    assert v.verdict == "veto"
    # Each violation contributes a reason
    assert len(v.reasons) >= 3


def test_capital_base_drives_position_cap() -> None:
    """The FIXED_MAX_NOTIONAL cap is 20% of CAPITAL_BASE_USD."""
    cap_lo, cap_hi = GUARDRAIL_CAPS["FIXED_MAX_NOTIONAL"]
    assert cap_hi == 0.20 * CAPITAL_BASE_USD


# ─────────────────────────────────────────────────────────────────────────
# Phase 2 — LLM judgment (mocked)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _StubClient:
    """A drop-in replacement for LLMClient used in tests."""
    response: LLMResponse
    last_user: str = ""

    @property
    def has_api_key(self) -> bool:
        return True

    def call(self, *, ctx, model, system, user, max_tokens=1024, **kw) -> LLMResponse:
        self.last_user = user
        return self.response

    def check_budget(self, raise_on_hard: bool = True):
        return None

    def budget_state(self):  # pragma: no cover
        from src.self_improve.llm_client import BudgetState
        return BudgetState(0.0, 15.0, 30.0, 0)


def _stub_response(text: str, *, degraded: bool = False, error: str | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="claude-haiku-4-5",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.001,
        duration_s=0.05,
        degraded=degraded,
        error=error,
    )


def test_evaluate_short_circuits_on_phase1_veto() -> None:
    """If preflight vetoes, the LLM must NOT be called."""
    stub = _StubClient(response=_stub_response("{\"verdict\": \"approve\"}"))
    p = Proposal(
        description="bad",
        config_changes={"FIXED_MAX_NOTIONAL": 99999.0},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.verdict == "veto"
    assert v.phase2_ran is False
    assert stub.last_user == ""  # never called


def test_evaluate_runs_llm_when_phase1_passes() -> None:
    stub = _StubClient(
        response=_stub_response('{"verdict": "approve", "concerns": []}'),
    )
    p = Proposal(
        description="bump XRP floor to 0.65",
        config_changes={"MIN_CONFIDENCE": 0.65},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.phase2_ran is True
    assert v.verdict == "approve"
    assert "bump XRP floor to 0.65" in stub.last_user


def test_evaluate_propagates_llm_veto() -> None:
    stub = _StubClient(
        response=_stub_response(
            '{"verdict": "veto", "concerns": ["disables ADX guard"]}'
        ),
    )
    p = Proposal(
        description="disable ADX guard for testing",
        config_changes={"MIN_CONFIDENCE": 0.50},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.verdict == "veto"
    assert any("disables ADX guard" in r for r in v.reasons)


def test_evaluate_handles_llm_approve_with_warning() -> None:
    stub = _StubClient(
        response=_stub_response(
            '{"verdict": "approve_with_warning", "concerns": ["aggressive"]}'
        ),
    )
    p = Proposal(
        description="tweak",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.verdict == "approve_with_warning"
    assert v.llm_concerns == ["aggressive"]


def test_evaluate_degrades_when_llm_unavailable() -> None:
    """No API key → LLM step returns degraded → verdict becomes
    approve_with_warning (preflight passed but LLM judgment unavailable)."""
    stub = _StubClient(
        response=_stub_response("", degraded=True, error="no API key"),
    )
    p = Proposal(
        description="tweak",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.verdict == "approve_with_warning"
    assert any("LLM judgment unavailable" in r for r in v.reasons)


def test_evaluate_handles_invalid_json_from_llm() -> None:
    """Garbage LLM response → graceful degradation, not crash."""
    stub = _StubClient(
        response=_stub_response("here's my analysis: looks fine!"),
    )
    p = Proposal(
        description="tweak",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    # Garbage → approve_with_warning (errs on the side of holding)
    assert v.verdict == "approve_with_warning"


def test_evaluate_strips_code_fence() -> None:
    """LLMs love wrapping JSON in ```json. Parser must handle it."""
    stub = _StubClient(
        response=_stub_response(
            "```json\n{\"verdict\": \"approve\", \"concerns\": []}\n```"
        ),
    )
    p = Proposal(
        description="ok",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    v = evaluate(p, client=stub, db_path=":memory:")
    assert v.verdict == "approve"


def test_evaluate_skip_llm_returns_preflight_only() -> None:
    p = Proposal(
        description="ok",
        config_changes={"MIN_CONFIDENCE": 0.55},
    )
    v = evaluate(p, client=None, db_path=":memory:", skip_llm=True)
    assert v.phase2_ran is False
    assert v.verdict == "approve"
