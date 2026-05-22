"""Researcher tests — mocked LLM, structured proposal output."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.self_improve.llm_client import LLMResponse
from src.self_improve.researcher import (
    FORBIDDEN_AREAS,
    ResearcherContext,
    propose,
)


@dataclass
class _StubClient:
    text: str = ""
    degraded: bool = False
    error: str = ""
    has_api_key: bool = True
    last_user: str = ""

    def call(self, *, ctx, model, system, user, max_tokens=1024, **kw):
        self.last_user = user
        return LLMResponse(
            text=self.text,
            model=model,
            input_tokens=10, output_tokens=10,
            cache_read_tokens=0, cache_write_tokens=0,
            cost_usd=0.001, duration_s=0.1,
            degraded=self.degraded,
            error=self.error,
        )


def _ctx() -> ResearcherContext:
    return ResearcherContext(
        triggers_fired=[{
            "id": "T3", "metric": "profit_factor_last_20_XRPUSDT",
            "value": 0.3, "threshold": 0.7,
            "window": "last-20-closes", "symbol": "XRPUSDT",
            "rationale": "XRP PF 0.30",
        }],
        portfolio_metrics={"net_pnl_usd": -200, "sharpe": -1.5},
        per_symbol_metrics={"XRPUSDT": {"net_pnl_usd": -196}},
        recent_closes=[
            {"ts": "2026-05-22T10:00:00", "symbol": "XRPUSDT",
             "side": "LONG", "pnl": -10, "reason": "SL", "confidence": 0.5},
        ],
        recent_decisions=[],
        config_fingerprint={"MIN_CONFIDENCE": 0.45},
    )


def test_propose_no_api_key_returns_no_change() -> None:
    """When the LLM is degraded, Researcher emits no_change."""
    stub = _StubClient(degraded=True, error="no API key")
    out = propose(_ctx(), client=stub)
    assert out.verdict == "no_change"
    assert "no API key" in out.error


def test_propose_parses_valid_propose_response() -> None:
    payload = {
        "verdict": "propose",
        "hypothesis": "XRP confidence floor is too low",
        "proposal": {
            "description": "Raise XRP confidence floor to 0.65",
            "config_changes": {"SYMBOL_MIN_CONFIDENCE": {"XRPUSDT": 0.65}},
            "category": "config_tune",
            "rationale": "XRP entries cluster at low conf",
            "expected_impact": "block ~30 trades, save ~$150",
        },
        "expected_impact": {
            "metric": "net_pnl_usd", "delta_estimate": 150,
            "confidence_band": "medium",
        },
        "confidence": 0.7,
        "alternatives_considered": [
            "full XRP blocklist", "lower threshold to 0.55",
        ],
    }
    stub = _StubClient(text=json.dumps(payload))
    out = propose(_ctx(), client=stub)
    assert out.verdict == "propose"
    assert out.proposal is not None
    assert out.proposal.description == "Raise XRP confidence floor to 0.65"
    assert out.confidence == 0.7
    assert len(out.alternatives_considered) == 2


def test_propose_parses_no_change_response() -> None:
    payload = {
        "verdict": "no_change",
        "hypothesis": "data looks fine, recent perf is recovering",
        "confidence": 0.6,
        "alternatives_considered": ["wait", "tighten more"],
    }
    stub = _StubClient(text=json.dumps(payload))
    out = propose(_ctx(), client=stub)
    assert out.verdict == "no_change"
    assert out.proposal is None
    assert "recovering" in out.hypothesis


def test_propose_parses_escalate_response() -> None:
    payload = {
        "verdict": "escalate",
        "hypothesis": "Need to change position sizing",
        "escalation_reason": "wants to lower FIXED_MAX_NOTIONAL — forbidden",
        "confidence": 0.8,
    }
    stub = _StubClient(text=json.dumps(payload))
    out = propose(_ctx(), client=stub)
    assert out.verdict == "escalate"
    assert "forbidden" in out.escalation_reason


def test_propose_auto_escalates_forbidden_change() -> None:
    """LLM gives a 'propose' verdict but touches FIXED_MAX_NOTIONAL —
    Researcher must auto-escalate."""
    payload = {
        "verdict": "propose",
        "hypothesis": "h",
        "proposal": {
            "description": "lower max notional",
            "config_changes": {"FIXED_MAX_NOTIONAL": 500.0},
            "category": "config_tune",
            "rationale": "r",
            "expected_impact": "ei",
        },
        "expected_impact": {},
        "confidence": 0.6,
        "alternatives_considered": ["a1", "a2"],
    }
    stub = _StubClient(text=json.dumps(payload))
    out = propose(_ctx(), client=stub)
    assert out.verdict == "escalate"
    assert "FIXED_MAX_NOTIONAL" in out.escalation_reason


def test_propose_handles_invalid_json() -> None:
    stub = _StubClient(text="here is my plan: do nothing")
    out = propose(_ctx(), client=stub)
    assert out.verdict == "no_change"
    assert "not valid JSON" in out.error


def test_propose_strips_code_fence() -> None:
    payload = json.dumps({
        "verdict": "no_change",
        "hypothesis": "nothing to do",
        "confidence": 0.5,
    })
    stub = _StubClient(text=f"```json\n{payload}\n```")
    out = propose(_ctx(), client=stub)
    assert out.verdict == "no_change"


def test_propose_handles_unknown_verdict() -> None:
    stub = _StubClient(
        text=json.dumps({"verdict": "yolo", "hypothesis": "?"})
    )
    out = propose(_ctx(), client=stub)
    assert out.verdict == "no_change"
    assert "unknown verdict" in out.error


def test_forbidden_areas_includes_risk_constants() -> None:
    assert "FIXED_MAX_NOTIONAL" in FORBIDDEN_AREAS
    assert "STOP_LOSS_PCT" in FORBIDDEN_AREAS
    assert "MAX_LEVERAGE" in FORBIDDEN_AREAS
