"""P3 two-sided apply-surface tests (PROFITABILITY_PLAN.md §3/P3).

Covers the wiring added on top of safety_envelopes:

  * runtime_overrides.apply_envelopes — two-sided, range-checked apply to
    the live globals, with per-key loader round-trips.
  * runtime_overrides.check_apply_allowed — the live-apply guard: envelope
    range-check, legacy tighten-only, blocklist-removal rejection,
    unknown-key rejection.
  * researcher prompt allowed-areas generated FROM the envelope table.
  * forward_sim.forward_config_from_overrides — override→sim-config mapping.
"""

from __future__ import annotations

import pytest

from src.self_improve import runtime_overrides as ro
from src.self_improve import safety_envelopes as se
from src.self_improve.researcher import _build_allowed_areas


# ─────────────────────────────────────────────────────────────────────────
# apply_envelopes — two-sided, range-checked, loader round-trip per key
# ─────────────────────────────────────────────────────────────────────────


# A current-value snapshot mirroring the live globals the loader passes in.
def _current() -> dict:
    return {
        "STRUCT_MIN_CONFIDENCE": 0.0,
        "STRUCT_SYMBOL_MIN_CONFIDENCE": {},
        "STRUCT_SYMBOL_DIRECTIONAL_CONF": {},
        "TRAILING_DISTANCE_PCT": 0.005,
        "TRAILING_BREAKEVEN_PCT": 0.008,
        "STAGNANT_HOURS": 6.0,
        "COOLDOWN_SECONDS": 1800,
        "MIN_HOLD_SECONDS": 3600,
        "WHIPSAW_COOLDOWN_HOURS": 2.0,
        "ADX_GUARD_MIN": 20,
        "EXHAUSTION_ATR_THRESHOLD": 3.0,
    }


# (key, in-range value, the field's [min, max]) — one mid-range sample per
# scalar envelope key, chosen DIFFERENT from the current default so it round-
# trips to an applied change.
_SCALAR_CASES = [
    ("STRUCT_MIN_CONFIDENCE", 0.55),
    ("TRAILING_DISTANCE_PCT", 0.008),
    ("TRAILING_BREAKEVEN_PCT", 0.012),
    ("STAGNANT_HOURS", 9.0),
    ("COOLDOWN_SECONDS", 3600),
    ("MIN_HOLD_SECONDS", 5400),
    ("WHIPSAW_COOLDOWN_HOURS", 4.0),
    ("ADX_GUARD_MIN", 25.0),
    ("EXHAUSTION_ATR_THRESHOLD", 2.5),
]


@pytest.mark.parametrize("key,value", _SCALAR_CASES)
def test_apply_envelopes_scalar_round_trip(key: str, value: float) -> None:
    res = ro.apply_envelopes(overrides={key: value}, current=_current())
    assert key in res["values"], res
    assert res["values"][key] == float(value)
    assert any(key in a for a in res["applied"])
    assert res["skipped"] == []


def test_apply_envelopes_is_two_sided_can_lower_a_struct_floor() -> None:
    # Lowering is allowed within range (supersedes monotonic-tightening).
    cur = _current()
    cur["STRUCT_MIN_CONFIDENCE"] = 0.70
    res = ro.apply_envelopes(
        overrides={"STRUCT_MIN_CONFIDENCE": 0.40}, current=cur)
    assert res["values"]["STRUCT_MIN_CONFIDENCE"] == 0.40


def test_apply_envelopes_out_of_range_is_skipped_not_applied() -> None:
    # TRAILING_DISTANCE_PCT envelope is [0.003, 0.010]; 0.05 is out.
    res = ro.apply_envelopes(
        overrides={"TRAILING_DISTANCE_PCT": 0.05}, current=_current())
    assert "TRAILING_DISTANCE_PCT" not in res["values"]
    assert any("outside envelope" in s for s in res["skipped"])


def test_apply_envelopes_per_symbol_floor_merges() -> None:
    cur = _current()
    cur["STRUCT_SYMBOL_MIN_CONFIDENCE"] = {"BTCUSDT": 0.50}
    res = ro.apply_envelopes(
        overrides={"STRUCT_SYMBOL_MIN_CONFIDENCE": {"ETHUSDT": 0.60}},
        current=cur,
    )
    merged = res["values"]["STRUCT_SYMBOL_MIN_CONFIDENCE"]
    assert merged == {"BTCUSDT": 0.50, "ETHUSDT": 0.60}


def test_apply_envelopes_nested_directional_merges() -> None:
    res = ro.apply_envelopes(
        overrides={"STRUCT_SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"long": 0.7}}},
        current=_current(),
    )
    assert res["values"]["STRUCT_SYMBOL_DIRECTIONAL_CONF"] == {
        "ETHUSDT": {"LONG": 0.7}
    }


def test_apply_envelopes_ignores_non_envelope_keys() -> None:
    res = ro.apply_envelopes(
        overrides={"MIN_CONFIDENCE": 0.9, "SYMBOL_SIDE_BLOCKLIST_ADD": [["x", "y"]]},
        current=_current(),
    )
    assert res["values"] == {}  # legacy keys handled by tighten path, not here


def test_apply_envelopes_noop_when_value_equals_current() -> None:
    res = ro.apply_envelopes(
        overrides={"TRAILING_DISTANCE_PCT": 0.005}, current=_current())
    assert res["values"] == {}
    assert any("equals current" in s for s in res["skipped"])


# ─────────────────────────────────────────────────────────────────────────
# check_apply_allowed — the live-apply guard
# ─────────────────────────────────────────────────────────────────────────


def _baseline() -> dict:
    return {
        "min_confidence": 0.0,
        "symbol_min_confidence": {},
        "symbol_directional_conf": {},
        "symbol_side_blocklist": set(),
    }


def test_check_apply_allows_in_range_envelope() -> None:
    v = ro.check_apply_allowed(
        overrides={"TRAILING_DISTANCE_PCT": 0.008},
        baseline_legacy=_baseline(),
    )
    assert v == []


def test_check_apply_rejects_out_of_range_envelope() -> None:
    v = ro.check_apply_allowed(
        overrides={"ADX_GUARD_MIN": 99.0}, baseline_legacy=_baseline())
    assert any("outside envelope" in x for x in v)


def test_check_apply_rejects_blocklist_removal() -> None:
    v = ro.check_apply_allowed(
        overrides={se.BLOCKLIST_REMOVE_KEY: [["XRPUSDT", "LONG"]]},
        baseline_legacy=_baseline(),
    )
    assert any("Chen-only" in x for x in v)


def test_check_apply_rejects_unknown_key() -> None:
    v = ro.check_apply_allowed(
        overrides={"TOTALLY_MADE_UP": 1}, baseline_legacy=_baseline())
    assert any("unknown key" in x for x in v)


def test_check_apply_rejects_empty() -> None:
    v = ro.check_apply_allowed(overrides={}, baseline_legacy=_baseline())
    assert v and "empty" in v[0]


def test_check_apply_legacy_tightening_still_enforced() -> None:
    # Lowering the legacy MIN_CONFIDENCE floor below baseline must be refused.
    base = _baseline()
    base["min_confidence"] = 0.60
    v = ro.check_apply_allowed(
        overrides={"MIN_CONFIDENCE": 0.40}, baseline_legacy=base)
    assert any("loosen" in x for x in v)


def test_check_apply_mixed_envelope_plus_legacy_tighten_ok() -> None:
    base = _baseline()
    v = ro.check_apply_allowed(
        overrides={"TRAILING_DISTANCE_PCT": 0.008, "MIN_CONFIDENCE": 0.50},
        baseline_legacy=base,
    )
    assert v == []


def test_check_apply_works_without_baseline() -> None:
    # baseline=None still range-checks envelopes + blocks removals.
    v = ro.check_apply_allowed(
        overrides={"TRAILING_DISTANCE_PCT": 0.99}, baseline_legacy=None)
    assert any("outside envelope" in x for x in v)


# ─────────────────────────────────────────────────────────────────────────
# Researcher prompt — generated FROM the envelope table (no drift)
# ─────────────────────────────────────────────────────────────────────────


def test_researcher_prompt_generated_from_envelope_table() -> None:
    text = _build_allowed_areas()
    # Every envelope key must appear in the prompt — the single source of
    # truth, so the prompt can never drift from the apply guard.
    for key in se.ENVELOPES:
        assert key in text, f"{key} missing from researcher allowed-areas"
    # Exit knobs that the OLD hand-listed prompt told the model to escalate
    # are now first-class allowed keys.
    assert "TRAILING_DISTANCE_PCT" in text
    assert "STAGNANT_HOURS" in text
    # Forbidden keys must NOT be presented as tunable.
    assert "FIXED_MAX_NOTIONAL" not in se.ENVELOPES
    assert "STOP_LOSS_PCT" not in se.ENVELOPES


def test_researcher_prompt_mentions_blocklist_removal_is_escalate() -> None:
    text = _build_allowed_areas()
    assert "REMOV" in text.upper()
    assert "escalat" in text.lower()
