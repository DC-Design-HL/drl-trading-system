"""Tests for src/self_improve/safety_envelopes.py (PROFITABILITY_PLAN.md P3).

The envelope guard is the apply-time safety boundary for the autonomous
loop, so its contract is tested exhaustively: in-range passes, out-of-range
rejects, unknown keys reject, nested per-symbol dicts are fully checked, and
malformed values are rejected rather than silently passed.
"""

from __future__ import annotations

import pytest

from src.self_improve import safety_envelopes as se


def test_unknown_key_rejected() -> None:
    chk = se.check_envelope("MAX_NOTIONAL", 9999)
    assert chk.ok is False
    assert "not in ENVELOPES" in chk.reason


def test_sizing_leverage_keys_absent() -> None:
    # Structurally off-limits — must never be applyable.
    for forbidden in ("LEVERAGE", "MAX_NOTIONAL", "POSITION_SIZE_PCT",
                      "STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "TRADING_HALTED"):
        assert not se.is_envelope_key(forbidden)
        assert se.check_envelope(forbidden, 1).ok is False


def test_in_range_scalar_passes() -> None:
    assert se.check_envelope("TRAILING_DISTANCE_PCT", 0.005).ok is True
    # Boundaries are inclusive.
    assert se.check_envelope("TRAILING_DISTANCE_PCT", 0.003).ok is True
    assert se.check_envelope("TRAILING_DISTANCE_PCT", 0.010).ok is True


def test_out_of_range_scalar_rejected() -> None:
    assert se.check_envelope("TRAILING_DISTANCE_PCT", 0.002).ok is False
    assert se.check_envelope("TRAILING_DISTANCE_PCT", 0.011).ok is False
    assert se.check_envelope("ADX_GUARD_MIN", 31.0).ok is False
    assert se.check_envelope("COOLDOWN_SECONDS", 800).ok is False


def test_per_symbol_dict_all_leaves_checked() -> None:
    ok = {"BTCUSDT": 0.5, "ETHUSDT": 0.7}
    assert se.check_envelope("STRUCT_SYMBOL_MIN_CONFIDENCE", ok).ok is True
    bad = {"BTCUSDT": 0.5, "ETHUSDT": 0.99}  # 0.99 > 0.95
    chk = se.check_envelope("STRUCT_SYMBOL_MIN_CONFIDENCE", bad)
    assert chk.ok is False
    assert "0.99" in chk.reason


def test_nested_per_symbol_per_side_dict() -> None:
    val = {"BTCUSDT": {"LONG": 0.6, "SHORT": 0.8}}
    assert se.check_envelope("STRUCT_SYMBOL_DIRECTIONAL_CONF", val).ok is True
    bad = {"BTCUSDT": {"LONG": 0.6, "SHORT": 1.2}}
    assert se.check_envelope("STRUCT_SYMBOL_DIRECTIONAL_CONF", bad).ok is False


def test_bool_and_non_numeric_rejected() -> None:
    assert se.check_envelope("ADX_GUARD_MIN", True).ok is False
    assert se.check_envelope("ADX_GUARD_MIN", "20").ok is False
    assert se.check_envelope("STRUCT_MIN_CONFIDENCE", None).ok is False


def test_empty_dict_rejected() -> None:
    assert se.check_envelope("STRUCT_SYMBOL_MIN_CONFIDENCE", {}).ok is False


def test_validation_engine_lookup() -> None:
    assert se.validation_engine("STRUCT_MIN_CONFIDENCE") == se.VALIDATION_REPLAY
    assert se.validation_engine("TRAILING_DISTANCE_PCT") == se.VALIDATION_FORWARD
    with pytest.raises(KeyError):
        se.validation_engine("NOPE")


def test_allowed_areas_text_is_generated_from_table() -> None:
    txt = se.allowed_areas_text()
    # Every envelope key appears; no hand-listing drift possible.
    for key in se.ENVELOPES:
        assert key in txt
    # Blocklist removal is called out as escalation-only.
    assert "REMOVING from the blocklist" in txt
    assert "off-limits" in txt


def test_blocklist_keys_defined() -> None:
    assert se.BLOCKLIST_ADD_KEY and se.BLOCKLIST_REMOVE_KEY
    # The remove key is NOT an envelope key (never auto-applied).
    assert not se.is_envelope_key(se.BLOCKLIST_REMOVE_KEY)
