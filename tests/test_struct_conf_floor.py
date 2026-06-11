"""STRUCT_* confidence-floor surface (PROFITABILITY_PLAN.md P1).

These tests cover the live-side behaviour: the resolver helper, the
zero-baseline no-op invariant, the runtime-overrides round-trip with
the existing escape hatch, and the live↔harness parity that is the
regression test for the F1 dead-knob bug.

Run with:
    python3 -m pytest tests/test_struct_conf_floor.py -q
"""

from __future__ import annotations

import os
from collections import namedtuple

import pytest

# Importing the live bot is heavy (PyTorch, stable-baselines3). The
# imports below are the same idiom used in tests/test_symbol_blocklist.py
# and tests/test_position_sync_regression.py — no per-test re-imports.
import live_trading_htf as live  # noqa: E402
from src.self_improve import backtest_harness as bh  # noqa: E402
from src.self_improve.runtime_overrides import (  # noqa: E402
    check_tightening_only,
    tighten_overrides,
)


# ─── _resolve_struct_floor precedence ───────────────────────────────────


def _reset_struct_globals() -> None:
    live.STRUCT_MIN_CONFIDENCE = 0.0
    live.STRUCT_SYMBOL_MIN_CONFIDENCE = {}
    live.STRUCT_SYMBOL_DIRECTIONAL_CONF = {}


@pytest.fixture(autouse=True)
def _isolation() -> None:
    """Reset module globals before every test — these are mutated."""
    _reset_struct_globals()
    yield
    _reset_struct_globals()


def test_baselines_are_no_op() -> None:
    """0.0 / {} baselines never produce a floor — the deploy is byte-identical
    to pre-P1 behaviour for entry decisions."""
    floor, label = live._resolve_struct_floor("ETHUSDT", "LONG")
    assert floor is None
    assert label == ""


def test_global_floor_applies_when_set() -> None:
    live.STRUCT_MIN_CONFIDENCE = 0.6
    floor, label = live._resolve_struct_floor("BTCUSDT", "SHORT")
    assert floor == 0.6
    assert label == "STRUCT_MIN_CONFIDENCE"


def test_per_symbol_overrides_global() -> None:
    live.STRUCT_MIN_CONFIDENCE = 0.6
    live.STRUCT_SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.7}
    eth, eth_label = live._resolve_struct_floor("ETHUSDT", "LONG")
    btc, btc_label = live._resolve_struct_floor("BTCUSDT", "LONG")
    assert eth == 0.7
    assert eth_label == "STRUCT_SYMBOL_MIN_CONFIDENCE[ETHUSDT]"
    assert btc == 0.6
    assert btc_label == "STRUCT_MIN_CONFIDENCE"


def test_directional_overrides_per_symbol() -> None:
    live.STRUCT_MIN_CONFIDENCE = 0.6
    live.STRUCT_SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.7}
    live.STRUCT_SYMBOL_DIRECTIONAL_CONF = {"ETHUSDT": {"LONG": 0.8}}
    long_floor, long_label = live._resolve_struct_floor("ETHUSDT", "LONG")
    short_floor, short_label = live._resolve_struct_floor("ETHUSDT", "SHORT")
    assert long_floor == 0.8
    assert long_label == "STRUCT_SYMBOL_DIRECTIONAL_CONF[ETHUSDT][LONG]"
    # SHORT side has no directional → falls back to per-symbol 0.7
    assert short_floor == 0.7


def test_floor_resolver_treats_zero_as_unset() -> None:
    """An explicit 0.0 is not a meaningful floor — same as absent.

    Otherwise a researcher proposing STRUCT_SYMBOL_MIN_CONFIDENCE={"X":0.0}
    would create a spurious 🚫 log marker on every entry."""
    live.STRUCT_SYMBOL_MIN_CONFIDENCE = {"BTCUSDT": 0.0}
    floor, _ = live._resolve_struct_floor("BTCUSDT", "LONG")
    assert floor is None


# ─── tighten_overrides: STRUCT_* keys ───────────────────────────────────


def test_tighten_struct_global_raises_only() -> None:
    """Raising STRUCT_MIN_CONFIDENCE 0.0→0.6 applies; lowering 0.6→0.3 skips."""
    res = tighten_overrides(
        overrides={"STRUCT_MIN_CONFIDENCE": 0.6},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_min_confidence=0.0,
    )
    assert res["struct_min_confidence"] == 0.6
    assert any("STRUCT_MIN_CONFIDENCE" in a for a in res["applied"])

    res2 = tighten_overrides(
        overrides={"STRUCT_MIN_CONFIDENCE": 0.3},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_min_confidence=0.6,
    )
    assert res2["struct_min_confidence"] == 0.6
    assert any("would loosen" in s for s in res2["skipped"])


def test_tighten_struct_per_symbol_raise_and_new_key() -> None:
    res = tighten_overrides(
        overrides={"STRUCT_SYMBOL_MIN_CONFIDENCE": {"ETHUSDT": 0.7}},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_symbol_min_confidence={},
    )
    assert res["struct_symbol_min_confidence"]["ETHUSDT"] == 0.7
    # A "no-op" (equal) is benign, not loosening
    res2 = tighten_overrides(
        overrides={"STRUCT_SYMBOL_MIN_CONFIDENCE": {"ETHUSDT": 0.7}},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_symbol_min_confidence={"ETHUSDT": 0.7},
    )
    assert any("equals baseline" in s for s in res2["skipped"])


def test_tighten_struct_directional_per_side() -> None:
    res = tighten_overrides(
        overrides={
            "STRUCT_SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"LONG": 0.8}},
        },
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_symbol_directional_conf={},
    )
    assert res["struct_symbol_directional_conf"]["ETHUSDT"]["LONG"] == 0.8


def test_check_tightening_only_accepts_struct_raise() -> None:
    violations = check_tightening_only(
        overrides={"STRUCT_MIN_CONFIDENCE": 0.6},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_min_confidence=0.0,
    )
    assert violations == []


def test_check_tightening_only_refuses_struct_lower() -> None:
    violations = check_tightening_only(
        overrides={"STRUCT_MIN_CONFIDENCE": 0.3},
        min_confidence=0.45,
        symbol_min_confidence={},
        symbol_directional_conf={},
        symbol_side_blocklist=set(),
        struct_min_confidence=0.6,
    )
    assert violations  # at least one violation
    assert any("would loosen" in v for v in violations)


def test_legacy_keys_still_applyable_for_non_structure_mode() -> None:
    """The legacy MIN_CONFIDENCE family must stay applyable so that a
    future return to model-first mode still has its surface."""
    from src.self_improve.runtime_overrides import APPLYABLE_KEYS
    assert "MIN_CONFIDENCE" in APPLYABLE_KEYS
    assert "SYMBOL_MIN_CONFIDENCE" in APPLYABLE_KEYS
    assert "SYMBOL_DIRECTIONAL_CONF" in APPLYABLE_KEYS
    assert "STRUCT_MIN_CONFIDENCE" in APPLYABLE_KEYS
    assert "STRUCT_SYMBOL_MIN_CONFIDENCE" in APPLYABLE_KEYS
    assert "STRUCT_SYMBOL_DIRECTIONAL_CONF" in APPLYABLE_KEYS


# ─── Live ↔ Harness parity (F1 regression) ──────────────────────────────


# Lightweight stand-in for backtest_harness.TradePair — only the fields
# _block_reason and _resolve_struct_floor read, so we don't need to drag
# in a full DB.
_FakePair = namedtuple(
    "_FakePair",
    "open_id close_id symbol side open_ts close_ts confidence pnl close_reason",
)


def _make_pair(
    symbol: str,
    side: str,
    conf: float,
    *,
    open_ts: str = "2026-05-10T00:00:00",
) -> _FakePair:
    return _FakePair(
        open_id=1,
        close_id=2,
        symbol=symbol,
        side=side,
        open_ts=open_ts,
        close_ts=open_ts,
        confidence=conf,
        pnl=0.0,
        close_reason="",
    )


def _live_would_block(pair: _FakePair) -> bool:
    """Replica of the live floor check at the end of _get_structure_direction."""
    if pair.open_ts < bh.STRUCTURE_FIRST_LIVE_SINCE:
        return False  # live wouldn't have run STRUCT_* on pre-cutoff
    floor, _ = live._resolve_struct_floor(pair.symbol, pair.side)
    if floor is None:
        return False
    return float(pair.confidence) < float(floor)


def test_live_and_harness_agree_global_floor() -> None:
    """The F1 regression: the same fixture, the same threshold, the
    live check and the harness check must agree on every pair."""
    overrides = {"STRUCT_MIN_CONFIDENCE": 0.6}
    live.STRUCT_MIN_CONFIDENCE = 0.6  # mirror live state to overrides
    pairs = [
        _make_pair("BTCUSDT", "LONG", 0.3),   # blocked both
        _make_pair("BTCUSDT", "LONG", 0.6),   # kept both (not strictly less)
        _make_pair("ETHUSDT", "SHORT", 0.59), # blocked both
        _make_pair("SOLUSDT", "SHORT", 0.95), # kept both
    ]
    for p in pairs:
        live_block = _live_would_block(p)
        harness_reason = bh._block_reason(p, overrides)
        assert (harness_reason is not None) == live_block, (
            f"divergence on {p.symbol} {p.side} conf={p.confidence}: "
            f"live={live_block} harness={harness_reason!r}"
        )


def test_live_and_harness_agree_per_symbol_and_directional() -> None:
    overrides = {
        "STRUCT_SYMBOL_MIN_CONFIDENCE": {"ETHUSDT": 0.7},
        "STRUCT_SYMBOL_DIRECTIONAL_CONF": {"ETHUSDT": {"LONG": 0.8}},
    }
    live.STRUCT_SYMBOL_MIN_CONFIDENCE = {"ETHUSDT": 0.7}
    live.STRUCT_SYMBOL_DIRECTIONAL_CONF = {"ETHUSDT": {"LONG": 0.8}}
    pairs = [
        _make_pair("ETHUSDT", "LONG", 0.79),   # blocked by directional
        _make_pair("ETHUSDT", "LONG", 0.80),   # kept
        _make_pair("ETHUSDT", "SHORT", 0.69),  # blocked by per-symbol
        _make_pair("ETHUSDT", "SHORT", 0.70),  # kept
        _make_pair("BTCUSDT", "LONG", 0.0),    # no rule → kept
    ]
    for p in pairs:
        live_block = _live_would_block(p)
        harness_reason = bh._block_reason(p, overrides)
        assert (harness_reason is not None) == live_block, (
            f"divergence on {p.symbol} {p.side} conf={p.confidence}: "
            f"live={live_block} harness={harness_reason!r}"
        )


def test_harness_does_not_apply_struct_to_pre_cutoff_pairs() -> None:
    """A pre-2026-04-13 OPEN had its confidence populated from PPO model
    output. Applying a STRUCT_* floor to it is a category error."""
    pre_cutoff = _make_pair(
        "BTCUSDT", "LONG", 0.10,
        open_ts="2026-03-01T00:00:00",  # before STRUCTURE_FIRST_LIVE_SINCE
    )
    reason = bh._block_reason(pre_cutoff, {"STRUCT_MIN_CONFIDENCE": 0.99})
    assert reason is None


def test_harness_warns_on_pre_cutoff_window_with_struct_override(
    tmp_path,
) -> None:
    """run_backtest must emit a warning when STRUCT_* is evaluated over
    a window containing pre-cutoff pairs — silent mis-filtering was F1."""
    import sqlite3
    db = tmp_path / "warn.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, symbol TEXT, action TEXT,
            data TEXT, price REAL, pnl REAL, confidence REAL,
            reason TEXT, created_at TEXT, is_testnet INTEGER DEFAULT 0
        )
        """
    )
    # One pre-cutoff pair
    conn.execute(
        "INSERT INTO trades(timestamp,symbol,action,price,pnl,confidence,"
        "reason,is_testnet) VALUES (?,?,?,?,?,?,?,?)",
        ("2026-03-01T00:00:00", "BTCUSDT", "OPEN_LONG",
         100.0, None, 0.7, None, 1),
    )
    conn.execute(
        "INSERT INTO trades(timestamp,symbol,action,price,pnl,confidence,"
        "reason,is_testnet) VALUES (?,?,?,?,?,?,?,?)",
        ("2026-03-01T06:00:00", "BTCUSDT", "CLOSE_LONG",
         110.0, 10.0, 1.0, "TP", 1),
    )
    conn.commit()
    conn.close()

    req = bh.BacktestRequest(
        start_date="2026-02-01T00:00:00",
        end_date="2026-06-01T00:00:00",
        config_overrides={"STRUCT_MIN_CONFIDENCE": 0.99},
        db_path=str(db),
        capital_base=5000.0,
    )
    result = bh.run_backtest(req)
    assert any("pre-structure-first" in w for w in result.warnings)


# ─── Override-file round-trip ───────────────────────────────────────────


def test_override_file_round_trip_blocks_matching_entry(tmp_path, monkeypatch) -> None:
    """Write an override file → loader picks it up → live globals reflect it
    → _resolve_struct_floor blocks a confidence below the new floor.

    The DRL_SKIP_RUNTIME_OVERRIDES escape hatch is used in reverse here
    via direct loader invocation, mirroring tests/test_self_improve."""
    import json

    override_file = tmp_path / "active_overrides.json"
    override_file.write_text(json.dumps({
        "experiment_id": 99,
        "config_changes": {
            "STRUCT_SYMBOL_MIN_CONFIDENCE": {"BTCUSDT": 0.75},
        },
    }))

    monkeypatch.setattr(live, "_RUNTIME_OVERRIDE_FILE", override_file)
    # Ensure kill switch is not set in this tmp world
    monkeypatch.setattr(
        live,
        "_AUTONOMY_KILL_SWITCH",
        tmp_path / "does_not_exist",
    )
    # And make sure the env-level escape hatch is off for this call
    monkeypatch.delenv("DRL_SKIP_RUNTIME_OVERRIDES", raising=False)

    summary = live._apply_runtime_overrides()
    assert summary.get("experiment_id") == 99
    assert any(
        "STRUCT_SYMBOL_MIN_CONFIDENCE[BTCUSDT]" in a
        for a in summary.get("applied", [])
    )
    floor, label = live._resolve_struct_floor("BTCUSDT", "LONG")
    assert floor == 0.75
    assert "STRUCT_SYMBOL_MIN_CONFIDENCE[BTCUSDT]" in label


def test_loader_no_op_when_skip_env_set(tmp_path, monkeypatch) -> None:
    """DRL_SKIP_RUNTIME_OVERRIDES=1 short-circuits regardless of file content."""
    override_file = tmp_path / "active_overrides.json"
    override_file.write_text('{"config_changes": {"STRUCT_MIN_CONFIDENCE": 0.9}}')
    monkeypatch.setattr(live, "_RUNTIME_OVERRIDE_FILE", override_file)
    monkeypatch.setenv("DRL_SKIP_RUNTIME_OVERRIDES", "1")
    summary = live._apply_runtime_overrides()
    assert summary == {}
    assert live.STRUCT_MIN_CONFIDENCE == 0.0  # untouched
