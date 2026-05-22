"""Backtest harness tests — synthetic scenarios + smoke-tests against
live DB to confirm the harness reproduces known historical findings.

The synthetic tests are the formal unit tests — they use a fresh tmp
DB seeded with controlled OPEN/CLOSE pairs so the expected metrics are
exact. The live-DB smoke tests are conditional (skip if the DB or
expected scenario data isn't present), and check that the harness
reproduces facts established in past analyses (the May-20 XRP
deep-dive, the unfiltered baseline, etc.).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.self_improve.migrate import migrate
from src.self_improve.backtest_harness import (
    BacktestRequest,
    pair_open_close,
    run_backtest,
)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic DB fixtures
# ─────────────────────────────────────────────────────────────────────────


def _new_trades_db(path: Path) -> None:
    """Build a tmp DB with the live `trades` schema + self-improve schema."""
    with sqlite3.connect(str(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                action TEXT,
                data TEXT,
                price REAL,
                pnl REAL,
                confidence REAL,
                reason TEXT,
                created_at TEXT,
                is_testnet INTEGER DEFAULT 0
            )
            """
        )
        migrate(conn)


def _insert(
    db: Path,
    *,
    ts: str,
    symbol: str,
    action: str,
    pnl: float = 0.0,
    confidence: float = 0.5,
    is_testnet: int = 1,
    reason: str = "",
) -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            INSERT INTO trades(timestamp, symbol, action, data, price,
                               pnl, confidence, reason, is_testnet)
            VALUES (?, ?, ?, '{}', 100.0, ?, ?, ?, ?)
            """,
            (ts, symbol, action, pnl, confidence, reason, is_testnet),
        )


def _build_simple_scenario(tmp_path: Path) -> Path:
    """A scenario with 4 trades on 2 symbols:
      * BTC LONG, conf=0.80, pnl=+10
      * BTC LONG, conf=0.50, pnl=-5
      * XRP LONG, conf=0.40, pnl=-12
      * XRP LONG, conf=0.70, pnl=+3

    Useful for verifying filter behavior with known-exact expected
    outcomes.
    """
    db = tmp_path / "trades.db"
    _new_trades_db(db)

    # Pairs are inserted in chronological order; the pairing logic walks
    # the table in (timestamp, id) order.
    for i, (ts_open, ts_close, sym, conf, pnl) in enumerate([
        ("2026-05-10T00:00:00", "2026-05-10T01:00:00", "BTCUSDT", 0.80, 10.0),
        ("2026-05-11T00:00:00", "2026-05-11T01:00:00", "BTCUSDT", 0.50, -5.0),
        ("2026-05-12T00:00:00", "2026-05-12T01:00:00", "XRPUSDT", 0.40, -12.0),
        ("2026-05-13T00:00:00", "2026-05-13T01:00:00", "XRPUSDT", 0.70, 3.0),
    ]):
        _insert(db, ts=ts_open, symbol=sym, action="OPEN_LONG", confidence=conf)
        _insert(db, ts=ts_close, symbol=sym, action="CLOSE_LONG", pnl=pnl, reason="TP")
    return db


# ─────────────────────────────────────────────────────────────────────────
# pair_open_close behavior
# ─────────────────────────────────────────────────────────────────────────


def test_pair_open_close_simple(tmp_path: Path) -> None:
    db = _build_simple_scenario(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        pairs = pair_open_close(
            conn,
            start_date="2026-05-01T00:00:00",
            end_date="2026-05-31T00:00:00",
        )
    assert len(pairs) == 4
    pnls = sorted(p.pnl for p in pairs)
    assert pnls == [-12.0, -5.0, 3.0, 10.0]


def test_pair_open_close_filters_outside_window(tmp_path: Path) -> None:
    """A CLOSE outside [start, end] should be dropped."""
    db = _build_simple_scenario(tmp_path)
    with sqlite3.connect(str(db)) as conn:
        pairs = pair_open_close(
            conn,
            start_date="2026-05-11T00:00:00",  # excludes 5/10 close
            end_date="2026-05-12T23:59:59",    # excludes 5/13 close
        )
    # Only 2 closes (5/11 and 5/12) remain
    assert len(pairs) == 2


def test_pair_open_close_ignores_mainnet(tmp_path: Path) -> None:
    """Hard rule: only is_testnet=1 — replay must not touch mainnet rows."""
    db = tmp_path / "trades.db"
    _new_trades_db(db)
    _insert(db, ts="2026-05-10T00:00:00", symbol="BTCUSDT",
            action="OPEN_LONG", confidence=0.7, is_testnet=0)
    _insert(db, ts="2026-05-10T01:00:00", symbol="BTCUSDT",
            action="CLOSE_LONG", pnl=999.0, is_testnet=0)
    with sqlite3.connect(str(db)) as conn:
        pairs = pair_open_close(
            conn,
            start_date="2026-05-01T00:00:00",
            end_date="2026-05-31T00:00:00",
        )
    assert pairs == []


# ─────────────────────────────────────────────────────────────────────────
# Override behavior
# ─────────────────────────────────────────────────────────────────────────


def test_no_overrides_keeps_everything(tmp_path: Path) -> None:
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={},
        db_path=str(db),
    )
    res = run_backtest(req)
    assert res.n_input_pairs == 4
    assert res.n_kept_pairs == 4
    assert res.n_blocked_pairs == 0
    assert res.portfolio_metrics["net_pnl_usd"] == pytest.approx(-4.0)


def test_min_confidence_blocks_low_conf_opens(tmp_path: Path) -> None:
    """Floor at 0.60 should block the BTC@0.50 and XRP@0.40 opens."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={"MIN_CONFIDENCE": 0.60},
        db_path=str(db),
    )
    res = run_backtest(req)
    assert res.n_blocked_pairs == 2
    # Surviving: BTC@0.80 (+10) and XRP@0.70 (+3) → net +13
    assert res.portfolio_metrics["net_pnl_usd"] == pytest.approx(13.0)
    blocked_syms = {(b["symbol"], b["confidence"]) for b in res.blocked_trades}
    assert ("BTCUSDT", 0.50) in blocked_syms
    assert ("XRPUSDT", 0.40) in blocked_syms


def test_per_symbol_confidence_takes_precedence_over_global(tmp_path: Path) -> None:
    """Per-symbol XRP floor at 0.80 (only) — global is unset; XRP loses both."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={"SYMBOL_MIN_CONFIDENCE": {"XRPUSDT": 0.80}},
        db_path=str(db),
    )
    res = run_backtest(req)
    # Both XRP opens (0.40, 0.70) blocked; BTC opens (0.80, 0.50) kept
    assert res.n_blocked_pairs == 2
    blocked_syms = {b["symbol"] for b in res.blocked_trades}
    assert blocked_syms == {"XRPUSDT"}


def test_blocklist_blocks_by_symbol_side(tmp_path: Path) -> None:
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "SYMBOL_SIDE_BLOCKLIST_ADD": [["XRPUSDT", "LONG"]],
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    # Both XRP LONG opens blocked; BTC LONG opens kept
    assert res.n_blocked_pairs == 2
    assert all(b["symbol"] == "XRPUSDT" for b in res.blocked_trades)


def test_blocklist_does_not_block_other_sides(tmp_path: Path) -> None:
    """Adding XRPUSDT:SHORT to blocklist must NOT block XRP LONG trades."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "SYMBOL_SIDE_BLOCKLIST_ADD": [["XRPUSDT", "SHORT"]],
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    assert res.n_blocked_pairs == 0


def _build_mixed_side_scenario(tmp_path: Path) -> Path:
    """A scenario with both LONG and SHORT trades on BTC at varying
    confidences — used to exercise SYMBOL_DIRECTIONAL_CONF, which is
    side-specific.

      * BTC LONG,  conf=0.70, pnl=+8
      * BTC LONG,  conf=0.92, pnl=+15
      * BTC SHORT, conf=0.70, pnl=-6
      * BTC SHORT, conf=0.90, pnl=+12
    """
    db = tmp_path / "trades.db"
    _new_trades_db(db)
    rows = [
        ("2026-05-10T00:00:00", "2026-05-10T01:00:00", "LONG",  0.70, 8.0),
        ("2026-05-11T00:00:00", "2026-05-11T01:00:00", "LONG",  0.92, 15.0),
        ("2026-05-12T00:00:00", "2026-05-12T01:00:00", "SHORT", 0.70, -6.0),
        ("2026-05-13T00:00:00", "2026-05-13T01:00:00", "SHORT", 0.90, 12.0),
    ]
    for ts_open, ts_close, side, conf, pnl in rows:
        open_action = f"OPEN_{side}"
        close_action = f"CLOSE_{side}"
        _insert(db, ts=ts_open, symbol="BTCUSDT", action=open_action, confidence=conf)
        _insert(db, ts=ts_close, symbol="BTCUSDT", action=close_action, pnl=pnl, reason="TP")
    return db


def test_directional_floor_blocks_only_target_side(tmp_path: Path) -> None:
    """SYMBOL_DIRECTIONAL_CONF on BTC SHORT @0.85 should block the
    low-conf SHORT (0.70) but neither LONG nor the high-conf SHORT."""
    db = _build_mixed_side_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "SYMBOL_DIRECTIONAL_CONF": {"BTCUSDT": {"SHORT": 0.85}},
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    # Exactly the 0.70 SHORT is blocked
    assert res.n_blocked_pairs == 1
    blocked = res.blocked_trades[0]
    assert blocked["symbol"] == "BTCUSDT"
    assert blocked["side"] == "SHORT"
    assert blocked["confidence"] == pytest.approx(0.70)
    assert "directional_floor" in blocked["reason"]
    # Net of kept: +8 (LONG@0.70) + 15 (LONG@0.92) + 12 (SHORT@0.90) = +35
    assert res.portfolio_metrics["net_pnl_usd"] == pytest.approx(35.0)


def test_directional_floor_other_side_unaffected(tmp_path: Path) -> None:
    """Setting BTC SHORT floor must not affect BTC LONG."""
    db = _build_mixed_side_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            # Aggressive 0.99 floor — would kill every LONG if it leaked
            "SYMBOL_DIRECTIONAL_CONF": {"BTCUSDT": {"SHORT": 0.99}},
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    # Both SHORTs gone, both LONGs kept
    assert res.n_blocked_pairs == 2
    kept_sides = {p["side"] for p in res.trade_log}
    assert kept_sides == {"LONG"}


def test_directional_floor_precedes_per_symbol_floor(tmp_path: Path) -> None:
    """When both SYMBOL_DIRECTIONAL_CONF and SYMBOL_MIN_CONFIDENCE apply,
    the side-specific reason should be the one reported."""
    db = _build_mixed_side_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "SYMBOL_DIRECTIONAL_CONF": {"BTCUSDT": {"SHORT": 0.85}},
            "SYMBOL_MIN_CONFIDENCE": {"BTCUSDT": 0.80},
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    short_blocked = [b for b in res.blocked_trades if b["side"] == "SHORT"]
    assert short_blocked, "expected the low-conf SHORT to be blocked"
    assert "directional_floor" in short_blocked[0]["reason"]


def test_unknown_override_key_emits_warning(tmp_path: Path) -> None:
    """Regression: unrecognized keys must not silently degrade to no-op.

    The 2026-05-22 experiment #1 incident showed the Researcher proposing
    a real config name the harness didn't implement; harness silently
    treated it as baseline and the gate passed trivially. From now on
    unknown keys MUST surface in result.warnings so the orchestrator can
    reject the experiment."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "DEFINITELY_NOT_A_REAL_KEY": {"foo": "bar"},
            "ANOTHER_FAKE": 0.5,
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    unknown_warnings = [
        w for w in res.warnings if w.startswith("unrecognized override key")
    ]
    assert len(unknown_warnings) == 2
    joined = " ".join(unknown_warnings)
    assert "DEFINITELY_NOT_A_REAL_KEY" in joined
    assert "ANOTHER_FAKE" in joined
    # And the harness should still have run baseline behavior (4 kept)
    assert res.n_kept_pairs == 4


def test_per_symbol_then_global_then_blocklist_order(tmp_path: Path) -> None:
    """Combine multiple overrides — blocklist hit reported first."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={
            "MIN_CONFIDENCE": 0.60,
            "SYMBOL_SIDE_BLOCKLIST_ADD": [["XRPUSDT", "LONG"]],
        },
        db_path=str(db),
    )
    res = run_backtest(req)
    # XRP entries hit blocklist; BTC@0.50 hits global floor
    assert res.n_blocked_pairs == 3
    xrp_reasons = [b["reason"] for b in res.blocked_trades if b["symbol"] == "XRPUSDT"]
    assert all("blocklist" in r for r in xrp_reasons)
    btc_low = [b for b in res.blocked_trades if b["symbol"] == "BTCUSDT"]
    assert len(btc_low) == 1
    assert "global_min_conf" in btc_low[0]["reason"]


# ─────────────────────────────────────────────────────────────────────────
# Determinism + JSON serialization
# ─────────────────────────────────────────────────────────────────────────


def test_result_is_json_serializable(tmp_path: Path) -> None:
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={"MIN_CONFIDENCE": 0.60},
        db_path=str(db),
    )
    import json
    from src.self_improve.backtest_harness import serialize
    body = serialize(run_backtest(req))
    parsed = json.loads(body)
    assert "portfolio_metrics" in parsed
    assert "trade_log" in parsed
    assert "blocked_trades" in parsed
    assert parsed["n_input_pairs"] == 4


def test_same_inputs_same_outputs(tmp_path: Path) -> None:
    """Determinism: two runs with the same request produce identical metrics."""
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={"MIN_CONFIDENCE": 0.55},
        db_path=str(db),
    )
    r1 = run_backtest(req)
    r2 = run_backtest(req)
    assert r1.portfolio_metrics == r2.portfolio_metrics
    assert r1.trade_log == r2.trade_log
    assert r1.n_kept_pairs == r2.n_kept_pairs


def test_warning_on_empty_input(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    _new_trades_db(db)  # schema but no rows
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        db_path=str(db),
    )
    res = run_backtest(req)
    assert res.n_input_pairs == 0
    assert any("no OPEN/CLOSE pairs" in w for w in res.warnings)


def test_warning_on_all_blocked(tmp_path: Path) -> None:
    db = _build_simple_scenario(tmp_path)
    req = BacktestRequest(
        start_date="2026-05-01T00:00:00",
        end_date="2026-05-31T00:00:00",
        config_overrides={"MIN_CONFIDENCE": 0.99},  # blocks all 4
        db_path=str(db),
    )
    res = run_backtest(req)
    assert res.n_kept_pairs == 0
    assert any("every input pair was blocked" in w for w in res.warnings)


# ─────────────────────────────────────────────────────────────────────────
# Historical scenario tests against the live DB
# ─────────────────────────────────────────────────────────────────────────
# These are SMOKE tests — they confirm that the harness reproduces facts
# established in past manual analyses. If the live DB or the expected
# data isn't present, the test is skipped (CI-friendly).


_LIVE_DB = Path(__file__).resolve().parents[2] / "data" / "trading.db"
_LIVE_DB_AVAILABLE = _LIVE_DB.exists()


@pytest.mark.skipif(
    not _LIVE_DB_AVAILABLE, reason="live data/trading.db not available"
)
def test_scenario_xrp_blocklist_reproduces_may20_finding() -> None:
    """Applying the XRP blocklist retroactively over the May-1 → May-20
    window should match the May-20 deep-dive: ~+$196 PnL improvement,
    36 XRP trades blocked."""
    baseline = run_backtest(
        BacktestRequest(
            start_date="2026-05-01T12:29:00",
            end_date="2026-05-20T13:00:00",
            config_overrides={},
            label="baseline-pre-blocklist",
            db_path=str(_LIVE_DB),
        )
    )
    with_block = run_backtest(
        BacktestRequest(
            start_date="2026-05-01T12:29:00",
            end_date="2026-05-20T13:00:00",
            config_overrides={
                "SYMBOL_SIDE_BLOCKLIST_ADD": [
                    ["XRPUSDT", "LONG"],
                    ["XRPUSDT", "SHORT"],
                ],
            },
            label="retroactive-xrp-blocklist",
            db_path=str(_LIVE_DB),
        )
    )

    blocked_xrp = with_block.n_input_pairs - with_block.n_kept_pairs
    # May-20 analysis: 36 XRP closes
    assert 33 <= blocked_xrp <= 40, (
        f"expected ~36 XRP trades blocked, got {blocked_xrp}"
    )
    delta = (
        with_block.portfolio_metrics["net_pnl_usd"]
        - baseline.portfolio_metrics["net_pnl_usd"]
    )
    # May-20 analysis: +$196.11 saved
    assert 180 <= delta <= 215, (
        f"expected +$196 saved by XRP blocklist, got +${delta:.2f}"
    )


@pytest.mark.skipif(
    not _LIVE_DB_AVAILABLE, reason="live data/trading.db not available"
)
def test_scenario_min_conf_tightening_blocks_xrp() -> None:
    """Raising MIN_CONFIDENCE to 0.65 over the May-1 → May-20 window
    should block ALL XRP trades (per May-20 finding: 33/36 XRP entries
    had conf < 0.55; all 36 had conf < 0.65) and ~zero BTC/ETH/SOL
    trades."""
    res = run_backtest(
        BacktestRequest(
            start_date="2026-05-01T12:29:00",
            end_date="2026-05-20T13:00:00",
            config_overrides={"MIN_CONFIDENCE": 0.65},
            label="floor-0.65",
            db_path=str(_LIVE_DB),
        )
    )
    blocked_by_symbol: dict[str, int] = {}
    for b in res.blocked_trades:
        blocked_by_symbol[b["symbol"]] = blocked_by_symbol.get(b["symbol"], 0) + 1
    # XRP should dominate the blocked set
    assert blocked_by_symbol.get("XRPUSDT", 0) >= 33


@pytest.mark.skipif(
    not _LIVE_DB_AVAILABLE, reason="live data/trading.db not available"
)
def test_scenario_no_overrides_matches_full_history() -> None:
    """No overrides means: kept count = input count, blocked count = 0,
    PnL = sum of all CLOSE.pnl in the window."""
    res = run_backtest(
        BacktestRequest(
            start_date="2026-05-01T12:29:00",
            end_date="2026-05-22T00:00:00",
            config_overrides={},
            label="raw-history",
            db_path=str(_LIVE_DB),
        )
    )
    assert res.n_blocked_pairs == 0
    assert res.n_kept_pairs == res.n_input_pairs
    # Sanity: > 100 closes in 3 weeks (real-world cadence is ~5 closes/day)
    assert res.n_kept_pairs > 100
