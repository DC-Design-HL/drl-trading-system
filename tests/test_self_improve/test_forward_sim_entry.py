"""Forward-sim entry-side tests (PROFITABILITY_PLAN.md P2.B).

Covers the deterministic helpers (resolve_struct_floor, derive_direction)
and a small end-to-end smoke run over a synthetic kline cache. Exit logic
arrives in P2.C — only entries are asserted here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.self_improve import forward_sim as fs


UTC = timezone.utc
_5M_NS = 5 * 60 * 1_000_000_000


# ─── Pure helpers ───────────────────────────────────────────────────────


def test_resolve_struct_floor_baseline_is_no_op() -> None:
    cfg = fs.ForwardSimConfig()
    floor, label = fs.resolve_struct_floor(cfg, "ETHUSDT", "LONG")
    assert floor is None
    assert label == ""


def test_resolve_struct_floor_precedence_matches_live() -> None:
    cfg = fs.ForwardSimConfig(
        struct_min_confidence=0.6,
        struct_symbol_min_confidence={"ETHUSDT": 0.7},
        struct_symbol_directional_conf={"ETHUSDT": {"LONG": 0.8}},
    )
    long_f, long_l = fs.resolve_struct_floor(cfg, "ETHUSDT", "LONG")
    short_f, _ = fs.resolve_struct_floor(cfg, "ETHUSDT", "SHORT")
    btc_f, _ = fs.resolve_struct_floor(cfg, "BTCUSDT", "LONG")
    assert long_f == 0.8
    assert "STRUCT_SYMBOL_DIRECTIONAL_CONF" in long_l
    assert short_f == 0.7
    assert btc_f == 0.6


def test_derive_direction_alignment() -> None:
    side, _ = fs.derive_direction(
        {"trend": "bullish", "last_signal_direction": "bullish"}
    )
    assert side == "LONG"
    side, _ = fs.derive_direction(
        {"trend": "bearish", "last_signal_direction": "bearish"}
    )
    assert side == "SHORT"


def test_derive_direction_misalignment_skips() -> None:
    side, reason = fs.derive_direction(
        {"trend": "bullish", "last_signal_direction": "bearish"}
    )
    assert side is None
    assert "disagree" in reason


def test_derive_direction_ranging_skips() -> None:
    side, reason = fs.derive_direction(
        {"trend": "ranging", "last_signal_direction": "bullish"}
    )
    assert side is None


# ─── Post-close time gates (cooldown / anti-whipsaw, P2.D timing) ────────

_T0 = pd.Timestamp("2026-06-01T00:00:00Z")


def test_post_close_no_history_allows() -> None:
    assert fs._post_close_block(
        "LONG", _T0, cooldown_until_ts=None, last_close_dir=0,
        last_close_pnl=0.0, last_close_ts=None, whipsaw_cooldown_hours=2.0,
    ) is None


def test_post_close_cooldown_blocks_all_sides_until_elapsed() -> None:
    cd_until = _T0 + pd.Timedelta(minutes=30)
    # Inside the cooldown window → blocked regardless of side.
    for side in ("LONG", "SHORT"):
        assert fs._post_close_block(
            side, _T0 + pd.Timedelta(minutes=10),
            cooldown_until_ts=cd_until, last_close_dir=-1,
            last_close_pnl=-5.0, last_close_ts=_T0,
            whipsaw_cooldown_hours=2.0,
        ) == "cooldown"
    # After the cooldown elapses (and no whipsaw reversal) → allowed.
    assert fs._post_close_block(
        "SHORT", _T0 + pd.Timedelta(minutes=31),
        cooldown_until_ts=cd_until, last_close_dir=-1,
        last_close_pnl=-5.0, last_close_ts=_T0, whipsaw_cooldown_hours=2.0,
    ) is None


def test_post_close_whipsaw_blocks_reversal_after_loss() -> None:
    # Last close was a LOSING LONG; an opposite SHORT within 2h is whipsaw.
    assert fs._post_close_block(
        "SHORT", _T0 + pd.Timedelta(hours=1),
        cooldown_until_ts=None, last_close_dir=1, last_close_pnl=-3.0,
        last_close_ts=_T0, whipsaw_cooldown_hours=2.0,
    ) == "whipsaw"
    # Same-side (LONG) re-entry is NOT a whipsaw (only the 30m cooldown gates it).
    assert fs._post_close_block(
        "LONG", _T0 + pd.Timedelta(hours=1),
        cooldown_until_ts=None, last_close_dir=1, last_close_pnl=-3.0,
        last_close_ts=_T0, whipsaw_cooldown_hours=2.0,
    ) is None
    # Past the whipsaw window → reversal allowed.
    assert fs._post_close_block(
        "SHORT", _T0 + pd.Timedelta(hours=2, minutes=1),
        cooldown_until_ts=None, last_close_dir=1, last_close_pnl=-3.0,
        last_close_ts=_T0, whipsaw_cooldown_hours=2.0,
    ) is None


def test_post_close_whipsaw_only_after_a_loss() -> None:
    # Winning last close → no whipsaw block even on an immediate reversal.
    assert fs._post_close_block(
        "SHORT", _T0 + pd.Timedelta(minutes=5),
        cooldown_until_ts=None, last_close_dir=1, last_close_pnl=+8.0,
        last_close_ts=_T0, whipsaw_cooldown_hours=2.0,
    ) is None


def test_post_close_cooldown_takes_precedence_over_whipsaw() -> None:
    # Both would fire; live checks cooldown first, so 'cooldown' wins.
    assert fs._post_close_block(
        "SHORT", _T0 + pd.Timedelta(minutes=10),
        cooldown_until_ts=_T0 + pd.Timedelta(minutes=30),
        last_close_dir=1, last_close_pnl=-3.0, last_close_ts=_T0,
        whipsaw_cooldown_hours=2.0,
    ) == "cooldown"


# ─── Funding accrual (P2.E) ─────────────────────────────────────────────

# Three 8h funding stamps at 00:00 / 08:00 / 16:00 on 2026-06-01.
_F_TS = np.array([
    int(pd.Timestamp("2026-06-01T00:00:00Z").value),
    int(pd.Timestamp("2026-06-01T08:00:00Z").value),
    int(pd.Timestamp("2026-06-01T16:00:00Z").value),
], dtype="int64")
_F_RATE = np.array([0.0001, 0.0001, 0.0001], dtype="float64")
_EN = int(pd.Timestamp("2026-05-31T23:00:00Z").value)  # before all three


def test_funding_none_is_zero() -> None:
    assert fs._funding_cost(1000.0, "LONG", _EN, _EN + 1, None, None) == 0.0


def test_funding_long_pays_short_receives() -> None:
    exit_ns = int(pd.Timestamp("2026-06-01T20:00:00Z").value)  # crosses all 3
    long_cost = fs._funding_cost(1000.0, "LONG", _EN, exit_ns, _F_TS, _F_RATE)
    short_cost = fs._funding_cost(1000.0, "SHORT", _EN, exit_ns, _F_TS, _F_RATE)
    # 3 stamps × 0.0001 × 1000 = 0.30; LONG pays it, SHORT receives it.
    assert long_cost == pytest.approx(0.30)
    assert short_cost == pytest.approx(-0.30)


def test_funding_only_boundaries_inside_window() -> None:
    # Window (00:30, 12:00] contains only the 08:00 stamp.
    en = int(pd.Timestamp("2026-06-01T00:30:00Z").value)
    ex = int(pd.Timestamp("2026-06-01T12:00:00Z").value)
    cost = fs._funding_cost(1000.0, "LONG", en, ex, _F_TS, _F_RATE)
    assert cost == pytest.approx(0.10)


def test_funding_no_boundary_crossed_is_zero() -> None:
    # Intra-window with no funding stamp inside.
    en = int(pd.Timestamp("2026-06-01T09:00:00Z").value)
    ex = int(pd.Timestamp("2026-06-01T15:00:00Z").value)
    assert fs._funding_cost(1000.0, "LONG", en, ex, _F_TS, _F_RATE) == 0.0


# ─── End-to-end smoke over a synthetic cache ────────────────────────────


def _seed_synthetic_cache(
    tmp_path: Path, symbol: str = "BTCUSDT", seed: int = 42,
) -> None:
    """Write a small but valid OHLCV cache so run_forward_sim has data
    to walk over. The signal logic itself is tested elsewhere; here we
    only assert the orchestration glue runs end-to-end without errors.
    """
    # 12 days of 5m bars + a 2-day decision window = 14 days total.
    start_ns = int(datetime(2026, 5, 25, tzinfo=UTC).timestamp() * 1_000_000_000)
    n_5m = 14 * 24 * 12
    ts = np.arange(n_5m, dtype=np.int64) * _5M_NS + start_ns
    # Synthetic random-walk OHLCV
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.standard_normal(n_5m) * 0.5)
    opens = np.roll(closes, 1); opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n_5m))
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n_5m))
    df = pd.DataFrame({
        "ts": ts,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.ones(n_5m),
    })
    base = tmp_path
    base.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base / f"{symbol}_5m.parquet", engine="pyarrow")

    # Coarser timeframes — just resample
    for tf, hours in (("15m", 0.25), ("1h", 1.0), ("4h", 4.0)):
        step = int(hours * 12)
        sub = df.iloc[::step].reset_index(drop=True)
        sub.to_parquet(base / f"{symbol}_{tf}.parquet", engine="pyarrow")


def test_run_forward_sim_smoke(tmp_path: Path) -> None:
    _seed_synthetic_cache(tmp_path)
    start = datetime(2026, 6, 6, tzinfo=UTC)
    end = datetime(2026, 6, 7, tzinfo=UTC)

    result = fs.run_forward_sim(
        symbols=("BTCUSDT",),
        start=start, end=end,
        cache_base=tmp_path,
    )
    assert result.mode == "forward"
    sym = result.per_symbol["BTCUSDT"]
    # Decisions = decision-interval ticks within the window.
    assert sym.n_decisions > 0
    # Entries may be 0 on a random-walk; the contract is that buckets
    # add up to <= decision count (some bars skip before reaching a gate).
    total_skipped = (
        sym.skipped_by_trend + sym.skipped_by_blocklist
        + sym.skipped_by_struct_floor + sym.skipped_by_s5_unimplemented
        + sym.skipped_by_struct_first_adx
        + sym.skipped_by_exhaustion + sym.skipped_by_rsi
        + sym.skipped_by_cooldown + sym.skipped_by_whipsaw
    )
    assert total_skipped + len(sym.entries) == sym.n_decisions


def test_run_forward_sim_blocklist_blocks_entries(tmp_path: Path) -> None:
    """A symbol-side combo in the blocklist must never produce entries."""
    _seed_synthetic_cache(tmp_path, symbol="SOLUSDT")
    start = datetime(2026, 6, 6, tzinfo=UTC)
    end = datetime(2026, 6, 7, tzinfo=UTC)
    # SOL LONG is in default blocklist → any LONG candidates must skip
    result = fs.run_forward_sim(
        symbols=("SOLUSDT",),
        start=start, end=end,
        cache_base=tmp_path,
    )
    sym = result.per_symbol["SOLUSDT"]
    # If any entries did fire, none of them are SOL LONG
    assert all(e.side != "LONG" for e in sym.entries)


def test_run_forward_sim_struct_floor_blocks(tmp_path: Path) -> None:
    _seed_synthetic_cache(tmp_path)
    # Strangle every entry with an impossibly high floor — should produce 0
    cfg = fs.ForwardSimConfig(struct_min_confidence=1.0)
    result = fs.run_forward_sim(
        symbols=("BTCUSDT",),
        start=datetime(2026, 6, 6, tzinfo=UTC),
        end=datetime(2026, 6, 7, tzinfo=UTC),
        cache_base=tmp_path,
        config=cfg,
    )
    assert len(result.per_symbol["BTCUSDT"].entries) == 0


def test_run_forward_sim_eth_s5_runs(tmp_path: Path) -> None:
    """ETH is S5 in the live config. After P2.D the OB-proximity and
    ADX-directional helpers ARE replicated, so ETH should produce a
    result whose buckets add up correctly. Exact entry counts depend on
    the random-walk fixture, so we only assert the bookkeeping holds."""
    _seed_synthetic_cache(tmp_path, symbol="ETHUSDT")
    result = fs.run_forward_sim(
        symbols=("ETHUSDT",),
        start=datetime(2026, 6, 6, tzinfo=UTC),
        end=datetime(2026, 6, 7, tzinfo=UTC),
        cache_base=tmp_path,
    )
    sym = result.per_symbol["ETHUSDT"]
    total_skipped = (
        sym.skipped_by_trend + sym.skipped_by_blocklist
        + sym.skipped_by_struct_floor + sym.skipped_by_s5_unimplemented
        + sym.skipped_by_struct_first_adx
        + sym.skipped_by_exhaustion + sym.skipped_by_rsi
        + sym.skipped_by_cooldown + sym.skipped_by_whipsaw
    )
    assert total_skipped + len(sym.entries) == sym.n_decisions


def test_to_json_round_trip(tmp_path: Path) -> None:
    _seed_synthetic_cache(tmp_path)
    result = fs.run_forward_sim(
        symbols=("BTCUSDT",),
        start=datetime(2026, 6, 6, tzinfo=UTC),
        end=datetime(2026, 6, 7, tzinfo=UTC),
        cache_base=tmp_path,
    )
    blob = result.to_json()
    assert blob["mode"] == "forward"
    assert "BTCUSDT" in blob["per_symbol"]
    assert "blocklist" in blob["config"]


def test_determinism(tmp_path: Path) -> None:
    """Same cache + same config → byte-identical entry timestamps + sides."""
    _seed_synthetic_cache(tmp_path)
    args = dict(
        symbols=("BTCUSDT",),
        start=datetime(2026, 6, 6, tzinfo=UTC),
        end=datetime(2026, 6, 7, tzinfo=UTC),
        cache_base=tmp_path,
    )
    r1 = fs.run_forward_sim(**args)
    r2 = fs.run_forward_sim(**args)
    e1 = [(e.ts, e.side, round(e.confidence, 5)) for e in r1.per_symbol["BTCUSDT"].entries]
    e2 = [(e.ts, e.side, round(e.confidence, 5)) for e in r2.per_symbol["BTCUSDT"].entries]
    assert e1 == e2


# ─── Golden windows (PROFITABILITY_PLAN.md P2) ──────────────────────────
#
# Committed expected metrics over three fixed (symbol, seed, window)
# combinations. These pin the full entry+exit+PnL pipeline so any
# unintended behaviour change is caught. The plan calls for "fixed
# historical windows"; the real kline cache is not committed (large +
# data/), so we use the deterministic synthetic cache instead — same
# regression-guard purpose. Regenerate the constants ONLY for an
# intentional behaviour change, and call it out in the commit.

_GOLDEN = [
    # (symbol, seed, start, end, n_decisions, n_entries, n_trades, net_pnl)
    ("BTCUSDT", 42, (2026, 6, 6), (2026, 6, 8), 192, 31, 31, -118.0665),
    ("BTCUSDT", 7,  (2026, 6, 5), (2026, 6, 9), 286, 21, 21, -333.2550),
    ("ETHUSDT", 42, (2026, 6, 6), (2026, 6, 9), 192, 5, 5, 11.6667),
]


@pytest.mark.parametrize(
    "symbol,seed,start,end,n_dec,n_entries,n_trades,net_pnl", _GOLDEN,
)
def test_golden_window(
    tmp_path, symbol, seed, start, end, n_dec, n_entries, n_trades, net_pnl,
) -> None:
    _seed_synthetic_cache(tmp_path, symbol=symbol, seed=seed)
    r = fs.run_forward_sim(
        symbols=(symbol,),
        start=datetime(*start, tzinfo=UTC),
        end=datetime(*end, tzinfo=UTC),
        cache_base=tmp_path,
    )
    sr = r.per_symbol[symbol]
    assert sr.n_decisions == n_dec
    assert len(sr.entries) == n_entries
    assert len(sr.trades) == n_trades
    assert sr.net_pnl_usd == pytest.approx(net_pnl, abs=1e-3)
