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


# ─── End-to-end smoke over a synthetic cache ────────────────────────────


def _seed_synthetic_cache(tmp_path: Path, symbol: str = "BTCUSDT") -> None:
    """Write a small but valid OHLCV cache so run_forward_sim has data
    to walk over. The signal logic itself is tested elsewhere; here we
    only assert the orchestration glue runs end-to-end without errors.
    """
    # 12 days of 5m bars + a 2-day decision window = 14 days total.
    start_ns = int(datetime(2026, 5, 25, tzinfo=UTC).timestamp() * 1_000_000_000)
    n_5m = 14 * 24 * 12
    ts = np.arange(n_5m, dtype=np.int64) * _5M_NS + start_ns
    # Synthetic random-walk OHLCV
    rng = np.random.default_rng(42)
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
