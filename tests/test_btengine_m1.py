"""M1 tests — config loading, kline cache day-keying, strategy registry."""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


REPO = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────

def test_config_loads_minimal_yaml(tmp_path):
    from src.btengine.config import BacktestConfig
    p = tmp_path / "c.yaml"
    p.write_text("""
run_id: t1
window: { start: 2026-04-01, end: 2026-04-15 }
symbols: [BTCUSDT]
strategy: structure_first_v3
""".strip())
    cfg = BacktestConfig.from_yaml(p)
    assert cfg.run_id == "t1"
    assert cfg.window.start == _dt.date(2026, 4, 1)
    assert cfg.window.end == _dt.date(2026, 4, 15)
    assert cfg.symbols == ["BTCUSDT"]
    assert cfg.intervals.primary == "15m"  # default from live_constants
    assert cfg.guards.enabled == []        # default empty


def test_config_rejects_inverted_window(tmp_path):
    from src.btengine.config import BacktestConfig
    p = tmp_path / "c.yaml"
    p.write_text("""
run_id: t
window: { start: 2026-05-01, end: 2026-04-01 }
symbols: [BTCUSDT]
strategy: x
""".strip())
    with pytest.raises(ValueError, match="must be after start"):
        BacktestConfig.from_yaml(p)


def test_config_rejects_unknown_interval(tmp_path):
    from src.btengine.config import BacktestConfig
    p = tmp_path / "c.yaml"
    p.write_text("""
run_id: t
window: { start: 2026-04-01, end: 2026-04-15 }
symbols: [BTCUSDT]
strategy: x
intervals: { primary: 7m }
""".strip())
    cfg = BacktestConfig.from_yaml(p)
    with pytest.raises(ValueError, match="not supported"):
        cfg.validate()


def test_config_warns_about_future_end_date(tmp_path):
    from src.btengine.config import BacktestConfig
    p = tmp_path / "c.yaml"
    future = (_dt.date.today() + _dt.timedelta(days=30)).isoformat()
    p.write_text(f"""
run_id: t
window: {{ start: 2026-04-01, end: {future} }}
symbols: [BTCUSDT]
strategy: x
""".strip())
    cfg = BacktestConfig.from_yaml(p)
    warnings = cfg.validate()
    assert any("future" in w for w in warnings)


def test_config_resolves_output_dir(tmp_path):
    from src.btengine.config import BacktestConfig
    p = tmp_path / "c.yaml"
    # Quote the dir value because '${run_id}' contains '{' which confuses
    # YAML flow-mapping parsing if left unquoted.
    p.write_text("""
run_id: my_run_42
window: { start: 2026-04-01, end: 2026-04-15 }
symbols: [BTCUSDT]
strategy: x
output:
  dir: 'runs/${run_id}'
""".strip())
    cfg = BacktestConfig.from_yaml(p)
    out = cfg.resolve_output_dir()
    assert out.name == "my_run_42"
    assert out.parent.name == "runs"


# ──────────────────────────────────────────────────────────────────
# Kline cache
# ──────────────────────────────────────────────────────────────────

def _kline_payload(start_ms: int, n: int, interval_ms: int):
    """Synthetic Binance klines payload: list of 12-tuples."""
    out = []
    for i in range(n):
        ts = start_ms + i * interval_ms
        c = 60000 + i * 0.5
        out.append([ts, c, c+1, c-1, c+0.2, "10",
                    ts + interval_ms - 1, "600000", 5,
                    "5", "300000", "0"])
    return out


def test_kline_cache_writes_parquet_per_day(tmp_path):
    from src.btengine.data.kline_cache import KlineCache, _INTERVAL_SECONDS
    cache = KlineCache(cache_dir=tmp_path)
    # Mock urlopen to return deterministic data for one specific day
    target_day = _dt.date(2026, 4, 1)
    day_start_ms = int(_dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc).timestamp() * 1000)
    interval_ms = _INTERVAL_SECONDS["15m"] * 1000
    payload = _kline_payload(day_start_ms, 96, interval_ms)  # 96 bars / day at 15m

    # context-manager mock
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    import json as _json
    cm.read = MagicMock(return_value=_json.dumps(payload).encode())

    with patch("urllib.request.urlopen", return_value=cm):
        df = cache.get("BTCUSDT", "15m",
                       _dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc),
                       _dt.datetime(2026, 4, 2, tzinfo=_dt.timezone.utc))
    assert len(df) == 96
    # Parquet should be written
    p = tmp_path / "BTCUSDT" / "15m" / "2026-04-01.parquet"
    assert p.exists()


def test_kline_cache_hits_disk_on_second_call(tmp_path):
    """Second call for the same day must NOT touch the network."""
    from src.btengine.data.kline_cache import KlineCache, _INTERVAL_SECONDS
    cache = KlineCache(cache_dir=tmp_path)
    day_start_ms = int(_dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc).timestamp() * 1000)
    interval_ms = _INTERVAL_SECONDS["1h"] * 1000
    payload = _kline_payload(day_start_ms, 24, interval_ms)
    call_count = {"n": 0}

    def _factory(*a, **k):
        call_count["n"] += 1
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        import json as _json
        cm.read = MagicMock(return_value=_json.dumps(payload).encode())
        return cm

    with patch("urllib.request.urlopen", side_effect=_factory):
        cache.get("ETHUSDT", "1h",
                  _dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc),
                  _dt.datetime(2026, 4, 2, tzinfo=_dt.timezone.utc))
        # Second call — same range
        cache.get("ETHUSDT", "1h",
                  _dt.datetime(2026, 4, 1, tzinfo=_dt.timezone.utc),
                  _dt.datetime(2026, 4, 2, tzinfo=_dt.timezone.utc))
    # Only the first call should have hit the network
    assert call_count["n"] == 1


# ──────────────────────────────────────────────────────────────────
# Strategy registry
# ──────────────────────────────────────────────────────────────────

def test_strategy_registry_decorator():
    from src.btengine.strategy.base import register_strategy, Strategy, get_strategy

    @register_strategy("test_strat_xyz")
    class _T(Strategy):
        pass

    assert get_strategy("test_strat_xyz") is _T
    # Registering twice raises
    with pytest.raises(ValueError, match="already registered"):
        @register_strategy("test_strat_xyz")
        class _T2(Strategy):
            pass


def test_strategy_registry_unknown_name_raises():
    from src.btengine.strategy.base import get_strategy
    with pytest.raises(KeyError, match="not registered"):
        get_strategy("definitely_does_not_exist_zzz")


# ──────────────────────────────────────────────────────────────────
# Live constants are non-empty (smoke)
# ──────────────────────────────────────────────────────────────────

def test_live_constants_match_live_bot():
    """The constants in btengine must match what the live bot actually uses
    today. Spot-check the most-likely-to-drift values."""
    from src.btengine import live_constants as LC
    import live_trading_htf as live

    assert LC.SYMBOL_SIDE_BLOCKLIST == frozenset(live.SYMBOL_SIDE_BLOCKLIST), \
        "btengine SYMBOL_SIDE_BLOCKLIST drifted from live_trading_htf"
    assert LC.REVERSAL_BLOCK_LONG_CANARY_SYMBOLS == frozenset(live.REVERSAL_BLOCK_LONG_CANARY_SYMBOLS), \
        "btengine REVERSAL_BLOCK_LONG_CANARY_SYMBOLS drifted from live_trading_htf"
    assert LC.REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT == live.REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT
    assert LC.ADX_GUARD_MIN == live.ADX_GUARD_MIN
    assert LC.ADX_GUARD_MAX == live.ADX_GUARD_MAX
    # Fee should match too (audit found 4 scripts double-counting)
    # The live module names this differently; check for any of the expected
    # variants.
    live_fee = getattr(live, "TRADING_FEE", None) or getattr(live, "FEE_PCT", None)
    if live_fee is not None:
        assert abs(LC.TRADING_FEE_TAKER - live_fee) < 1e-6, \
            f"btengine TRADING_FEE_TAKER {LC.TRADING_FEE_TAKER} != live {live_fee}"
