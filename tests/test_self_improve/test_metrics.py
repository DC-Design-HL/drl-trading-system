"""Tests for the pure-function metrics module.

Synthetic inputs only — the metrics functions must work without any
SQLite or live-data dependency. If any of these tests rely on real
trade data, the design has leaked.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.self_improve.metrics import (
    TradeClose,
    consecutive_losses,
    filter_window,
    max_drawdown_pct,
    net_pnl,
    parse_ts,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    trailing_consecutive_losses,
    win_rate,
)

UTC = timezone.utc


def _t(ts: str, symbol: str = "BTCUSDT", side: str = "LONG", pnl: float = 1.0) -> TradeClose:
    return TradeClose(ts=parse_ts(ts), symbol=symbol, side=side, pnl=pnl)


def test_parse_ts_naive_treated_as_utc() -> None:
    dt = parse_ts("2026-05-22T12:00:00")
    assert dt.tzinfo is UTC
    assert dt.hour == 12


def test_parse_ts_with_z() -> None:
    dt = parse_ts("2026-05-22T12:00:00Z")
    assert dt.tzinfo is UTC


def test_filter_window_keeps_only_recent() -> None:
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    trades = [
        _t("2026-05-22T11:00:00"),  # 1h ago — kept
        _t("2026-05-20T12:00:00"),  # 2d ago — kept
        _t("2026-05-15T12:00:00"),  # 7d ago — boundary kept
        _t("2026-05-13T12:00:00"),  # 9d ago — dropped
    ]
    out = filter_window(trades, now=now, days=7)
    assert len(out) == 3


def test_net_pnl() -> None:
    trades = [_t("2026-05-22T00:00:00", pnl=10.0), _t("2026-05-22T01:00:00", pnl=-3.0)]
    assert net_pnl(trades) == pytest.approx(7.0)


def test_win_rate_empty() -> None:
    assert win_rate([]) == 0.0


def test_win_rate_mixed() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=10.0),
        _t("2026-05-22T01:00:00", pnl=-3.0),
        _t("2026-05-22T02:00:00", pnl=5.0),
        _t("2026-05-22T03:00:00", pnl=0.0),  # 0 is not a win
    ]
    assert win_rate(trades) == pytest.approx(0.5)


def test_profit_factor_basic() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=10.0),
        _t("2026-05-22T01:00:00", pnl=-5.0),
    ]
    assert profit_factor(trades) == pytest.approx(2.0)


def test_profit_factor_no_losses_is_inf() -> None:
    trades = [_t("2026-05-22T00:00:00", pnl=10.0)]
    assert profit_factor(trades) == math.inf


def test_profit_factor_no_wins_is_zero() -> None:
    trades = [_t("2026-05-22T00:00:00", pnl=-10.0)]
    assert profit_factor(trades) == 0.0


def test_profit_factor_empty_is_zero() -> None:
    assert profit_factor([]) == 0.0


def test_consecutive_losses_chronological() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=-1.0),
        _t("2026-05-22T01:00:00", pnl=-1.0),
        _t("2026-05-22T02:00:00", pnl=5.0),
        _t("2026-05-22T03:00:00", pnl=-1.0),
        _t("2026-05-22T04:00:00", pnl=-1.0),
        _t("2026-05-22T05:00:00", pnl=-1.0),
    ]
    assert consecutive_losses(trades) == 3  # the second streak


def test_trailing_consecutive_losses_zero_when_recent_win() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=-1.0),
        _t("2026-05-22T01:00:00", pnl=5.0),  # most recent — breaks the streak
    ]
    assert trailing_consecutive_losses(trades) == 0


def test_trailing_consecutive_losses_streak() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=5.0),
        _t("2026-05-22T01:00:00", pnl=-1.0),
        _t("2026-05-22T02:00:00", pnl=-1.0),
        _t("2026-05-22T03:00:00", pnl=-1.0),
    ]
    assert trailing_consecutive_losses(trades) == 3


def test_max_drawdown_pct_no_loss_is_zero() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=10.0),
        _t("2026-05-22T01:00:00", pnl=5.0),
    ]
    assert max_drawdown_pct(trades, capital_base=1000.0) == 0.0


def test_max_drawdown_pct_simple() -> None:
    trades = [
        _t("2026-05-22T00:00:00", pnl=100.0),   # equity peak = 1100
        _t("2026-05-22T01:00:00", pnl=-50.0),   # equity = 1050 → DD = 50/1000 = 5%
        _t("2026-05-22T02:00:00", pnl=-30.0),   # equity = 1020 → DD = 80/1000 = 8% ← max
        _t("2026-05-22T03:00:00", pnl=200.0),   # recovers
    ]
    dd = max_drawdown_pct(trades, capital_base=1000.0)
    assert dd == pytest.approx(8.0)


def test_sharpe_zero_with_too_few_days() -> None:
    """Need at least 2 distinct days for Sharpe."""
    trades = [_t("2026-05-22T00:00:00", pnl=10.0)]
    assert sharpe_ratio(trades) == 0.0


def test_sharpe_zero_with_constant_pnl() -> None:
    """All days the same → zero variance → Sharpe is 0 by convention."""
    trades = [
        _t("2026-05-20T00:00:00", pnl=5.0),
        _t("2026-05-21T00:00:00", pnl=5.0),
        _t("2026-05-22T00:00:00", pnl=5.0),
    ]
    assert sharpe_ratio(trades) == 0.0


def test_sharpe_positive_when_consistently_winning() -> None:
    """Three consecutive winning days with low variance → positive Sharpe."""
    base = datetime(2026, 5, 20, tzinfo=UTC)
    trades = [
        TradeClose(ts=base + timedelta(days=i), symbol="X", side="LONG", pnl=p)
        for i, p in enumerate((10.0, 12.0, 11.0, 13.0, 9.0))
    ]
    sh = sharpe_ratio(trades, capital_base=1000.0)
    assert sh > 0


def test_sortino_inf_when_no_downside() -> None:
    base = datetime(2026, 5, 20, tzinfo=UTC)
    trades = [
        TradeClose(ts=base + timedelta(days=i), symbol="X", side="LONG", pnl=p)
        for i, p in enumerate((10.0, 12.0, 11.0))
    ]
    assert sortino_ratio(trades) == math.inf


def test_summarize_returns_all_keys() -> None:
    out = summarize([_t("2026-05-22T00:00:00", pnl=5.0)])
    assert set(out.keys()) >= {
        "net_pnl_usd", "num_closes", "win_rate", "profit_factor",
        "sharpe", "sortino", "max_drawdown_pct",
    }
