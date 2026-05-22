"""Trigger evaluation tests — synthetic scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.self_improve.metrics import TradeClose
from src.self_improve.triggers import THRESHOLDS, evaluate

UTC = timezone.utc


def _trade(days_ago: float, pnl: float, symbol: str = "BTCUSDT", side: str = "LONG") -> TradeClose:
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    return TradeClose(
        ts=now - timedelta(days=days_ago),
        symbol=symbol,
        side=side,
        pnl=pnl,
    )


def test_no_triggers_when_healthy() -> None:
    """Steady winners over 7 days → no triggers."""
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    trades = [
        # 7d worth of wins across symbols, low variance
        _trade(0.5, pnl=8.0, symbol="BTCUSDT"),
        _trade(1.5, pnl=6.0, symbol="ETHUSDT"),
        _trade(2.5, pnl=7.0, symbol="SOLUSDT"),
        _trade(3.5, pnl=5.0, symbol="BTCUSDT"),
        _trade(4.5, pnl=9.0, symbol="ETHUSDT"),
        _trade(5.5, pnl=6.0, symbol="SOLUSDT"),
    ]
    hits = evaluate(trades, now=now, capital_base=5000.0)
    assert hits == []


def test_t2_pnl_drawdown_fires() -> None:
    """7d net PnL of -$200 on $5k = -4% → T2 fires (threshold -3%)."""
    now = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
    trades = [
        _trade(1.0, pnl=-50.0, symbol="BTCUSDT"),
        _trade(2.0, pnl=-50.0, symbol="ETHUSDT"),
        _trade(3.0, pnl=-50.0, symbol="BTCUSDT"),
        _trade(4.0, pnl=-50.0, symbol="ETHUSDT"),
    ]
    hits = evaluate(trades, now=now, capital_base=5000.0)
    ids = {h.id for h in hits}
    assert "T2" in ids


def test_t4_consecutive_losses_fires() -> None:
    """3 consecutive BTC LONG losses, most recent → T4 fires."""
    trades = [
        _trade(5.0, pnl=5.0,  symbol="BTCUSDT", side="LONG"),  # last win
        _trade(4.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
        _trade(3.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
        _trade(2.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
    ]
    hits = evaluate(trades, capital_base=5000.0)
    t4 = [h for h in hits if h.id == "T4"]
    assert t4, "T4 should fire on trailing 3-loss streak"
    assert any("LONG" in h.metric for h in t4)


def test_t4_does_not_fire_on_old_streak() -> None:
    """Streak ended in the past, recent close was a win → no T4."""
    trades = [
        _trade(5.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
        _trade(4.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
        _trade(3.0, pnl=-5.0, symbol="BTCUSDT", side="LONG"),
        _trade(2.0, pnl=10.0, symbol="BTCUSDT", side="LONG"),  # snapped streak
    ]
    hits = evaluate(trades, capital_base=5000.0)
    t4 = [h for h in hits if h.id == "T4"]
    assert not t4


def test_t3_per_symbol_pf_fires() -> None:
    """XRP with 10 closes at PF=0.5 → T3 fires."""
    # 10 trades: 4 wins of +$5, 6 losses of -$10 → PF = 20/60 = 0.33
    trades = [
        *[_trade(d, pnl=5.0,  symbol="XRPUSDT", side="LONG") for d in (10, 9, 8, 7)],
        *[_trade(d, pnl=-10.0, symbol="XRPUSDT", side="LONG") for d in (6, 5, 4, 3, 2, 1)],
    ]
    hits = evaluate(trades, capital_base=5000.0)
    t3 = [h for h in hits if h.id == "T3" and h.symbol == "XRPUSDT"]
    assert t3
    assert t3[0].value < THRESHOLDS["T3_symbol_pf_min"]


def test_t3_does_not_fire_below_min_closes() -> None:
    """Only 5 closes — below T3's min of 10 → no T3 even if PF is bad."""
    trades = [
        _trade(5.0, pnl=-10.0, symbol="XRPUSDT"),
        _trade(4.0, pnl=-10.0, symbol="XRPUSDT"),
        _trade(3.0, pnl=-10.0, symbol="XRPUSDT"),
        _trade(2.0, pnl=-10.0, symbol="XRPUSDT"),
        _trade(1.0, pnl=-10.0, symbol="XRPUSDT"),
    ]
    hits = evaluate(trades, capital_base=5000.0)
    assert not [h for h in hits if h.id == "T3"]


def test_thresholds_overridable() -> None:
    """Caller can override thresholds (e.g. for tighter canary monitoring)."""
    trades = [
        _trade(1.0, pnl=-50.0),
        _trade(2.0, pnl=-50.0),
    ]
    # Default threshold -3% → -100/5000 = -2% does NOT fire.
    base_hits = evaluate(trades, capital_base=5000.0)
    assert not [h for h in base_hits if h.id == "T2"]

    # Tightened threshold -1% → -2% DOES fire.
    tight_hits = evaluate(
        trades,
        capital_base=5000.0,
        thresholds={"T2_pnl_pct_min": -0.01},
    )
    assert [h for h in tight_hits if h.id == "T2"]


def test_t5_drawdown_fires() -> None:
    """6% drawdown over 30d → T5 fires."""
    trades = [
        _trade(20.0, pnl=200.0),  # peak +200 vs base 5000
        _trade(15.0, pnl=-150.0),
        _trade(10.0, pnl=-150.0),  # DD = (200 - (-100))/5000 = 6% from peak
    ]
    hits = evaluate(trades, capital_base=5000.0)
    assert [h for h in hits if h.id == "T5"]
