"""
Regression test for the widened stagnant_exit band.

On 2026-04-25 the lower bound was widened from -0.3% to -1.0% based on a
35-day backtest (456 reconstructed round trips). The widening should:
  * still admit any pnl% inside [-1.0%, +0.5%]
  * still reject anything outside that band
  * specifically: a -0.5% drifter (previously OUTSIDE band, NOT closed)
    should now be IN-band and eligible for STAGNANT_EXIT.
"""

from live_trading_htf import STAGNANT_HOURS, STAGNANT_PCT_MAX, STAGNANT_PCT_MIN


def _in_band(pnl_pct: float) -> bool:
    return STAGNANT_PCT_MIN <= pnl_pct / 100 <= STAGNANT_PCT_MAX


def test_widened_band_constants() -> None:
    """The deployed band is [-1.0%, +0.5%]; if you tighten it, update the backtest."""
    assert STAGNANT_PCT_MIN == -0.010, (
        f"STAGNANT_PCT_MIN={STAGNANT_PCT_MIN} — backtest "
        f"(scripts/backtest_max_hold_30d.py) showed -0.010 was best of bands tested"
    )
    assert STAGNANT_PCT_MAX == 0.005
    assert STAGNANT_HOURS == 6.0


def test_drifter_at_minus_half_pct_is_in_band() -> None:
    """A position at -0.5% pnl after 6h was OUT of band before this change.
    With the widened band it should be IN-band, eligible for stagnant exit.
    Pre-widen: rule did not fire here (lower bound -0.3%). Post-widen: rule fires.
    """
    assert _in_band(-0.5)


def test_minus_one_and_a_half_pct_still_out_of_band() -> None:
    """At -1.5% the position is too far gone for stagnant treatment — let SL/TP play."""
    assert not _in_band(-1.5)


def test_winning_trade_above_max_not_caught() -> None:
    """A winning trade at +1% pnl should NEVER be force-closed by stagnant rule."""
    assert not _in_band(1.0)


def test_at_band_edges() -> None:
    """Edge inclusive."""
    assert _in_band(-1.0)
    assert _in_band(0.5)
    assert not _in_band(-1.0001)
    assert not _in_band(0.5001)
