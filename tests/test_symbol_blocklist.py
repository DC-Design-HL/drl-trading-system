"""
Regression test for SYMBOL_SIDE_BLOCKLIST contents.

The blocklist gates entry decisions in HTFLiveBot.execute_trade via a set
membership check (live_trading_htf.py:2346). Entries are added when a
symbol/side combo shows net-negative expectancy over a recent window and
the model needs retraining (which only happens on Chen's Mac).

Removing any of these without first restoring the symbol's expected
profitability re-opens a known loss source — do not delete entries here
without (a) backtest justification on fresh data and (b) sign-off.
"""

from live_trading_htf import SYMBOL_SIDE_BLOCKLIST


def test_sol_long_blocked() -> None:
    """SOL LONG blocklisted 2026-05-11 after May-9 dev-rollback bleed."""
    assert ("SOLUSDT", "LONG") in SYMBOL_SIDE_BLOCKLIST


def test_xrp_both_sides_blocked() -> None:
    """XRP LONG + SHORT blocklisted 2026-05-20.

    Since May-1 reset (n=36): WR 41.7%, net -$196.11. 33/36 entries had
    model confidence <0.55. Backtest: full blocklist saves the maximum
    +$196; no confidence floor produced a net-positive XRP slice in the
    observed window. Reversible after retraining on Mac.
    """
    assert ("XRPUSDT", "LONG") in SYMBOL_SIDE_BLOCKLIST
    assert ("XRPUSDT", "SHORT") in SYMBOL_SIDE_BLOCKLIST


def test_btc_and_eth_not_blocked() -> None:
    """BTC and ETH (both sides) remain tradeable — net-positive on fresh data."""
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for side in ("LONG", "SHORT"):
            assert (symbol, side) not in SYMBOL_SIDE_BLOCKLIST, (
                f"{symbol} {side} unexpectedly blocked"
            )


def test_blocklist_uses_canonical_side_strings() -> None:
    """The gate at line 2346 compares against 'LONG'/'SHORT' — guard typos."""
    for _symbol, side in SYMBOL_SIDE_BLOCKLIST:
        assert side in {"LONG", "SHORT"}, f"non-canonical side string: {side!r}"
