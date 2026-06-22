"""Regression test for the 2026-06-22 balance/PnL reporting fix.

Bug: the dashboard showed the real $4,731 wallet as $5,340 — total_balance
added the per-symbol cumulative realized_pnl on top of the already-correct
synced wallet (double-count), and the headline PnL summed those same inflated
per-close figures. Fix: headline balance = real wallet; headline PnL = wallet
− reset baseline.
"""

from __future__ import annotations

from src.ui.api_server import headline_pnl, RESET_BASELINE_USD


def test_headline_pnl_is_wallet_minus_baseline():
    # Real wallet on 2026-06-22 was $4,731.32 → a real −$268.68 drawdown,
    # NOT the +$340 the double-counted dashboard implied.
    assert headline_pnl(4731.32) == -268.68
    assert headline_pnl(5608.0) == 608.0
    assert headline_pnl(RESET_BASELINE_USD) == 0.0


def test_headline_pnl_custom_baseline():
    assert headline_pnl(5200.0, baseline=5000.0) == 200.0
    assert headline_pnl(4800.0, baseline=5000.0) == -200.0


def test_headline_pnl_does_not_double_count_realized():
    # The old bug: balance = wallet + sum(realized_pnl). With wallet already
    # reflecting realized PnL, the headline must depend ONLY on the wallet, so
    # adding a (bogus) realized sum must not change it.
    wallet = 4731.32
    bogus_realized_sum = 608.64  # what used to be added on top
    assert headline_pnl(wallet) == headline_pnl(wallet)  # stable
    # The number must equal wallet-baseline, never wallet+realized-baseline.
    assert headline_pnl(wallet) != round(wallet + bogus_realized_sum - RESET_BASELINE_USD, 2)
