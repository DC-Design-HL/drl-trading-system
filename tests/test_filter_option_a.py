"""
Regression tests for Option A — deployed 2026-04-27 after equity drew down
-8.18% in 42h on the 3× sizing rollout.

Three deltas:
  1. RISK_POOL_PCT  0.30 → 0.20   (3× → 2× sizing)
  2. SYMBOL_SIDE_BLOCKLIST: 4 historically loser combos blocked at entry
  3. FAKE_BOS_ENTRY_GUARD_ENABLED: structure detector's fake_bos/fake_choch
     flags now also gate entry (previously only used for SL adjustments)
"""

from live_trading_htf import (
    FAKE_BOS_ENTRY_GUARD_ENABLED,
    FIXED_MAX_NOTIONAL,
    RISK_BUDGET_PARTS,
    RISK_POOL_PCT,
    SYMBOL_SIDE_BLOCKLIST,
)


def test_sizing_stepped_back_to_2x() -> None:
    """RISK_POOL_PCT must be 0.20 on this branch (was 0.30 before drawdown).
    If you tighten or loosen this, re-run scripts/backtest_30pct_target.py
    against the current trade DB and update the expected drawdown numbers
    in docs/skills/aggressive-sizing-30pct.md.
    """
    assert RISK_POOL_PCT == 0.20, (
        f"RISK_POOL_PCT={RISK_POOL_PCT} — expected 0.20 (2× sizing). "
        f"3× was rolled back on 2026-04-27 after -8.18% drawdown in 42h."
    )
    assert RISK_BUDGET_PARTS == 20
    # FIXED_MAX_NOTIONAL stays at 6000 — at 2× the typical notional is
    # ~$3,300 well below the cap, but raising the cap was part of the
    # aggressive bundle and there's no reason to drop it back.
    assert FIXED_MAX_NOTIONAL == 6000.0


def test_symbol_side_blocklist_contents() -> None:
    """Blocklist must contain exactly the 4 combos identified by backtest:
    BTC SHORT, ETH SHORT, ETH LONG, SOL LONG.
    """
    assert SYMBOL_SIDE_BLOCKLIST == {
        ("BTCUSDT", "SHORT"),
        ("ETHUSDT", "SHORT"),
        ("ETHUSDT", "LONG"),
        ("SOLUSDT", "LONG"),
    }


def test_blocklist_is_directional() -> None:
    """The asymmetry matters: SOL SHORT is profitable (75.6% WR historically)
    but SOL LONG is not. The block must be by (symbol, side) tuple, not
    by symbol alone.
    """
    # SOL SHORT must NOT be in blocklist — it's the strongest performer
    assert ("SOLUSDT", "SHORT") not in SYMBOL_SIDE_BLOCKLIST
    # XRP both sides profitable — neither should be blocked
    assert ("XRPUSDT", "LONG") not in SYMBOL_SIDE_BLOCKLIST
    assert ("XRPUSDT", "SHORT") not in SYMBOL_SIDE_BLOCKLIST
    # BTC LONG profitable — should not be blocked
    assert ("BTCUSDT", "LONG") not in SYMBOL_SIDE_BLOCKLIST


def test_fake_bos_entry_guard_enabled() -> None:
    """The fake_bos entry guard fixes a code asymmetry where the bot used
    fake_bos for SL adjustments but not for entry decisions. Must be on
    by default after this rollout.
    """
    assert FAKE_BOS_ENTRY_GUARD_ENABLED is True
