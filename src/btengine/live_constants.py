"""Single source of truth for production constants.

Imported by both `live_trading_htf.py` (so live behavior matches what is
defined here) and the btengine framework (so backtest behavior matches
live). When you change a number here, both worlds update together.

When the live bot starts up it should re-import these so a hot edit
propagates. Currently the live bot defines them inline — Phase 1 leaves
that arrangement intact and merely *reads* the same values; Phase 2
inverts the dependency so the live bot imports from here.

For the audit context that motivated this: see
docs/backtest_redesign_proposal.md. Discrepancies discovered:
  * 4 backtest scripts double-counted exit fees
  * 9 different ADX implementations across scripts
  * 17 scripts missing the Apr 27 SYMBOL_SIDE_BLOCKLIST guard
"""

from __future__ import annotations

# ── Fees & slippage (Binance Futures testnet) ──────────────────────────
TRADING_FEE_TAKER = 0.0004    # 0.04% per side (taker)
TRADING_FEE_MAKER = 0.0002    # 0.02% per side (maker)
SLIPPAGE_PCT = 0.0005         # 0.05% per side, conservative

# ── Position sizing ────────────────────────────────────────────────────
INITIAL_BALANCE = 5000.0
RISK_POOL_PCT = 0.20          # Option A baseline (post-Apr 27 rollback from 30%)
FIXED_MAX_NOTIONAL = 6000     # Absolute cap per trade
LEVERAGE = 40                 # Reference; actual leverage may be reduced for liq safety
LIQ_BUFFER_PCT = 0.0050       # Extra cushion above SL distance

# ── Stop-loss / take-profit (ATR-driven) ───────────────────────────────
ATR_SL_FLOOR_MULT = 1.5       # SL ≥ 1.5 × ATR
ATR_TP_FLOOR_MULT = 3.0       # TP ≥ 3.0 × ATR
BASE_SL_PCT = 0.015           # 1.5% baseline before ATR floor
BASE_TP_PCT = 0.030           # 3.0% baseline before ATR floor

# ── Partial TP ladder ──────────────────────────────────────────────────
PARTIAL_TP1_R = 1.0           # First partial at 1R (1× SL distance)
PARTIAL_TP1_FRACTION = 0.40   # Close 40% at 1R
PARTIAL_TP2_R = 2.0           # Second partial at 2R
PARTIAL_TP2_FRACTION = 0.35   # Close 35% at 2R
# Remaining 25% trails

# ── Trailing stop ──────────────────────────────────────────────────────
TRAILING_ACTIVATE_PCT = 0.005   # Activate at +0.5% profit
TRAILING_DISTANCE_PRE_TP1 = 0.003   # 0.3% trail before TP1 hit
TRAILING_DISTANCE_POST_TP1 = 0.005  # 0.5% trail after TP1 hit

# ── Stagnant exit ──────────────────────────────────────────────────────
STAGNANT_HOURS = 6
STAGNANT_PCT_MIN = -0.010   # -1.0%
STAGNANT_PCT_MAX = 0.005    # +0.5%

# ── Entry / regime guards ──────────────────────────────────────────────
MIN_CONFIDENCE = 0.45
ADX_GUARD_MIN = 20
ADX_GUARD_MAX = 60
RSI_GUARD_OB_THRESHOLD = 70   # OB threshold for SHORT entries
RSI_GUARD_OS_THRESHOLD = 30   # OS threshold for LONG entries
RSI_GUARD_EXTREME_OB = 80
RSI_GUARD_EXTREME_OS = 20

# ── USDT.D guard ───────────────────────────────────────────────────────
USDT_D_GUARD_ENABLED = True
USDT_D_THRESHOLD_PCT = 0.7    # 4-symbol basket must drop > 0.7% to flag rising
USDT_D_LOOKBACK_HOURS = 2
USDT_D_CACHE_TTL_SECONDS = 600

# ── Funding-rate LONG guard ────────────────────────────────────────────
FUNDING_LONG_GUARD_ENABLED = True
FUNDING_LONG_GUARD_MAX = 0.05    # Block LONG if 8h funding > +0.05%

# ── Whale-NEUTRAL guard ────────────────────────────────────────────────
WHALE_NEUTRAL_GUARD_ENABLED = True

# ── Extreme-position news guard (deployed Apr 30) ──────────────────────
EXT_POS_NEWS_GUARD_ENABLED = True
EXT_POS_NEWS_SENTIMENT_THRESHOLD = 0.5
EXT_POS_NEWS_LOOKBACK_HOURS = 4

# ── Symbol × side blocklist (deployed Apr 27) ──────────────────────────
SYMBOL_SIDE_BLOCKLIST = frozenset({
    ("BTCUSDT", "SHORT"),
    ("ETHUSDT", "SHORT"),
    ("ETHUSDT", "LONG"),
    ("SOLUSDT", "LONG"),
})

# ── Per-symbol confidence overrides ────────────────────────────────────
SYMBOL_MIN_CONFIDENCE = {
    "ETHUSDT": 0.80,
}

# ── REVERSE_CLOSE_LONG canary (deployed Apr 23, expanded to all 4 May 1) ─
REVERSAL_BLOCK_LONG_CANARY_SYMBOLS = frozenset({
    "XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT",
})
REVERSAL_BLOCK_LONG_REGIME_GATE_MIN_SLOPE_PCT = -0.5
BTC_REGIME_CACHE_TTL_SECONDS = 900

# ── Whipsaw cooldown ───────────────────────────────────────────────────
WHIPSAW_COOLDOWN_HOURS = 2

# ── Bar cadence ────────────────────────────────────────────────────────
PRIMARY_INTERVAL = "15m"
HTF_INTERVALS = ("1h", "4h")
MIN_BARS_BETWEEN_TRADES = 4   # 1h cooldown on 15m bars
