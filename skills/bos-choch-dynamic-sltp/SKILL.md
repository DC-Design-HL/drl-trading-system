# BOS/CHOCH Dynamic SL/TP Adjustment

## Overview

This module adds **Break of Structure (BOS)** and **Change of Character (CHOCH)** detection to dynamically adjust Stop Loss and Take Profit levels on open positions across all trading bots.

## Architecture

### Detection Module: `src/signals/bos_choch.py`

The `MarketStructure` class provides:

| Method | Purpose |
|--------|---------|
| `detect_swing_points(df)` | Find swing highs/lows using N-bar lookback-on-both-sides |
| `determine_trend(swings)` | Classify trend as bullish/bearish/ranging from swing sequence |
| `detect_bos(df, swings, trend)` | Detect Break of Structure (trend continuation) |
| `detect_choch(df, swings, trend)` | Detect Change of Character (trend reversal) |
| `is_fake_bos(df, signal)` | Validate BOS via wick rejection, volume divergence, rapid reversal |
| `is_fake_choch(df, signal)` | Validate CHOCH (same 3 criteria) |
| `get_signals(df_15m, df_1h, df_4h)` | Multi-timeframe aggregation → single signal dict |

### Signal Output

```python
{
    'bos_bullish': bool,      # Bullish Break of Structure
    'bos_bearish': bool,      # Bearish Break of Structure
    'choch_bullish': bool,    # Bullish Change of Character (reversal)
    'choch_bearish': bool,    # Bearish Change of Character (reversal)
    'fake_bos': bool,         # Detected fake/failed BOS
    'fake_choch': bool,       # Detected fake/failed CHOCH
    'last_swing_high': float, # Most recent swing high level
    'last_swing_low': float,  # Most recent swing low level
    'trend': str,             # 'bullish' | 'bearish' | 'ranging'
    'confidence': float,      # 0.0-1.0 (multi-TF confirmation score)
}
```

### Multi-Timeframe Confidence Scoring

| Condition | Confidence Δ |
|-----------|-------------|
| 15m signal only | Base 0.50 |
| 1H confirms same BOS direction | +0.15 |
| 1H confirms same CHOCH direction | +0.10 |
| 4H confirms same BOS direction | +0.20 |
| 4H confirms same CHOCH direction | +0.15 |
| 1H contradicts 15m | -0.10 |
| 4H contradicts 15m CHOCH | -0.15 |

## Integration Points

### Affected Bots

| Bot | File | Exit Method | Exchange Orders |
|-----|------|-------------|-----------------|
| `drl-htf-agent` (BTC testnet) | `live_trading_htf.py` | `_check_sl_tp()` | ✅ Real orders via Algo API |
| `drl-htf-eth` (ETH testnet) | `live_trading_htf.py` | `_check_sl_tp()` | ✅ Real orders via Algo API |
| `drl-htf-partial` (Paper S2) | `live_trading_htf_partial.py` | `_check_partial_tp()` | ❌ Paper only |
| `drl-htf-hybrid` (Paper S3) | `live_trading_htf_hybrid.py` | `_check_hybrid_exit()` | ❌ Paper only |

### SL/TP Adjustment Rules

**CRITICAL: Adjustments ONLY apply when position is profitable (unrealized PnL > 0).**

#### LONG Positions (when profitable)

| Signal | Action | Rationale |
|--------|--------|-----------|
| BOS bullish | Trail SL → last swing low + 0.2% buffer; extend TP → swing high | Trend continuation = let winner run |
| CHOCH bearish | Tighten SL → lock 75% of profit from entry | Reversal warning = protect gains |
| Fake BOS bearish | IGNORE — hold position | Fake breakout = noise |
| Fake CHOCH bullish | IGNORE — don't over-tighten | False reversal signal |

#### SHORT Positions (when profitable)

| Signal | Action | Rationale |
|--------|--------|-----------|
| BOS bearish | Trail SL → last swing high - 0.2% buffer; extend TP → swing low | Trend continuation |
| CHOCH bullish | Tighten SL → lock 75% of profit | Reversal warning |
| Fake BOS bullish | IGNORE | Noise |
| Fake CHOCH bearish | IGNORE | Noise |

### Safety Rules

1. **SL never moves to a worse position** — only trails toward profit
2. **BOS/CHOCH adjustments are layered ON TOP of basic trailing** (break-even at +1%, lock 50% at +2%)
3. **Bot-side WS is the PRIMARY SL authority** — checks `sl_price` on every price tick and executes MARKET close instantly
4. **Exchange SL is a crashguard backstop** — only updated when change exceeds 0.1% AND 60-second cooldown has elapsed
5. **Paper bots only adjust internal state** — no exchange interaction

### SL/TP Architecture (Testnet Bots)

```
WebSocket price tick
  → _check_sl_tp(price)
    → Update internal sl_price (every tick, no threshold)
    → If price <= sl_price: MARKET close via close_position_market() [INSTANT]
    → If sl_price changed ≥0.1%: log + alert + sync exchange crashguard [THROTTLED]
```

**Two-layer SL system:**

| Layer | Authority | Speed | Purpose |
|-------|-----------|-------|---------|
| Bot-side WS | PRIMARY | Instant (~ms) | Real-time SL check on every tick → MARKET close |
| Exchange STOP_MARKET | CRASHGUARD | Exchange trigger | Safety net if bot crashes — uses last synced SL level |

**Exchange SL update throttling** (in `FuturesTestnetExecutor`):
- `MIN_SL_CHANGE_PCT = 0.001` (0.1%) — minimum price change to trigger exchange update
- `SL_UPDATE_COOLDOWN_SECS = 60` — maximum 1 exchange update per 60 seconds
- Prevents the old problem: 10+ API cancel+place calls in 8 seconds for $7 SL changes

**Bot-side SL logging/alert throttling** (in `live_trading_htf.py`):
- `MIN_SLTP_LOG_PCT = 0.001` (0.1%) — minimum change to log/alert
- Internal `sl_price` still updates on every tick (no threshold for the actual exit check)

### Clean-Line Validation Rule (Primary Filter)

**This is the #1 validation rule for ALL BOS/CHOCH signals on 5m timeframe.**

A structure break is only valid if the line from **Candle A** (the swing point) to **Candle B** (the breaking candle) is **clean and direct**:

| Element | Definition |
|---------|-----------|
| **Candle A** | The swing point candle — line starts at its **wick** (high for swing high, low for swing low) |
| **Candle B** | The breaking candle — line ends at its **body** (open/close area that closes beyond the level) |
| **Clean line** | The straight path from A's wick to B's body does NOT intersect any intermediate candle wicks |

**Invalid condition**: If ANY candle between A and B has a wick that touches or crosses the line (i.e., any intermediate high reaches the swing high level for bullish, or any intermediate low reaches the swing low level for bearish), the BOS/CHOCH is **rejected**.

**Why this works**:
- A clean line = genuine impulse/momentum behind the structural break
- Intermediate wicks crossing = price struggled and was rejected multiple times, meaning the "break" is a grind, not a true shift
- Filters out the majority of fake/weak structural breaks that trap traders

**Implementation**: `MarketStructure._is_clean_break()` in `src/signals/bos_choch.py`

### Fake Signal Detection (Secondary Filter — 2 criteria)

1. **Wick Rejection**: >60% of breakout candle range is wick beyond the broken level
2. **Rapid Reversal**: Price closes back below/above the broken level within 3 bars

## Data Flow

```
Each Iteration (15-min cycle):
  1. Fetch 15m OHLCV (existing)
  2. Fetch 1H + 4H OHLCV
  3. Run MarketStructure.get_signals() → cached result
  4. _check_sl_tp() reads cached signals → adjusts internal sl_price

On every WebSocket price tick (~1/sec):
  1. _check_sl_tp(price) → recalculates trailing SL (no threshold)
  2. If price <= sl_price → MARKET close via close_position_market() [instant]
  3. If SL change ≥ 0.1% → log + alert + sync exchange crashguard [throttled, max 1/60s]
```

## Files

| Path | Role |
|------|------|
| `src/signals/__init__.py` | Signal package init |
| `src/signals/bos_choch.py` | Core BOS/CHOCH detection + clean-line validation |
| `src/api/futures_executor.py` | `close_position_market()`, throttled `update_sl()`, `should_update_exchange_sl()` |
| `src/api/testnet_executor.py` | Bridge: `update_sl_tp()` → futures executor |
| `live_trading_htf.py` | `_check_sl_tp()` — bot-side WS SL, separated internal tracking from log/exchange sync |
| `live_trading_htf_partial.py` | BOS/CHOCH in `_check_partial_tp()` |
| `live_trading_htf_hybrid.py` | BOS/CHOCH in `_check_hybrid_exit()` |
| `research/bos_choch_deep_research.md` | Deep research document |
| `docs/TODO-ws-testnet-integration.md` | Pre-change state documentation |

## Logging

SL/TP adjustments are logged only when change ≥ 0.1% (prevents spam):

```
🔄 SL adjusted: $84000.00 → $84500.00 (reason=BOS_bullish(conf=0.65), profit=1.50%, peak=$85200.00)
🔄 TP adjusted: $87000.00 → $88500.00 (reason=BOS_bullish(conf=0.65))
🔄 SL crashguard updated: BTCUSDT → $84500.00 (orderId=123 algo=True)
📉 MARKET close: BTCUSDT LONG qty=0.025 (orderId=456)
```
