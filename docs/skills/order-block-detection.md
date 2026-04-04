# Order Block Detection (Smart Money Concepts)

## What It Does
Detects institutional Order Block (OB) zones using market structure analysis (BOS/CHoCH). Order Blocks are price zones where large players previously placed significant orders, creating support/resistance levels.

## Status
**BUILT but NOT connected to live trading.** Standalone module only. Requires Chen's approval before integration.

## How It Works

### Market Structure Detection
1. Find **swing highs/lows** using pivot points (configurable length, default 5 bars)
2. Detect **BOS** (Break of Structure) = price breaks previous swing in trend direction (continuation)
3. Detect **CHoCH** (Change of Character) = price breaks previous swing AGAINST trend (reversal)

### Order Block Creation
- When CHoCH confirmed → mark the **last candle before the move** as an Order Block
- **Bullish OB** = last bearish candle before bullish CHoCH → buy zone (support)
- **Bearish OB** = last bullish candle before bearish CHoCH → sell zone (resistance)
- OBs on BOS are weaker than CHoCH OBs
- Zone = [candle_low, candle_body_top] with ATR-based sizing

### Mitigation
- When price returns and closes through the OB zone → OB is "used up" (mitigated) and removed
- Methods: close-based (default), wick-based, or average-based

### Confidence Scoring
Based on: volume relative to average, test count, CHoCH vs BOS origin, freshness (age)

## Key Files
- `src/signals/order_blocks.py` — Core detector (28KB)
- `src/signals/order_blocks_config.py` — Configuration
- `tests/test_order_blocks.py` — Test suite (all passing)

## API
```python
from src.signals.order_blocks import OrderBlockDetector

det = OrderBlockDetector()
det.update_order_blocks("BTCUSDT")

# Get active OBs
obs = det.get_active_obs("BTCUSDT")

# Check if price is in an OB zone
should_enter, ob, confidence = det.check_ob_entry("BTCUSDT", "LONG", current_price)
```

## Planned Integration (NOT YET LIVE)

### Tiered Position Sizing
- **T1** (price IN OB zone): 1.5x dollar risk — highest confidence entry
- **T2** (price NEAR OB zone, within 1x zone size): 1.0x dollar risk — normal
- **T3** (no OB nearby): 0.5x dollar risk — lowest confidence, reduce exposure

### Backtest Results (Mar 24-31, 70 trades)
- T1 trades: 71.4% WR, $120 raw PnL
- T2 trades: 75.0% WR, $41 raw PnL  
- T3 trades: 40.4% WR, -$87 raw PnL
- Full stack (OB tiers + orderbook guard + SMA combo): **62.5% WR, $221 PnL** vs $74 baseline

### Rollout Plan
1. ✅ Week 1 (Mar 31): Orderbook guard only (LIVE)
2. ⏰ Week 2 (Apr 7): OB tiered sizing in SHADOW MODE (log only)
3. ⏰ Week 3 (Apr 14): Enable actual tiered sizing if shadow looks good

## Multi-Timeframe Support
Supports 15m, 1h (default), and 4h timeframes. Can be extended to multi-TF confluence.

## Data Source
Binance public API klines — no auth needed. Fetches 500 hourly candles for sufficient history.
