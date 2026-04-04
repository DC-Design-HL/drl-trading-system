# Macro Trend Guard (BTC SMA + RSI Combo)

## What It Does
Blocks trades that go against the BTC macro trend. Uses a combination of BTC's 6h vs 24h SMA crossover AND RSI confirmation to determine macro direction.

## Status
**BACKTESTED but NOT deployed.** Planned as an optional Layer 2 guard to add on top of the orderbook guard.

## How It Works
Every hour, compute:
1. **BTC 6h SMA** vs **BTC 24h SMA** — if 6h > 24h by >0.2%, trend is bullish
2. **BTC 14-period RSI** on 1h candles — confirms trend direction

### Rules
- **Block SHORT** when BTC SMA6 > SMA24 by >0.2% AND RSI > 55 (bullish macro)
- **Block LONG** when BTC SMA6 < SMA24 by >0.2% AND RSI < 45 (bearish macro)
- Otherwise: neutral → allow all trades

## Data Source
Binance public API: `GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=30`

## Backtest Results (Mar 24-31)

### Standalone
- 79 trades → 46 taken, 33 blocked
- Win rate: 45.6% → **56.5%** (+10.9 points)
- Blocked 33 trades: 10 winners, 23 losers (net +13)
- Catches **3/3 whipsaw shorts** from March 30

### Combined with Orderbook Guard (Full Stack)
- Part of the "full stack" that achieved 62.5% WR and $221 PnL
- Adds macro-level filtering on top of microstructure (orderbook) filtering

## Why Not Deployed Yet
- The orderbook guard alone catches the whipsaw shorts
- Adding SMA+RSI blocks 42% of trades — may be too aggressive
- Need more data (>1 week) to validate it's not overfitting
- Chen wants to review before any new logic goes live

## Configuration (planned)
```python
MACRO_GUARD_ENABLED = False  # Toggle
MACRO_SMA_SHORT = 6          # Short SMA period (hours)
MACRO_SMA_LONG = 24          # Long SMA period (hours)
MACRO_SMA_THRESHOLD = 0.2    # Minimum SMA divergence (%)
MACRO_RSI_BULL = 55          # RSI above this = bullish confirmation
MACRO_RSI_BEAR = 45          # RSI below this = bearish confirmation
```

## Implementation Notes
- Uses BTC as the macro filter for ALL pairs (BTC, ETH, SOL, XRP)
- This is intentional — when BTC trends, alts follow
- Could be extended to per-pair macro filtering later
