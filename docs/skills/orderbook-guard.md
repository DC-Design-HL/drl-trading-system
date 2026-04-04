# Orderbook Guard (Golden Guard)

## What It Does
Blocks trades when the Binance Futures orderbook bias contradicts the trade direction. If the bot wants to SHORT but buyers dominate the orderbook (bullish), the trade is blocked. Vice versa for LONGs.

## How It Works
- Fetches orderbook depth from **Binance Futures API** (`fapi.binance.com/fapi/v1/depth`)
- Takes **3 snapshots** averaged together (0.3s apart) to reduce noise/spoofing
- Computes `imbalance_10` = (bid_vol - ask_vol) / total at top 10 levels
- **Bias threshold: 0.25** on imbalance_10:
  - `> +0.25` = bullish → blocks SHORTs
  - `< -0.25` = bearish → blocks LONGs
  - Between = neutral → allows all
- Falls back to SPOT orderbook if futures API fails
- **Fail-open**: if no data available, trade proceeds

## Key Files
- `live_trading_htf.py` → `_check_orderbook_guard()` method + `ORDERBOOK_GUARD_ENABLED` constant
- `src/features/orderbook_imbalance.py` → `get_orderbook_imbalance()` function

## Configuration
- `ORDERBOOK_GUARD_ENABLED = True/False` in `live_trading_htf.py`
- `_OB_BIAS_THRESHOLD = 0.25` in `orderbook_imbalance.py`
- `_OB_SAMPLES = 3` — number of snapshots to average
- `_OB_SAMPLE_DELAY = 0.3` — seconds between samples
- `_OB_CACHE_TTL = 10` — cache duration in seconds

## Applies To
ALL confidence tiers including Tier 1 autonomous. Runs BEFORE the signal gate.

## Backtest Results (Mar 24-31, 67 trades)
- Win rate: 46.3% → **51.8%**
- PnL: $53 → **$170** (+217%)
- Blocked 11 trades: 9 losers, 2 winners
- Caught **3/3 whipsaw shorts** from March 30

## Deployment
- Deployed: 2026-03-31
- Git: `dev` branch
- Services: All 4 HTF bots (BTC, ETH, SOL, XRP)
- Each bot gets its own per-symbol orderbook reading

## Logs
- Block: `🛡️ ORDERBOOK GUARD BLOCK: {symbol} blocked — orderbook bias is {BIAS} (contradicts {DIRECTION})`
- Pass: Debug level only

## Kill Switch
Set `ORDERBOOK_GUARD_ENABLED = False` in `live_trading_htf.py` and restart bots.
