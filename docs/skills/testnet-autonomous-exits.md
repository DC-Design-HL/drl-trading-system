# Testnet Autonomous Exits

## Purpose

Ensure that position exits (SL/TP/trailing) are handled by Binance testnet exchange orders — not by our bot polling and closing. Our bot only decides when to OPEN a position and mirrors it to testnet with SL/TP orders. The exchange closes the position autonomously.

## Current State (as of 2026-03-22)

- **BinanceConnector** (`src/api/binance.py`): Spot-only testnet via ccxt. No futures support.
- **TestnetExecutor** (`src/api/testnet_executor.py`):
  - LONG: Places real market buy + OCO (TP + SL) orders on spot testnet. OCO handles exit. ✅
  - SHORT: **Conceptual/simulated** — spot testnet can't short. Sells any held base currency, tracks position in memory. Exit is a no-op. ❌
  - `update_sl_tp()`: Updates OCO for LONGs (trailing SL). No-op for simulated shorts. 
- **Live bot** (`live_trading_htf.py`):
  - On SL/TP trigger: calls `_close_position()` → `_mirror_testnet()` to tell testnet executor to close.
  - Problem: bot is doing the exit decision AND execution — testnet should handle exit autonomously via exchange orders.

## Architecture: How It Should Work

### For LONG positions (spot testnet supports this):
1. Bot decides OPEN_LONG → `mirror_trade()` places market BUY on testnet
2. Bot sets SL/TP → `mirror_trade()` places OCO SELL order (TP limit + SL stop-limit)
3. Bot adjusts trailing SL → `update_sl_tp()` cancels old OCO, places new one
4. **Testnet OCO fills autonomously** when price hits SL or TP
5. Bot **polls testnet** to detect the fill and sync internal state — does NOT send a CLOSE

### For SHORT positions (requires futures testnet):
**Option A — Switch to Futures Testnet** (recommended):
- Binance Futures Testnet at `https://testnet.binancefuture.com` supports real shorting
- Use `ccxt.binance` with `defaultType: 'future'` and futures testnet URLs
- Place real SHORT market orders + SL/TP orders
- Exchange handles exit autonomously

**Option B — Simulate on Spot** (current, limited):
- Spot testnet can't short, so exits must be bot-driven
- This defeats the purpose of autonomous exits

## Implementation Checklist

### Phase 1: Fix LONG autonomous exits (spot testnet)
- [ ] OCO orders already placed on open — verify they work end-to-end
- [ ] Add a **position sync loop** that polls testnet order status:
  - Check if OCO filled (SL or TP hit)
  - If filled, update bot internal state (`position=0`, record PnL)
  - Log which side filled (SL vs TP) and at what price
- [ ] Remove bot-side close mirroring for LONGs — let OCO handle it
- [ ] Handle edge case: bot trailing SL updates OCO, but price gaps through

### Phase 2: Add futures testnet for SHORT autonomous exits
- [ ] Add `BinanceFuturesConnector` or extend `BinanceConnector` with futures mode
- [ ] Futures testnet config: `https://testnet.binancefuture.com/fapi/v1`
- [ ] On OPEN_SHORT: place real futures SHORT market order + SL/TP orders
- [ ] On trailing SL adjustment: cancel old SL, place new one (futures supports separate SL/TP orders, not just OCO)
- [ ] Position sync loop polls futures positions for autonomous fill detection
- [ ] Remove conceptual/simulated short logic

### Phase 3: Unified sync loop
- [ ] Single background thread or async task that:
  - Polls all open testnet orders every 5-10 seconds
  - Detects fills, partial fills, cancellations
  - Syncs bot state accordingly
  - Logs everything for audit trail
- [ ] Bot's `_close_position()` no longer mirrors to testnet — it only updates internal state
- [ ] Testnet is the source of truth for position exits

## Key Files
- `src/api/binance.py` — BinanceConnector (spot)
- `src/api/testnet_executor.py` — TestnetExecutor (mirror logic)
- `live_trading_htf.py` — Live bot (lines 838-847: `_mirror_testnet`)

## Constraints
- Binance spot testnet only supports: BTC, ETH, BNB, LTC, TRX, XRP, SOL, ADA, DOGE paired with USDT
- Futures testnet requires separate API keys from https://testnet.binancefuture.com
- ccxt dropped `set_sandbox_mode` for futures — must set URLs manually
- OCO orders on spot testnet can be flaky — always verify fill status

## Anti-Patterns (Don't Do This)
- ❌ Bot calculates exit price and sends CLOSE to testnet — defeats autonomous exit purpose
- ❌ Simulated/conceptual shorts with local PnL tracking — must use real orders
- ❌ Trusting bot internal state over exchange state — testnet is source of truth for exits
