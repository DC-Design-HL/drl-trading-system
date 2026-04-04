# Binance Futures Testnet

## Purpose

Execute real LONG and SHORT futures positions on Binance USDM Futures paper trading
(`demo-fapi.binance.com`). Exchange handles all exits autonomously via SL/TP orders.
The bot only opens positions and updates SL/TP — it never sends explicit close orders.

## Environment Variables (in .env)

```
BINANCE_FUTURES_API_KEY=<key>
BINANCE_FUTURES_API_SECRET=<secret>
BINANCE_FUTURES_BASE_URL=https://demo-fapi.binance.com
FUTURES_LEVERAGE=5        # optional, default 5
```

## Architecture

```
live_trading_htf.py
    -> TestnetExecutor.mirror_trade()          # audit log + delegate
           -> FuturesTestnetExecutor.open_long/short()
                  -> BinanceFuturesConnector   # direct REST / HMAC-SHA256
                         -> demo-fapi.binance.com
```

### On OPEN_LONG / OPEN_SHORT
1. Set leverage via /fapi/v1/leverage
2. Size position: available_balance x 0.25 x confidence_scale
3. Place MARKET BUY (LONG) or MARKET SELL (SHORT)
4. Place STOP_MARKET closePosition=true (SL)
5. Place TAKE_PROFIT_MARKET closePosition=true (TP)
6. Track SL/TP order IDs for cancellation on trailing SL updates

### On Trailing SL Update (live_trading_htf.py -> update_sl_tp)
1. Cancel old STOP_MARKET order by tracked ID
2. Place new STOP_MARKET at updated SL price

### On CLOSE_LONG / CLOSE_SHORT (from bot)
- No-op: the exchange has already exited or will exit autonomously
- Audit log entry written with note: "Exit handled autonomously by exchange SL/TP orders"

## Key Files

- src/api/binance_futures.py   - Direct REST connector (HMAC-SHA256, no ccxt)
- src/api/futures_executor.py  - FuturesTestnetExecutor: open/update/query
- src/api/testnet_executor.py  - TestnetExecutor: delegates to futures, spot fallback
- src/ui/api_server.py         - Testnet API endpoints (all data from exchange)
- live_trading_htf.py          - Bot: mirrors opens, skips close mirrors
- tests/test_binance_futures.py   - Unit tests
- tests/test_futures_executor.py  - Unit tests

## API Endpoints Used

- POST /fapi/v1/order (MARKET)            - Open position
- POST /fapi/v1/order (STOP_MARKET)       - Place SL
- POST /fapi/v1/order (TAKE_PROFIT_MARKET)- Place TP
- DELETE /fapi/v1/order                   - Cancel SL/TP for update
- DELETE /fapi/v1/allOpenOrders           - Emergency cancel all
- GET /fapi/v2/positionRisk               - Open positions (entry, mark, PnL)
- GET /fapi/v2/account                    - Wallet balance, unrealized PnL
- GET /fapi/v1/userTrades                 - Trade fill history (realized PnL)
- GET /fapi/v1/openOrders                 - SL/TP order status
- POST /fapi/v1/leverage                  - Set leverage per symbol
- GET /fapi/v1/ticker/price               - Latest price
- GET /fapi/v1/premiumIndex               - Mark price (fallback to ticker)

## FuturesTestnetExecutor Methods

```python
executor = FuturesTestnetExecutor()            # from env vars
executor = FuturesTestnetExecutor(connector)   # inject mock for tests

# Opening positions
executor.open_long("BTCUSDT", usdt_amount, sl, tp, leverage=5)
executor.open_short("BTCUSDT", usdt_amount, sl, tp, leverage=5)

# Updating SL/TP (trailing SL)
executor.update_sl("BTCUSDT", "LONG", new_sl)
executor.update_tp("BTCUSDT", "LONG", new_tp)

# Syncing state from exchange
positions = executor.sync_positions()  # also clears stale tracking for closed symbols

# All data from exchange (no local calculation)
portfolio  = executor.get_portfolio()          # wallet balance, available, unrealized PnL
positions  = executor.get_positions()          # open positions (entry/mark/liq price, PnL)
trades     = executor.get_trade_history("BTCUSDT", limit=500)
pnl        = executor.get_pnl_summary(symbol="BTCUSDT")
```

## Financial Data Source of Truth

All financial data comes directly from Binance exchange APIs:
- Unrealized PnL: totalUnrealizedProfit from /fapi/v2/account
- Realized PnL: sum of realizedPnl from /fapi/v1/userTrades
- Positions: entry price, mark price, liquidation price from /fapi/v2/positionRisk
- Wallet balance: totalWalletBalance from /fapi/v2/account

Never computed locally.

## Anti-Patterns

- Do NOT use ccxt for futures (use direct REST BinanceFuturesConnector)
- Do NOT calculate PnL locally (always from exchange)
- Do NOT simulate shorts (real SELL orders on futures)
- Do NOT send CLOSE orders from the bot (exchange exits autonomously)
- Do NOT reconstruct positions from local trade logs (query /fapi/v2/positionRisk)
- Do NOT use spot testnet for shorts (no shorting supported)

## Demo Account

- Base URL: https://demo-fapi.binance.com
- Balance: $5000 USDT, 0.01 BTC, $5000 USDC (as of 2026-03-22)
- Keys: BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET in .env
