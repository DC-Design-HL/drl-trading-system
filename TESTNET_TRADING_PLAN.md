# Testnet Trading Implementation Plan

## Goal
Mirror the DRL bot's LONG/SHORT trading decisions to Binance Testnet with real order execution and full dashboard visibility.

## Architecture

### Constraints
- Binance Testnet (testnet.binance.vision) is **SPOT only** — no futures/perpetuals
- LONG positions: execute real BUY orders on testnet
- CLOSE LONG: execute real SELL orders on testnet
- SHORT/CLOSE SHORT: recorded as conceptual trades (spot can't truly short); if base currency held, it is sold

### Components

#### 1. `src/api/testnet_executor.py` (NEW)
- `TestnetExecutor` class — singleton used by orchestrator and api_server
- `mirror_trade(bot_trade, bot_result)` — translates bot decision → real testnet order
- `get_current_positions()` — live positions with current prices + unrealized PNL
- `get_trades(limit)` — read from `logs/testnet_trades.json`
- `get_pnl_summary()` — realized + unrealized PNL, win rate
- Stores each trade in `logs/testnet_trades.json` (line-delimited JSON)
- Uses `BinanceConnector` for all API calls
- Reads `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` from env

#### 2. `src/ui/api_server.py` (MODIFIED — 4 new endpoints)
- `GET /api/testnet/trades` — trade history from testnet_trades.json
- `GET /api/testnet/positions` — open positions with live prices + unrealized PNL
- `GET /api/testnet/pnl` — realized + unrealized PNL summary + equity curve data
- `POST /api/testnet/execute` — manually trigger a testnet trade (for testing)

#### 3. `live_trading_multi.py` (MODIFIED — auto-execution hook)
- `MultiAssetOrchestrator.__init__`: instantiate `TestnetExecutor` if `TESTNET_MIRROR=true`
- `run_single_cycle()`: after each bot decision, call `self.testnet_executor.mirror_trade()`
- Log both dry-run result and testnet execution result
- Guard with try/except so testnet failures never block the main trading loop

#### 4. `src/ui/app.py` (MODIFIED — enhanced Testnet tab)
New sections added to the Testnet tab (all data from API endpoints):
- **Open Positions table**: symbol, side, entry price, current price, unrealized PNL, SL, TP
- **Trade History table**: timestamp, symbol, action, price, amount, PNL, order_id
- **PNL Summary**: realized, unrealized, total, win rate, total trades
- **Equity Curve chart**: cumulative PNL over time (Plotly line chart)
- **Live Order Book**: open/pending orders from `/api/testnet/orders`

#### 5. `tests/test_testnet_trading.py` (NEW)
- Test testnet connectivity
- Test place a small market order on testnet
- Test `/api/testnet/trades` returns list
- Test `/api/testnet/positions` returns list
- Test PNL calculation

## Data Flow

```
Bot run_iteration()
    └─> execute_trade() → trade dict
                └─> [if TESTNET_MIRROR=true]
                        └─> TestnetExecutor.mirror_trade()
                                ├─> BinanceConnector.place_market_order()  (50%)
                                ├─> BinanceConnector.place_limit_order()   (50%)
                                └─> _save_trade() → logs/testnet_trades.json

Dashboard (app.py Testnet Tab)
    ├─> GET /api/testnet/status    (existing — balance/portfolio)
    ├─> GET /api/testnet/positions (new — open bot-mirrored positions)
    ├─> GET /api/testnet/trades    (new — trade history)
    ├─> GET /api/testnet/pnl       (new — PNL + equity curve)
    └─> GET /api/testnet/orders    (existing — open orders)
```

## Trade Record Schema
```json
{
  "symbol": "BTCUSDT",
  "ccxt_symbol": "BTC/USDT",
  "action": "OPEN_LONG_SPLIT",
  "side": "BUY",
  "price": 43250.50,
  "filled_price": 43251.00,
  "amount": 0.00578,
  "sl": 41087.98,
  "tp": 46000.25,
  "confidence": 0.72,
  "timestamp": "2026-03-19T14:32:00.000Z",
  "order_id": "12345678",
  "limit_order_id": "12345679",
  "limit_price": 43034.87,
  "limit_amount": 0.00579,
  "executed": true,
  "error": null,
  "pnl": null,
  "dry_run": false
}
```

## Environment Variables
- `TESTNET_MIRROR=true` — enables auto-execution hook (default: false)
- `BINANCE_TESTNET_API_KEY` — testnet API key (already set)
- `BINANCE_TESTNET_API_SECRET` — testnet API secret (already set)
- `BINANCE_TESTNET_PROXY_URL` — optional Cloudflare proxy (already set)

## Risk Controls
- TestnetExecutor failures are caught and logged — never block main loop
- Minimum trade value: $10 USDT
- Max position: 25% of testnet USDT balance × confidence scale
- Split entry: 50% market + 50% limit (mirrors bot logic)
- No automatic SL/TP orders placed (bot logic manages exits and mirrors them)
