# Testnet Real Portfolio — No Local Calculations

## Purpose

All portfolio value, PnL, balance, and position data displayed in the UI must come directly from Binance testnet API calls. Zero local calculation of financial data. The system must never compute PnL, balances, or portfolio values from trade logs or internal state.

## Current State (as of 2026-03-22)

### What's already from testnet ✅
- **Portfolio value** (`/api/testnet/status`): Fetches real balances from `BinanceConnector.get_all_balances()`, gets live ticker prices, calculates value. Source: Binance spot testnet API.
- **Spot wallet holdings**: Real balances from testnet.
- **USDT balance**: Real from testnet.

### What's calculated locally ❌ (must fix)
- **Realized PnL** (`/api/testnet/pnl`): Summed from `testnet_trades.json` file — locally recorded trade PnL values, not from exchange.
- **Unrealized PnL**: Calculated by `get_current_positions()` using local entry price vs live ticker price. Entry price comes from our trade log, not exchange.
- **Win rate / trade stats**: Computed from local trade log file.
- **Equity curve**: Built from cumulative local PnL records.
- **Bot-Mirrored Positions** (`/api/testnet/positions`): Reconstructed from local `testnet_trades.json` — not queried from exchange.

## Architecture: How It Should Work

### Balance & Portfolio
- Query `GET /fapi/v2/account` (futures) or spot balance endpoint
- Return `totalWalletBalance`, `availableBalance`, `totalUnrealizedProfit` directly from exchange response
- Portfolio value = exchange-reported total, not sum of local calculations

### Open Positions
- **Spot**: Query account balances + live tickers (already done for status endpoint)
- **Futures**: Query `GET /fapi/v2/positionRisk` — returns `entryPrice`, `markPrice`, `unRealizedProfit`, `positionAmt`, `leverage`, `liquidationPrice` directly from Binance
- Display exchange-reported values as-is

### PnL
- **Unrealized PnL**: From exchange position endpoint (`unRealizedProfit` field)
- **Realized PnL**: From exchange trade history endpoint (`GET /fapi/v1/userTrades`) — sum actual fills
- **Alternatively**: Track `totalWalletBalance` changes over time (balance at start vs now = total realized PnL)
- Equity curve: periodic snapshots of exchange-reported balance, not cumulative trade PnL

### Trade History
- Query exchange `GET /fapi/v1/userTrades` or spot `GET /api/v3/myTrades`
- Display actual fill prices, quantities, commissions from exchange
- Local trade log is for bot decision audit only, NOT for financial display

## Implementation Checklist

### Phase 1: Positions from exchange
- [ ] Replace `get_current_positions()` — query exchange instead of reading `testnet_trades.json`
- [ ] For spot: use `get_all_balances()` + tickers (already working in status endpoint)
- [ ] For futures: add `get_futures_positions()` → `GET /fapi/v2/positionRisk`
- [ ] UI shows exchange-reported entry price, mark price, unrealized PnL, liquidation price

### Phase 2: PnL from exchange
- [ ] Replace `get_pnl_summary()` — all values from exchange APIs
- [ ] Realized PnL: query exchange trade history, sum actual realized profits
- [ ] Unrealized PnL: from position endpoint directly
- [ ] Win rate: computed from actual exchange trade fills, not local log
- [ ] Equity curve: store periodic balance snapshots (from exchange) in MongoDB or local DB

### Phase 3: Remove local financial calculations
- [ ] `testnet_trades.json` becomes a **bot decision log** only — record what the bot decided, for debugging
- [ ] Remove `_load_positions_from_trades()` logic that reconstructs positions from local file
- [ ] Remove local PnL summing from trade log
- [ ] API endpoints serve exchange data, never locally computed financial values
- [ ] Add clear separation: bot audit log ≠ financial source of truth

## Key Files
- `src/api/testnet_executor.py` — `get_current_positions()`, `get_pnl_summary()` (lines 244-342)
- `src/ui/api_server.py` — `/api/testnet/status` (line 594), `/api/testnet/positions` (line 744), `/api/testnet/pnl` (line 759)
- `src/ui/app.py` — UI display (line 2717: Bot-Mirrored Open Positions)
- `src/api/binance.py` — BinanceConnector (needs futures methods)

## API Endpoints Reference (Binance Testnet)

### Spot Testnet (`https://testnet.binance.vision`)
- `GET /api/v3/account` — balances
- `GET /api/v3/myTrades` — trade history
- `GET /api/v3/openOrders` — open orders

### Futures Testnet (`https://testnet.binancefuture.com`)
- `GET /fapi/v2/account` — full account info (wallet balance, unrealized PnL)
- `GET /fapi/v2/positionRisk` — all positions with entry/mark/liquidation prices
- `GET /fapi/v1/userTrades` — trade fill history
- `GET /fapi/v1/openOrders` — open orders (SL/TP)
- `GET /fapi/v1/allOrders` — all order history

## Anti-Patterns (Don't Do This)
- ❌ Sum PnL from local trade log files — use exchange trade history
- ❌ Calculate unrealized PnL as `(current_price - entry_price) * amount` locally — use exchange-reported value
- ❌ Reconstruct positions from local open/close trade records — query exchange positions
- ❌ Display locally computed portfolio value — use exchange wallet balance
- ❌ Trust local `testnet_trades.json` for any financial display — it's an audit log only
- ❌ Hardcode initial balance and subtract/add trades — exchange balance is the truth

## Validation
After implementation, verify:
1. Stop the bot — UI still shows correct positions/PnL from exchange
2. Manually place a trade on testnet web UI — it appears in our dashboard
3. Exchange-side OCO fills — our UI reflects the closed position and updated PnL without bot involvement
4. Portfolio value matches what Binance testnet website shows
