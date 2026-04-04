# Testnet Validation Checklist

Validates that the 3 core testnet integration requirements are working correctly on Binance Futures paper trading (demo-fapi.binance.com).

## ⛔ CRITICAL RULE: NEVER OPEN A POSITION WITHOUT SL AND TP

**This is non-negotiable.** Every position opened on futures testnet MUST have:
1. **TP order** placed on exchange (LIMIT reduceOnly) — verified via `get_open_orders()`
2. **SL monitoring** active via WebSocket — verified by checking bot logs for "WS price monitor connected"

If TP placement fails, the position MUST be immediately closed. No unprotected positions are allowed.

### Enforcement
- `open_long()` / `open_short()` in `futures_executor.py` MUST verify TP order was placed (orderId is not None)
- If TP placement returns None or fails → immediately close the market order that was just placed
- Log a CRITICAL error if a position exists without a corresponding TP order
- On bot startup, `_sync_order_tracking()` must check: for every open position, a TP order exists. If not, place one or close the position.

---

## Requirement 1: Internal trigger → REAL futures testnet position

### What it should do
When `live_trading_htf.py` decides to open a position, the full chain must place a REAL MARKET order on demo-fapi:

```
live_trading_htf._mirror_testnet(trade)
  → testnet_executor.mirror_trade(trade, {})
    → _execute_futures_open(...)
      → FuturesTestnetExecutor.open_long/open_short(symbol, usdt_amount, sl, tp)
        → BinanceFuturesConnector.place_market_order(symbol, side, quantity)
          → POST https://demo-fapi.binance.com/fapi/v1/order?type=MARKET
```

### Code paths
- `live_trading_htf.py:860` — `_mirror_testnet(trade)`
- `src/api/testnet_executor.py:247` — `_execute_futures_open()` in `mirror_trade()`
- `src/api/futures_executor.py:101` — `place_market_order()` call inside `open_long()`
- `src/api/binance_futures.py:194` — `place_market_order()` HTTP POST

### How to verify
```bash
# Query positions on demo-fapi
python3 -c "
import os, sys; sys.path.insert(0, '.')
os.environ['BINANCE_FUTURES_API_KEY'] = '<key>'
os.environ['BINANCE_FUTURES_API_SECRET'] = '<secret>'
from src.api.binance_futures import BinanceFuturesConnector
c = BinanceFuturesConnector(api_key=os.environ['BINANCE_FUTURES_API_KEY'],
    api_secret=os.environ['BINANCE_FUTURES_API_SECRET'],
    base_url='https://demo-fapi.binance.com')
print(c.get_positions())
"
```
**Expected**: Open positions with positionAmt != 0, entryPrice matching bot logs.

### Common failure modes
- `TESTNET_MIRROR=true` not set in `.env` → `testnet_executor` is None, `_mirror_testnet` returns immediately
- `BINANCE_FUTURES_API_KEY` / `BINANCE_FUTURES_API_SECRET` not set → `get_futures_executor()` returns None, falls back to spot mode (no real longs/shorts)
- Insufficient balance (availableBalance < MIN_TRADE_VALUE_USDT=10 USDT) → `error` in result, `executed=False`
- Zero quantity (usdt_amount / mark_price rounds to 0 at qty_precision) → `error` in result

---

## Requirement 2: Every position has TP and SL configured

### What it should do
For every open position:
- **TP**: A LIMIT reduceOnly order at the TP price must exist on the exchange (TAKE_PROFIT_MARKET falls back to this on demo-fapi due to -4120)
- **SL**: Bot-side WebSocket monitoring handles SL — demo-fapi does NOT support STOP_MARKET (`place_stop_loss_order` returns sentinel `{"orderId": None, "status": "TESTNET_NOT_SUPPORTED"}`)

### Code paths
- TP placement: `src/api/futures_executor.py:120` — `place_take_profit_order()` in `open_long()`
- TP fallback to LIMIT: `src/api/binance_futures.py:296` — catches -4120, queries position size, places LIMIT reduceOnly
- SL sentinel: `src/api/binance_futures.py:243` — catches -4120 for STOP_MARKET, returns sentinel
- Bot SL monitoring: `live_trading_htf.py:1004` — WebSocket price tick calls `_check_sl_tp(price)`
- Startup sync: `src/api/futures_executor.py:_sync_order_tracking()` — loads TP/SL order IDs from exchange on init

### How to verify
```bash
# Check open orders (expect LIMIT reduceOnly TP for each open position)
python3 -c "
import os, sys; sys.path.insert(0, '.')
os.environ['BINANCE_FUTURES_API_KEY'] = '<key>'
os.environ['BINANCE_FUTURES_API_SECRET'] = '<secret>'
from src.api.binance_futures import BinanceFuturesConnector
c = BinanceFuturesConnector(api_key=os.environ['BINANCE_FUTURES_API_KEY'],
    api_secret=os.environ['BINANCE_FUTURES_API_SECRET'],
    base_url='https://demo-fapi.binance.com')
for o in c.get_open_orders():
    print(o.get('symbol'), o.get('type'), o.get('side'), o.get('price'), o.get('reduceOnly'))
"
# Expected: LIMIT SELL/BUY with reduceOnly=True for each open position
```

### Sync mechanism (restart safety)
After bot restart `_sl_orders`/`_tp_orders` dicts are empty. `_sync_order_tracking()` (called in `__init__`) re-populates them from exchange open orders:
- `LIMIT reduceOnly` → `_tp_orders[symbol] = orderId`
- `STOP_MARKET reduceOnly/closePosition` → `_sl_orders[symbol] = orderId`

This prevents `update_tp()` from stacking duplicate TP orders after restart.

To place a TP for a position that has none:
```python
executor.ensure_tp_order(symbol, side, tp_price)
```

### Common failure modes
- TP order missing after restart: fixed by `_sync_order_tracking()` — confirms TP order exists before placing a new one
- TAKE_PROFIT_MARKET rejected with -4120: expected, handled by LIMIT reduceOnly fallback in `place_take_profit_order()`
- STOP_MARKET rejected with -4120: expected, returns sentinel, SL is monitored by bot WebSocket
- TP order filled (position closed by exchange): `_sync_order_tracking()` on next init won't find the order
- Wrong SL direction: `open_long` uses SELL side for SL; `open_short` uses BUY side

---

## Requirement 3: ALL testnet tab data from exchange API — zero local calculations

### What it should do
Every `/api/testnet/*` Flask endpoint must source ALL financial data from exchange API calls:

| Endpoint | Exchange call | Data returned |
|----------|---------------|---------------|
| `GET /api/testnet/status` | `GET /fapi/v2/account` | wallet balance, unrealized PnL, available balance |
| `GET /api/testnet/positions` | `GET /fapi/v2/positionRisk` | entry price, mark price, unrealized PnL, liquidation price |
| `GET /api/testnet/pnl` | `/fapi/v2/account` + `/fapi/v1/userTrades` | realized PnL, unrealized PnL, win rate, equity curve |
| `GET /api/testnet/trades` | `GET /fapi/v1/userTrades` | exchange trade fills with realizedPnl |

### Code paths
- `src/ui/api_server.py:594` — `/api/testnet/status` → `FuturesTestnetExecutor.get_portfolio()`
- `src/ui/api_server.py:722` — `/api/testnet/positions` → `FuturesTestnetExecutor.get_positions()`
- `src/ui/api_server.py:737` — `/api/testnet/pnl` → `FuturesTestnetExecutor.get_pnl_summary()`
- `src/ui/api_server.py:705` — `/api/testnet/trades` → `FuturesTestnetExecutor.get_trade_history()`
- `src/api/futures_executor.py:314` — `get_portfolio()` calls `/fapi/v2/account` (no local calc)
- `src/api/futures_executor.py:336` — `get_positions()` passes through exchange fields unchanged
- `src/api/futures_executor.py:407` — `get_pnl_summary()` uses `/fapi/v2/account` unrealized + `/fapi/v1/userTrades` realized

### How to verify
```bash
# Start the API server, then:
curl http://127.0.0.1:5001/api/testnet/status | python3 -m json.tool
curl http://127.0.0.1:5001/api/testnet/positions | python3 -m json.tool
curl http://127.0.0.1:5001/api/testnet/pnl | python3 -m json.tool
curl http://127.0.0.1:5001/api/testnet/trades | python3 -m json.tool

# Cross-check directly against exchange:
python3 -c "
import os, sys; sys.path.insert(0, '.')
os.environ['BINANCE_FUTURES_API_KEY'] = '<key>'
os.environ['BINANCE_FUTURES_API_SECRET'] = '<secret>'
from src.api.futures_executor import FuturesTestnetExecutor
fx = FuturesTestnetExecutor()
print('portfolio:', fx.get_portfolio())
print('positions:', fx.get_positions())
print('pnl:', fx.get_pnl_summary())
"
```
**Expected**: Values match between API server response and direct exchange query (minor timing differences OK).

### What to check for local calculations
Search for these anti-patterns (none should exist in the futures code path):
```bash
grep -n "testnet_trades.json\|_load_positions_from_trades\|realized_pnl.*sum\|unrealized.*entry.*current" src/ui/api_server.py
```
These patterns exist only in `testnet_executor.py`'s spot fallback (used when no `BINANCE_FUTURES_API_KEY`), not in the futures code path.

### Common failure modes
- `BINANCE_FUTURES_API_KEY` not set on server → endpoints return `{"error": "Futures testnet not configured"}`
- Exchange rate limit / geo-block → configure Cloudflare proxy via `BINANCE_TESTNET_PROXY_URL`
- `get_pnl_summary()` shows 0 trades: `get_trade_history()` queries symbols in `_sl_orders`+`_tp_orders`; if both empty after restart, defaults to `["BTCUSDT", "ETHUSDT"]` — pass `symbol=` explicitly for full history

---

## Known demo-fapi Limitations

| Feature | Production futures | demo-fapi behavior |
|---------|-------------------|-------------------|
| STOP_MARKET | ✅ Supported | ❌ Error -4120, sentinel returned |
| TAKE_PROFIT_MARKET | ✅ Supported | ❌ Error -4120, falls back to LIMIT reduceOnly |
| OCO orders | ✅ Supported | ❌ Not supported |
| closePosition=true | ✅ Supported | ⚠️ Only works with STOP/TP_MARKET |
| WebSocket streams | ✅ Supported | ✅ Supported (via stream.binance.com) |

**SL workaround**: `live_trading_htf.py` runs a WebSocket monitor (`_start_ws_monitor`) subscribing to `{symbol}@aggTrade`. On each trade tick, `_check_sl_tp(price)` is called and triggers a bot-side close if SL is breached.

**TP workaround**: LIMIT reduceOnly at the TP price. For a LONG: LIMIT SELL above market rests until price rises to TP, then fills. Semantically identical to TAKE_PROFIT_MARKET in practice.

---

## Running Tests
```bash
python3 -m pytest tests/test_binance_futures.py tests/test_futures_executor.py -v
```
All 108 tests should pass. Key test classes:
- `TestSyncOrderTracking` — validates `_sync_order_tracking()` startup reload
- `TestEnsureTpOrder` — validates idempotent TP placement
- `TestPlaceTakeProfitOrder` — validates LIMIT reduceOnly fallback
- `TestPlaceStopLossOrder` — validates -4120 sentinel return
