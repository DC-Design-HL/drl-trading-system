# Binance USDM Futures API — Comprehensive Reference

> Verified against **demo-fapi.binance.com** on 2026-03-22.
> Account: `BINANCE_FUTURES_API_KEY` / `BINANCE_FUTURES_API_SECRET` in `.env`

---

## Table of Contents

1. [Base URLs & Environments](#1-base-urls--environments)
2. [Authentication](#2-authentication)
3. [Rate Limits](#3-rate-limits)
4. [Order Types — Complete Reference](#4-order-types--complete-reference)
5. [demo-fapi Testnet Limitations](#5-demo-fapi-testnet-limitations)
6. [REST API Endpoints](#6-rest-api-endpoints)
7. [SL/TP Orders — Definitive Guide](#7-sltp-orders--definitive-guide)
8. [Position Management](#8-position-management)
9. [WebSocket Streams](#9-websocket-streams)
10. [Error Codes](#10-error-codes)
11. [Our BinanceFuturesConnector](#11-our-binancefuturesconnector)
12. [Python Code Examples](#12-python-code-examples)

---

## 1. Base URLs & Environments

| Environment | REST Base URL | WebSocket Base URL |
|---|---|---|
| **Production** | `https://fapi.binance.com` | `wss://fstream.binance.com` |
| **demo-fapi (paper trading)** | `https://demo-fapi.binance.com` | `wss://dstream.binance.com` |
| **Real testnet** | `https://testnet.binancefuture.com` | `wss://stream.binancefuture.com` |

**demo-fapi** is a paper trading environment with real market data but simulated execution. It has significant order type restrictions (see §5).

---

## 2. Authentication

### Signing Signed Requests

All `TRADE` and `USER_DATA` endpoints require:
1. `X-MBX-APIKEY` header with your API key
2. `timestamp` query param (milliseconds since epoch)
3. `recvWindow` query param (optional, default 5000ms; we use 60000ms)
4. `signature` query param — HMAC-SHA256 of the full query string

```python
import hashlib, hmac, time
from urllib.parse import urlencode

def sign(params: dict, secret: str) -> str:
    query = urlencode(params)
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={sig}"

# Always add timestamp BEFORE signing
params["timestamp"] = int(time.time() * 1000)
params["recvWindow"] = 60000
url = f"{BASE_URL}{path}?{sign(params, secret)}"
```

### Timestamp Validity Window

```
Request valid if: timestamp < serverTime + 1000ms
             AND: serverTime - timestamp <= recvWindow
```

Use `GET /fapi/v1/time` to get server time if clocks drift.

### Security Types

| Type | Requirement |
|---|---|
| NONE | No auth needed |
| MARKET_DATA | API key only (`X-MBX-APIKEY` header) |
| USER_STREAM | API key only |
| TRADE | API key + signed |
| USER_DATA | API key + signed |

---

## 3. Rate Limits

### IP Rate Limits

| Header | Limit | Reset |
|---|---|---|
| `x-mbx-used-weight-1m` | 2400 requests/min | Rolling 1min |

HTTP 429 is returned when exceeded; HTTP 418 after repeated violations (ban: 2min–3days).

### Order Rate Limits

| Header | Limit |
|---|---|
| `X-MBX-ORDER-COUNT-10S` | 300 orders per 10 seconds |
| `X-MBX-ORDER-COUNT-1M` | 1200 orders per minute |

### Endpoint Weights (common)

| Endpoint | Weight |
|---|---|
| GET /fapi/v1/ping | 1 |
| GET /fapi/v1/ticker/price | 1 (single symbol) / 2 (all) |
| GET /fapi/v1/premiumIndex | 1 (single) |
| GET /fapi/v2/account | 5 |
| GET /fapi/v2/positionRisk | 5 |
| GET /fapi/v1/openOrders | 1 (with symbol) / 40 (all) |
| POST /fapi/v1/order | 1 (order) + 1 (10s) + 1 (1m) |
| POST /fapi/v1/batchOrders | 5 (IP) + 5 (10s) + 1 (1m) |
| DELETE /fapi/v1/order | 1 |

---

## 4. Order Types — Complete Reference

### Mandatory Parameters Per Order Type

| Type | Required Params | Optional Params |
|---|---|---|
| `MARKET` | symbol, side, type, quantity | reduceOnly, closePosition (if no qty), positionSide |
| `LIMIT` | symbol, side, type, quantity, price, timeInForce | reduceOnly, positionSide, newClientOrderId |
| `STOP` | symbol, side, type, quantity, price, stopPrice | timeInForce, reduceOnly, workingType |
| `STOP_MARKET` | symbol, side, type, stopPrice | quantity OR closePosition, workingType, reduceOnly |
| `TAKE_PROFIT` | symbol, side, type, quantity, price, stopPrice | timeInForce, reduceOnly, workingType |
| `TAKE_PROFIT_MARKET` | symbol, side, type, stopPrice | quantity OR closePosition, workingType, reduceOnly |
| `TRAILING_STOP_MARKET` | symbol, side, type, callbackRate | quantity OR closePosition, activationPrice, workingType |

### Parameter Descriptions

| Parameter | Type | Values | Notes |
|---|---|---|---|
| `symbol` | STRING | e.g. `BTCUSDT` | Uppercase, no slash |
| `side` | ENUM | `BUY` / `SELL` | |
| `positionSide` | ENUM | `BOTH` (one-way) / `LONG` / `SHORT` (hedge) | Defaults to `BOTH` |
| `type` | ENUM | See above | |
| `timeInForce` | ENUM | `GTC`, `IOC`, `FOK`, `GTX`, `GTD` | Required for LIMIT |
| `quantity` | DECIMAL | | Cannot use with `closePosition=true` |
| `price` | DECIMAL | | Required for LIMIT, STOP, TAKE_PROFIT |
| `reduceOnly` | STRING | `"true"` / `"false"` | Not available in Hedge mode |
| `stopPrice` | DECIMAL | | For STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET |
| `workingType` | ENUM | `CONTRACT_PRICE` (default) / `MARK_PRICE` | Trigger price type |
| `closePosition` | STRING | `"true"` / `"false"` | Closes entire position; conflicts with `quantity` |
| `activationPrice` | DECIMAL | | For TRAILING_STOP_MARKET; defaults to current mark price |
| `callbackRate` | DECIMAL | `0.1`–`5.0` (%) | Required for TRAILING_STOP_MARKET |
| `newClientOrderId` | STRING | regex `^[\.A-Z\:/a-z0-9_-]{1,36}$` | Custom order ID |
| `newOrderRespType` | ENUM | `ACK` (default) / `RESULT` | RESULT returns final fill info |
| `priceMatch` | ENUM | `NONE`, `OPPONENT`, `OPPONENT_5`, `OPPONENT_10`, `OPPONENT_20`, `QUEUE`, `QUEUE_5`, `QUEUE_10`, `QUEUE_20` | For LIMIT/STOP/TAKE_PROFIT |
| `recvWindow` | LONG | max 60000 | |
| `timestamp` | LONG | ms since epoch | |

### Order Type Details

#### MARKET
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "type": "MARKET",
  "quantity": 0.002
}
```
Fills immediately at best available price. On demo-fapi, minimum notional = 100 USDT.

#### LIMIT
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "LIMIT",
  "quantity": 0.002,
  "price": 70000.0,
  "timeInForce": "GTC",
  "reduceOnly": "true"
}
```
Rests until price reaches the limit. GTC = Good Till Cancel.

#### STOP_MARKET
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "STOP_MARKET",
  "stopPrice": 67000.0,
  "closePosition": "true",
  "workingType": "MARK_PRICE"
}
```
Triggers a market close when mark price crosses stopPrice. **NOT available on demo-fapi.**

#### TAKE_PROFIT_MARKET
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "TAKE_PROFIT_MARKET",
  "stopPrice": 71000.0,
  "closePosition": "true",
  "workingType": "MARK_PRICE"
}
```
Triggers a market close when mark price crosses stopPrice (in favorable direction). **NOT available on demo-fapi.**

#### STOP (stop-limit)
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "STOP",
  "quantity": 0.002,
  "price": 66800.0,
  "stopPrice": 67000.0,
  "timeInForce": "GTC",
  "reduceOnly": "true"
}
```
Becomes a LIMIT order when stopPrice is crossed. **NOT available on demo-fapi.**

#### TAKE_PROFIT (stop-limit)
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "TAKE_PROFIT",
  "quantity": 0.002,
  "price": 70900.0,
  "stopPrice": 70800.0,
  "timeInForce": "GTC",
  "reduceOnly": "true"
}
```
Becomes a LIMIT order when stopPrice is crossed (TP direction). **NOT available on demo-fapi.**

#### TRAILING_STOP_MARKET
```json
{
  "symbol": "BTCUSDT",
  "side": "SELL",
  "type": "TRAILING_STOP_MARKET",
  "quantity": 0.002,
  "callbackRate": 1.0,
  "activationPrice": 69000.0,
  "workingType": "MARK_PRICE",
  "reduceOnly": "true"
}
```
Trails behind the highest/lowest mark price by callbackRate%. **NOT available on demo-fapi.**

---

## 5. demo-fapi Testnet Limitations

### Verified on 2026-03-22

| Order Type | demo-fapi | Real Testnet | Production |
|---|---|---|---|
| MARKET | ✅ Works | ✅ | ✅ |
| LIMIT | ✅ Works | ✅ | ✅ |
| STOP | ❌ -4120 | ✅ | ✅ |
| STOP_MARKET | ❌ -4120 | ✅ | ✅ |
| TAKE_PROFIT | ❌ -4120 | ✅ | ✅ |
| TAKE_PROFIT_MARKET | ❌ -4120 | ✅ | ✅ |
| TRAILING_STOP_MARKET | ❌ -4120 | ✅ | ✅ |

**Error -4120:** `"Order type not supported for this endpoint. Please use the Algo Order API endpoints instead."`

The Algo Order API endpoints also return -4120 on demo-fapi — none of the trigger/conditional order types are available on paper trading.

### What Works for SL/TP on demo-fapi

| Purpose | Strategy | Works? | Notes |
|---|---|---|---|
| LONG TP | LIMIT SELL at TP price (above market) | ✅ | Rests until price rises to TP |
| SHORT TP | LIMIT BUY at TP price (below market) | ✅ | Rests until price drops to TP |
| LONG SL | LIMIT SELL at SL price (below market) | ❌ | Fills immediately — not SL behavior |
| SHORT SL | LIMIT BUY at SL price (above market) | ❌ | Never fills until price drops to level |
| **Any SL** | **Bot-side monitoring** | ✅ | Watch mark price via WebSocket; place MARKET when SL hit |

### Minimum Notional

MARKET and LIMIT orders require notional ≥ 100 USDT (unless `reduceOnly=true`).

### Account Defaults on demo-fapi

- Default leverage: 20x (set via `/fapi/v1/leverage`)
- Margin type: cross
- Position mode: one-way (BOTH)
- Starting balance: ~5000 USDT (paper)

---

## 6. REST API Endpoints

### Market Data (No Auth)

| Method | Path | Params | Description |
|---|---|---|---|
| GET | `/fapi/v1/ping` | — | Test connectivity |
| GET | `/fapi/v1/time` | — | Server time |
| GET | `/fapi/v1/exchangeInfo` | — | All symbols, filters, rate limits |
| GET | `/fapi/v1/depth` | symbol, limit | Order book (limit: 5,10,20,50,100,500,1000) |
| GET | `/fapi/v1/trades` | symbol, limit | Recent trades |
| GET | `/fapi/v1/klines` | symbol, interval, startTime, endTime, limit | OHLCV candles |
| GET | `/fapi/v1/ticker/price` | symbol (optional) | Latest price |
| GET | `/fapi/v1/ticker/bookTicker` | symbol (optional) | Best bid/ask |
| GET | `/fapi/v1/premiumIndex` | symbol (optional) | Mark price + funding rate |
| GET | `/fapi/v1/fundingRate` | symbol, startTime, endTime, limit | Funding rate history |

### Account & Trade (Signed)

| Method | Path | Weight | Description |
|---|---|---|---|
| POST | `/fapi/v1/order` | 1 | Place new order |
| POST | `/fapi/v1/batchOrders` | 5 | Place up to 5 orders |
| GET | `/fapi/v1/order` | 1 | Query specific order |
| DELETE | `/fapi/v1/order` | 1 | Cancel order |
| DELETE | `/fapi/v1/allOpenOrders` | 1 | Cancel all open orders for symbol |
| GET | `/fapi/v1/openOrders` | 1/40 | Open orders (1 with symbol, 40 without) |
| GET | `/fapi/v1/allOrders` | 5 | All orders (recent) |
| GET | `/fapi/v2/account` | 5 | Full account snapshot |
| GET | `/fapi/v3/account` | 5 | Account V3 (newer) |
| GET | `/fapi/v2/balance` | 5 | Asset balances |
| GET | `/fapi/v3/balance` | 5 | Balance V3 |
| GET | `/fapi/v2/positionRisk` | 5 | Position info (all or by symbol) |
| GET | `/fapi/v1/userTrades` | 5 | Trade fill history |
| POST | `/fapi/v1/leverage` | 1 | Set leverage |
| POST | `/fapi/v1/marginType` | 1 | Set margin type (ISOLATED/CROSS) |
| GET | `/fapi/v1/positionSide/dual` | 30 | Get position mode |
| POST | `/fapi/v1/positionSide/dual` | 1 | Set position mode |
| GET | `/fapi/v1/leverageBracket` | 1 | Leverage brackets per symbol |
| GET | `/fapi/v1/income` | 30 | Income history |
| POST | `/fapi/v1/listenKey` | 1 | Create user data stream listen key |
| PUT | `/fapi/v1/listenKey` | 1 | Keepalive listen key |
| DELETE | `/fapi/v1/listenKey` | 1 | Close listen key |

### Endpoint Details

#### POST /fapi/v1/leverage
```
Required: symbol, leverage (1–125)
Response: { symbol, leverage, maxNotionalValue }
```

#### GET /fapi/v2/positionRisk
```
Optional: symbol
Response array per position:
  symbol, positionAmt, entryPrice, breakEvenPrice, markPrice,
  unRealizedProfit, liquidationPrice, leverage, marginType,
  isolatedMargin, positionSide, notional, updateTime
```

#### GET /fapi/v2/account
```
Response: {
  totalWalletBalance,    # USDT wallet
  availableBalance,      # free margin
  totalMarginBalance,    # wallet + unrealizedPnl
  totalUnrealizedProfit,
  totalPositionInitialMargin,
  assets: [...],
  positions: [...]
}
```

---

## 7. SL/TP Orders — Definitive Guide

### Production (fapi.binance.com) — Full Support

**LONG position: entry at 68000**
```python
# SL: trigger MARKET SELL when mark price drops to 67000
conn.place_stop_loss_order("BTCUSDT", "SELL", 67000, close_position=True)
# → POST /fapi/v1/order with type=STOP_MARKET, stopPrice=67000,
#     workingType=MARK_PRICE, closePosition=true

# TP: trigger MARKET SELL when mark price rises to 71000
conn.place_take_profit_order("BTCUSDT", "SELL", 71000, close_position=True)
# → POST /fapi/v1/order with type=TAKE_PROFIT_MARKET, stopPrice=71000,
#     workingType=MARK_PRICE, closePosition=true
```

**SHORT position: entry at 68000**
```python
# SL: trigger MARKET BUY when mark price rises to 69000
conn.place_stop_loss_order("BTCUSDT", "BUY", 69000, close_position=True)

# TP: trigger MARKET BUY when mark price drops to 65000
conn.place_take_profit_order("BTCUSDT", "BUY", 65000, close_position=True)
```

### demo-fapi — Workaround

**LONG position: entry at 68000**
```python
# SL: ❌ STOP_MARKET not supported → returns sentinel {orderId: None}
#     Bot monitors mark price and places MARKET SELL when price < SL
sl_order = conn.place_stop_loss_order("BTCUSDT", "SELL", 67000, close_position=True)
# sl_order = {"orderId": None, "status": "TESTNET_NOT_SUPPORTED", ...}

# TP: ✅ TAKE_PROFIT_MARKET not supported → fallback LIMIT SELL at TP price
#     LIMIT SELL above market rests until price rises to TP level
tp_order = conn.place_take_profit_order("BTCUSDT", "SELL", 71000, close_position=True)
# tp_order = {"orderId": 12345678, "type": "LIMIT", "status": "NEW", ...}
```

**SHORT position: entry at 68000**
```python
# SL: ❌ returns sentinel → bot monitoring
sl_order = conn.place_stop_loss_order("BTCUSDT", "BUY", 69000, close_position=True)

# TP: ✅ LIMIT BUY below market rests until price drops to TP level
tp_order = conn.place_take_profit_order("BTCUSDT", "BUY", 65000, close_position=True)
```

### Why LIMIT Cannot Simulate SL

For a LONG position at 68000 with SL at 67000:

- **LIMIT SELL at 67000**: Semantics = "sell at 67000 or better (higher)". Current bid ~68000 is ABOVE 67000, so this fills **immediately** at market price. This is NOT a stop-loss.
- **STOP_MARKET at 67000**: Semantics = "when price crosses DOWN through 67000, then market sell". This correctly waits.

**Conclusion**: LIMIT cannot simulate SL. Only STOP_MARKET works for SL. Use bot-side monitoring on demo-fapi.

### SL/TP with closePosition vs quantity

`closePosition=true`:
- Exchange closes the ENTIRE position regardless of current size
- Cannot be combined with `quantity`
- Works for one-way mode positions

`quantity` + `reduceOnly=true`:
- Closes a specific quantity
- Required if you want partial closes
- Required in Hedge mode

---

## 8. Position Management

### Position Mode

**One-way mode (default)**: `positionSide=BOTH`
- One position per symbol
- Side determined by positionAmt sign (+ = long, - = short)
- `reduceOnly=true` on orders that should reduce position

**Hedge mode**: `positionSide=LONG` or `SHORT`
- Two simultaneous positions per symbol
- Can't use `reduceOnly`
- Must specify `positionSide` on every order

Set mode:
```
POST /fapi/v1/positionSide/dual  →  { "dualSidePosition": "true" }  # hedge
                                    { "dualSidePosition": "false" } # one-way
```

### Margin Type

```
POST /fapi/v1/marginType  →  { "symbol": "BTCUSDT", "marginType": "ISOLATED" }
                               { "symbol": "BTCUSDT", "marginType": "CROSS" }
```

### Leverage

```
POST /fapi/v1/leverage  →  { "symbol": "BTCUSDT", "leverage": 5 }
Response: { "symbol": "BTCUSDT", "leverage": 5, "maxNotionalValue": "500000000" }
```

### Open Position Flow (one-way, LONG)

```
1. POST /fapi/v1/leverage  { symbol, leverage }
2. POST /fapi/v1/order     { symbol, side=BUY, type=MARKET, quantity=X }
3. POST /fapi/v1/order     { symbol, side=SELL, type=STOP_MARKET, stopPrice=SL,
                             closePosition=true, workingType=MARK_PRICE }
4. POST /fapi/v1/order     { symbol, side=SELL, type=TAKE_PROFIT_MARKET,
                             stopPrice=TP, closePosition=true, workingType=MARK_PRICE }
```

### Position Sizing

```python
mark_price = conn.get_mark_price("BTCUSDT")   # from /fapi/v1/premiumIndex
qty_precision = conn.get_qty_precision("BTCUSDT")  # from /fapi/v1/exchangeInfo
quantity = round(usdt_amount / mark_price, qty_precision)
```

---

## 9. WebSocket Streams

### Connection URLs

```
# Single stream
wss://fstream.binance.com/ws/<streamName>

# Combined streams
wss://fstream.binance.com/stream?streams=<stream1>/<stream2>

# demo-fapi WebSocket
wss://dstream.binance.com/ws/<streamName>

# User data stream
wss://fstream.binance.com/ws/<listenKey>
```

### Subscribe/Unsubscribe at Runtime

```json
// Subscribe
{"method": "SUBSCRIBE", "params": ["btcusdt@markPrice", "btcusdt@kline_1m"], "id": 1}

// Unsubscribe
{"method": "UNSUBSCRIBE", "params": ["btcusdt@markPrice"], "id": 2}

// List subscriptions
{"method": "LIST_SUBSCRIPTIONS", "id": 3}
```

### Market Data Streams

| Stream | Description |
|---|---|
| `<symbol>@markPrice` | Mark price + funding rate (every 3s) |
| `<symbol>@markPrice@1s` | Mark price every 1s |
| `<symbol>@aggTrade` | Aggregated trade ticks |
| `<symbol>@kline_<interval>` | OHLCV candles (1m, 3m, 5m, 15m, 1h, 4h, 1d) |
| `<symbol>@depth<levels>` | Partial order book (5, 10, 20 levels) |
| `<symbol>@depth` | Differential order book updates |
| `<symbol>@ticker` | 24-hour rolling stats |
| `<symbol>@bookTicker` | Best bid/ask |
| `<symbol>@forceOrder` | Liquidation order updates |
| `!markPrice@arr` | All symbols mark price stream |

### User Data Stream

**Setup:**
```python
# 1. Create listen key
resp = session.post("https://fapi.binance.com/fapi/v1/listenKey",
                    headers={"X-MBX-APIKEY": api_key})
listen_key = resp.json()["listenKey"]

# 2. Connect WebSocket
ws_url = f"wss://fstream.binance.com/ws/{listen_key}"

# 3. Keepalive (every 30 minutes)
session.put(f"https://fapi.binance.com/fapi/v1/listenKey",
            headers={"X-MBX-APIKEY": api_key})
```

### ORDER_TRADE_UPDATE Event

```json
{
  "e": "ORDER_TRADE_UPDATE",
  "E": 1691000000000,   // event time ms
  "T": 1691000000000,   // transaction time ms
  "o": {
    "s": "BTCUSDT",     // symbol
    "c": "myOrder123",  // clientOrderId
    "S": "BUY",         // side
    "o": "LIMIT",       // type
    "f": "GTC",         // timeInForce
    "q": "0.002",       // quantity
    "p": "70000.0",     // price
    "ap": "70000.0",    // avg fill price
    "sp": "0",          // stopPrice
    "x": "TRADE",       // execType: NEW/CANCELED/TRADE/AMENDMENT/EXPIRED
    "X": "FILLED",      // status: NEW/PARTIALLY_FILLED/FILLED/CANCELED/EXPIRED
    "i": 12345678,      // orderId
    "l": "0.002",       // last filled qty
    "z": "0.002",       // cumulative filled qty
    "L": "70000.0",     // last execution price
    "N": "USDT",        // commission asset
    "n": "0.014",       // commission
    "T": 1691000000000, // trade time
    "t": 987654,        // trade id (-1 if no fill)
    "m": false,         // maker?
    "R": true,          // reduceOnly?
    "ps": "BOTH",       // positionSide
    "rp": "0.0"         // realized profit
  }
}
```

### ACCOUNT_UPDATE Event

```json
{
  "e": "ACCOUNT_UPDATE",
  "E": 1691000000000,
  "T": 1691000000000,
  "a": {
    "m": "ORDER",   // event reason: DEPOSIT/WITHDRAW/ORDER/FUNDING_FEE/etc
    "B": [{"a": "USDT", "wb": "5000.0", "cw": "4900.0", "bc": "0"}],
    "P": [{
      "s": "BTCUSDT", "pa": "0.002", "ep": "68000.0",
      "cr": "0.0", "up": "0.5", "mt": "cross", "iw": "0", "ps": "BOTH"
    }]
  }
}
```

---

## 10. Error Codes

### Most Common

| Code | Name | Cause | Fix |
|---|---|---|---|
| -4120 | STOP_ORDER_SWITCH_ALGO | Order type not supported (demo-fapi) | Use LIMIT or bot monitoring |
| -4164 | ORDER_REJECT_NOTIONAL_EXCEED | Notional < 100 USDT | Increase quantity |
| -1121 | INVALID_SYMBOL | Symbol doesn't exist or wrong format | Check symbol name |
| -1111 | BAD_PRECISION | Too many decimals | Round to symbol precision |
| -1100 | ILLEGAL_CHARS | Invalid characters in param | Check param values |
| -2010 | NEW_ORDER_REJECTED | Various — check msg | See msg for details |
| -2011 | CANCEL_REJECTED | Order already filled/canceled | Check order status first |
| -1125 | INVALID_LISTEN_KEY | Listen key expired | Re-create with POST /fapi/v1/listenKey |
| -1003 | TOO_MANY_REQUESTS | Rate limit exceeded | Back off, check headers |
| -4114 | INVALID_CLIENT_TRAN_ID_LEN | clientTranId > 64 chars | Shorten |

### HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (check body for error code) |
| 401 | Unauthorized (bad API key) |
| 403 | WAF blocked |
| 408 | Backend timeout — unknown if executed |
| 429 | Rate limit — back off |
| 418 | IP banned (repeated 429s) |
| 503 | Server error — unknown if executed; do NOT retry without checking |

---

## 11. Our BinanceFuturesConnector

**File:** `src/api/binance_futures.py`
**Config:** `BINANCE_FUTURES_API_KEY`, `BINANCE_FUTURES_API_SECRET`, `BINANCE_FUTURES_BASE_URL`

### Methods

| Method | Description |
|---|---|
| `ping()` | Test connectivity |
| `get_account()` | Full account snapshot (balance, positions) |
| `get_positions()` | All positions with positionAmt != 0 |
| `get_position(symbol)` | Single position or None if flat |
| `place_market_order(symbol, side, quantity)` | MARKET order |
| `place_stop_loss_order(symbol, side, stop_price, quantity=None, close_position=True)` | STOP_MARKET or sentinel on demo-fapi |
| `place_take_profit_order(symbol, side, stop_price, quantity=None, close_position=True)` | TAKE_PROFIT_MARKET or LIMIT fallback on demo-fapi |
| `cancel_order(symbol, order_id)` | Cancel by orderId |
| `cancel_all_orders(symbol)` | Cancel all open orders for symbol |
| `get_open_orders(symbol=None)` | All open orders |
| `get_trade_history(symbol, limit=500)` | Trade fill history |
| `set_leverage(symbol, leverage)` | Set leverage |
| `get_ticker(symbol)` | Latest price |
| `get_mark_price(symbol)` | Mark price (falls back to last price) |
| `get_qty_precision(symbol)` | quantityPrecision from exchangeInfo |
| `get_price_precision(symbol)` | pricePrecision from exchangeInfo |

### demo-fapi Behavior of SL/TP Methods

**`place_stop_loss_order` on demo-fapi:**
- Attempts STOP_MARKET
- Gets -4120 → logs INFO, returns `{"orderId": None, "status": "TESTNET_NOT_SUPPORTED"}`
- Caller checks `orderId is not None` before storing

**`place_take_profit_order` on demo-fapi:**
- Attempts TAKE_PROFIT_MARKET
- Gets -4120 → falls back to LIMIT reduceOnly at stop_price
- If `close_position=True` and no `quantity`, calls `get_position()` for current qty
- Returns real exchange order response with valid orderId

---

## 12. Python Code Examples

### Basic Setup

```python
import os
from src.api.binance_futures import BinanceFuturesConnector

conn = BinanceFuturesConnector(
    api_key=os.getenv("BINANCE_FUTURES_API_KEY"),
    api_secret=os.getenv("BINANCE_FUTURES_API_SECRET"),
    base_url=os.getenv("BINANCE_FUTURES_BASE_URL", "https://demo-fapi.binance.com"),
)
```

### Open LONG with SL/TP

```python
from src.api.futures_executor import FuturesTestnetExecutor

ex = FuturesTestnetExecutor()
result = ex.open_long(
    symbol="BTCUSDT",
    usdt_amount=500.0,    # position size in USDT
    sl=67000.0,           # stop loss price
    tp=71000.0,           # take profit price
    leverage=5,
)

if result["executed"]:
    print(f"Order: {result['order_id']}")
    print(f"Qty: {result['quantity']} @ ${result['mark_price']}")
    print(f"SL order: {result['sl_order_id']}")  # None on demo-fapi
    print(f"TP order: {result['tp_order_id']}")  # real order ID
else:
    print(f"Failed: {result['error']}")
```

### Open SHORT with SL/TP

```python
result = ex.open_short(
    symbol="ETHUSDT",
    usdt_amount=300.0,
    sl=3200.0,   # stop loss ABOVE entry for shorts
    tp=2800.0,   # take profit BELOW entry for shorts
    leverage=3,
)
```

### Update Trailing SL

```python
# Cancel old SL, place new one higher (trailing LONG)
ex.update_sl("BTCUSDT", "LONG", 68500.0)

# Cancel old SL, place new one lower (trailing SHORT)
ex.update_sl("ETHUSDT", "SHORT", 3050.0)
```

### Direct API Calls

```python
# Check current balance
account = conn.get_account()
balance = float(account["availableBalance"])
print(f"Available: ${balance:.2f}")

# Get mark price
price = conn.get_mark_price("BTCUSDT")

# Get all open positions
positions = conn.get_positions()
for p in positions:
    print(f"{p['symbol']}: {p['positionAmt']} @ {p['entryPrice']}")

# Cancel all orders for symbol
conn.cancel_all_orders("BTCUSDT")

# Set leverage
conn.set_leverage("BTCUSDT", 5)
```

### WebSocket Price Monitoring (for bot-side SL)

```python
import asyncio, json, websockets

async def monitor_price(symbol: str, sl_price: float, on_sl_hit):
    stream = f"{symbol.lower()}@markPrice@1s"
    url = f"wss://dstream.binance.com/ws/{stream}"  # demo-fapi WS
    async with websockets.connect(url) as ws:
        async for msg in ws:
            data = json.loads(msg)
            mark = float(data["p"])
            # For LONG: SL hit when price drops below SL
            if mark <= sl_price:
                await on_sl_hit(symbol, mark)
                break
```

### Manual Order Placement (raw)

```python
# Place LIMIT SELL reduceOnly (TP for LONG)
order = conn._post("/fapi/v1/order", {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "type": "LIMIT",
    "quantity": 0.002,
    "price": 71000.0,
    "timeInForce": "GTC",
    "reduceOnly": "true",
})
print(f"TP order placed: {order['orderId']}")

# Market close position
close = conn._post("/fapi/v1/order", {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "type": "MARKET",
    "quantity": 0.002,
    "reduceOnly": "true",
})
```

---

## Summary: demo-fapi Cheat Sheet

```
What works:
  MARKET buy/sell          → ✅ normal
  LIMIT buy/sell           → ✅ normal (min notional 100 USDT unless reduceOnly)
  LIMIT reduceOnly         → ✅ TP for LONG and SHORT (rests until price reaches level)
  cancel_order             → ✅ normal
  cancel_all_orders        → ✅ normal
  set_leverage             → ✅ normal
  get_positions            → ✅ normal
  get_account              → ✅ normal

What doesn't work:
  STOP_MARKET              → ❌ -4120
  TAKE_PROFIT_MARKET       → ❌ -4120
  STOP (stop-limit)        → ❌ -4120
  TAKE_PROFIT (stop-limit) → ❌ -4120
  TRAILING_STOP_MARKET     → ❌ -4120

SL/TP strategy on demo-fapi:
  TP: place_take_profit_order() → auto-falls back to LIMIT (✅ works)
  SL: place_stop_loss_order()  → returns sentinel {orderId: None}
      → rely on bot WebSocket monitoring for SL execution
```
