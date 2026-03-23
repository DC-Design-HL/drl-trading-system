# Binance Futures Algo Order Migration

## Background

As of 2025, Binance migrated all conditional order types from the standard
`/fapi/v1/order` endpoint to a dedicated **Algo Order API**. Placing orders
of these types via the old endpoint now returns error code **`-4120`**
(`STOP_ORDER_SWITCH_ALGO`).

### Affected Order Types

- `STOP_MARKET` — used for stop-loss
- `TAKE_PROFIT_MARKET` — used for take-profit
- `STOP` — limit stop order
- `TAKE_PROFIT` — limit take-profit order
- `TRAILING_STOP_MARKET` — trailing stop

**Regular `MARKET` and `LIMIT` orders are NOT affected** and continue to
work on `/fapi/v1/order`.

---

## Algo Order API Endpoints

All endpoints require HMAC-SHA256 signing (same as standard signed endpoints).

### Place Algo Order

```
POST /fapi/v1/algoOrder
```

**Weight:** 1

| Parameter     | Type    | Required | Description |
|---------------|---------|----------|-------------|
| algoType      | ENUM    | YES      | Only `CONDITIONAL` is supported |
| symbol        | STRING  | YES      | Trading pair (e.g. `BTCUSDT`) |
| side          | ENUM    | YES      | `BUY` or `SELL` |
| type          | ENUM    | YES      | `STOP_MARKET`, `TAKE_PROFIT_MARKET`, `STOP`, `TAKE_PROFIT`, `TRAILING_STOP_MARKET` |
| positionSide  | ENUM    | NO       | `BOTH` (One-way), `LONG`/`SHORT` (Hedge). Default `BOTH` |
| timeInForce   | ENUM    | NO       | `GTC`, `IOC`, `FOK`, `GTX`. Default `GTC` |
| quantity      | DECIMAL | NO       | Cannot be sent with `closePosition=true` |
| price         | DECIMAL | NO       | Limit price (for STOP/TAKE_PROFIT limit types) |
| triggerPrice  | DECIMAL | NO       | Price at which the order triggers |
| workingType   | ENUM    | NO       | `MARK_PRICE` or `CONTRACT_PRICE`. Default `CONTRACT_PRICE` |
| closePosition | STRING  | NO       | `"true"` / `"false"`. Close-All, used with STOP_MARKET/TAKE_PROFIT_MARKET |
| priceProtect  | STRING  | NO       | `"TRUE"` / `"FALSE"`. Default `"FALSE"` |
| reduceOnly    | STRING  | NO       | `"true"` / `"false"`. Cannot combine with Hedge Mode or closePosition |
| activatePrice | DECIMAL | NO       | For TRAILING_STOP_MARKET only |
| callbackRate  | DECIMAL | NO       | For TRAILING_STOP_MARKET, 0.1–10 (1 = 1%) |
| clientAlgoId  | STRING  | NO       | Custom ID, auto-generated if omitted |
| priceMatch    | ENUM    | NO       | For LIMIT/STOP/TAKE_PROFIT orders |
| selfTradePreventionMode | ENUM | NO | Default `NONE` |
| goodTillDate  | LONG   | NO       | Cancel time for GTD orders |
| recvWindow    | LONG    | NO       | |
| timestamp     | LONG    | YES      | |

**Response:**
```json
{
  "algoId": 2146760,
  "clientAlgoId": "6B2I9XVcJpCjqPAJ4YoFX7",
  "algoType": "CONDITIONAL",
  "orderType": "TAKE_PROFIT_MARKET",
  "symbol": "BTCUSDT",
  "side": "SELL",
  "algoStatus": "NEW",
  "triggerPrice": "90000.000",
  ...
}
```

**Key difference from legacy:** Response uses `algoId` instead of `orderId`.

### Cancel Algo Order

```
DELETE /fapi/v1/algoOrder
```

| Parameter    | Type   | Required | Description |
|--------------|--------|----------|-------------|
| algoId       | LONG   | NO*      | |
| clientAlgoId | STRING | NO*      | |
| recvWindow   | LONG   | NO       | |
| timestamp    | LONG   | YES      | |

*Either `algoId` or `clientAlgoId` must be sent.

### Cancel All Open Algo Orders

```
DELETE /fapi/v1/algoOpenOrders
```

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| symbol    | STRING | YES      | |
| recvWindow| LONG   | NO       | |
| timestamp | LONG   | YES      | |

### Query Open Algo Orders

```
GET /fapi/v1/openAlgoOrders
```

**Weight:** 1 per symbol, 40 without symbol filter.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| algoType  | STRING | NO       | |
| symbol    | STRING | NO       | |
| algoId    | LONG   | NO       | |
| recvWindow| LONG   | NO       | |
| timestamp | LONG   | YES      | |

### Query Specific Algo Order

```
GET /fapi/v1/algoOrder
```

| Parameter    | Type   | Required |
|--------------|--------|----------|
| algoId       | LONG   | NO*      |
| clientAlgoId | STRING | NO*      |
| recvWindow   | LONG   | NO       |
| timestamp    | LONG   | YES      |

*Either `algoId` or `clientAlgoId` must be sent.

### Query All Algo Orders (Historical)

```
GET /fapi/v1/allAlgoOrders
```

**Weight:** 5

| Parameter | Type   | Required |
|-----------|--------|----------|
| symbol    | STRING | YES      |
| algoId    | LONG   | NO       |
| startTime | LONG   | NO       |
| endTime   | LONG   | NO       |
| page      | INT    | NO       |
| limit     | INT    | NO       |
| recvWindow| LONG   | NO       |
| timestamp | LONG   | YES      |

---

## How SL/TP Are Placed in Our System

### Order Placement Flow (Algo-First Strategy)

When placing SL or TP orders, the system uses this cascade:

1. **Try Algo Order API** (`POST /fapi/v1/algoOrder`)
   - SL: `type=STOP_MARKET`, `workingType=MARK_PRICE`, `closePosition=true`
   - TP: `type=TAKE_PROFIT_MARKET`, `workingType=MARK_PRICE`, `closePosition=true`
   - On success, `algoId` is stored as the order ID with `_algo_order=True` flag

2. **Fall back to Legacy API** (`POST /fapi/v1/order`)
   - Same parameters but via the old endpoint
   - Works if the environment hasn't migrated to algo orders yet

3. **SL: Return sentinel** (orderId=None) if both fail with `-4120`
   - Bot-side WebSocket price monitoring handles SL execution
   
4. **TP: Fall back to LIMIT reduceOnly** if `-4120` on both paths
   - LIMIT orders at TP price correctly simulate TP behavior

### Order Cancellation

- Orders are tracked with `_algo_sl_flags` / `_algo_tp_flags` dicts
- When cancelling (e.g., trailing SL update), the system checks if the order
  is algo (`is_algo=True`) and uses the appropriate cancel endpoint
- `cancel_order()` auto-retries as algo cancel if standard cancel returns
  `-2011` (Unknown order)
- `cancel_all_orders()` cancels both standard AND algo orders

### Startup Sync

`_sync_order_tracking()` on executor construction:
1. Fetches standard open orders → identifies SL (STOP_MARKET) and TP (LIMIT reduceOnly)
2. Fetches algo open orders → identifies SL (STOP_MARKET/STOP) and TP (TAKE_PROFIT_MARKET/TAKE_PROFIT)
3. Sets `_algo_sl_flags` / `_algo_tp_flags` for proper cancel routing

---

## Common Errors and Fixes

### Error `-4120` (STOP_ORDER_SWITCH_ALGO)

**Cause:** Trying to place STOP_MARKET, TAKE_PROFIT_MARKET, etc. via `/fapi/v1/order`.

**Fix:** Use `/fapi/v1/algoOrder` with `algoType=CONDITIONAL` instead.

### Error `-2011` (Unknown order) on Cancel

**Cause:** Trying to cancel an algo order via `/fapi/v1/order` (wrong endpoint).

**Fix:** Use `DELETE /fapi/v1/algoOrder` with `algoId` parameter. Our system
automatically retries as algo cancel when this error occurs.

### Algo Order Not Found in Open Orders

**Cause:** Checking `/fapi/v1/openOrders` — algo orders don't appear there.

**Fix:** Also check `/fapi/v1/openAlgoOrders` for conditional/algo orders.

### `algoId` vs `orderId`

**Key difference:** Algo orders return `algoId` not `orderId`. Our connector
normalizes this by setting `result["orderId"] = algoId` and marking with
`_algo_order = True` so the executor knows which cancel endpoint to use.

---

## Code Locations

- **Connector:** `src/api/binance_futures.py` — `BinanceFuturesConnector`
  - `place_algo_order()` — generic algo order placement
  - `cancel_algo_order()` — cancel specific algo order
  - `cancel_all_algo_orders()` — cancel all algo orders for symbol
  - `get_open_algo_orders()` — query open algo orders
  - `get_algo_order()` — query specific algo order
  - `get_all_algo_orders()` — query historical algo orders
  - `place_stop_loss_algo()` — convenience: STOP_MARKET via algo
  - `place_take_profit_algo()` — convenience: TAKE_PROFIT_MARKET via algo
  - `place_stop_loss_order()` — algo-first with legacy fallback
  - `place_take_profit_order()` — algo-first with legacy fallback

- **Executor:** `src/api/futures_executor.py` — `FuturesTestnetExecutor`
  - `_sync_order_tracking()` — syncs both standard and algo orders
  - `_algo_sl_flags` / `_algo_tp_flags` — track which orders are algo
  - All SL/TP operations use `is_algo` flag for proper cancel routing

## References

- [New Algo Order](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/New-Algo-Order)
- [Cancel Algo Order](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Cancel-Algo-Order)
- [Cancel All Algo Open Orders](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Cancel-All-Algo-Open-Orders)
- [Current All Algo Open Orders](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Current-All-Algo-Open-Orders)
- [Query Algo Order](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Query-Algo-Order)
- [Query All Algo Orders](https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Query-All-Algo-Orders)
