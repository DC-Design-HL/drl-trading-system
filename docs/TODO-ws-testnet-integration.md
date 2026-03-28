# TODO: WebSocket-Based Testnet SL/TP Management

## Date Created: 2026-03-28
## Status: PLANNED
## Risk Level: HIGH — changes core exit logic for live testnet positions

---

## Current State (BEFORE changes)

### How SL/TP Works Now

**Architecture**: Hybrid — exchange algo orders + bot-side WebSocket monitoring

1. **Position Open**: Bot opens MARKET order via REST API (`POST /fapi/v1/order`)
2. **SL Placement**: Bot places STOP_MARKET algo order on exchange (`POST /fapi/v1/algoOrder`)
3. **TP Placement**: Bot places TAKE_PROFIT_MARKET algo order on exchange (`POST /fapi/v1/algoOrder`)
4. **Trailing SL**: On every WebSocket price tick, bot recalculates SL → cancels old algo order → places new algo order via REST API
5. **Exit**: Exchange triggers SL/TP autonomously when price hits the level

### Files Involved

| File | Role |
|------|------|
| `src/api/binance_futures.py` | REST API connector (HMAC-signed requests to demo-fapi.binance.com) |
| `src/api/futures_executor.py` | Trade execution: `open_long()`, `open_short()`, `update_sl()`, `update_tp()` |
| `src/api/testnet_executor.py` | Bridge: `mirror_trade()` routes bot decisions to futures executor |
| `live_trading_htf.py` | Main bot loop: `_check_sl_tp()` called on every WS price tick |

### Current SL/TP Trailing Flow (per price tick)

```
WebSocket tick ($67,000.50)
  → _check_sl_tp(67000.50)
    → Calculate new trailing SL: peak * (1 - 0.005) = $66,665.25
    → Is $66,665.25 > current SL ($66,660.00)? YES
    → Call update_sl():
      1. Cancel existing algo order (DELETE /fapi/v1/algoOrder, algoId=123)
      2. Place new algo order (POST /fapi/v1/algoOrder, triggerPrice=$66,665.25)
      3. Track new algoId
    → Alert queued for Telegram
```

### The Problem

- **Every price tick** triggers a SL recalculation and potential exchange update
- Updates happen every ~1 second with ~$7 changes (0.01% on BTC)
- Each update = 2 API calls (cancel + place) = massive rate limit pressure
- ETH hits -2021 errors ("Order would immediately trigger") in rapid-fire loops
- 10+ SL updates in 8 seconds is common during trending moves

### Current Constants

```python
TRAILING_BREAKEVEN_PCT = 0.01    # Activate trailing at +1% profit
TRAILING_DISTANCE_PCT = 0.005    # Trail 0.5% behind peak price
```

### Current Exchange Order Types Used

- `STOP_MARKET` via Algo API → SL (closePosition=true, workingType=MARK_PRICE)
- `TAKE_PROFIT_MARKET` via Algo API → TP (closePosition=true, workingType=MARK_PRICE)
- Fallback: Legacy `/fapi/v1/order` if algo fails
- Last resort for TP: `LIMIT` reduceOnly

### Order Tracking State

```python
self._sl_orders: Dict[str, int] = {}       # symbol → algoId
self._tp_orders: Dict[str, int] = {}       # symbol → algoId  
self._algo_sl_flags: Dict[str, bool] = {}  # symbol → True if algo order
self._algo_tp_flags: Dict[str, bool] = {}  # symbol → True if algo order
```

### Startup Safety

- `_sync_order_tracking()`: Re-populates order tracking from exchange on restart
- `ensure_sl_tp_for_open_positions()`: Places missing SL/TP for orphaned positions

---

## Proposed Change: WebSocket-Only SL/TP

### Concept

Instead of placing/canceling algo orders on Binance for every SL trail:

1. **Only place exchange SL/TP orders ONCE at position open** (initial SL and TP)
2. **Bot monitors SL/TP via WebSocket price ticks** (already doing this)
3. **When bot-side SL triggers**: Send a MARKET close order via REST API
4. **Only update exchange SL order when change is significant** (e.g., >0.1% / >$67 on BTC) AND with a cooldown (e.g., max 1 update per 60 seconds)
5. **Exchange SL remains as safety net** — if bot crashes, exchange SL catches the position

### Benefits

- Eliminates API rate limit abuse (cancel+place loops)
- Eliminates -2021 errors on ETH
- Trailing SL is instant (no API roundtrip latency)
- Exchange SL acts as a "crash protection" backstop, not the primary exit mechanism
- TP can still be exchange-managed (no change needed for TP)

### Risks

- If bot crashes AND exchange SL is stale (not updated recently), position exits at an old SL level
- WebSocket disconnection could miss SL triggers (mitigated by exchange backstop)
- Need robust WS reconnection logic (already exists)

### Safety Measures

- Exchange SL stays in place as crashguard — always kept within reasonable distance
- Update exchange SL periodically (every 60s) or when change exceeds threshold
- If WS disconnects for >30s, force-update exchange SL to latest bot-side SL
- Log all bot-side SL triggers clearly for audit

---

## Rollback Plan

If anything breaks:
1. `git revert <commit>` to undo changes
2. `systemctl restart drl-htf-agent drl-htf-eth` to restart bots
3. Bots will resume using exchange algo orders for all SL/TP

## Git State Before Changes

```
Branch: dev
Last commit: 8638d56 feat: BOS/CHOCH clean-line validation + diagonal chart lines (wick→body)
```

---

## Implementation Checklist

- [ ] Add minimum SL change threshold (0.1%) to `update_sl()` in futures_executor.py
- [ ] Add cooldown timer (60s) between exchange SL updates
- [ ] Bot-side `_check_sl_tp()` triggers MARKET close when SL hit (instead of relying solely on exchange)
- [ ] Exchange SL remains as crashguard backstop
- [ ] Update exchange SL periodically when change is significant
- [ ] Test with BTC testnet position
- [ ] Test with ETH testnet position
- [ ] Verify bot restart still syncs properly
- [ ] Verify exchange SL catches position if bot is killed mid-trade
- [ ] Update skill docs
