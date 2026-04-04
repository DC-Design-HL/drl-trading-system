---
name: testnet-position-sync
description: Diagnose and fix Binance Futures testnet position desyncs between the HTF trading bots and the exchange. Use when positions on Binance testnet don't match what the bot thinks, when close/open alerts fire but exchange state doesn't change, when direction mismatches occur (bot says LONG, exchange has SHORT), when ghost SL/TP alerts appear for non-existent positions, or when partial TP levels are wrong.
---

# Testnet Position Sync

## Quick Diagnosis

Run the sync check script to compare exchange vs bot state:

```bash
bash skills/testnet-position-sync/scripts/sync-check.sh
```

## Known Desync Causes & Fixes

### 1. Direction Mismatch (bot=LONG, exchange=SHORT)

**Cause**: `_mirror_testnet()` previously skipped REVERSE_CLOSE actions. The old SHORT wasn't closed on the exchange before opening the new LONG. The OPEN_LONG just partially reduced the SHORT.

**Fix in code** (already deployed): `_mirror_testnet()` now sends a market close order for REVERSE_CLOSE before returning. The subsequent OPEN is mirrored as a fresh position.

**Manual fix**: Restart the bot — `_sync_position_from_exchange()` detects direction mismatch and syncs to exchange state with recalculated SL/TP.

```bash
systemctl restart drl-htf-agent  # BTC
systemctl restart drl-htf-eth    # ETH
```

### 2. Ghost SL/TP Alerts (alerts for non-existent position)

**Cause**: Bot state file has `position: 1` or `-1` but exchange is flat. Happens when exchange closes a position (TP/SL hit) but the bot's internal state wasn't updated.

**Fix in code** (already deployed): `_sync_position_from_exchange()` checks if exchange position exists. If exchange is flat but bot has position → resets bot to flat.

**Manual fix**: Delete the state file and restart:
```bash
rm logs/htf_trading_state_ETHUSDT.json  # ETH
rm logs/htf_trading_state.json          # BTC
systemctl restart drl-htf-eth
systemctl restart drl-htf-agent
```

### 3. Units Mismatch (bot has wrong quantity)

**Cause**: Partial TPs executed on bot side but failed on exchange, or emergency closes changed exchange quantity without updating bot.

**Manual fix**: Edit the state file to match exchange, disable stale partial TP levels:
```python
import json
with open('logs/htf_trading_state.json') as f:
    s = json.load(f)
s['position_units'] = <EXCHANGE_AMOUNT>
s['initial_position_units'] = <EXCHANGE_AMOUNT>
s['partial_tp_level'] = 2        # Skip partial TPs
s['partial_tp1_price'] = 0.0
s['partial_tp2_price'] = 0.0
with open('logs/htf_trading_state.json', 'w') as f:
    json.dump(s, f, indent=2)
```
Then restart the bot.

### 4. SL/TP = $0 (unprotected position)

**Cause**: Position detected from exchange sync but SL/TP were zeroed during a state reset.

**Fix in code** (already deployed): `_load_state()` auto-recalculates SL/TP from entry price when they are zero:
- LONG: SL = entry × (1 - STOP_LOSS_PCT), TP = entry × (1 + TAKE_PROFIT_PCT)
- SHORT: SL = entry × (1 + STOP_LOSS_PCT), TP = entry × (1 - TAKE_PROFIT_PCT)

### 5. Emergency Close Kills New Position

**Cause**: Binance testnet rejects SL/TP orders with errors -4509, -2021, -2022. The safety rule "never have position without SL/TP" triggered an emergency market close on the new position.

**Fix in code** (already deployed): These errors are recognized as testnet limitations. Position stays open, bot-side WS monitoring handles SL/TP instead.

**Testnet error codes handled**:
- `-4509`: TIF GTE can only be used with open positions
- `-2021`: Order would immediately trigger
- `-2022`: ReduceOnly Order is rejected

### 6. False Liquidation Risk Alerts

**Cause**: Liquidation check used `self.position` (bot's internal direction) which could be wrong during desyncs. A SHORT with liq=$278K was flagged as "LONG liquidation risk".

**Fix in code** (already deployed): Check now infers direction from liquidation price vs entry price:
- liq < entry → LONG (liquidation below = price dropping)
- liq > entry → SHORT (liquidation above = price rising)

## Sync Check Procedure

1. Get exchange positions:
```python
from src.api.binance_futures import BinanceFuturesConnector
c = BinanceFuturesConnector(api_key=KEY, api_secret=SECRET)
for p in c.get_positions():
    amt = float(p.get('positionAmt', 0))
    if amt != 0:
        print(f"{p['symbol']}: amt={amt}, entry=${p['entryPrice']}")
```

2. Compare with bot state files:
   - BTC: `logs/htf_trading_state.json`
   - ETH: `logs/htf_trading_state_ETHUSDT.json`
   - Other: `logs/htf_trading_state_<SYMBOL>.json`

3. Check fields: `position` (1=LONG, -1=SHORT, 0=FLAT), `position_price`, `position_units`, `sl_price`, `tp_price`

4. If mismatch → edit state file or restart bot (sync runs on startup and every iteration)

## Key Files

| File | Purpose |
|------|---------|
| `live_trading_htf.py` | Bot logic, state management, `_sync_position_from_exchange()` |
| `src/api/futures_executor.py` | Exchange order execution, SL/TP placement, emergency close logic |
| `logs/htf_trading_state.json` | BTC bot state |
| `logs/htf_trading_state_ETHUSDT.json` | ETH bot state |
| `trade_alerter.py` | Telegram alert service |
| `logs/htf_pending_alerts.jsonl` | Alert queue file |
| `logs/.alerter_offset` | Alerter position in queue |

## Alerter Spam Fix

If alerts are spamming, skip the alerter past the spam:
```bash
wc -c logs/htf_pending_alerts.jsonl | awk '{print $1}' > logs/.alerter_offset
systemctl restart drl-trade-alerter
```
