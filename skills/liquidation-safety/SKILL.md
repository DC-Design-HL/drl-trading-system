# Liquidation Price Safety Check

## Purpose

Validates that the bot's Stop Loss (SL) price always triggers **before** the
exchange liquidation price. This is a critical safety net to prevent liquidation
during extreme market moves, flash crashes, or if leverage is ever increased.

## Logic

A **delta buffer** of 1% of entry price is maintained between the SL and
liquidation price to account for slippage, latency, and flash-crash scenarios.

### LONG positions

```
Liquidation price must be LOWER than (SL price − delta)
If liquidation >= SL − delta → ⚠️ ALERT
```

### SHORT positions

```
Liquidation price must be HIGHER than (SL price + delta)
If liquidation <= SL + delta → ⚠️ ALERT
```

### Edge cases

- **1x leverage longs**: Liquidation = $0 → check is gracefully skipped.
- **1x leverage shorts**: Liquidation = 2× entry → effectively impossible, but
  checked for correctness.
- **Dry-run / paper trade**: Uses a simulated liquidation formula:
  - LONG: `entry × (1 − 1/leverage)`
  - SHORT: `entry × (1 + 1/leverage)`

## Where the check runs

| File | Location | Trigger |
|------|----------|---------|
| `live_trading_htf.py` | `_check_sl_tp()` | Every iteration + every WS tick |
| `live_trading_htf.py` | `_load_state()` | On bot startup (state restore) |
| `src/api/futures_executor.py` | `open_long()` / `open_short()` | After position is opened |
| `src/api/futures_executor.py` | `get_liquidation_price()` | On-demand from `/fapi/v2/positionRisk` |
| `src/api/futures_executor.py` | `validate_sl_vs_liquidation()` | Reusable validation method |
| `live_trading_htf_partial.py` | `_check_partial_tp()` | Every iteration (simulated) |
| `live_trading_htf_hybrid.py` | `_check_hybrid_exit()` | Every iteration (simulated) |

## Alert format

When risk is detected, a `LIQUIDATION_RISK` entry is written to
`logs/htf_pending_alerts.jsonl`. The `trade_alerter.py` formats it as:

```
⚠️ LIQUIDATION RISK — ETHUSDT LONG
🔴 Liquidation: $1,950.00
🛑 SL: $1,960.00
💰 Entry: $2,000.00
📐 Buffer: $10.00 (0.5%)
⚡ Recommended: Tighten SL or reduce position size
📈 Strategy: HTF Standard
🕐 2026-03-23 14:30 UTC
```

## Alert JSON schema

```json
{
  "timestamp": "2026-03-23T14:30:00",
  "strategy": "htf",
  "trade": {
    "type": "LIQUIDATION_RISK",
    "symbol": "ETHUSDT",
    "direction": "LONG",
    "liquidation_price": 1950.00,
    "sl_price": 1960.00,
    "entry_price": 2000.00,
    "delta": 20.00,
    "buffer": 10.00,
    "buffer_pct": 0.5,
    "simulated": false
  },
  "signals": {},
  "position": { ... }
}
```

## Key methods

### `FuturesTestnetExecutor.get_liquidation_price(symbol)`
Fetches the real liquidation price from `/fapi/v2/positionRisk`.

### `FuturesTestnetExecutor.validate_sl_vs_liquidation(symbol, side, entry, sl)`
Returns a dict with `safe`, `liquidation_price`, `delta`, `buffer`, `buffer_pct`.

### `HTFLiveBot._get_liquidation_price()`
Wrapper that calls the executor's method. Returns 0.0 for dry-run/no-connection.

### `HTFLiveBot._check_liquidation_safety()`
Periodic check called every iteration. Logs CRITICAL and writes alert on risk.

### `HTFPartialBot._get_simulated_liquidation_price(leverage)`
### `HTFHybridBot._get_simulated_liquidation_price(leverage)`
Paper trade formula: `entry × (1 ± 1/leverage)`.

## When this matters

- At **1x leverage**, this check is mostly academic (longs can't be liquidated).
- If leverage is **ever increased** (2x, 3x, 5x+), this check becomes the
  primary defense against catastrophic liquidation.
- The 1% delta buffer protects against:
  - Network latency in SL execution
  - Exchange matching engine delays
  - Flash crashes that gap past the SL price
  - Funding rate spikes that move the liquidation price
