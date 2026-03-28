# API Response Schemas

## GET /api/metrics/overview

```json
{
  "uptime": "2h 15m",
  "uptime_seconds": 8100,
  "api": { /* same as /api/metrics/api */ },
  "errors": { /* same as /api/metrics/errors */ },
  "health": { /* same as /api/metrics/health */ },
  "trades": { /* same as /api/metrics/trades */ }
}
```

## GET /api/metrics/api

```json
{
  "outbound": {
    "binance/fapi/v1/order": {
      "total_calls": 45,
      "success": 42,
      "errors": 3,
      "error_rate": "6.7%",
      "avg_latency_ms": 850.3,
      "last_call": "2026-03-28T15:30:00",
      "last_error": "Binance Futures API 400: {'code': -2021, 'msg': 'Order would immediately trigger.'}"
    }
  },
  "inbound": {},
  "outbound_rate_per_min": 2.4,
  "inbound_rate_per_min": 0.0
}
```

## GET /api/metrics/errors

```json
{
  "binance_api_error": {
    "count": 3,
    "first_seen": "2026-03-28T14:00:00",
    "last_seen": "2026-03-28T15:30:00",
    "sample": "Binance Futures API 400: ..."
  },
  "binance_would_trigger": {
    "count": 12,
    "first_seen": "2026-03-28T13:40:00",
    "last_seen": "2026-03-28T13:44:00",
    "sample": "Order would immediately trigger"
  }
}
```

Error type classification (in `binance_futures.py`):
- `binance_api_error` — generic API error
- `binance_algo_migration` — error -4120 (STOP_ORDER_SWITCH_ALGO)
- `binance_would_trigger` — error -2021 (Order would immediately trigger)
- `binance_gte_position` — error -4509 (TIF GTE requires open position)
- `binance_timeout` — request timeout
- `binance_connection` — connection error

## GET /api/metrics/health

```json
{
  "Binance REST (Futures)": {
    "status": "healthy",
    "response_time_ms": 815.7,
    "last_check": "2026-03-28T15:29:42",
    "last_success": "2026-03-28T15:29:42",
    "last_error": null,
    "consecutive_failures": 0
  },
  "Bot: htf-btc": {
    "status": "healthy",
    "response_time_ms": 9.4,
    "last_check": "2026-03-28T15:29:43",
    "last_success": "2026-03-28T15:29:43",
    "last_error": null,
    "consecutive_failures": 0
  }
}
```

Status values: `"healthy"`, `"degraded"`, `"down"`, `"unknown"`

## GET /api/metrics/trades

```json
{
  "testnet": {
    "total_trades": 25,
    "wins": 15,
    "losses": 10,
    "win_rate": "60.0%",
    "total_pnl": 142.50,
    "best_trade": 45.20,
    "worst_trade": -22.10,
    "avg_pnl": 5.70,
    "trades_today": 3,
    "pnl_today": 18.50,
    "recent_trades": [
      {
        "timestamp": "2026-03-28T15:30:00",
        "symbol": "BTCUSDT",
        "action": "CLOSE_LONG",
        "direction": "LONG",
        "pnl": 11.16,
        "entry_price": 66797.37,
        "exit_price": 66827.64,
        "reason": "SL"
      }
    ]
  },
  "paper": { /* same structure */ }
}
```
