---
name: system-dashboard
description: System monitoring dashboard for the DRL trading system. Use when adding metrics, health checks, API tracking, error monitoring, or trade statistics to the dashboard. Also use when debugging connectivity issues, checking service health, or investigating API rate limits. Covers the metrics collector, health checker, API instrumentation, Streamlit UI system tab, and all /api/metrics/* endpoints.
---

# System Dashboard

## Architecture

Zero external services — everything runs inside the existing stack:

```
Python bots → MetricsCollector (in-memory, thread-safe singleton)
                    ↓
Health Checker (background thread, 60s interval)
                    ↓
Flask API (/api/metrics/*) 
                    ↓
Streamlit UI → "🖥️ System Dashboard" expander (top of page)
```

## Components

### 1. Metrics Collector — `src/metrics/collector.py`

Thread-safe singleton (`get_collector()`). Tracks:

| Category | What's Tracked | Key Methods |
|----------|---------------|-------------|
| API calls | Inbound/outbound counts, latency, error rate per endpoint | `record_api_call()`, `get_api_stats()` |
| Errors | Count by type, first/last seen, sample message | `record_error()`, `get_error_stats()` |
| Service health | Status, latency, consecutive failures per service | `update_service_health()`, `get_health_stats()` |
| Trades | Wins/losses, PnL, win rate, recent trades (testnet + paper) | `record_trade()`, `get_trade_stats()` |

**Instrumentation pattern** (add to any API client):
```python
from src.metrics.collector import get_collector
t0 = time.time()
try:
    result = api_call()
    get_collector().record_api_call("outbound", "service/endpoint", success=True, latency_ms=(time.time()-t0)*1000)
except Exception as exc:
    get_collector().record_api_call("outbound", "service/endpoint", success=False, latency_ms=(time.time()-t0)*1000, error=str(exc))
    get_collector().record_error("service_error_type", str(exc))
    raise
```

Already instrumented: `src/api/binance_futures.py` (all GET/POST/DELETE calls).

### 2. Health Checker — `src/metrics/health_checker.py`

Background thread started by the API server. Checks every 60 seconds:

| Service | Check Method | What's Tested |
|---------|-------------|---------------|
| Binance REST (Futures) | `GET /fapi/v1/ping` | demo-fapi.binance.com reachable |
| Binance WebSocket | TCP connect | stream.binance.com:9443 |
| Binance Spot (OHLCV) | `GET /api/v3/ping` | data-api.binance.vision |
| MongoDB Atlas | `admin.command("ping")` | pymongo connection |
| Cloudflare Tunnel | `systemctl is-active` | tunnel service running |
| API Server (local) | `GET /api/status` | Flask server on :5001 |
| Bot: htf-btc | `systemctl is-active` | drl-htf-agent.service |
| Bot: htf-eth | `systemctl is-active` | drl-htf-eth.service |
| Bot: htf-partial | `systemctl is-active` | drl-htf-partial.service |
| Bot: htf-hybrid | `systemctl is-active` | drl-htf-hybrid.service |

To add a new health check: add a function `_check_<name>() -> tuple(status, latency_ms, error)` and register it in `run_health_checks()`.

### 3. API Endpoints — `src/ui/api_server.py`

| Endpoint | Returns |
|----------|---------|
| `GET /api/metrics/overview` | Full system overview (all categories) |
| `GET /api/metrics/api` | API call stats (inbound/outbound) |
| `GET /api/metrics/errors` | Error counts by type |
| `GET /api/metrics/health` | Service health status |
| `GET /api/metrics/health/check` | Run health checks NOW and return results |
| `GET /api/metrics/trades` | Trade statistics (testnet + paper) |

### 4. Streamlit UI — `src/ui/app.py`

`render_system_dashboard()` function, displayed in a collapsible expander at the top of the page (above all trading tabs).

Four sections:
1. **🏥 Service Health** — green/red per service with latency
2. **📡 API Statistics** — calls/min rate, per-endpoint breakdown table
3. **❌ Errors** — error type table with counts and samples
4. **📊 Trade Statistics** — testnet + paper side-by-side with recent trades

## Adding New Metrics

### New API to track
1. Add `get_collector().record_api_call()` in the API client's request method
2. Add `get_collector().record_error()` in error handlers
3. Data automatically appears in the dashboard

### New service health check
1. Add `_check_<name>()` function in `health_checker.py`
2. Register in `run_health_checks()` dict
3. Health status automatically appears in dashboard

### New error type
1. Call `get_collector().record_error("error_type_name", message)` where errors occur
2. Errors automatically appear in the dashboard table

## Files

| Path | Role |
|------|------|
| `src/metrics/__init__.py` | Package init |
| `src/metrics/collector.py` | Thread-safe metrics singleton |
| `src/metrics/health_checker.py` | Background health check thread |
| `src/api/binance_futures.py` | Instrumented with metrics recording |
| `src/ui/api_server.py` | `/api/metrics/*` endpoints + health checker startup |
| `src/ui/app.py` | `render_system_dashboard()` in top-level expander |
