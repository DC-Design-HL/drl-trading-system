# HF Client-Only Fix Log
Date: 2026-03-19

## Problem
`chen470/drl-trading-bot-dev2` HF Space crashed on load with `KeyError: 'close'` at
`app.py line 1837` because `load_real_market_data` was calling `/api/market` which
returns a single price dict (not OHLCV candlestick data).

Additional issues: multiple server-side imports (`src.features.whale_wallet_registry`,
`subprocess` bot control, `storage.log_trade`, local file deletion) that crash or
silently corrupt state on HF where only `src.ui` and `src.data.storage` are available.

## Fixes Applied

### `src/ui/api_server.py`
- Added `GET /api/ohlcv?symbol=&interval=&limit=` endpoint — fetches real candlestick
  data from Binance `/api/v3/klines` and returns `[{time, open, high, low, close, volume}, ...]`

### `src/ui/app.py`
1. **`IS_CLIENT_MODE` flag** — `bool(os.environ.get('API_SERVER_URL'))`, true on HF
2. **`load_real_market_data`** — now calls `/api/ohlcv` (returns proper DatetimeIndex
   DataFrame with OHLCV columns); fixes the line 1837 `df['close']` crash
3. **`load_trading_log`** — uses `GET /api/trades` in client mode, local storage otherwise
4. **`get_trading_state`** — uses `GET /api/state` + `GET /api/trades` in client mode;
   all `src.features.whale_wallet_registry` imports moved to `_load_whale_alerts_local()`
   (server-only helper with `try/except ImportError` guard)
5. **Trading Controls** — subprocess calls (pgrep/pkill/Popen for bot start/stop) and
   `storage.log_trade()` disabled in client mode; shows info message instead
6. **Clear Trade Log** — local file deletion guarded behind `not IS_CLIENT_MODE`
7. **Live Portfolio tab** — uses `load_trading_log()` (API-aware) instead of
   `storage.get_trades()` directly; bot online-status inferred from trade timestamps
8. **Backtest tab** — moved UI from inside `tab_testnet` (wrong tab) to `tab_backtest`;
   shows graceful "not available in client mode" message on HF

## Verification
- `python3 -c "import src.ui.app"` — no errors (only expected Streamlit warnings)
- `grep "from src\.(backtest|brain|env|features|models|api)"` — only guarded occurrences
- HF container logs after factory restart: ZERO errors, clean startup in ~80s

---

# Live Chart "No market data available" Fix
Date: 2026-03-19

## Problem
The Live Chart tab showed "No market data available" even though `/api/ohlcv` works.

## Root Cause
`load_real_market_data` had a bare `except Exception: pass` that silently swallowed
all errors. On HF the SDK is `streamlit` (no Flask server, only `app_file: src/ui/app.py`),
so there is NO local Flask server at `localhost:5001`. The HTTP request always failed,
the exception was swallowed, and the function returned an empty DataFrame.

## Fix (`src/ui/app.py` — `load_real_market_data`)
1. **Added logging** — replaced `except Exception: pass` with `_logger.warning(...)` so
   failures are visible in HF container logs
2. **Direct Binance fallback** — after the Flask API attempt fails, the function now
   directly calls `https://data-api.binance.vision/api/v3/klines` (the same endpoint
   the Flask server itself uses, no auth required) and parses the raw Binance klines
   format into the same DataFrame schema. This works regardless of whether Flask is up.

## Tests Added (`tests/test_api/test_ohlcv.py` — 27 tests, all passing)
- `/api/ohlcv` endpoint: all symbols (BTCUSDT, ETHUSDT, SOLUSDT), all intervals
  (1m, 5m, 15m, 30m, 1h, 4h, 1d), time-in-seconds validation, error handling
- `load_real_market_data`: returns non-empty DataFrame with correct columns/index,
  falls back to direct Binance when Flask unavailable, handles total failure gracefully
- Chart rendering: does not crash with valid data, returns "No market data available"
  for empty DataFrame, embeds historical candle data before WebSocket connects

## Deploy
- Commit: `43aa9d33` pushed to `hf-dev2 hf-clean:main`
- Factory restart at 2026-03-19 19:45Z
- Build logs: DONE (no errors)
- Run logs: clean Streamlit startup on port 8501, zero errors

---
## 2026-03-19 — Market Analysis / Agent Status / Testnet Fixes

### Issues Fixed

**Issue 1: Market Analysis "Unable to load"**
- Added non-200 HTTP status code handling in `render_market_analysis_fragment`
  (was silently returning `{}`, now shows an error card with HTTP status)

**Issue 2: Agent Status "Model: Not found" / "0.0% Win Rate"**
- Root cause: `model_path.exists()` always False on HuggingFace (model file not deployed to HF Space)
- Fix: In `IS_CLIENT_MODE`, call `/api/model` endpoint instead of checking local filesystem
  — returns `model_exists`, `model_date`, `win_rate`, `total_return`, `total_trades` from server
- Model status and trade stats now reflect actual server-side data

**Issue 3: Testnet "Cannot reach API server"**
- Added `if tn_resp.status_code == 200:` check before `.json()` call
  to handle non-200 responses cleanly (previously would throw on 404/500)

**Issue 4: Short timeouts causing "Current Price: $0.00" / stale data**
- Increased all `timeout=1` to `timeout=5` for API calls in fragments:
  - `render_sidebar_metrics_fragment` (state)
  - `render_position_fragment` (state, trades)
  - `render_agent_status_fragment` (state, trades)
  - `render_position_fragment` trades call (was timeout=2, now 5)

### E2E Tests Added
- `tests/test_integration/test_e2e_api.py` — 14 `@pytest.mark.e2e` tests
  - `/api/state`: non-empty dict, has assets/balance, position fields
  - `/api/market`: price data, regime/whale present, app.py parse simulation
  - `/api/testnet/status`: HTTP 200, all required keys, parse simulation
  - `/api/ohlcv`: 500 candles, candle structure validation
  - `/api/model`: win_rate, total_trades, model_exists fields

### Deploy
- Branch: hf-clean → pushed to hf-dev2/main (commit 08ab9ef)
- Factory restart triggered, container started CLEAN (zero errors)

---

## No-Mock-Data Audit & Fix — 2026-03-19

### Policy
All data points in the UI must be real. No fake/generated/padded data.

### Files Changed
- `src/ui/api_server.py`
- `src/ui/app.py`
- `src/ui/testnet_server.py`
- `src/ui/testnet_client.py`

### Fixes Applied

#### 1. Fake Whale Alerts (CRITICAL)
**api_server.py lines 128-162**: Removed 35-line block that generated up to 50 fake whale transactions with `random.uniform` amounts, `random.choice` chains, `random.randbytes` wallet addresses, and `random.randint` timestamps. Disguised as "seamless backfill for rich visual experience".

**app.py `_load_whale_alerts_local()`**: Removed identical fake generator. Removed `import random as _random`.

After fix: whale alerts show only real on-chain data from `data/whale_wallets/*.json`. If none available, shows empty list.

#### 2. Hardcoded Initial Capital ($5k/asset → state balance)
Removed `initial_capital = max(len(raw_assets), 1) * 5000 if raw_assets else 20000` pattern from:
- `api_server.py` /api/state endpoint (was overriding `total_balance` with fake value)
- `app.py` asset view path
- `app.py` global view path  
- `app.py` sidebar Agent Status section
- `app.py` Live Portfolio tab (was `4 * 5000 = $20,000`)

After fix: balance uses `state.get('total_balance', state.get('balance'))` — real stored state only.

#### 3. Hardcoded 10000 as Initial Capital for Return %
- `api_server.py` /api/model: `total_return` now `None` (cannot compute without real initial capital)
- `api_server.py` /api/testnet: `pnl_pct` and `pnl_usdt` now `None`
- `app.py` Live Portfolio tab: equity curve uses absolute PnL points (not % of fake capital)
- `app.py` Live Portfolio tab: Realized PNL and Open PNL cards now show absolute `$` amounts
- `app.py` per-asset table: PnL% column shows `—` (dollar column still shows real value)
- `app.py` per-asset table: Best/Worst columns show absolute `$` not fake `%`
- `app.py` testnet performance tab: Total Return shows "N/A"
- `app.py` sidebar: `portfolio_balance` default of 10000 removed, shows `—` if unavailable
- `testnet_server.py`: P&L % metric shows "N/A"
- `testnet_client.py`: JS `initialValue = 10000` removed, P&L % shows "N/A"

#### 4. Manual Trade Log Balance
`app.py` Open Long / Open Short / Close Position buttons: `state.get('balance', 10000)` → `state.get('balance')` — no fake 10000 logged in trade records.

### Verification
```
grep -r "random\.uniform|random\.choice|needed_mock|initial_capital|backfill.*gap" src/ui/
# → No matches found
```

### Deployment
- Commit: `faa23b1`
- Branch: `hf-clean` → force pushed to `hf-dev2/main`
- Space status: RUNNING, sha matches, domain READY
