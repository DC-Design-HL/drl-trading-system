---
name: hf-client-debug
description: Debug and fix HuggingFace Spaces client-server architecture issues for the DRL trading system. Use when the HF Space UI shows errors like connection refused, module not found, or data not loading. The HF Space is a CLIENT-ONLY Streamlit UI — all data and logic must come from the remote API server via API_SERVER_URL. Common issues include hardcoded localhost URLs, server-side imports on the client, and local filesystem access attempts.
---

# HF Client Debug

The DRL trading system uses a split architecture:
- **Local server**: Flask API on port 5001, exposed via Cloudflare Tunnel
- **HF Space**: Streamlit client-only UI, connects to server via `API_SERVER_URL` env var

## Common Failure Patterns

### 1. Connection Refused to localhost:5001
**Symptom**: `HTTPConnectionPool(host='127.0.0.1', port=5001): Connection refused`
**Cause**: Hardcoded `localhost:5001` URLs in `app.py` instead of using `get_api_url()`
**Fix**: Search entire `app.py` for `127.0.0.1:5001` and `localhost:5001`, replace with `get_api_url()`

### 2. Module Not Found (ccxt, stable_baselines3, etc.)
**Symptom**: `ModuleNotFoundError: No module named 'ccxt'`
**Cause**: Client-side code imports server-side modules (trading, ML, exchange libs)
**Fix**: Move the functionality to a server-side API endpoint. Client calls the endpoint via HTTP instead of importing directly.

### 3. Local Filesystem Access Fails
**Symptom**: `Storage Path: /app/logs/... Exists: False`
**Cause**: Client tries to read local files that only exist on the server
**Fix**: Expose the data through API endpoints; client reads via HTTP

## Pre-Push Checklist (MANDATORY before reporting "deployed")

1. Search `app.py` for ANY direct imports from `src/` that require heavy deps — wrap in try/except
2. Search for hardcoded `127.0.0.1` or `localhost`
3. Verify `get_api_url()` returns `API_SERVER_URL` env var on HF
4. Test each API endpoint from the tunnel URL

## Post-Push Verification (MANDATORY before telling Chen it works)

1. Wait for build to complete (check build logs)
2. Check container logs for ANY errors or tracebacks
3. If space shows RUNNING_APP_STARTING or RUNNING, wait until fully RUNNING
4. **Factory restart** after any code push to clear Streamlit cache
5. Check container logs AGAIN after restart
6. **Load the UI page** and check for client-side errors (KeyError, ModuleNotFoundError, etc.)
7. **Interact with each tab** — verify Market Analysis, Testnet, and other tabs render without errors
8. If ANY error is found in logs or UI, fix it and re-push. Repeat until clean.
9. Only report "deployed and working" if container logs AND UI are completely error-free

## Auto-Fix Loop
After every deploy, Claude Code must:
1. Check container logs for errors
2. If errors found → fix code → push → factory restart → check again
3. Keep looping until zero errors
4. This is non-negotiable — never report success with known errors

## Debugging Checklist

1. Check HF container logs for import errors or crashes
2. Search `app.py` for ANY direct imports from `src/` that require heavy deps
3. Search for hardcoded `127.0.0.1` or `localhost`
4. Verify `get_api_url()` returns `API_SERVER_URL` env var on HF
5. Test each API endpoint from the tunnel URL
6. After every fix, push and verify build + container logs

## UI Feature Verification (check after every deploy)

After container logs are clean, verify these UI features work:
1. **Live Chart** — candle chart renders with real data (not "No market data available")
2. **Live Portfolio** — shows positions and PnL
3. **Performance** — metrics render
4. **Testnet tab** — connects to Binance testnet, shows balance
5. **Market Analysis** — shows analysis data from API, no connection errors

If any feature shows empty/error state, check the corresponding API endpoint and fix.

## Tunnel URL Management
- Cloudflare quick tunnels change URL on restart
- The `drl-tunnel-watcher` service auto-updates the HF Space variable
- **CRITICAL**: Watcher must point to the CURRENT dev space (chen470/drl-trading-bot-dev2)
- If tunnel URL changes, the HF Space must be restarted to pick up the new env var
- After any tunnel restart, verify: `curl -s $TUNNEL_URL/api/ping` returns ok

## E2E Integration Tests (must pass before deploy)
1. `/api/ping` returns ok
2. `/api/state` returns non-empty state with assets, positions, balance
3. `/api/market?symbol=BTCUSDT` returns price data
4. `/api/ohlcv?symbol=BTCUSDT&interval=1h&limit=10` returns candle array
5. `/api/testnet/status` returns connected:true with balances
6. `/api/trades` returns trade list
7. Market Analysis section parses /api/state correctly
8. Testnet tab parses /api/testnet/status correctly
9. Live Chart gets OHLCV data and renders (no empty DataFrame)

## Principle
**The HF client should ONLY need**: streamlit, requests, pandas, numpy, plotly, python-dotenv
**Everything else** (ccxt, stable-baselines3, torch, sklearn, etc.) stays server-side only.
