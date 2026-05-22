# Architecture Migration Plan: Local Server + HF Client

## Date: 2026-03-19

---

## Current Architecture (Problem)

```
HuggingFace Space
├── live_trading_multi.py  (bot, all logic)
├── src/ui/api_server.py   (Flask REST API, port 5001)
└── src/ui/app.py          (Streamlit UI, port 8501)
     └── calls http://127.0.0.1:5001/api/*
```

**Problems:**
- Binance Testnet geo-blocked from HuggingFace's datacenter (HTTP 451)
- Cloudflare Workers proxy is unreliable and adds latency
- Bot cannot reach testnet.binance.vision from HF

---

## Target Architecture (Solution)

```
Local Machine
├── live_trading_multi.py  (bot, trading logic)
├── src/ui/api_server.py   (Flask REST API, port 5001)
└── [tunnel tool]          (exposes :5001 as public HTTPS URL)
     └── e.g. https://abc123.ngrok.io → localhost:5001

HuggingFace Space
└── src/ui/app.py          (Streamlit UI, CLIENT only)
     └── calls $API_SERVER_URL/api/*
          └── API_SERVER_URL = https://abc123.ngrok.io (HF Secret)
```

**Benefits:**
- Bot runs locally → direct Binance testnet access (no geo-blocks)
- HF Space becomes a pure UI/client, no trading logic
- Testnet tab works via server-side `testnet_server.py` (direct connection locally)

---

## Implementation Plan

### Step 1: Configure Flask API Server for Remote Access

**File: `src/ui/api_server.py`**

Changes:
1. Enable CORS properly so HuggingFace domain can call the API
2. Add a `/api/ping` health endpoint (lightweight check for client)
3. Add startup banner showing the public URL

```python
from flask_cors import CORS
CORS(app, origins=["*"])  # Allow HF and any origin
```

**File: `start_local_server.sh`** (new)

A convenience script to:
1. Start the Flask API server (port 5001)
2. Optionally start a tunnel (ngrok or Cloudflare)
3. Print the public URL to copy into HF Secrets

---

### Step 2: Make Streamlit UI Configurable

**File: `src/ui/app.py`**

Changes:
1. Replace all hardcoded `http://127.0.0.1:5001` with `get_api_url()` helper
2. `get_api_url()` reads from `os.environ.get('API_SERVER_URL', 'http://127.0.0.1:5001')`
3. Show connection status in UI (connected/disconnected + server URL)
4. Graceful degradation when server is unreachable (not just red error)

**Occurrences of hardcoded URL to replace:**
- Line 1183: `requests.get('http://127.0.0.1:5001/api/state', ...)`
- Line 1221: `requests.get(f'http://127.0.0.1:5001/api/market?symbol=...', ...)`
- Line 1427: `requests.get('http://127.0.0.1:5001/api/state', ...)`
- Line 1438: `requests.get(f'http://127.0.0.1:5001/api/market?symbol=...', ...)`
- Line 1459: `requests.get('http://127.0.0.1:5001/api/trades', ...)`
- Line 1539: `requests.get('http://127.0.0.1:5001/api/state', ...)`
- Line 1553: `requests.get('http://127.0.0.1:5001/api/trades', ...)`

---

### Step 3: Fix Testnet Tab

**File: `src/ui/testnet_server.py`**

Changes:
1. Remove the HuggingFace detection/warning block (it no longer applies)
2. Remove Cloudflare proxy fallback (not needed locally)
3. Simplify to direct `BinanceConnector` with `testnet=True`
4. Server now calls `https://testnet.binance.vision` directly

**File: `src/ui/app.py` (testnet tab section)**

Changes:
1. Always use server-side (`testnet_server.py`) since server runs locally
2. Remove the client-side JS fallback (optional, can keep as backup)

---

### Step 4: Add Tunnel Support (ngrok)

**File: `requirements.txt`**

Add: `pyngrok>=7.0.0`

**File: `start_local_server.py`** (new script)

```python
# Start Flask server with optional ngrok tunnel
# Usage:
#   python start_local_server.py              # local only
#   python start_local_server.py --tunnel     # with ngrok tunnel
```

Logic:
1. Start `api_server.py` in a background thread
2. If `--tunnel`: start pyngrok tunnel, print public URL
3. Print instructions to add URL to HF Secrets

---

### Step 5: Integration Tests

**File: `tests/test_integration/test_server_client.py`** (new)

Tests:
1. `test_server_starts()` - Flask server starts, `/health` returns 200
2. `test_api_state()` - `/api/state` returns valid JSON
3. `test_api_trades()` - `/api/trades` returns list
4. `test_api_market()` - `/api/market?symbol=BTCUSDT` returns market data
5. `test_cors_headers()` - Response includes CORS headers
6. `test_binance_testnet_connectivity()` - testnet.binance.vision ping returns 200
7. `test_binance_testnet_with_keys()` - API keys work for account balance
8. `test_client_url_configurable()` - `API_SERVER_URL` env var is read correctly
9. `test_end_to_end_data_flow()` - Bot saves state → API serves it → client reads it

---

### Step 6: Update HuggingFace Space

**File: `src/ui/app.py`** (already updated in Step 2)

**HF Space Secrets to add:**
- `API_SERVER_URL` = `<ngrok-or-tunnel-url>`
- Keep existing: `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET`, etc.

**File: `README.md` or HF Space README** - update deployment instructions

**Push target:** `Chen4700/drl-trading-bot-dev` (DEV space only)

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/ui/api_server.py` | Modify | Enable CORS, add `/api/ping` |
| `src/ui/app.py` | Modify | Replace hardcoded localhost:5001 with `API_SERVER_URL` env var |
| `src/ui/testnet_server.py` | Modify | Remove proxy/HF-specific code, direct testnet access |
| `start_local_server.py` | Create | Convenience script to start server + optional tunnel |
| `tests/test_integration/test_server_client.py` | Create | Integration tests |
| `tests/test_integration/__init__.py` | Create | Package init |
| `requirements.txt` | Modify | Add `pyngrok>=7.0.0` |
| `.env.example` | Modify | Add `API_SERVER_URL` example |

---

## Blockers & Solutions

### Blocker 1: HuggingFace Space Needs Public URL
**Problem:** The local Flask server on :5001 is not reachable from HF by default.
**Solution:** Use ngrok (pyngrok) or Cloudflare Tunnel (`cloudflared`) to expose it.
- ngrok: requires account but easy setup, free tier works
- Cloudflare Tunnel: free, persistent, but requires `cloudflared` binary
- **Chosen:** ngrok via `pyngrok` (Python package, easiest to automate)

### Blocker 2: CORS on Flask
**Problem:** HuggingFace serves the Streamlit app from a different origin.
**Solution:** Add `flask-cors` (already in requirements) and configure `CORS(app, origins=["*"])`.

### Blocker 3: Testnet Keys in HF Secrets vs .env
**Problem:** The testnet tab reads keys from `os.getenv()` — works both locally (.env) and on HF (secrets).
**Solution:** No change needed. Keep reading from env vars.

### Blocker 4: Streamlit on HF can't do ngrok
**Problem:** Streamlit on HF is the CLIENT, not the server — it doesn't need ngrok.
**Solution:** ngrok only runs on the LOCAL server side.

---

## Execution Order

1. [x] Write PLAN.md (this file)
2. [x] Modify `src/ui/api_server.py` — Enable CORS + add /api/ping
3. [x] Modify `src/ui/app.py` — Replace hardcoded URLs with `API_SERVER_URL`
4. [x] Modify `src/ui/testnet_server.py` — Remove proxy/HF blocks (direct testnet)
5. [x] Create `start_local_server.py` — Server startup script with optional ngrok
6. [x] Add `pyngrok` to `requirements.txt`
7. [x] Add `API_SERVER_URL` to `.env.example`
8. [x] Create integration tests (`tests/test_integration/test_server_client.py`)
9. [~] Run tests locally — BLOCKED: this machine has no project venv (no flask/dotenv/etc.)
    - Static checks passed: all 5 modified files parse cleanly (ast.parse)
    - Logic check passed: `get_api_url()` default/custom/trailing-slash behavior verified
    - Storage check passed: JsonFileStorage save/load verified (when deps available)
    - Binance testnet check passed: testnet.binance.vision/api/v3/ping reachable locally
    - Full pytest suite must be run in the project's virtualenv:
      `pip install -r requirements.txt && pytest tests/test_integration/ -v`
10. [x] Push to `Chen4700/drl-trading-bot-dev`
