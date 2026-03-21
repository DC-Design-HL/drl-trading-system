# Deploy Log — HF Dev Space (Chen4700/drl-trading-bot-dev)

**Date:** 2026-03-19
**Goal:** Switch HF Space from Docker (full trading stack) to Streamlit SDK (client-only UI)

---

## What Was Done

### 1. README.md frontmatter updated
Changed from Docker SDK to Streamlit SDK:
```yaml
# Before
sdk: docker
app_port: 7860

# After
sdk: streamlit
sdk_version: 1.42.0
app_file: src/ui/app.py
```

### 2. Created requirements-hf.txt
New lightweight requirements file for HF UI-only deployment:
- `streamlit>=1.42.0`
- `streamlit-lightweight-charts>=0.1.0`
- `pandas`, `numpy`, `requests`, `plotly`
- `python-dotenv`, `pymongo`, `dnspython`, `certifi`

**Why pymongo is needed:** `src/data/storage.py` has a bare `import pymongo` at the top level (not guarded). The app falls back to `JsonFileStorage` if `STORAGE_TYPE` is not set to `mongo`, but the import must succeed.

### 3. Created `hf-deploy` branch
- `main` stays untouched (full requirements for local dev)
- `hf-deploy` branch replaces `requirements.txt` with lightweight HF version
- Force-pushed `hf-deploy → main` on `huggingface` remote

### 4. Build Result
- Build completed in ~21 seconds (pip install phase)
- All 11 lightweight deps installed cleanly
- No errors in build log

### 5. Runtime Result
- Stage: **RUNNING**
- SDK: **streamlit** (confirmed via HF API)
- Streamlit startup confirmed: "You can now view your Streamlit app in your browser."

---

## API Tunnel Status

**Tunnel URL:** `https://advertise-refined-doom-bicycle.trycloudflare.com`
**Test:** `curl -s .../api/ping`
**Result:** `{"ok":true,"timestamp":"2026-03-19T13:21:45.580803"}` ✓

The HF Space needs `API_SERVER_URL` set to this tunnel URL as a HF Space variable.
**Check:** Go to the dev space Settings → Variables and verify `API_SERVER_URL = https://advertise-refined-doom-bicycle.trycloudflare.com`

---

## Known Issues / Notes

1. **Cloudflare Tunnel is ephemeral** — The tunnel URL changes every time `cloudflared tunnel` is restarted. When that happens, update `API_SERVER_URL` in the HF Space variables.

2. **storage.py import** — `import pymongo` is at module level in `src/data/storage.py`. Must be in requirements. If the space should truly be API-only with no local storage, this import can be guarded later.

3. **LFS files** — The HF space still contains model `.zip`/`.pkl` files (883 MB via Git LFS). These aren't needed for a client-only space. To save space, consider adding them to `.gitattributes` as not-LFS on the deploy branch, or using `.hfignore`.

4. **Production space** — NOT touched. Only `Chen4700/drl-trading-bot-dev` was modified.

---

## Git State

- Branch `hf-deploy` pushed to `huggingface` remote as `main` (force push)
- `origin` (GitHub) `main` branch unchanged
- Commit: `3857657` — "Deploy: Switch HF Space to Streamlit SDK (client-only)"

---

# Session 2 — 2026-03-19

**Goal:** Fix testnet `/api/testnet/status` hang + push to HF dev space

## Fixes Applied

### 1. BinanceConnector timeout (`src/api/binance.py`)
- Added `'timeout': 10000` (10s) to ccxt config — prevents indefinite hangs on slow/blocked proxy
- Added `'fetchMarkets': False` and `'loadMarketsAlways': False` — disables ccxt's lazy market-loading which could block on testnet's limited exchange info

### 2. Load dotenv in API server (`src/ui/api_server.py`)
- Added `load_dotenv()` at module startup — ensures `.env` vars (incl. `BINANCE_TESTNET_PROXY_URL`) are available for local dev

### 3. Testnet symbol whitelist (`src/ui/api_server.py`)
- Binance spot testnet only supports ~10 pairs (BTC, ETH, BNB, LTC, TRX, XRP, SOL, ADA, DOGE vs USDT)
- When iterating wallet balances, skip ticker fetch for currencies not in this whitelist
- Fixes "binance Invalid symbol" errors that accumulated and made the endpoint appear to hang

## Build Result
- Build logs: CACHED (all deps unchanged, only app files copied) — completed in ~7s
- Streamlit confirmed running: "You can now view your Streamlit app in your browser."
- Commit: `75e6612`

## Verification Results
- **HF Space stage:** PAUSED — flagged as abusive by HF moderation (not a code issue)
  - Space must be manually unpaused at huggingface.co/spaces/Chen4700/drl-trading-bot-dev
- **Cloudflare tunnel:** DOWN (error 1033) — `cloudflared` is not running locally
  - Restart tunnel: `cloudflared tunnel --url http://localhost:5001`
  - Update `API_SERVER_URL` HF Space variable after restart
- **Integration tests:** **22/22 PASSED** (71s) — all endpoints, testnet structure, app.py API migration verified

## Git State
- Branch `hf-deploy` (commit `75e6612`) force-pushed to `huggingface` remote as `main`
- `origin` (GitHub) `main` branch unchanged — fixes remain on `hf-deploy` only
