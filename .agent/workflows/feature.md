---
description: Implement a new feature in the trading system or dashboard
---

# Feature Development Workflow

// turbo-all

## 1. Plan (PLANNING mode)
- Understand the requirement fully before writing any code
- Identify which files need to change:
  - `src/features/` → new signals, analyzers
  - `src/ui/app.py` → dashboard UI changes
  - `src/ui/api_server.py` → new API endpoints
  - `live_trading_multi.py` → bot logic changes
- Create a brief implementation plan (what changes, which files, why)
- Ask the user ONE clarifying question if scope is unclear — do not proceed blind

## 2. Implement (EXECUTION mode)
- Work file by file in dependency order (backend before frontend)
- For new analyzers: follow the pattern in `src/features/order_flow.py`
  - Constructor with cache
  - `_get_request_config()` for proxy support
  - OKX fallback if using Binance Futures API
- For new dashboard cards: follow the pattern in `render_market_analysis_fragment()`
- For new API endpoints: add to `src/ui/api_server.py` with a 60s cache if it calls external APIs

## 3. Test Locally
- Test new backend code with the venv:
  ```
  BINANCE_PROXY=$(grep BINANCE_PROXY .env | cut -d= -f2) ./venv/bin/python3 -c "..."
  ```
- Verify the feature works end-to-end before deploying

## 4. Deploy
```
git add <changed files>
git commit -m "Feature: <short description>"
git push origin main
```

## 5. Verify on HF
- Wait ~90s for build to complete
- Check the new feature on the live dashboard
- Confirm no existing features broke

## Done Criteria
- Feature works on the live HF dashboard
- No regressions in existing functionality
- Proxy/bandwidth usage is minimal (caching in place)
