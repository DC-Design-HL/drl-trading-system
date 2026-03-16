# Cloudflare Proxy Timestamp Fix

## Problem
When using the Cloudflare Workers proxy for Binance testnet API, requests were failing with:
```
{"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}
```

## Root Cause
- Binance API requires signed requests to have a timestamp within a `recvWindow` (default: 5000ms)
- When requests go through Cloudflare proxy, there's additional network latency
- By the time the request reaches Binance, the timestamp is too old and gets rejected

## Solution
Added `recvWindow: 60000` (60 seconds) to the ccxt configuration in `src/api/binance.py`:

```python
config = {
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': rate_limit,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
        'recvWindow': 60000,  # 60 seconds - max allowed by Binance (handles proxy latency)
        'fetchCurrencies': False,
        'fetchMarkets': True,
    }
}
```

## What This Does
- Increases the validity window for signed requests from 5 seconds to 60 seconds
- 60 seconds is the maximum allowed by Binance API
- All authenticated API calls (account, orders, trades) will now include `recvWindow=60000`
- Handles proxy latency while still maintaining security

## Testing
To verify the fix works:
1. Restart the Streamlit app (it should auto-reload)
2. Go to the Testnet tab
3. Check if the balance call succeeds
4. Look for: ✅ instead of ❌ for "Failed to get balances"

## Files Changed
- `src/api/binance.py` - Added recvWindow parameter

## References
- Binance API Docs: https://binance-docs.github.io/apidocs/spot/en/#timing-security
- recvWindow: Maximum 60000ms (60 seconds)
