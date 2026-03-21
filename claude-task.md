Implement real LONG/SHORT trading on Binance Testnet with full visibility in the Testnet tab.

## Context
- The DRL trading bot already makes LONG/SHORT decisions (PPO agent + composite scorer + 3-tier decision system)
- Currently the bot's trades are DRY RUN only (logged but not executed on the exchange)
- Binance Testnet API keys are in .env: BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET
- The testnet connects to https://testnet.binance.vision/
- The server runs locally and can reach Binance testnet directly (no proxy needed)
- The BinanceConnector class in src/api/binance.py handles exchange connectivity

## Requirements

### 1. Testnet Trade Execution
- Mirror the EXACT same trading logic the bot uses for dry-run trades (from live_trading_multi.py)
- When the bot decides to OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT — execute the same trade on Binance testnet
- Use the same position sizing, SL/TP logic, split entry (50% market + 50% limit)
- Store testnet trades separately from the bot's dry-run trades (different MongoDB collection: testnet_trades)
- Each testnet trade should record: symbol, action, side, price, amount, order_id, timestamp, pnl, sl, tp, confidence

### 2. Testnet Trade API Endpoints (add to api_server.py)
- GET /api/testnet/trades — all testnet trade history
- GET /api/testnet/positions — current open positions on testnet
- GET /api/testnet/pnl — realized + unrealized PNL from testnet trades
- POST /api/testnet/execute — manually trigger a testnet trade (for testing)

### 3. Testnet Tab UI (update app.py testnet section)
Show full visibility:
- **Open Positions table**: symbol, side (LONG/SHORT), entry price, current price, unrealized PNL, SL, TP
- **Trade History table**: all executed testnet trades with timestamp, symbol, action, price, amount, PNL, order_id
- **PNL Summary**: total realized PNL, total unrealized PNL, win rate, total trades
- **Equity Curve chart**: plot cumulative PNL over time from testnet trades
- **Live Order Book**: show any open/pending orders on testnet
- Make sure ALL data comes from the API endpoints (client-mode compatible)

### 4. Auto-Execution Hook
- Add a hook in the trading loop (live_trading_multi.py or api_server.py) that:
  - Listens for bot trade decisions
  - Mirrors each decision to testnet in real-time
  - Logs both the bot's dry-run result and the testnet execution result
- This should be toggleable via env var: TESTNET_MIRROR=true/false

### 5. Integration Tests
- Test testnet trade execution (place a small market buy on testnet)
- Test /api/testnet/trades returns trade history
- Test /api/testnet/positions returns open positions
- Test PNL calculation from testnet trades

## Important Rules
- NO mock/fake data — all trades must be real testnet executions
- NO hardcoded values — all prices, amounts from real API responses
- Guard all None values in f-strings (ui-type-safety skill)
- Push to HF dev space after changes
- Check container logs — auto-fix loop until zero errors

## Git/Deploy
- Work on hf-clean branch
- Push to chen470/drl-trading-bot-dev2
- HF_TOKEN from .env
- Factory restart after push
- Verify container logs clean

Write plan to TESTNET_TRADING_PLAN.md first, then implement.
