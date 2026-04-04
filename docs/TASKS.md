# TASKS.md — Active Task Tracker

## In Progress

### TASK-022: Wyckoff Full Implementation
- **Requested**: 2026-04-02 17:04 UTC
- **Status**: ⏸️ ON HOLD — detector done, backtest inconclusive, need more data
- **Last update**: 2026-04-02 20:47 UTC
  - ✅ Detector: 14/14 events, 12/12 tests, all 4 symbols
  - ✅ Backtest: 47 trades on 4H/1H/15m — not ready as guard
  - ✅ Docs: `docs/wyckoff-analysis-summary.md` + skill updated
  - ⏸️ ON HOLD: resume when 500+ labeled trades available (~2-3 months)

### TASK-023: ADX Exhaustion Guard Decision
- **Requested**: 2026-04-02 15:24 UTC
- **Status**: ⏳ COLLECTING DATA — Decision at Apr 4 15:24 UTC
- **Last update**: 2026-04-02 15:24 UTC
  - ✅ Full ADX band analysis done (53 trades)
  - ✅ Proposal doc: `docs/adx-exhaustion-guard-proposal.md`
  - ✅ Reminder set for Apr 4 15:24 UTC
  - ⏳ Collecting 48h more data before go/no-go

### TASK-024: Apr 2 Guard Deployments
- **Requested**: 2026-04-02 07:28 UTC
- **Status**: ✅ DEPLOYED
- **Last update**: 2026-04-02 15:05 UTC
  - ✅ Rescue override disabled (RESCUE_ENABLED=False)
  - ✅ ADX guard raised to 20 (ADX_GUARD_MIN=20)
  - ✅ Trailing activation +1%→+0.5% (TRAILING_BREAKEVEN_PCT=0.005)
  - ✅ Trailing distance 0.5%→0.3% (TRAILING_DISTANCE_PCT=0.003)
  - ✅ SQLite storage replacing broken MongoDB
  - ✅ All pushed to dev, all 4 bots restarted

### TASK-025: Whale Model Retrain
- **Requested**: 2026-04-02 11:53 UTC
- **Status**: ⏳ WAITING — Chen to train on Mac M3
- **Last update**: 2026-04-02 14:50 UTC
  - ✅ Stale wallet investigation complete (Robinhood dead since Feb 14)
  - ✅ 1,750 new labeled sequences pushed to dev
  - ✅ Reminder set for Apr 2 18:50 UTC
  - ⏳ Chen will pull, train, and push new model

### TASK-019: SOL Model Training + Deployment
- **Requested**: 2026-03-29 20:40 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-30 09:50 UTC
  - ✅ SOL 15m data downloaded: 70,081 bars (Mar 2024 → Mar 2026, $71-$295)
  - ✅ Walk-forward complete: 9/9 folds, Avg OOS Sharpe 5.47, +21.4% return, 100% positive
  - ✅ Best fold: #3 (Sharpe 13.37, +56.9% return)
  - ✅ systemd service `drl-htf-sol` created, enabled, running
  - ✅ Loaded best model fold_03, WebSocket live, SOL @ $84.21

### TASK-020: XRP Model Training + Deployment
- **Requested**: 2026-03-30 09:51 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-30 16:08 UTC
  - ✅ XRP 15m data: 70,081 bars (Mar 2024 → Mar 2026, $0.62-$3.40)
  - ✅ Trained on Chen's Mac M3 Pro (walk-forward, 9 folds)
  - ✅ Results: Avg OOS Sharpe 6.44, +56.6% return, 100% positive folds
  - ✅ Best fold: #3 (Sharpe 12.4, +130% return)
  - ✅ systemd service `drl-htf-xrp` created, enabled, running
  - ✅ First tick: XRP @ $1.34, WebSocket live
  - ⚠️ Stopped paper trade bots (hybrid/partial) to free memory for XRP

### TASK-021: Championship Model Shadow Comparison
- **Requested**: 2026-03-30 13:16 UTC
- **Status**: ✅ DEPLOYED + RUNNING
- **Last update**: 2026-03-31 06:32 UTC
  - ✅ Championship training complete on Mac M3: 5.4M steps, 101 min, 12 folds × 3 phases
  - ✅ Final metrics: OOS Sharpe 2.1, Win Rate 58%, Calibration Error 0.04
  - ✅ Shadow comparator (`championship_shadow.py`) — compares championship vs live bot every 15min
  - ✅ Logs to MongoDB `championship_shadow` collection + JSONL
  - ✅ systemd service `drl-championship-shadow` running
  - ✅ First comparison: Live=LONG(97%), Champ=HOLD(40%) — championship more conservative
  - ✅ Fixed ensemble confidence calibrator bug (list→ndarray)
  - ⚠️ Service was stopped since Mar 30 16:07 — **RESTARTED** on Mar 31 06:32
  - ⏳ Collecting comparison data — will analyze after ~50+ data points

### TASK-017: Whale Behavior LSTM — Training + Alert Integration + Live Data
- **Requested**: 2026-03-29 05:07 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-29 10:30 UTC
  - ✅ Skill updated: `skills/whale-behavior-prediction/SKILL.md`
  - ✅ Gradient accumulation added for memory-efficient training
  - ✅ MPS (Apple Silicon) support for Mac training
  - ✅ Model trained on Chen's M3 Mac — SELL 73%, BUY 17%, Dir 58.5%
  - ✅ `WhaleIntentPredictor` created — reads live wallet data
  - ✅ Trade alerts show whale behavior confidence (DISPLAY ONLY, not decision-making)
  - ✅ Real-time WebSocket collector via Alchemy (`drl-whale-ws` service)
  - ✅ Replaced hourly Etherscan poller — now instant detection
  - ✅ Live data flowing: Binance Hot Wallet txs detected within seconds
  - ⏳ Monitoring phase: watching correlation between whale signals and trade outcomes

### TASK-001: Fix HF Space UI + Testnet + Push & Verify
- **Requested**: 2026-03-19 13:32 UTC
- **Status**: ✅ DEPLOYED + VERIFIED — All tabs fixed, container logs clean
- **Last update**: 2026-03-19 20:09 UTC
  - ✅ Market Analysis: reads model info from /api/model, error cards for failures
  - ✅ Testnet: proper error handling + status code checks
  - ✅ Agent Status: model info from API (not local filesystem)
  - ✅ Timeouts increased 1s→5s for remote API calls
  - ✅ 14 E2E integration tests added (test_e2e_api.py)
  - ✅ Tunnel watcher fixed to point to correct space
  - ✅ Container logs: ZERO errors
  - ✅ All mock/fake data removed (8 instances across 4 files)
  - ✅ Mock data audit: zero mock generators remain in src/ui/
  - ⏳ Waiting for Chen to confirm all tabs show real data only
- **Space URL**: https://huggingface.co/spaces/chen470/drl-trading-bot-dev2

### TASK-003: Testnet Real Trading + Full Visibility
- **Requested**: 2026-03-19 22:00 UTC
- **Status**: ✅ DEPLOYED — UI moved to local server, HF no longer needed
- **Last update**: 2026-03-21 05:35 UTC
  - ✅ TESTNET_MIRROR=true activated
  - ✅ /api/testnet/trades — working
  - ✅ /api/testnet/pnl — working
  - ✅ Streamlit UI running as systemd service (drl-trading-ui) on port 8501
  - ✅ Caddy reverse proxy: http://116.203.196.107 → Streamlit UI
  - ✅ API routes: http://116.203.196.107/api/* → Flask server
  - ⏳ HTF bot not yet started (needs `live_trading_htf.py` to run)

### TASK-004: Model Retraining Plan (Anti-Overfitting) + Algorithm Upgrade
- **Requested**: 2026-03-20 05:29 UTC
- **Status**: ✅ COMPLETE — BTC comparison done, QRDQN wins
- **Last update**: 2026-03-20 10:45 UTC
- **Deliverables**:
  - ✅ `RETRAINING_PLAN.md` — Full overfitting analysis + walk-forward methodology
  - ✅ `ALGORITHM_RESEARCH.md` — Algorithm comparison research
  - ✅ `download_historical_data.py` — 3-year data downloader
  - ✅ `train_walkforward_v2.py` — Production walk-forward script (PPO + RecurrentPPO + QRDQN)
  - ✅ Historical data: 26,304 candles × 4 assets (Mar 2023 → Mar 2026)
  - ✅ sb3-contrib installed (RecurrentPPO + QRDQN)
- **BTC Comparison Results** (OOS walk-forward, 8 folds):
  - QRDQN: Sharpe 0.59, +2.6% return (WINNER)
  - PPO: Sharpe 0.09, +0.6% return
  - RecurrentPPO: Sharpe -0.60, 0.0% return

### TASK-005: Multi-Timeframe Hierarchical DRL System
- **Requested**: 2026-03-20 10:30 UTC
- **Status**: 🔄 INTEGRATION COMPLETE — Server live, HF blocked (same flag issue)
- **Last update**: 2026-03-21 05:20 UTC
- **Concept**: 1D→4H→1H→15M hierarchical deep-dive (trend on daily, trigger on 15M)
- **Deliverables (training)**:
  - ✅ `src/features/htf_features.py` (1,462 lines) — 117-dim feature engine across 4 TFs
  - ✅ `src/env/htf_env.py` (701 lines) — Multi-TF gymnasium environment
  - ✅ `src/brain/htf_agent.py` (651 lines) — Hierarchical agent with curriculum training
  - ✅ `train_htf.py` (651 lines) — Training script with auto data download + resampling
  - ✅ `train_htf_walkforward.py` (960 lines) — Walk-forward validation script
- **Deliverables (integration — NEW)**:
  - ✅ `live_trading_htf.py` — Live HTF bot (loads best fold, 50% position, testnet mirror)
  - ✅ API endpoints: `/api/htf/status`, `/api/htf/trades`, `/api/htf/performance`
  - ✅ UI tab "🔮 HTF Agent" added to `app.py`
  - ✅ All 3 files syntax-verified, committed, pushed to HF
  - ✅ API server restarted — all 3 endpoints responding
  - ⏳ Needs systemd service file (drl-htf-agent.service) to run HTF bot as daemon
  - ✅ UI moved to local server — no longer depends on HF
- **50% Position Size Results (8 folds OOS)**:
  - Avg Sharpe: 3.85, Avg Return: +14.8%/2mo, MaxDD: 5.95%, Positive: 87.5%

### TASK-007: Exchange-Side OCO Orders for Testnet SL/TP
- **Requested**: 2026-03-22 06:08 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-23 12:00 UTC
  - ✅ Migrated from legacy orders to Binance Algo Order API (`/fapi/v1/algoOrder`)
  - ✅ SL/TP visible in Binance testnet UI as proper TP/SL
  - ✅ TIF GTE bug fixed (removed timeInForce for market-type algos, added retry+delay)
  - ✅ -4130 stale order conflict handled (clear stale algos and retry)
  - ✅ Only updates changed orders (skip if price diff < $0.01)
  - ✅ Validated: cancel+replace does NOT affect entry price (6 tests on live exchange)
  - ✅ Skill: `skills/binance-algo-orders/SKILL.md`

### TASK-008: WebSocket Real-Time SL/TP for Paper Trade Bots
- **Requested**: 2026-03-22 06:13 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-23 04:32 UTC
  - ✅ WS monitor thread on all bots, real-time SL/TP on every tick
  - ✅ Services restarted and running

### TASK-009: ETH Model Training + Deployment
- **Requested**: 2026-03-22 07:17 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-22 12:53 UTC
  - ✅ ETH 15m data downloaded (70,301 bars, Mar 2024 → Mar 2026)
  - ✅ 9/9 folds complete — Avg Sharpe 6.58, Avg Return +19.8%, 100% positive folds
  - ✅ Best fold: #8 (Sharpe 9.90, +34.2% return, 4.1% MaxDD)
  - ✅ live_trading_htf.py updated: --symbol flag, HTF_SYMBOL env var, symbol-aware model finder
  - ✅ WebSocket stream uses correct symbol (ethusdt@aggTrade)
  - ✅ systemd service `drl-htf-eth` created and running
  - ✅ First trade: OPEN_LONG @ $2,080.05 (conf 0.97, SL $2,048, TP $2,142)

### TASK-010: Futures Testnet Migration
- **Requested**: 2026-03-22 12:17 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-22 13:38 UTC
  - ✅ Futures API keys saved in .env (demo-fapi.binance.com)
  - ✅ BinanceFuturesConnector: direct REST, HMAC signing, all endpoints
  - ✅ FuturesTestnetExecutor: real LONG/SHORT, SL/TP, position sync, portfolio from exchange
  - ✅ API server: /api/testnet/* endpoints use futures exchange data
  - ✅ TestnetExecutor delegates to FuturesTestnetExecutor
  - ✅ _mirror_testnet skips CLOSE (exchange handles exits autonomously)
  - ✅ 92 tests — all passing
  - ✅ TP fallback: LIMIT reduceOnly on demo-fapi (STOP_MARKET not supported)
  - ✅ SL handled by bot WebSocket (demo-fapi limitation)
  - ✅ Skills: binance-futures-testnet, testnet-autonomous-exits, testnet-real-portfolio
  - ✅ Validated: UI data matches Binance Futures API exactly

### TASK-011: BOS/CHOCH Dynamic SL/TP
- **Requested**: 2026-03-23 08:02 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-23 08:14 UTC
  - ✅ `src/signals/bos_choch.py` (565 lines) — detection module
  - ✅ Integrated into all 4 bots via `_check_sl_tp()`
  - ✅ Only adjusts when position is profitable
  - ✅ Multi-timeframe: 15m + 1H + 4H
  - ✅ Research: `research/bos_choch_deep_research.md`
  - ✅ Skill: `skills/bos-choch-dynamic-sltp/SKILL.md`

### TASK-012: Direct Telegram Trade Alerts
- **Requested**: 2026-03-23 14:00 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-23 19:10 UTC
  - ✅ `trade_alerter.py` — zero AI token usage, direct Telegram Bot API
  - ✅ systemd service `drl-trade-alerter` running + enabled
  - ✅ Alert types: OPEN, CLOSE, PARTIAL, SL/TP UPDATE, LIQUIDATION_RISK
  - ✅ Includes position size in USDT
  - ✅ Only testnet alerts — paper trade (hybrid/partial) skipped
  - ✅ Skill: `skills/trade-alerts/SKILL.md`

### TASK-013: Liquidation Safety Check
- **Requested**: 2026-03-23 14:50 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-23 14:56 UTC
  - ✅ SL vs liquidation validation every iteration (1% delta buffer)
  - ✅ Telegram alerts on liquidation risk
  - ✅ Applied to all 4 bots
  - ✅ Skill: `skills/liquidation-safety/SKILL.md`

### TASK-016: Market Signal Gate for Low-Confidence Trades
- **Requested**: 2026-03-27 18:29 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-27 18:44 UTC
  - ✅ Signal gate implemented in `live_trading_htf.py`
  - ✅ Tier 1 (conf ≥ 0.80): model autonomous
  - ✅ Tier 2 (conf < 0.80): needs 2/4 signal confirmations
  - ✅ 4 signal sources verified (MTF, Order Flow, Regime, Orderbook) — all producing real data
  - ✅ Backtested against 18 trades over 3 days — would have blocked 3 losers totaling -$51
  - ✅ Also gates reversals, not just flat→open entries
  - ✅ Fail-open: if API is down, trade is allowed (safety)
  - ✅ Both BTC + ETH bots restarted
  - ✅ Pushed to dev (commit 9d39999)

### TASK-015: Ghost ETH SL Alerts + ETH Signal Investigation
- **Requested**: 2026-03-26 11:36 UTC
- **Status**: 🔄 IN PROGRESS — Ghost alerts fixed, investigation complete, awaiting Chen's decision on VecNormalize fix
- **Last update**: 2026-03-26 11:51 UTC
  - ✅ Ghost SL alerts: fixed `_sync_position_from_exchange()` to detect stale positions and reset to flat
  - ✅ Exchange sync now runs every iteration (not just on trade open)
  - ✅ ETH bot restarted — stale SHORT detected and reset to FLAT
  - ✅ Pushed to dev (commit 9ca78a8)
  - ✅ Deep signal investigation: `docs/eth-signal-investigation.md`
  - 🔴 ROOT CAUSE: VecNormalize broken — model receives raw obs instead of normalized (shape 117 vs 4 mismatch)
  - 🔴 Also found: `compute_alignment()` bug — compares 1H with itself instead of 15M
  - ✅ VecNormalize FIXED — was using CartPole (4-dim) dummy env, now uses correct 117-dim env
  - ✅ Both BTC + ETH bots confirmed loading VecNormalize correctly
  - ✅ Added ranging regime filter: ADX < 20 → require confidence > 0.80
  - ✅ Added momentum exhaustion filter: skip entry if price > 3 ATR from 20-bar VWAP
  - ✅ Both bots restarted, pushed to dev (commits 9ca78a8, e580b80)
  - ⏳ Monitoring ETH model behavior with proper normalization

### TASK-014: SL/TP Redesign — Phase 1
- **Requested**: 2026-03-24 09:34 UTC
- **Status**: ✅ COMPLETE + DEPLOYED
- **Last update**: 2026-03-24 12:11 UTC
  - ✅ Full analysis: docs/sl-tp-redesign-plan.md (85% SL hit rate, 15% win rate identified)
  - ✅ Partial TP: 40% at 1R → 35% at 2R → 25% trails (live_trading_htf.py + live_trading_multi.py)
  - ✅ ATR floor: SL >= 1.5×ATR, TP >= 3.0×ATR (risk_manager.py)
  - ✅ Regime multipliers recalibrated: high_vol SL×1.5/TP×1.3, trending TP×1.8, ranging SL×1.2/TP×0.8
  - ✅ MFE/MAE tracking on every trade close (MongoDB + JSONL)
  - ✅ Position size cap: max $3,000 USDT notional per trade
  - ✅ Time-based stagnant exit: close after 6h if PnL in [-0.3%, +0.5%]
  - ✅ futures_executor: new place_partial_tp_order() method
  - ✅ Pushed to dev branch (commit fb2d601)
  - ✅ Both bots (BTC + ETH) restarted — positions preserved, running clean
  - ⏳ Phase 2 (retraining with unified params) pending after 2 weeks of Phase 1 data

## Completed

### ✅ Architecture migration: Local Server + HF Client
- Server running as systemd service (drl-trading-server)
- Cloudflare Tunnel running (drl-trading-tunnel)  
- Tunnel URL auto-updater running (drl-tunnel-watcher)
- Models pulled (662MB LFS)

### ✅ Skills created
- claude-code-runner (root workaround)
- drl-trading-workflow (deployment rules)
- hf-client-debug (troubleshooting patterns)

### ✅ Project analysis completed
- Full analysis in project-analysis.md
- All 12 agents/bots identified

### ✅ TASK-006: Market Analysis Audit + Test Coverage + Skill
- **Requested**: 2026-03-21 08:05 UTC
- **Completed**: 2026-03-21 08:15 UTC
- 4 bugs fixed, 110 tests, audit report, skill created
