# Market Analysis Component Audit

**Date:** 2026-03-21  
**Auditor:** Builder Agent  
**Server:** http://127.0.0.1:5001/api/market

---

## Executive Summary

The market analysis pipeline has **10 signal components** and **4 model components**. After a deep code audit and live API testing, I found:

- **4 bugs fixed** during this audit
- **3 components returning real, useful data** (Regime, Order Flow, Funding)
- **2 components returning partial data** (Whale Tracker, Alternative Data)
- **3 components not surfaced in API** (Correlation Engine, Risk Manager, MTF live)
- **2 components requiring trained models** (Whale Pattern, Cross-chain)
- **2 model components requiring heavy ML models** (TFT Forecaster, Ensemble Orchestrator)

### Live API Response Analysis (BTCUSDT)

| Signal | Status | Real Data? | Notes |
|--------|--------|-----------|-------|
| `price` | ✅ Working | Yes | $70,727 (real Binance spot) |
| `regime` | ✅ Working | Yes | TRENDING_UP, ADX=29.3 |
| `funding` | ✅ Working | Yes | -0.0025% from OKX |
| `order_flow` | ✅ Working | Yes | 3-layer signal, real trades |
| `whale` | ⚠️ Partial | Partially | Score=0, direction=NEUTRAL (flow_metrics empty) |
| `mtf` | ❌ Static | No | Always returns "Syncing..." |
| `forecast` | ❌ Null | No | Disabled to prevent API deadlock |
| `news` | ❌ Null | No | Disabled per user request |

---

## Component-by-Component Audit

### 1. whale_tracker.py — Whale Flow Analysis

**Data Sources:**
- OKX funding rate & L/S ratio (free API, working)
- OKX open interest (free API, working)
- OKX top trader ratio (free API, working)
- Fear & Greed Index from alternative.me (free API, working)
- Whale Alert API (requires API key, optional)
- BSCScan API (requires API key, V2 paid-only)
- BinanceWhaleStream (WebSocket, not started in API server)

**Signal Logic:** ✅ Sound
- Weighted composite of 7 sub-signals
- Confidence = agreement / total signals
- Squeeze detection (funding + OI + flow divergence)
- Contrarian Fear & Greed signal interpretation is correct

**Bugs Found & Fixed:**
1. 🐛 **BinanceOITracker.oi_history never initialized** — `get_oi_signal()` would crash with `AttributeError: 'BinanceOITracker' object has no attribute 'oi_history'`. Fixed by adding `self.oi_history = deque(maxlen=100)` to `__init__`.
2. 🐛 **BinanceTopTraderClient.get_large_transactions returns `bias`/`strength` but WhaleTracker reads `signal` key** — Always returned 0.0. Fixed to convert bias+strength to numeric signal.
3. 🐛 **API server reads `signals.get('direction')` and `signals.get('score')` but WhaleTracker returns `recommendation` and `combined_score`** — Whale always showed NEUTRAL/0 in dashboard. Fixed key mapping.

**Error Handling:** ✅ Good — Each sub-signal has try/except, returns None on failure.

**Staleness:** ⚠️ Moderate — 30s signal cache, 300s for sub-components. WebSocket stream not started in API server mode (correctly avoided for leak prevention).

**Real Value:** ⚠️ Medium — In API-server fallback mode (no live trading state), the whale signal has limited data since WhaleStream isn't running. The OKX-based signals (funding, L/S ratio, top trader) do provide real value.

**Output Format:** ✅ After fix, returns proper `{score, direction, confidence, bullish, bearish, neutral, flow_metrics}`.

---

### 2. order_flow.py — Order Flow Analysis

**Data Sources:**
- Binance Spot REST trades (`data-api.binance.vision`, free, working)
- OKX trades fallback (free, working)
- OKX funding rate (for FundingRateAnalyzer)
- OKX open interest (for liquidation danger)

**Signal Logic:** ✅ Excellent
- 3-layer composite: CVD (50%) + Taker Ratio (30%) + Notable Orders (20%)
- CVD computed from OHLCV candle body ratios — solid approximation
- Taker buy/sell ratio from 1000 recent trades
- Notable orders at $5K threshold (reasonable for spot)
- CVD Divergence detection (price vs volume delta)

**Error Handling:** ✅ Good — Fallbacks from Binance to OKX.

**Staleness:** ✅ Good — 30s cache, fresh trade data each call.

**Real Value:** ✅ High — All three layers provide actionable trading signals.

**Live Data Verified:** BTC order_flow shows score=0.60 (bullish), taker ratio=95.3% buys, 16 notable buys / 0 sells. Data is real and reasonable.

**Output Format:** ✅ Well-structured with per-layer breakdown.

---

### 3. regime_detector.py — Market Regime Detection

**Data Sources:**
- Receives OHLCV DataFrame (fetched by API server from Binance)

**Signal Logic:** ✅ Excellent
- ADX-based trend detection with correct DI calculations
- ATR-based volatility ratio (current vs 50-period average)
- 5 regimes: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY
- Thresholds: ADX>25=trending, ADX<20=ranging — industry standard
- Volatility: >1.5x=high, <0.5x=low — reasonable

**Error Handling:** ✅ Returns UNKNOWN regime with insufficient data.

**Staleness:** N/A — computed fresh from OHLCV each call.

**Real Value:** ✅ High — Regime detection is one of the most valuable analysis signals.

**Live Data Verified:** BTC shows TRENDING_UP with ADX=29.3, volatility=0.6x — reasonable for current market.

**Output Format:** ✅ Clean `{type, adx, direction, volatility}`.

---

### 4. mtf_analyzer.py — Multi-Timeframe Analysis

**Data Sources:**
- Binance klines for 4h, 1h, 15m (free API)

**Signal Logic:** ✅ Sound
- EMA 9/21 crossover for trend direction
- RSI 14 for confirmation/warning
- All-timeframe alignment requirement for confluence

**Error Handling:** ✅ Returns partial data if timeframes missing.

**Staleness:** ⚠️ Issue — In the API server, MTF is NOT calculated in fallback mode. It always returns `{reason: 'Syncing...', aligned: False, bias: 'NEUTRAL'}` unless the live trading bot has populated the state file.

**Real Value:** ✅ High (when used) — Multi-timeframe confluence is a well-established trading concept.

**API Integration Issue:** The MTF analysis is not computed as a fallback like regime/funding/order_flow. The API server should call `MultiTimeframeAnalyzer.get_confluence()` when state data is missing.

---

### 5. whale_pattern_predictor.py — ML Whale Pattern Prediction

**Data Sources:**
- Trained WhalePatternLearner models (per chain: ETH, SOL, XRP)
- Cached whale wallet transaction data (CSV files)
- Wallet registry with tier-based weights

**Signal Logic:** ✅ Sound (when data is available)
- Loads trained models per blockchain
- Applies tier-based wallet accuracy weighting
- 5-minute prediction cache

**Error Handling:** ✅ Good — Returns `{signal: 0, confidence: 0, status: 'no_model/no_data/error'}`.

**Staleness:** ⚠️ Depends on background data collection. The predictor correctly avoids synchronous scraping in the prediction path. Relies on background cron for fresh wallet data.

**Real Value:** ⚠️ Medium — Depends on trained model quality and fresh wallet data. BTC uses ETH whale signals as proxy, which is a rough approximation.

---

### 6. on_chain_whales.py — On-Chain Whale Analytics

**Data Sources:**
- Etherscan API V2 (requires API key)
- Solana RPC (public, but heavy)
- XRPL RPC (public, working)

**Signal Logic:** ✅ Sound
- Parses block transactions for large transfers
- ETH: 100+ ETH threshold (~$250K+)
- SOL: 5000+ SOL threshold (~$500K+)  
- XRP: 500K+ XRP threshold (~$300K+)

**Error Handling:** ✅ Good — Each chain watcher has try/except.

**Staleness:** ⚠️ Only checks latest block, no rolling history.

**Real Value:** ⚠️ Low-Medium — Requires API keys to function. Solana watcher is essentially a stub (just checks one exchange wallet). Not directly surfaced in `/api/market`.

---

### 7. cross_chain_whale_flow.py — Cross-Chain Whale Flow

**Data Sources:**
- WhalePatternPredictor signals for ETH, SOL, XRP

**Signal Logic:** ✅ Sound
- Capital rotation detection (ETH↔SOL flow ratio)
- Cross-chain consensus scoring
- Risk sentiment (stablecoin flow proxy)
- Chain dominance shifts

**Error Handling:** ✅ Returns 0/neutral defaults on failure.

**Real Value:** ⚠️ Low — This is a meta-analyzer built on top of WhalePatternPredictor. When the underlying models lack data, this returns all zeros. Not directly surfaced in `/api/market`.

---

### 8. alternative_data.py — Fear & Greed, BTC Dominance

**Data Sources:**
- Fear & Greed Index from alternative.me (free, working)
- BTC/ETH Dominance from CoinGecko (free, rate-limited)

**Signal Logic:** ✅ Sound
- F&G normalized to [-1, +1] — correct contrarian mapping
- BTC dominance normalized around 50% center
- Altcoin season proxy from ETH/BTC ratio

**Error Handling:** ✅ Good — Disk cache fallback when API fails.

**Staleness:** ✅ Good — 12-hour cache TTL (F&G updates daily anyway).

**Real Value:** ✅ Medium — Good supplementary data, not directly surfaced in `/api/market` but used by the RL agent.

---

### 9. correlation_engine.py — Asset Correlation Analysis

**Data Sources:**
- Binance klines for ETH, BNB, SOL, ETH/BTC (free API)

**Signal Logic:** ✅ Excellent
- Rolling correlation (20-period) between BTC and alts
- ETH/BTC ratio tracking for rotation signals
- SOL beta calculation
- USDT dominance proxy (inverse of market momentum)
- Market regime detection (risk-on/risk-off/rotation)

**Error Handling:** ✅ Good — NaN filling for missing data.

**Real Value:** ✅ High — But only used during RL training, not surfaced in `/api/market`.

---

### 10. risk_manager.py — Risk Management Signals

**Data Sources:**
- Receives OHLCV DataFrame

**Signal Logic:** ✅ Excellent
- ATR-based adaptive SL/TP (2.5x ATR for SL, 4.0x for TP)
- Asset-specific parameters (BTC=1.5%, SOL=2.5%)
- Structural SL/TP using VWAP and swing highs/lows
- Kelly Criterion position sizing (half-Kelly for safety)
- Trailing stop management

**Error Handling:** ✅ Good — Falls back to base defaults.

**Real Value:** ✅ High — Essential for live trading, not directly surfaced in `/api/market`.

---

## Model Components

### 11. regime_classifier.py — HMM Regime Classifier

**Implementation:** Full HMM (GaussianHMM from hmmlearn)
- 4 states: BULL_TREND, BEAR_TREND, RANGE_CHOP, HIGH_VOL_BREAKOUT
- Auto-labels states based on return and volatility characteristics
- Provides transition probabilities (predictive edge)
- Trained models exist for BTC, ETH, SOL, XRP

**Real Value:** ✅ High — Transition probabilities provide genuine predictive value.

### 12. price_forecaster.py — TFT Price Forecasting

**Implementation:** Full Temporal Fusion Transformer
- Variable Selection Network + LSTM + Multi-Head Attention
- 25 features (returns, volume, technicals, temporal)
- 4 horizons (1h, 4h, 12h, 24h) with quantile outputs
- Trained models exist for all 4 assets

**API Integration:** Disabled in API server ("prevents API deadlock" from loading PyTorch on CPU).

**Real Value:** ✅ High (when used) — Well-architected TFT. Not surfaced in API.

### 13. confidence_engine.py — Signal Confidence Scoring

**Implementation:** Simple but effective
- Maps raw confidence [0-1] to position multiplier [0.25x-2.0x]
- Exponential scale-up for high confidence
- Tracks historical confidence vs outcome correlation

**Real Value:** ✅ Medium — Good for position sizing. Not directly surfaced.

### 14. ensemble_orchestrator.py — Ensemble Decision Making

**Implementation:** HMM-weighted specialist voting
- 4 PPO specialist agents (one per regime)
- Weights actions by HMM transition probabilities
- Requires trained specialist models

**Real Value:** ✅ High (when trained) — Sophisticated ensemble approach.

---

## Bugs Fixed During Audit

| # | Component | Bug | Severity | Fix |
|---|-----------|-----|----------|-----|
| 1 | `whale_tracker.py` | `BinanceOITracker.oi_history` never initialized — crashes `get_oi_signal()` | 🔴 High | Added `self.oi_history = deque(maxlen=100)` to `__init__` |
| 2 | `whale_tracker.py` | `BinanceTopTraderClient.get_large_transactions()` returns `bias`/`strength` but caller reads `signal` key — always 0 | 🟡 Medium | Convert bias+strength to numeric signal |
| 3 | `api_server.py` | Whale signal reads `direction`/`score` but tracker returns `recommendation`/`combined_score` — always NEUTRAL/0 | 🔴 High | Fixed key mapping |
| 4 | `api_server.py` | Cached `/api/market` responses leak `_fetched_at` internal field | 🟢 Low | Strip `_fetched_at` from cached response |

---

## Recommendations

### Immediate Fixes
1. ✅ Done — All 4 bugs fixed
2. Add MTF fallback calculation in API server (like regime/order_flow)
3. Consider loading TFT model in a background thread for forecast data

### Signal Quality Improvements
1. The whale signal is the weakest link — without WebSocket stream in API mode, flow_metrics is always empty
2. Order flow notable order threshold ($5K) captures real institutional activity
3. Regime detection is the strongest and most reliable signal

### Architecture Notes
- The API server correctly avoids heavy ML inference (TFT, WhalePatternPredictor)
- Cache TTL of 30s is good for live trading dashboard
- OKX fallback for funding/trades is reliable and well-implemented
