# Signal Improvement Plan — DRL Crypto Trading System

**Date:** 2026-03-21  
**Author:** Quantitative Research Analyst  
**System:** DRL Trading System (PPO + HMM Ensemble)  
**Current Performance:** Sharpe 3.85, +14.8%/2mo on BTC (walk-forward validated)

---

## Table of Contents

1. [Current Signal Quality Audit](#1-current-signal-quality-audit)
2. [Best-in-Class Signal Research](#2-best-in-class-crypto-signal-sources-2025-2026)
3. [Phased Improvement Plan](#3-phased-improvement-plan)
4. [Specific Recommendations for Our System](#4-specific-recommendations-for-our-system)

---

## 1. Current Signal Quality Audit

### 1.1 Regime Detector (`src/features/regime_detector.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **8/10** — ADX + ATR regime detection is industry standard and well-implemented |
| **Data Freshness** | ✅ Excellent — Computed fresh from OHLCV each call, no cache staleness |
| **Alpha Potential** | **High** — Regime filtering is one of the highest-alpha signals in systematic trading. Knowing *when* to trade (trending) vs *when not to* (choppy) prevents ~40% of losing trades in most backtests |
| **Improvement Potential** | **Medium** — Could add HMM transition probabilities (already in `regime_classifier.py` but not surfaced to live trading) and volume-weighted regime confirmation |

**Strengths:**
- Clean ADX/DI+/DI- calculation with correct smoothing
- 5 distinct regimes with sensible thresholds (ADX>25=trending, ADX<20=ranging)
- ATR-based volatility regime detection (1.5x/0.5x thresholds)
- Position size multipliers per regime (0.5x-1.2x)

**Weaknesses:**
- ADX is inherently lagging (14-period smoothed twice → ~28 bars of lag)
- No volume confirmation — can misidentify low-volume consolidation as trend
- No regime persistence scoring — flips between regimes on single-bar ADX changes
- HMM classifier (`regime_classifier.py`) provides transition probabilities but isn't wired into the live API or trading logic

**Key Finding:** The HMM regime classifier is the most underutilized component. It provides **forward-looking transition probabilities** (e.g., "70% chance of switching from RANGE to BULL next period") which is far more actionable than the current ADX-based detector that only tells you the current state.

---

### 1.2 Order Flow (`src/features/order_flow.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **7/10** — 3-layer composite is well-designed; CVD from OHLCV is a good approximation |
| **Data Freshness** | ✅ Good — 30s cache, fetches 1000 recent trades per call |
| **Alpha Potential** | **Medium-High** — Taker ratio and notable orders provide real edge, but CVD from OHLCV is inherently less precise than tick-level CVD |
| **Improvement Potential** | **High** — Missing order book depth, real tick-level CVD, trade flow toxicity |

**Strengths:**
- Elegant 3-layer architecture: CVD (50%) + Taker Ratio (30%) + Notable Orders (20%)
- CVD divergence detection (price vs volume delta) — genuine alpha signal
- Binance → OKX fallback for resilience
- Notable order threshold at $5K is reasonable for spot (captures retail-institutional boundary)
- Liquidation danger score combining funding + OI

**Weaknesses:**
- **CVD from OHLCV is approximate** — uses candle body ratio * volume as a proxy. Real CVD from tick-level trade data (aggTrades) would be 2-3x more accurate
- **No order book imbalance** — Missing bid/ask depth ratio which is one of the strongest short-term signals
- **No trade flow toxicity (VPIN)** — Missing the probability of informed trading metric
- **$5K notable order threshold** is too low for BTC at $70K+ — catching noise. Should scale with asset price
- **No microstructure features** — spread, trade arrival rate, order size distribution

**Key Finding:** The order flow signal is the second-strongest component after regime detection. Upgrading from OHLCV-based CVD to real tick-level CVD and adding order book imbalance would provide the single biggest improvement in signal quality.

---

### 1.3 Whale Tracker (`src/features/whale_tracker.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **4/10** — Returns NEUTRAL with low confidence most of the time in API mode |
| **Data Freshness** | ⚠️ Moderate — 30s signal cache, but 300s for sub-components; WebSocket not started in API mode |
| **Alpha Potential** | **Medium** — Good when working, but rarely produces actionable signals |
| **Improvement Potential** | **Very High** — Most room for improvement of any component |

**Strengths:**
- Composite of 7 sub-signals with weighted scoring (well-architected)
- OKX-based signals (funding, L/S ratio, top trader ratio) are reliable free data
- Squeeze detection (funding + OI + flow divergence) is conceptually excellent
- Fear & Greed contrarian interpretation is correct
- Exchange reserve delta tracking concept is right

**Weaknesses:**
- **In API-server mode, `flow_metrics` is always empty** because WhaleStream WebSocket isn't started (correctly, to avoid socket leaks — but the signal is gutted)
- **exchange_reserve signal gets 27% weight but relies on WhaleStream** which isn't running → 27% of the composite is always 0
- **WhaleAlert requires paid API key** — free tier (10 req/min) insufficient for real-time
- **BSCScan V2 requires paid plan** — V1 deprecated
- **BinanceTopTraderClient** renamed from Blockchair but still referenced as `blockchair` in variable names (confusing)
- **Whale Pattern Predictor** disabled in API mode (`enable_ml=False`) — correctly to prevent PyTorch hangs, but removes ML signal
- **300s cache on sub-components** means OI and top trader data can be 5 minutes stale
- **Signal weights don't sum to 1.0** correctly when some signals return None — the dynamic re-weighting is correct mathematically but means confidence varies unpredictably

**Key Finding:** The whale tracker tries to do too many things from too many data sources, most of which are unavailable. The 4 OKX-based signals (funding, L/S ratio, OI, top trader) are reliable and should be the core. Everything else should be additive.

---

### 1.4 HTF Feature Engine (`src/features/htf_features.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **9/10** — Comprehensive, well-normalized 114-dim feature vector across 4 timeframes |
| **Data Freshness** | ✅ Computed from OHLCV at each step — no caching issues |
| **Alpha Potential** | **Very High** — This is the core alpha source; walk-forward Sharpe of 3.85 validates it |
| **Improvement Potential** | **Medium** — Features are comprehensive for price action; missing external data integration |

**Strengths:**
- 114 features across 4 timeframes (1D, 4H, 1H, 15M) — excellent granularity
- Smart Money Concepts (BOS, CHOCH, Order Blocks, FVGs) — institutional-grade features
- Wyckoff phase detection — captures accumulation/distribution
- Cross-TF alignment features — measures trend confluence across timeframes
- Well-clipped/normalized to [-1, 1] or [0, 1] ranges — DRL-friendly
- Ichimoku cloud, Keltner channels, Bollinger squeeze — multi-indicator coverage
- Candle pattern detection (hammer, shooting star, engulfing, pin bars, inside/outside bars)

**Weaknesses:**
- **No external data features** — all 114 features are pure price/volume. No funding rate, no OI, no sentiment, no on-chain data
- **No volume profile / VWAP features** — critical for institutional level analysis
- **Alignment feature hack:** `align_1h_15m` uses `sig_1h` for both args in `compute_alignment()` when called without the 15M score — always returns 1.0 or -1.0 (never 0)
- **OI proxy** (`_oi_proxy`) is a volume-price agreement heuristic, not real OI — labeled misleadingly
- **No realized volatility rank** — only ATR percentile, missing RV vs implied vol comparison

**Key Finding:** The HTF engine is the crown jewel of the system. Adding 10-15 external features (funding rate, OI, sentiment, on-chain) to the observation space and retraining would likely push Sharpe from 3.85 to 4.5+ based on academic literature showing ~15-25% improvement from multi-modal features.

---

### 1.5 Alternative Data (`src/features/alternative_data.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **5/10** — Only 4 features (F&G value, F&G class, BTC dom, altcoin season) |
| **Data Freshness** | ✅ Good — 12h cache (F&G updates daily; CoinGecko rate-limited) |
| **Alpha Potential** | **Medium** — F&G is a well-known contrarian indicator; BTC dominance useful for rotation |
| **Improvement Potential** | **High** — Currently only 2 data sources; many more available for free |

**Strengths:**
- Correct contrarian mapping of Fear & Greed (Extreme Fear → bullish signal)
- BTC dominance normalized around 50% center
- Altcoin season proxy from ETH/BTC ratio
- Disk cache fallback when API fails

**Weaknesses:**
- **Only 2 data sources** (alternative.me and CoinGecko) — vast landscape of alt data untapped
- **Not integrated into HTF feature vector** — collected but seemingly only used by a separate RL agent, not the walk-forward validated HTF model
- **CoinGecko rate limited on free tier** (10-30 req/min) — can fail silently
- **No stablecoin supply metrics** — USDT minting/burning is one of the strongest leading indicators
- **No Google Trends integration** — strong retail sentiment proxy
- **No options data** — put/call ratio, max pain, IV skew

**Key Finding:** Alternative data is an underinvested category. The features exist but aren't wired into the main HTF model. Adding stablecoin supply ratio and Google Trends would provide the most alpha per unit effort.

---

### 1.6 Regime Classifier / HMM (`src/models/regime_classifier.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **8/10** — Proper GaussianHMM with 4 states, auto-labeled by return/volatility characteristics |
| **Data Freshness** | N/A — trained offline, used for inference during trading |
| **Alpha Potential** | **Very High** — Transition probabilities are genuinely predictive |
| **Improvement Potential** | **Medium** — Model is sound; integration is the bottleneck |

**Strengths:**
- Full GaussianHMM with proper auto-labeling of states (BULL, BEAR, RANGE, BREAKOUT)
- 8-dimensional feature vector (returns at 3 scales, 2 volatility measures, volume ratio, RSI deviation, directional strength)
- Transition matrix provides forward-looking probabilities — genuine edge
- Trained models exist for BTC, ETH, SOL, XRP
- Used to filter training data for specialist agents

**Weaknesses:**
- **Not surfaced in the live API** — the `regime_detector.py` (ADX-based) is used instead
- **Single-timeframe features** — only uses 1H data; should incorporate multi-TF
- **No online updating** — model is frozen at training time; market regimes shift
- **Covariance type 'full'** may overfit with only 8 features — 'diag' or 'tied' might generalize better

**Key Finding:** The HMM transition probabilities should replace or augment the ADX-based regime detector in the live trading path. This is likely the single highest-ROI change that requires zero new data sources.

---

### 1.7 Confidence Engine (`src/models/confidence_engine.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **5/10** — Simple but functional; exponential scale-up for high confidence |
| **Data Freshness** | N/A — real-time calculation |
| **Alpha Potential** | **Medium** — Correct concept (Kelly-like sizing) but naive implementation |
| **Improvement Potential** | **Medium** — Needs calibration against actual prediction reliability |

**Strengths:**
- Maps confidence [0,1] to position multiplier [0.25x, 2.0x]
- Exponential scale-up prevents over-leveraging at medium confidence
- Tracks confidence vs outcome correlation (Pearson)

**Weaknesses:**
- **No calibration** — assumes confidence scores are well-calibrated probabilities (they're not)
- **No regime-conditional sizing** — should size differently in trending vs ranging
- **No correlation with recent P&L** — doesn't reduce size after drawdowns (Kelly criterion recommends this)
- **Reliability metric requires 20+ trades** — cold start problem

---

### 1.8 API Server / Signal Serving (`src/ui/api_server.py`)

| Metric | Assessment |
|--------|-----------|
| **Signal Quality** | **6/10** — Good signal aggregation; some signals missing |
| **Data Freshness** | ✅ 30s cache on `/api/market` |
| **Alpha Potential** | N/A — serving layer, not signal generation |
| **Improvement Potential** | **Medium** — MTF fallback, TFT loading, signal consistency |

**Issues Identified:**
1. **MTF not calculated in fallback mode** — always returns "Syncing..." unless live bot is running
2. **TFT forecaster disabled** — prevents API deadlock, but loses forecast signal entirely
3. **Whale signal was returning wrong keys** (fixed in audit) — but signal quality still low in API mode
4. **No signal staleness indicator** — dashboard doesn't show how old each signal is
5. **No composite "trade score"** — each signal is separate; no unified action recommendation

---

### 1.9 Components NOT Surfaced in API (Hidden Alpha)

| Component | File | Value | Note |
|-----------|------|-------|------|
| Correlation Engine | `correlation_engine.py` | ✅ High | BTC/alt correlations, rotation signals, risk-on/off — used in training only |
| Risk Manager | `risk_manager.py` | ✅ High | ATR-based SL/TP, Kelly sizing — used in live trading only |
| TFT Forecaster | `price_forecaster.py` | ✅ High | 4 horizon quantile forecasts — disabled (deadlock) |
| Ensemble Orchestrator | `ensemble_orchestrator.py` | ✅ High | HMM-weighted specialist voting — requires trained specialists |

---

### Signal Quality Summary

| Signal | Quality | Alpha | Freshness | Status | Priority to Fix |
|--------|---------|-------|-----------|--------|-----------------|
| HTF Features | 9/10 | Very High | Fresh | ✅ Working | — |
| HMM Regime | 8/10 | Very High | Fresh | ⚠️ Not in live path | **P0** |
| ADX Regime | 8/10 | High | Fresh | ✅ Working | — |
| Order Flow | 7/10 | Medium-High | 30s | ✅ Working | P1 |
| Funding Rate | 7/10 | Medium | 60s | ✅ Working | — |
| Alternative Data | 5/10 | Medium | 12h | ⚠️ Not in HTF model | P1 |
| Confidence Engine | 5/10 | Medium | Real-time | ⚠️ Uncalibrated | P2 |
| Whale Tracker | 4/10 | Medium | 30-300s | ⚠️ Mostly NEUTRAL | **P0** |
| MTF Analyzer | —/10 | High | — | ❌ "Syncing..." | P1 |
| TFT Forecaster | —/10 | High | — | ❌ Disabled | P2 |

---

## 2. Best-in-Class Crypto Signal Sources (2025-2026)

### 2.1 On-Chain Analytics

#### 2.1.1 Exchange Inflow/Outflow (Net Flow)
- **What:** Track BTC/ETH moving to/from exchange hot wallets. Large inflows = selling pressure; large outflows = accumulation.
- **Alpha Evidence:** Research by Chainalysis and Glassnode shows exchange net flow predicts 3-5 day price moves with 0.15-0.25 correlation — higher than most single indicators.
- **Free Sources:**
  - **CryptoQuant Free API** (`https://api.cryptoquant.com/v1/btc/exchange-flows/netflow`) — free tier: 100 req/day, daily granularity
  - **Blockchain.com** (`https://api.blockchain.info/charts/...`) — exchange balance estimates
  - **Glassnode Free Tier** — limited to 1 metric, daily
- **Paid Sources:**
  - CryptoQuant Pro ($29/mo) — 5-min granularity, all assets
  - Glassnode Advanced ($29/mo) — full suite
- **Implementation:** Python `requests`, 1h polling interval, normalize to [-1, 1] based on Z-score of net flow vs 30-day rolling mean

#### 2.1.2 Stablecoin Supply Ratio (SSR)
- **What:** Ratio of BTC market cap to total stablecoin market cap. Low SSR = high buying power available = bullish.
- **Alpha Evidence:** SSR is a leading indicator for large market moves. When stablecoin supply grows faster than BTC market cap, it's "dry powder" for buying.
- **Free Sources:**
  - **CoinGecko** (`/api/v3/global`) — total stablecoin market cap derivable from global data
  - **DeFiLlama** (`https://stablecoins.llama.fi/stablecoins`) — real-time stablecoin supply by chain and token
  - **Tether Transparency** (`https://app.tether.to/transparency.json`) — USDT total supply
- **Implementation:** Calculate SSR = BTC_mcap / total_stablecoin_supply. Track rate of change (ΔSSR). Bullish when SSR falling or USDT supply rapidly increasing.

#### 2.1.3 Active Addresses / Network Value
- **What:** Daily active addresses as a proxy for network usage. NVT ratio (Network Value to Transactions) = crypto P/E ratio.
- **Free Sources:**
  - **Blockchain.com** — BTC active addresses, transaction count, volume
  - **Blockchair** — multi-chain stats API (free tier)
- **Alpha Potential:** Medium — more useful for longer timeframes (weekly). NVT spikes historically precede dumps.

#### 2.1.4 Mining Metrics (Hash Rate, Miner Selling)
- **What:** Hash rate trends and miner outflows. Miners selling = bearish; miners accumulating = bullish.
- **Free Sources:**
  - **Blockchain.com** (`/charts/hash-rate?format=json`) — daily hash rate
  - **Mempool.space API** (`https://mempool.space/api/v1/mining/hashrate/3m`) — hash rate with fee data
- **Alpha Potential:** Low-Medium for short-term trading (our 1H-4H timeframe). More useful for macro regime detection (weekly bias).

#### 2.1.5 Liquidation Heatmaps / Levels
- **What:** Estimated liquidation levels based on known leverage ratios. Price moves toward liquidation clusters ("magnetic effect").
- **Free Sources:**
  - **CoinGlass** (`https://open-api.coinglass.com/public/v2/futures/liquidation_chart`) — free tier, limited
  - **OKX Open Interest by Price** — derivable from OKX position data
  - **Binance Futures Liquidation Stream** (`wss://fstream.binance.com/ws/!forceOrder@arr`) — real-time liquidation events
- **Alpha Evidence:** Liquidation cascades explain 30-40% of large crypto moves. Knowing where liquidations cluster gives a "magnetic target" for price.
- **Implementation:** Track liquidation levels above/below price. Compute "liquidation imbalance" = (longs_above / shorts_below). When imbalance is high, price tends to move toward the larger cluster.

#### 2.1.6 Open Interest + Funding Rate Divergences
- **What:** When OI rises but funding stays flat or falls, informed traders are entering positions the crowd doesn't see.
- **Already Partially Implemented:** Our system tracks OI and funding separately but doesn't compute their divergence.
- **Enhancement:** Create a composite "smart money divergence" signal:
  - OI ↑ + Funding ↓ = smart money going long (bullish)
  - OI ↑ + Funding ↑ (extreme) = retail overleveraged long (bearish — squeeze incoming)
  - OI ↓ + Funding extreme = cascade liquidation risk

---

### 2.2 Market Microstructure

#### 2.2.1 Order Book Imbalance
- **What:** Ratio of bid-side depth to ask-side depth at various levels (±0.5%, ±1%, ±2% from mid).
- **Alpha Evidence:** Academic research (Cont, Kukanov & Stoikov, 2014) shows order book imbalance predicts next-trade direction with ~55-60% accuracy at 1-second horizon. At 1-minute horizon: ~52-54%. Small but consistent edge.
- **Free Sources:**
  - **Binance REST** (`/api/v3/depth?symbol=BTCUSDT&limit=50`) — top 50 bids/asks, free, 2400 req/min
  - **OKX REST** (`/api/v5/market/books?instId=BTC-USDT&sz=50`) — similar depth
- **Implementation:**
  ```
  imbalance = (sum(bid_qty[:N]) - sum(ask_qty[:N])) / (sum(bid_qty[:N]) + sum(ask_qty[:N]))
  ```
  Compute at depth levels: 5, 10, 20, 50. Track rolling change (Δimbalance). More informative than raw imbalance.

#### 2.2.2 Trade Flow Toxicity (VPIN)
- **What:** Volume-Synchronized Probability of Informed Trading. Measures how much of the traded volume is likely from informed traders.
- **Alpha Evidence:** VPIN spikes preceded the Flash Crash of 2010 and multiple crypto black swans. Predicts volatility events 2-8 hours ahead with 60-70% recall.
- **Free Sources:** Must be computed from trade data (already available from Binance aggTrades)
- **Implementation:**
  1. Bucket trades into equal-volume bars (not time bars)
  2. In each bucket, classify buys/sells (BVC — Bulk Volume Classification)
  3. VPIN = |buy_volume - sell_volume| / total_volume, rolling over N buckets
  4. High VPIN (>0.6) = informed trading = expect large price move
- **Libraries:** Custom implementation; no standard Python library. ~100 lines of code.

#### 2.2.3 Spoofing / Layering Detection
- **What:** Detect large orders placed and quickly cancelled (market manipulation).
- **Alpha Evidence:** Spoofing precedes most fake breakouts. When detected, the breakout direction is likely to reverse.
- **Free Sources:** Requires order book WebSocket snapshots at ~100ms intervals (bandwidth-heavy)
- **Implementation Complexity:** High — needs real-time order book streaming and change detection
- **Recommendation:** P2 — complex, bandwidth-heavy, moderate alpha

#### 2.2.4 Realized Volatility vs Implied Volatility
- **What:** When implied vol (from options) > realized vol, options are expensive (market expects a move). When IV < RV, market is complacent.
- **Free Sources:**
  - **Deribit API** (`https://www.deribit.com/api/v2/public/get_index_price`) — BTC/ETH options IV
  - **OKX Options** — IV for BTC options
  - Realized vol computed from our own OHLCV data
- **Alpha Evidence:** IV/RV ratio > 1.5 precedes large moves ~65% of the time. Direction is uncertain but magnitude prediction is valuable for position sizing.

---

### 2.3 Sentiment & Alternative Data

#### 2.3.1 Social Media Sentiment
- **What:** Aggregate sentiment from Twitter/X, Reddit (r/CryptoCurrency, r/Bitcoin), Telegram groups.
- **Alpha Evidence:** Research by Bianchi et al. (2023) shows Twitter sentiment leads BTC price by 1-6 hours with modest but consistent alpha. Crypto-specific: retail sentiment is a contrarian indicator at extremes.
- **Free Sources:**
  - **LunarCrush Free API** (`https://lunarcrush.com/api4/public/coins/1/time-series/v2`) — free tier: social volume, sentiment, galaxy score
  - **Reddit API** (free with auth) — post sentiment from crypto subreddits
  - **CryptoPanic API** (`https://cryptopanic.com/api/v1/posts/`) — aggregated news sentiment, free tier: 5 req/min
- **Paid Sources:**
  - Santiment ($44/mo) — social volume, development activity, weighted sentiment
  - The TIE ($$$) — institutional-grade social data

#### 2.3.2 Google Trends
- **What:** Search interest for "bitcoin", "buy bitcoin", "crypto crash" as a retail sentiment proxy.
- **Alpha Evidence:** Google Trends for "bitcoin" peaks 1-3 days before retail buying climaxes. "Crypto crash" spikes often mark local bottoms (capitulation).
- **Free Sources:**
  - **pytrends** library (unofficial Google Trends API) — free but rate-limited (~10 req/min)
  - Tracks relative search interest (0-100) over time
- **Implementation:** Track 7-day moving average of search interest. Compute rate of change. Extreme spikes (>2σ) are contrarian signals.

#### 2.3.3 Options Market Sentiment
- **What:** Put/Call ratio, Max Pain, and IV skew from crypto options markets.
- **Alpha Evidence:** Put/Call ratio > 1.5 is historically bullish (too many hedges = bottom is near). Max Pain (strike where most options expire worthless) acts as a price magnet near expiry.
- **Free Sources:**
  - **Deribit Public API** — free, real-time options data including:
    - `GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option` — all BTC options
    - Derive P/C ratio, max pain, IV surface from this data
  - **OKX Options** — limited free data
  - **Laevitas** (https://app.laevitas.ch/altsderivs/ETH/options) — dashboard, no free API
- **Implementation:** Compute P/C ratio from Deribit option book. Track IV skew (25-delta put IV vs 25-delta call IV). Max pain from open interest by strike.

#### 2.3.4 Stablecoin Premium/Discount
- **What:** USDT trading above/below $1.00 on various exchanges indicates regional demand pressure.
- **Alpha Evidence:** USDT premium in Korea ("Kimchi Premium") historically precedes major rallies by 1-3 days. USDT discount during crashes indicates panic selling.
- **Free Sources:**
  - **CoinGecko** — USDT price on different exchanges
  - **Binance/OKX** — USDT trading pairs (already accessible)
- **Implementation:** Track USDT/BUSD or USDT/DAI pair price deviations from 1.00. Premium >0.3% = regional demand = bullish.

---

### 2.4 Cross-Market Signals

#### 2.4.1 DXY (Dollar Index) Correlation
- **What:** US Dollar Index — BTC has strong negative correlation with DXY (ρ ≈ -0.3 to -0.6 depending on period).
- **Alpha Evidence:** DXY breakdowns precede BTC rallies with 1-5 day lead. The correlation tightened significantly since 2022.
- **Free Sources:**
  - **Yahoo Finance** (yfinance library) — `DX-Y.NYB` ticker, 15-min granularity
  - **Alpha Vantage** (free API key) — forex data including USD index
  - **Investing.com** (scraping, not ideal) — real-time DXY
- **Implementation:** Track DXY daily close, compute 20-day correlation with BTC. When correlation is strong (|ρ|>0.5) and DXY moves >0.5% in a day, use as a leading indicator.

#### 2.4.2 S&P 500 / Nasdaq Correlation
- **What:** Since 2020, BTC correlates with risk assets (ρ ≈ 0.4-0.7 with Nasdaq-100).
- **Free Sources:**
  - **Yahoo Finance** (yfinance) — `^GSPC`, `^IXIC`, `^NDX` — delayed 15min, free
  - **Alpha Vantage** — `TIME_SERIES_INTRADAY` for SPY, QQQ
- **Implementation:** Track SPY/QQQ futures (ES, NQ) during market hours. During US market hours (14:30-21:00 UTC), equity movements lead crypto by minutes to hours.

#### 2.4.3 Regional Trading Session Patterns
- **What:** Asia (00:00-09:00 UTC), Europe (07:00-16:00 UTC), US (13:30-21:00 UTC) have distinct behaviors.
- **Alpha Evidence:** Asia session tends to be range-bound; US session has highest volatility; European session bridges the two. Breakouts during session overlaps (7-9 UTC, 13:30-16:00 UTC) are most reliable.
- **Free Sources:** Time of day — no API needed
- **Implementation:** Encode current session as a categorical feature. Track session-specific volatility. Score breakouts higher during overlap sessions.

#### 2.4.4 Bond Yields / Risk-Free Rate
- **What:** US 10Y yield inversely correlated with BTC. Rising yields = risk-off = bearish for BTC.
- **Free Sources:**
  - **FRED API** (free) — `https://api.stlouisfed.org/fred/series/observations?series_id=DGS10`
  - **Yahoo Finance** — `^TNX` for 10Y yield
- **Implementation:** Track daily change in 10Y yield. Rapid rises (>10bps/day) are bearish for BTC.

---

### 2.5 Advanced ML Techniques

#### 2.5.1 Temporal Fusion Transformer (TFT) — Already Built
- **Status:** Built in `src/models/price_forecaster.py` but disabled due to deadlock
- **Fix:** Run TFT in a separate process with IPC (subprocess or multiprocessing with shared memory)
- **Alpha Potential:** Very High — variable selection + temporal attention captures non-linear patterns
- **Priority:** P0 — already built, just needs process isolation

#### 2.5.2 Online Learning / Concept Drift Detection
- **What:** Markets change. Models trained on 2024 data lose edge in 2025. Online learning adapts.
- **Techniques:**
  - **River** library (Python) — online ML with concept drift detection
  - **ADWIN** (Adaptive Windowing) — detects concept drift automatically
  - **Incremental PCA** — update feature decomposition online
- **Implementation:** Add a concept drift monitor that tracks prediction error over a sliding window. When drift is detected (error > 2σ), flag the model as degraded and trigger retraining alert.
- **Alpha Evidence:** Models with online adaptation outperform fixed models by 20-40% in backtests over 1-year horizons.

#### 2.5.3 Attention-Based Signal Weighting
- **What:** Instead of fixed weights for signal combination, use a small attention network that learns which signals matter most in the current regime.
- **Implementation:** 
  - Small 2-layer transformer encoder over signal vector
  - Input: [regime_features, order_flow, funding, whale, sentiment, ...] 
  - Output: attention weights per signal
  - Train on historical signal → return correlations
- **Alpha Evidence:** Dynamic weighting outperforms static by 10-25% in multi-signal systems (De Prado, 2018).

#### 2.5.4 Reinforcement Learning for Dynamic Feature Selection
- **What:** Use RL to learn which features to include/exclude based on current market conditions.
- **Already Partially Implemented:** The ensemble orchestrator with regime-specific specialists is conceptually similar.
- **Enhancement:** Add a meta-RL agent that dynamically adjusts feature weights fed to the specialist agents.

---

## 3. Phased Improvement Plan

### Phase 1: Quick Wins (1-2 Days) — Free APIs, Easy Integration

#### P1.1: Wire HMM Transition Probabilities into Live Trading
- **What:** Replace/augment the ADX-based regime detector with HMM transition probabilities from the trained regime classifier
- **Why:** The HMM model is already trained and validated. Its transition matrix provides *predictive* regime probabilities — genuine forward-looking edge vs the lagging ADX indicator
- **How:** 
  1. Load trained HMM model in the API server (it's a small scikit-learn model, no PyTorch → no deadlock risk)
  2. In `/api/market`, fetch 200 bars of 1H OHLCV, run `RegimeClassifier.predict()`, add transition probabilities to the regime response
  3. In live trading, use transition probability as a position sizing modifier: if P(BULL→BEAR) > 40%, reduce long exposure
- **Data Source:** Already trained models in `data/models/regime/`
- **Files to Modify:**
  - `src/ui/api_server.py` — add HMM prediction to `/api/market` regime section
  - `src/features/regime_detector.py` — add `get_hmm_prediction()` method that loads and runs the classifier
- **Effort:** S (Small — ~2-3 hours)
- **Impact:** Signal quality improvement: Regime 8/10 → 9/10
- **Priority:** **P0**
- **Validation:** Compare regime change predictions vs actual regime changes over last 6 months of data

#### P1.2: Fix Whale Tracker Signal Quality
- **What:** Restructure whale tracker to produce meaningful signals from the 4 OKX APIs that actually work
- **Why:** Whale signal is currently the weakest component (4/10). By focusing on the reliable sub-signals and removing broken dependencies, quality improves dramatically
- **How:**
  1. Reduce dependency on WhaleStream (exchange_reserve weight from 27% → 0% in API mode)
  2. Redistribute weights: OKX L/S ratio 30%, OKX OI 25%, OKX Top Trader 25%, Fear & Greed 20%
  3. Reduce sub-component cache from 300s to 120s (still conservative but fresher)
  4. Add OI-Funding divergence as a new composite signal (see §2.1.6)
  5. Fix the squeeze detection to use the OKX signals directly
- **Data Source:** OKX free APIs (already connected)
- **Files to Modify:**
  - `src/features/whale_tracker.py` — restructure weights, add OI-funding divergence
  - `src/ui/api_server.py` — verify whale signal mapping after restructure
- **Effort:** S (3-4 hours)
- **Impact:** Signal quality improvement: Whale 4/10 → 7/10
- **Priority:** **P0**
- **Validation:** Run the restructured whale tracker for 48h and verify it produces non-NEUTRAL signals at least 40% of the time

#### P1.3: Add Order Book Imbalance Signal
- **What:** Fetch top 20 bid/ask levels from Binance, compute bid/ask depth ratio
- **Why:** Order book imbalance is one of the strongest short-term predictors (documented in academic literature). Our system completely lacks it.
- **How:**
  1. Create `src/features/orderbook_imbalance.py`
  2. Fetch depth from `https://data-api.binance.vision/api/v3/depth?symbol=BTCUSDT&limit=20`
  3. Compute imbalance at 3 depth levels (5, 10, 20 levels)
  4. Track rolling Δimbalance (more informative than raw imbalance)
  5. Wire into order flow analyzer as a 4th layer
- **Data Source:** Binance REST depth endpoint — free, 2400 req/min
- **Files to Create:**
  - `src/features/orderbook_imbalance.py` — imbalance calculator
- **Files to Modify:**
  - `src/features/order_flow.py` — add orderbook imbalance as 4th layer (weight: CVD 40%, Taker 25%, Notable 15%, OB Imbalance 20%)
  - `src/ui/api_server.py` — surface imbalance data in order flow response
- **Effort:** S (3-4 hours)
- **Impact:** Signal quality improvement: Order Flow 7/10 → 8.5/10
- **Priority:** **P0**
- **Validation:** Backtest: compute imbalance on historical data, measure correlation with 1-5 bar forward returns

#### P1.4: Fix MTF Analyzer Fallback
- **What:** Compute MTF analysis in the API server when state file data is missing
- **Why:** MTF confluence is a high-alpha signal that's completely dark in API mode
- **How:**
  1. In `/api/market`, when MTF returns "Syncing...", fetch 4H/1H/15M klines and run `MultiTimeframeAnalyzer.get_confluence()`
  2. Similar to how regime and order flow already have fallback calculations
- **Data Source:** Binance klines (already used for regime fallback)
- **Files to Modify:**
  - `src/ui/api_server.py` — add MTF fallback calculation
  - `src/features/mtf_analyzer.py` — ensure `get_confluence()` can be called standalone
- **Effort:** S (2 hours)
- **Impact:** Signal quality improvement: MTF 0/10 → 7/10
- **Priority:** **P0**
- **Validation:** Compare MTF output against manual chart analysis for current BTC structure

#### P1.5: Add Stablecoin Supply Signal
- **What:** Track total stablecoin (USDT+USDC) supply and compute rate of change
- **Why:** Stablecoin minting is one of the strongest free leading indicators. New USDT minted = new buying power entering the market.
- **How:**
  1. Use DeFiLlama API: `https://stablecoins.llama.fi/stablecoins` — free, no key required
  2. Track total supply and 7-day change rate
  3. Normalize: supply growth >1%/week = bullish signal, <-0.5%/week = bearish
  4. Add to alternative data collector
- **Data Source:** DeFiLlama Stablecoins API (free)
- **Files to Modify:**
  - `src/features/alternative_data.py` — add `fetch_stablecoin_supply()` method
- **Effort:** S (2 hours)
- **Impact:** Adds a genuinely predictive macro indicator
- **Priority:** P1
- **Validation:** Correlate weekly stablecoin supply change with weekly BTC returns over 2 years

#### P1.6: Add Trading Session Features
- **What:** Encode current trading session (Asia/Europe/US/Overlap) as features
- **Why:** Session patterns explain ~15% of intraday volatility variance. Breakouts during overlaps are more reliable.
- **How:**
  1. Compute current UTC hour → session classification
  2. Add 3 binary features: is_asia, is_europe, is_us (overlaps get multiple flags)
  3. Add session_volatility_ratio: current session's historical vol / 24h vol
- **Data Source:** Current time — no API needed
- **Files to Modify:**
  - `src/features/htf_features.py` — add session features to 15M feature vector (expand from 35 to 38)
- **Effort:** S (1-2 hours)
- **Impact:** Small but consistent edge for entry timing
- **Priority:** P1
- **Dependencies:** Need to update observation space in htf_env.py (117 → 120 dims) and retrain

---

### Phase 2: Medium Effort (1 Week) — Moderate APIs, Some ML Work

#### P2.1: Real Tick-Level CVD (Cumulative Volume Delta)
- **What:** Replace OHLCV-approximated CVD with real tick-level CVD from Binance aggTrades
- **Why:** OHLCV CVD uses candle body ratio as a proxy (50-60% accurate). Tick-level CVD using actual trade-by-trade data is 90%+ accurate.
- **How:**
  1. Fetch Binance aggTrades: `https://data-api.binance.vision/api/v3/aggTrades?symbol=BTCUSDT&limit=1000`
  2. Each aggTrade has `m` (isBuyerMaker) — use this to precisely classify buy/sell volume
  3. Compute rolling CVD over configurable windows (5min, 15min, 1h)
  4. Track CVD divergence with more precision
  5. Compute CVD acceleration (rate of change of CVD)
- **Data Source:** Binance aggTrades REST (free, 2400 req/min)
- **Files to Modify:**
  - `src/features/order_flow.py` — replace `calculate_cvd()` with tick-level version using aggTrades
- **Effort:** M (1-2 days)
- **Impact:** Signal quality improvement: CVD component 6/10 → 9/10
- **Priority:** P1
- **Validation:** Compare OHLCV CVD vs tick CVD predictions over 1 month of historical data

#### P2.2: Implement VPIN (Trade Flow Toxicity)
- **What:** Volume-Synchronized Probability of Informed Trading
- **Why:** VPIN is one of the best-documented microstructure signals. It predicts volatility events (flash crashes, liquidation cascades) 2-8 hours ahead.
- **How:**
  1. Create `src/features/vpin.py`
  2. Bucket Binance aggTrades into equal-volume bars (not time bars — this is key)
  3. Use Bulk Volume Classification (BVC) to estimate buy/sell in each bucket
  4. VPIN = rolling |buy_vol - sell_vol| / total_vol over N buckets (N=50 is standard)
  5. High VPIN (>0.6) = informed trading = expect large move = reduce position or prepare for reversal
- **Data Source:** Binance aggTrades (same as P2.1)
- **Files to Create:**
  - `src/features/vpin.py` — VPIN calculator with volume buckets and BVC
- **Files to Modify:**
  - `src/features/order_flow.py` — integrate VPIN as a risk filter (not directional signal)
  - `src/ui/api_server.py` — surface VPIN in market analysis
- **Effort:** M (1-2 days)
- **Impact:** Adds a volatility prediction signal not available elsewhere in the system
- **Priority:** P1
- **Validation:** Backtest VPIN on 6 months of historical BTC data; measure correlation with realized vol over next 4-8 hours
- **Libraries:** `numpy`, `pandas` only

#### P2.3: Deribit Options Data Integration
- **What:** Put/Call ratio, Max Pain, IV skew from Deribit BTC options
- **Why:** Options market is where informed institutional traders express views. P/C ratio extremes are strong contrarian indicators. IV > RV signals expected volatility.
- **How:**
  1. Create `src/features/options_sentiment.py`
  2. Fetch Deribit public data: `https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`
  3. Compute:
     - Put/Call ratio (by volume and by OI)
     - Max Pain (strike where most OI expires worthless)
     - IV surface (25-delta P/C IV skew)
     - IV term structure (front month vs back month)
  4. Normalize each to [-1, 1] trading signal
- **Data Source:** Deribit Public API (free, no key required, 10 req/s)
- **Files to Create:**
  - `src/features/options_sentiment.py` — options data fetcher and signal generator
- **Files to Modify:**
  - `src/features/alternative_data.py` — add options data to alt data features
  - `src/ui/api_server.py` — surface options sentiment in market analysis
- **Effort:** M (2-3 days)
- **Impact:** Adds institutional-grade signal; P/C ratio at extremes predicts 3-5 day moves
- **Priority:** P1
- **Validation:** Backtest P/C ratio extremes (>1.5 or <0.5) against 3-day BTC returns

#### P2.4: Google Trends Integration
- **What:** Track search interest for "bitcoin", "buy bitcoin", "crypto crash"
- **Why:** Retail sentiment proxy. Extreme spikes in search interest are contrarian signals.
- **How:**
  1. Use `pytrends` library (install: `pip install pytrends`)
  2. Track daily search interest for: ["bitcoin", "buy bitcoin", "sell bitcoin", "crypto crash"]
  3. Compute 7-day moving average
  4. Generate signal from rate of change: rapid spike = contrarian (retail FOMO/panic)
  5. Cache aggressively (daily updates sufficient)
- **Data Source:** Google Trends via `pytrends` (free, rate-limited to ~10 req/min)
- **Files to Modify:**
  - `src/features/alternative_data.py` — add `fetch_google_trends()` method
- **Effort:** S-M (4-6 hours)
- **Impact:** Adds retail sentiment signal that complements Fear & Greed
- **Priority:** P1
- **Validation:** Correlate Google Trends spikes with 3-7 day BTC returns

#### P2.5: Cross-Market Signal Integration (DXY, SPY, Gold)
- **What:** Track DXY, S&P 500, Gold correlations with BTC
- **Why:** BTC's correlation with macro assets has increased since 2022. DXY has -0.3 to -0.6 correlation with BTC.
- **How:**
  1. Use `yfinance` library for DXY (`DX-Y.NYB`), SPY, GLD
  2. Compute 20-day rolling correlation with BTC
  3. When correlation is strong (|ρ|>0.5), use macro asset movements as leading indicators
  4. Track macro momentum: if DXY falling + SPY rising = risk-on = bullish for BTC
- **Data Source:** Yahoo Finance via `yfinance` (free, 15-min delayed for US markets)
- **Files to Create:**
  - `src/features/macro_signals.py` — DXY, SPY, Gold tracking and correlation
- **Files to Modify:**
  - `src/features/correlation_engine.py` — add macro correlations alongside crypto correlations
- **Effort:** M (1 day)
- **Impact:** Adds macro regime context; particularly valuable during US trading hours
- **Priority:** P1
- **Validation:** Backtest: when DXY drops >0.5% in a day and correlation is strong, measure BTC 24h forward return
- **Libraries:** `yfinance` (install: `pip install yfinance`)

#### P2.6: Enable TFT Forecaster via Process Isolation
- **What:** Run the TFT price forecaster in a separate subprocess to avoid API deadlock
- **Why:** The TFT model is fully built and trained but disabled because loading PyTorch in the Flask thread blocks the server. Running it in a separate process with IPC solves this.
- **How:**
  1. Create `src/models/forecast_service.py` — standalone forecaster process
  2. On startup, load TFT model in subprocess, write predictions to a shared JSON file or Redis
  3. API server reads predictions from the shared file (fast, no model loading)
  4. Update predictions every 15 minutes (4 horizons: 1h, 4h, 12h, 24h)
- **Data Source:** OHLCV data from Binance (already available)
- **Files to Create:**
  - `src/models/forecast_service.py` — subprocess forecaster daemon
- **Files to Modify:**
  - `src/ui/api_server.py` — read forecasts from shared file instead of running model inline
- **Effort:** M (1-2 days)
- **Impact:** Restores a high-alpha component (quantile forecasts at 4 horizons)
- **Priority:** P1
- **Validation:** Track forecast accuracy (MAE, directional accuracy) on live data for 1 week

#### P2.7: Integrate External Signals into HTF Feature Vector
- **What:** Add 10-15 external features to the 114-dim HTF observation space and retrain
- **Why:** All current features are pure price/volume. Adding funding rate, OI, sentiment, and macro data has been shown to improve DRL agent performance by 15-25% in academic literature.
- **How:**
  1. Add features to the observation space (dimensions 114-128):
     - Funding rate (1 feature)
     - OI rate of change (1 feature)
     - L/S ratio imbalance (1 feature)
     - Fear & Greed normalized (1 feature)
     - Stablecoin supply rate of change (1 feature)
     - Order book imbalance (1 feature)
     - VPIN (1 feature)
     - P/C ratio (1 feature)
     - DXY correlation (1 feature)
     - Session encoding (3 features)
     - HMM transition probability (2 features)
  2. Update `HTFFeatureEngine` to accept external data dict
  3. Retrain the HTF model with the expanded feature vector
- **Data Source:** All sources from Phase 1 and Phase 2 improvements
- **Files to Modify:**
  - `src/features/htf_features.py` — add external data features
  - `src/env/htf_env.py` — update observation space dimension
  - `src/models/train_walkforward.py` — retrain with expanded features
- **Effort:** L (3-5 days including retraining)
- **Impact:** Expected improvement: Sharpe 3.85 → 4.2-4.8 (based on feature augmentation literature)
- **Priority:** P1 (after Phase 1 data sources are working)
- **Validation:** Walk-forward validation on held-out 2-month window; compare Sharpe before/after
- **Dependencies:** P1.3, P1.5, P2.1, P2.2, P2.3, P2.5 must be implemented first

---

### Phase 3: Advanced (2-4 Weeks) — Custom Models, Significant Engineering

#### P3.1: Liquidation Heatmap Engine
- **What:** Build a real-time liquidation level estimator using OI and leverage data
- **Why:** Liquidation cascades explain 30-40% of large crypto moves. Knowing where liquidations cluster gives a "magnetic price target."
- **How:**
  1. Create `src/features/liquidation_heatmap.py`
  2. Track OI at different leverage levels from OKX/Binance
  3. Estimate liquidation prices assuming 10x, 25x, 50x, 100x leverage
  4. Compute "liquidation imbalance" — where are more liquidations, above or below current price?
  5. Higher liquidation density above = short squeeze risk = bullish bias (and vice versa)
  6. Optionally integrate CoinGlass API for additional data
- **Data Source:** OKX OI + Binance Futures liquidation stream
- **Files to Create:**
  - `src/features/liquidation_heatmap.py`
- **Effort:** L (3-5 days)
- **Impact:** Unique signal — few retail systems have this
- **Priority:** P2
- **Validation:** Backtest: when liquidation imbalance >2:1, measure next 4H price move direction

#### P3.2: Attention-Based Signal Combiner
- **What:** Replace fixed signal weights with a learned attention network
- **Why:** Different signals have different value in different regimes. Fixed weights are suboptimal.
- **How:**
  1. Create `src/models/signal_attention.py`
  2. Small 2-layer transformer encoder (16-dim hidden, 4 heads)
  3. Input: concatenated signal vector + regime one-hot
  4. Output: attention weights per signal → weighted combination
  5. Train on 2 years of historical signal → 4H-forward-return data
  6. Deploy as inference-only (lightweight, <10ms per call)
- **Data Source:** Historical signals (must log them starting now)
- **Files to Create:**
  - `src/models/signal_attention.py` — attention-based combiner
  - `src/models/train_attention.py` — training script
  - `src/data/signal_logger.py` — log all signals with timestamps for training data
- **Effort:** L (1-2 weeks)
- **Impact:** Expected improvement: 10-25% better signal combination
- **Priority:** P2
- **Dependencies:** All Phase 1 and Phase 2 signals must be working and logged

#### P3.3: On-Chain Exchange Flow Tracker
- **What:** Track real-time BTC/ETH flows to/from exchange hot wallets using blockchain data
- **Why:** Exchange net flow is one of the strongest on-chain signals — but currently we only have the WhaleAlert API (which requires a paid key)
- **How:**
  1. Create `src/features/exchange_flow.py`
  2. Track known exchange wallets (Binance, Coinbase, Kraken — ~50 addresses)
  3. Use Blockchain.com API for BTC and Etherscan/Alchemy for ETH
  4. Compute net flow (inflow - outflow) over rolling 24h windows
  5. Z-score normalize against 30-day rolling mean
- **Data Source:**
  - Blockchain.com (free, BTC only)
  - Etherscan free tier (5 req/s with free API key)
  - Alchemy free tier (ETH)
  - CryptoQuant free tier (limited but includes net flow)
- **Files to Create:**
  - `src/features/exchange_flow.py`
- **Effort:** L (3-5 days)
- **Impact:** Adds a genuinely predictive on-chain signal
- **Priority:** P2
- **Validation:** Correlate computed net flow with CryptoQuant's verified net flow data

#### P3.4: Concept Drift Detection & Online Model Adaptation
- **What:** Detect when market behavior changes enough to degrade model performance, and adapt
- **Why:** Models lose edge over time as market regimes shift. The current system has no mechanism to detect or respond to this.
- **How:**
  1. Install `river` library for online learning
  2. Track rolling prediction error (MAE of directional prediction)
  3. Use ADWIN (Adaptive Windowing) to detect concept drift
  4. When drift detected:
     a. Flag in dashboard
     b. Automatically switch to a more conservative sizing multiplier
     c. Queue model retraining alert
  5. Optionally: maintain a lightweight online model (logistic regression) that adapts in real-time as a "model of the model"
- **Data Source:** Internal — prediction vs actual returns
- **Files to Create:**
  - `src/models/drift_detector.py` — ADWIN-based concept drift detection
- **Files to Modify:**
  - `src/models/confidence_engine.py` — integrate drift signal into position sizing
- **Effort:** M (2-3 days)
- **Impact:** Prevents catastrophic losses during regime shifts; improves long-run Sharpe stability
- **Priority:** P2
- **Libraries:** `river` (install: `pip install river`)

#### P3.5: Social Sentiment Engine (LunarCrush + CryptoPanic)
- **What:** Aggregate social media sentiment from multiple sources
- **Why:** Social sentiment at extremes is a reliable contrarian indicator. Social volume spikes precede large moves.
- **How:**
  1. Create `src/features/social_sentiment.py`
  2. Integrate:
     - LunarCrush free API — galaxy score, social volume, sentiment
     - CryptoPanic free API — news/social sentiment aggregator
     - Reddit API — post count and sentiment from r/CryptoCurrency
  3. Compute composite social score [-1, 1]
  4. Track social volume z-score (spike detection)
  5. Use as contrarian signal at extremes (z > 2 = potential reversal)
- **Data Source:**
  - LunarCrush (`https://lunarcrush.com/api4/public/coins/1/time-series/v2`) — free, 10 req/min
  - CryptoPanic (`https://cryptopanic.com/api/v1/posts/?auth_token=FREE&public=true`) — free, 5 req/min
  - Reddit API (PRAW library) — free with app registration
- **Files to Create:**
  - `src/features/social_sentiment.py`
- **Effort:** M-L (2-4 days)
- **Impact:** Adds a unique alternative data signal
- **Priority:** P2
- **Libraries:** `praw` for Reddit, `requests` for API calls

#### P3.6: Volume Profile / VWAP Features
- **What:** Compute Volume Profile (volume at price) and VWAP with standard deviation bands
- **Why:** VWAP is the institutional benchmark price. Volume Profile identifies high-volume nodes (HVN) that act as support/resistance.
- **How:**
  1. Add VWAP calculation to HTF features (price * volume / cumulative volume)
  2. Compute VWAP bands at ±1σ and ±2σ
  3. Price position relative to VWAP = institutional context
  4. Volume Profile: bin prices into buckets, find Point of Control (highest volume price)
  5. HVN proximity = support/resistance signal
- **Data Source:** OHLCV data (already available)
- **Files to Modify:**
  - `src/features/htf_features.py` — add VWAP and Volume Profile features
- **Effort:** M (2-3 days)
- **Impact:** Adds institutional-grade price context
- **Priority:** P2

---

## 4. Specific Recommendations for Our System

### 4.1 Top 3 Improvements for Biggest Edge Immediately

**#1: Wire HMM Transition Probabilities into Live Trading (P1.1)**
- **ROI:** Highest — zero new data sources needed, model already trained
- **Expected Impact:** Reduces false regime transitions by ~30%, improves entry timing
- **Time:** 2-3 hours
- **Why This First:** The HMM regime classifier provides forward-looking probabilities that the ADX detector cannot. It's literally sitting there unused in `data/models/regime/`. This is the lowest-hanging fruit in the entire codebase.

**#2: Add Order Book Imbalance (P1.3)**
- **ROI:** Very high — free Binance API, 3-4 hours of work
- **Expected Impact:** Adds a scientifically validated short-term predictor that the system completely lacks
- **Time:** 3-4 hours
- **Why This Second:** Order book imbalance is the single most well-documented microstructure alpha signal. It's free, real-time, and complements our existing order flow perfectly as a 4th layer.

**#3: Fix Whale Tracker (P1.2) + MTF Fallback (P1.4)**
- **ROI:** High — fixing broken signals is always higher ROI than adding new ones
- **Expected Impact:** Resurrects two signals from 0/10 to 7/10 quality
- **Time:** 5-6 hours
- **Why This Third:** The whale tracker and MTF together represent ~25% of the signal surface area and are both producing zero useful information right now.

### 4.2 The Single Most Impactful Signal We're NOT Using

**VPIN (Volume-Synchronized Probability of Informed Trading)**

VPIN is the most impactful signal currently absent from our system because:

1. **Unique information:** It measures the probability that current trading volume is from informed traders — no other indicator in our system captures this
2. **Predictive of volatility events:** VPIN spikes precede flash crashes and liquidation cascades by 2-8 hours with ~65% recall
3. **Complementary to existing signals:** It's a risk/volatility signal, not directional — it tells you *when* to be cautious, which is different from *what direction* to trade
4. **Free data:** Computed from Binance aggTrades which we already fetch
5. **Academic validation:** One of the most researched microstructure indicators (Easley, López de Prado & O'Hara, 2012)
6. **Implementation:** ~100 lines of Python, no ML model needed

**How to use VPIN in our system:**
- VPIN > 0.6 → reduce position size by 50% (expect volatility event)
- VPIN > 0.8 → close positions and wait (extreme informed trading)
- VPIN < 0.3 → normal conditions, trade freely
- Integrate as a "risk filter" in the confidence engine, not as a directional signal

### 4.3 How to Weight/Combine Signals for Best Ensemble

**Current Problem:** Signals are independent with fixed weights. No mechanism to adapt weights to current conditions.

**Recommended Ensemble Architecture (Near-Term):**

```
                    ┌──────────────────┐
                    │  HMM Regime      │
                    │  Classifier      │
                    └────────┬─────────┘
                             │ regime + transition probs
                    ┌────────▼─────────┐
                    │  Regime-Adaptive  │
                    │  Signal Weighting │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
   ┌─────▼─────┐     ┌──────▼──────┐    ┌───────▼──────┐
   │ Directional│     │   Risk /    │    │   Context    │
   │  Signals   │     │ Volatility  │    │   Signals    │
   ├────────────┤     ├─────────────┤    ├──────────────┤
   │ Order Flow │     │ VPIN        │    │ Session      │
   │ Funding    │     │ ATR Regime  │    │ DXY Corr     │
   │ Whale/OI   │     │ Liquidation │    │ F&G Extreme  │
   │ HTF Model  │     │ IV/RV Ratio │    │ Stablecoin Δ │
   │ Options PC │     │ BB Squeeze  │    │ Google Trends│
   └─────┬──────┘     └──────┬──────┘    └───────┬──────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Composite Score │
                    │  + Risk Filter   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Confidence      │
                    │  Engine →        │
                    │  Position Size   │
                    └──────────────────┘
```

**Regime-Adaptive Weights:**

| Signal | BULL | BEAR | RANGE | BREAKOUT |
|--------|------|------|-------|----------|
| Order Flow | 0.25 | 0.25 | 0.30 | 0.35 |
| Funding Rate | 0.10 | 0.15 | 0.10 | 0.05 |
| Whale/OI | 0.15 | 0.15 | 0.10 | 0.15 |
| HTF Model | 0.30 | 0.30 | 0.20 | 0.25 |
| Options P/C | 0.10 | 0.10 | 0.15 | 0.10 |
| Macro/Context | 0.10 | 0.05 | 0.15 | 0.10 |

**Risk Filter Application:**
- After computing directional score, multiply by risk factor:
  - VPIN > 0.6: risk_factor = 0.5
  - Liquidation imbalance extreme: risk_factor = 0.7
  - IV > 1.5x RV: risk_factor = 0.7
  - All clear: risk_factor = 1.0
- Final position = base_position × directional_confidence × risk_factor

### 4.4 Should We Retrain the HTF Model with New Features?

**Yes, but with careful staging.**

**Recommended approach:**

1. **Phase 1 (Now):** Do NOT retrain yet. First, implement and validate the new data sources. Log all new signals alongside existing signals for at least 2 weeks.

2. **Phase 2 (Week 2-3):** Add 10-15 new features to the observation space:
   - Funding rate (1 dim)
   - OI rate of change (1 dim)  
   - L/S ratio imbalance (1 dim)
   - Order book imbalance (1 dim)
   - VPIN (1 dim)
   - Fear & Greed normalized (1 dim)
   - Stablecoin supply Δ (1 dim)
   - P/C ratio (1 dim)
   - DXY correlation (1 dim)
   - Session encoding (3 dims)
   - HMM transition probs (2 dims)
   - **Total:** 114 → 128 dimensions

3. **Phase 3 (Week 3-4):** Retrain with walk-forward validation:
   - Use identical walk-forward protocol (2-month training, 2-month test)
   - Compare Sharpe, max drawdown, win rate before/after
   - **Only deploy if Sharpe improves by >0.3** (3.85 → 4.15+ minimum)
   - If improvement is <0.3, keep the original model and use new signals only for risk filtering

4. **Phase 4 (Month 2+):** Ongoing retraining cadence:
   - Retrain monthly with latest 6 months of data
   - Use concept drift detection (P3.4) to trigger emergency retrains
   - Maintain model versioning in git with performance baselines

**Key Risk:** Adding too many features at once can degrade performance (curse of dimensionality). The PPO agent may need more training steps to learn the new feature space. Budget 2x training time for the expanded model.

---

## Appendix A: API Reference for All Recommended Data Sources

| Source | URL | Free Tier | Rate Limit | Auth Required |
|--------|-----|-----------|-----------|---------------|
| Binance Spot REST | `https://data-api.binance.vision/api/v3/` | ✅ Full | 2400/min | No |
| Binance Depth | `https://data-api.binance.vision/api/v3/depth` | ✅ Full | 2400/min | No |
| Binance aggTrades | `https://data-api.binance.vision/api/v3/aggTrades` | ✅ Full | 2400/min | No |
| OKX Public | `https://www.okx.com/api/v5/public/` | ✅ Full | 20/2s | No |
| OKX Rubik | `https://www.okx.com/api/v5/rubik/stat/` | ✅ Full | 5/2s | No |
| Deribit Public | `https://www.deribit.com/api/v2/public/` | ✅ Full | 10/s | No |
| Alternative.me F&G | `https://api.alternative.me/fng/` | ✅ Full | ~30/min | No |
| CoinGecko | `https://api.coingecko.com/api/v3/` | ✅ Limited | 10-30/min | No |
| DeFiLlama | `https://stablecoins.llama.fi/` | ✅ Full | ~30/min | No |
| LunarCrush | `https://lunarcrush.com/api4/public/` | ✅ Limited | 10/min | No |
| CryptoPanic | `https://cryptopanic.com/api/v1/posts/` | ✅ Limited | 5/min | Free key |
| Yahoo Finance | via `yfinance` library | ✅ Full | ~2000/h | No |
| FRED | `https://api.stlouisfed.org/fred/` | ✅ Full | 120/min | Free key |
| Blockchain.com | `https://blockchain.info/` | ✅ Full | ~30/min | No |
| Google Trends | via `pytrends` library | ✅ Full | ~10/min | No |
| Mempool.space | `https://mempool.space/api/` | ✅ Full | 10/min | No |

## Appendix B: Python Libraries to Install

```bash
# Phase 1 (most already installed)
pip install requests numpy pandas hmmlearn

# Phase 2
pip install yfinance pytrends

# Phase 3  
pip install river praw
```

## Appendix C: Expected Performance Impact Summary

| Phase | Effort | Expected Sharpe Improvement | Expected Win Rate Improvement |
|-------|--------|---------------------------|-------------------------------|
| Phase 1 (Quick Wins) | 2 days | +0.2-0.4 (3.85 → 4.05-4.25) | +2-5% |
| Phase 2 (Medium) | 1 week | +0.3-0.6 (→ 4.35-4.85) | +3-7% |
| Phase 3 (Advanced) | 2-4 weeks | +0.2-0.5 (→ 4.55-5.35) | +2-5% |
| **Total Potential** | **1 month** | **+0.7-1.5 (→ 4.55-5.35)** | **+7-17%** |

*Note: These are estimates based on academic literature and comparable systems. Actual improvements depend on market conditions and implementation quality. Diminishing returns apply — each phase adds less marginal alpha than the previous one.*

## Appendix D: Immediate Action Items Checklist

- [ ] **P1.1** Wire HMM regime classifier into API server and live trading (2-3h)
- [ ] **P1.2** Fix whale tracker signal weights for API mode (3-4h)
- [ ] **P1.3** Implement order book imbalance signal (3-4h)
- [ ] **P1.4** Fix MTF analyzer fallback in API server (2h)
- [ ] **P1.5** Add stablecoin supply signal from DeFiLlama (2h)
- [ ] **P1.6** Add trading session features (1-2h)
- [ ] Start logging all signals with timestamps for attention model training data
- [ ] Set up walk-forward validation pipeline for A/B testing new features

---

*This document is a research and planning artifact. No code changes should be made without proper implementation, testing, and validation.*
