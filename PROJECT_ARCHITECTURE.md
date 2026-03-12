# DRL Trading System - Project Architecture

**Last Updated:** March 12, 2026
**System Version:** Ultimate Agent (Whale-Fused PPO-LSTM)

---

## 📋 Executive Summary

This is an autonomous **Deep Reinforcement Learning (DRL) Trading System** that uses PPO-LSTM agents to trade cryptocurrency on Binance Testnet. The system features real-time whale wallet tracking, multi-timeframe analysis, advanced feature engineering (150+ features), and a Streamlit dashboard for monitoring.

### Key Capabilities
- 🧠 **PPO-LSTM Agent** with 150+ advanced features (Wyckoff, SMC, whale patterns)
- 🐋 **Whale Pattern Prediction** using ML models trained on verified whale wallets
- 📊 **Multi-Asset Trading** (BTC, ETH, SOL, XRP)
- 🔄 **Self-Improvement Loop** (fine-tunes on successful trades every 24 hours)
- 📈 **Real-time Dashboard** with TradingView-style charts
- 🛡️ **Advanced Risk Management** (circuit breaker, adaptive SL/TP, regime detection)
- ⚡ **Live Trading** on Binance Testnet with dry-run mode

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  Streamlit Dashboard (app.py) + API Server (api_server.py)      │
│  - Real-time charts (TradingView-style)                         │
│  - Whale analytics dashboard                                    │
│  - Bot status monitoring                                        │
│  - Market analysis cards (Whale, Funding, Order Flow)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                   LIVE TRADING ORCHESTRATOR                      │
│         live_trading_multi.py (MultiAssetTradingBot)            │
│  - Multi-threaded bot execution per asset                       │
│  - Global portfolio management                                  │
│  - 30-min cooldown after losses                                 │
│  - 4-hour minimum hold time                                     │
└─────┬───────────────────────────────────┬───────────────────────┘
      │                                   │
┌─────▼─────────────────┐       ┌────────▼──────────────────────┐
│   DRL BRAIN           │       │   MARKET INTELLIGENCE         │
│  (PPO-LSTM Agent)     │       │                               │
│                       │       │  1. Whale Tracker             │
│ • UltimateFeatureEng  │       │     - Pattern Predictor       │
│ • VecNormalize        │       │     - Wallet Collector        │
│ • 150+ features       │       │     - Registry (ETH/SOL/XRP)  │
│ • Model inference     │       │                               │
│                       │       │  2. Order Flow Analyzer       │
│ Models:               │       │     - CVD, OI, funding rates  │
│ ├─ ultimate_agent.zip │       │                               │
│ └─ vec_normalize.pkl  │       │  3. Multi-Timeframe Analyzer  │
│                       │       │     - 4h, 1d, 1w timeframes   │
│                       │       │                               │
│                       │       │  4. Regime Detector           │
│                       │       │     - Trending/Ranging        │
│                       │       │     - ADX/ATR-based           │
│                       │       │                               │
│                       │       │  5. TFT Price Forecaster      │
│                       │       │     - Neural price prediction │
│                       │       │     - Confidence scoring      │
└───────────────────────┘       └───────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────────────┐
│                      DATA PIPELINE                              │
│                                                                  │
│  1. Historical Data (Multi-Asset Fetcher)                       │
│     └─ CCXT → Binance API → CSV cache                          │
│                                                                  │
│  2. Whale Wallet Data                                           │
│     ├─ ETH: 8 verified wallets (Binance, Bitfinex, etc.)       │
│     ├─ SOL: 11 verified wallets                                │
│     └─ XRP: 13 verified wallets                                │
│                                                                  │
│  3. Alternative Data                                            │
│     ├─ Fear & Greed Index                                      │
│     └─ BTC Dominance                                            │
│                                                                  │
│  4. Storage Layer                                               │
│     ├─ SQLite (trading.db) - trades, state                     │
│     ├─ JSON - whale data, backtest reports                     │
│     └─ CSV - historical OHLCV                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
drl-trading-system/
├── .agent/                       # Agent workflow definitions
│   └── workflows/
│       ├── feature.md            # Feature development workflow
│       ├── fix.md                # Bug fix workflow
│       ├── train.md              # Model training workflow
│       └── deploy.md             # Deployment workflow
│
├── config/
│   └── config.yaml               # System configuration (exchange, risk, model params)
│
├── data/                         # All data storage
│   ├── historical/               # Cached OHLCV data (CSV)
│   ├── models/                   # Trained DRL models
│   │   ├── ultimate_agent.zip    # Main PPO model
│   │   ├── ultimate_agent_vec_normalize.pkl
│   │   ├── tft/                  # TFT price forecaster models
│   │   └── multi_asset/          # Asset-specific fine-tuned models
│   ├── whale_wallets/            # Whale wallet transaction data
│   │   ├── eth/                  # 8 verified ETH whale wallets
│   │   ├── sol/                  # 11 verified SOL whale wallets
│   │   └── xrp/                  # 13 verified XRP whale wallets
│   ├── alternative_cache/        # Fear/Greed, BTC dominance
│   ├── checkpoints/              # Training checkpoints
│   ├── backtest_report.json      # Latest backtest results
│   └── trading.db                # SQLite database (trades, positions, state)
│
├── src/                          # Source code
│   ├── env/                      # Gymnasium trading environments
│   │   ├── ultimate_env.py       # Main env (150+ features)
│   │   ├── advanced_env.py       # Advanced features env
│   │   ├── trading_env.py        # Base trading env
│   │   └── rewards.py            # Reward functions (Sharpe/Sortino)
│   │
│   ├── brain/                    # DRL agent & training
│   │   ├── agent.py              # PPO-LSTM wrapper
│   │   ├── trainer.py            # Training loops
│   │   └── replay_buffer.py      # Experience replay
│   │
│   ├── features/                 # Feature engineering modules
│   │   ├── ultimate_features.py  # 150+ feature engine (Wyckoff, SMC, etc.)
│   │   ├── whale_tracker.py      # Real-time whale monitoring (46KB)
│   │   ├── whale_pattern_predictor.py  # ML-based whale signal generator
│   │   ├── whale_wallet_collector.py   # Scrapes whale wallet data
│   │   ├── whale_wallet_registry.py    # Verified wallet addresses
│   │   ├── order_flow.py         # CVD, OI, funding rate analysis (25KB)
│   │   ├── mtf_analyzer.py       # Multi-timeframe analysis
│   │   ├── regime_detector.py    # Market regime classification
│   │   ├── risk_manager.py       # Adaptive risk management
│   │   ├── correlation_engine.py # Multi-asset correlation
│   │   └── on_chain_whales.py    # On-chain whale watcher
│   │
│   ├── models/                   # ML models (non-DRL)
│   │   ├── whale_pattern_learner.py  # Random Forest for whale patterns (30KB)
│   │   ├── price_forecaster.py   # TFT (Temporal Fusion Transformer)
│   │   ├── confidence_engine.py  # Signal confidence scoring
│   │   ├── regime_classifier.py  # Regime classification model
│   │   └── ensemble_orchestrator.py  # Model ensemble coordination
│   │
│   ├── data/                     # Data fetching & storage
│   │   ├── multi_asset_fetcher.py  # CCXT data fetcher
│   │   ├── storage.py            # Database abstraction layer
│   │   ├── candle_stream.py      # Real-time candle streaming
│   │   └── whale_stream.py       # Real-time whale data streaming
│   │
│   ├── api/                      # Exchange integration & execution
│   │   ├── binance.py            # Binance API wrapper
│   │   ├── executor.py           # Order execution engine
│   │   ├── risk_manager.py       # Pre-execution risk checks
│   │   └── portfolio_manager.py  # Global portfolio coordination
│   │
│   ├── backtest/                 # Backtesting engine
│   │   ├── engine.py             # Main backtest executor
│   │   └── data_loader.py        # Historical data loader
│   │
│   └── ui/                       # User interface
│       ├── app.py                # Streamlit dashboard (2470 lines)
│       ├── api_server.py         # Flask API server (21KB)
│       ├── charts.py             # TradingView chart components
│       └── components.py         # Reusable UI components
│
├── logs/                         # All logs
│   ├── trading_log.json          # Trade history
│   ├── multi_asset_state.json    # Bot state per asset
│   ├── tensorboard/              # TensorBoard training logs
│   └── training/                 # Training session logs
│
├── Main Scripts                  # Top-level executable scripts
│   ├── run.py                    # Legacy single-asset runner
│   ├── live_trading_multi.py     # Multi-asset live trading (68KB)
│   ├── train_ultimate.py         # Ultimate agent training (17KB)
│   ├── train_whale_patterns.py   # Train whale ML models
│   ├── backtest_strategy.py      # Strategy backtester (28KB)
│   ├── launch_dashboard.sh       # Start Streamlit UI
│   └── start.sh                  # Production startup script
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker deployment config
├── .env                          # Environment variables (API keys)
└── README.md                     # User-facing documentation
```

---

## 🧩 Core Components Deep Dive

### 1. **DRL Agent (PPO-LSTM)**

**Location:** `src/brain/agent.py`, `src/env/ultimate_env.py`

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Architecture:** LSTM policy network for temporal dependencies
- **Observation Space:** 153 dimensions (150 features + 3 position state variables)
- **Action Space:** Discrete(3) → [0: Hold, 1: Buy/Long, 2: Sell/Short]
- **Reward Function:** Sharpe/Sortino ratio-based (risk-adjusted returns)

**Training Pipeline:**
1. Fetch historical data (1-year, 1h timeframe)
2. Compute 150+ features via UltimateFeatureEngine
3. Create Gymnasium env (UltimateTradingEnv)
4. Wrap with VecNormalize for observation scaling
5. Train PPO agent (500k-2M timesteps)
6. Save model + VecNormalize stats

**Key Training Files:**
- `train_ultimate.py` - Main training script
- `train_multi_asset.py` - Multi-asset transfer learning
- `train_whale_patterns.py` - Whale pattern ML training

### 2. **Feature Engineering (150+ Features)**

**Location:** `src/features/ultimate_features.py`

**Feature Categories:**
1. **Technical Indicators** (40+ features)
   - RSI, MACD, Bollinger Bands, ATR, ADX
   - EMA crossovers (9/21, 50/200)
   - Volume indicators (OBV, MFI)

2. **Wyckoff Analysis** (20+ features)
   - Accumulation/Distribution phases
   - Spring/Upthrust detection
   - Volume spread analysis

3. **Smart Money Concepts (SMC)** (15+ features)
   - Order blocks
   - Fair value gaps
   - Liquidity zones

4. **Multi-Timeframe** (30+ features)
   - 4h, 1d, 1w trend alignment
   - Higher timeframe support/resistance

5. **Whale Patterns** (20+ features)
   - Exchange dump ratio
   - Accumulator hoard ratio
   - Flow velocity, momentum
   - Wallet-specific hit rates

6. **Market Regime** (10+ features)
   - Trending/Ranging classification
   - Volatility regime

7. **Correlation Features** (15+ features)
   - BTC dominance
   - Multi-asset correlation matrix
   - Fear & Greed Index

### 3. **Whale Tracking System**

**Location:** `src/features/whale_tracker.py`, `src/models/whale_pattern_learner.py`

**Verified Whale Wallets:**
- **ETH (8 wallets):** Binance, Bitfinex, Kraken hot/cold wallets
- **SOL (11 wallets):** Major exchange wallets + accumulators
- **XRP (13 wallets):** Ripple, exchanges, known whales

**Data Collection:**
1. `whale_wallet_collector.py` scrapes blockchain APIs (Etherscan, Solscan, XRPScan)
2. Stores transaction history in JSON files (`data/whale_wallets/`)
3. Updates every 1 hour (configurable)

**Pattern Learning:**
1. `whale_pattern_learner.py` trains Random Forest models per chain
2. Features: flow velocity, exchange dump ratio, accumulator hoard ratio, wallet-specific patterns
3. Predicts price impact (momentum signal: -1 to +1)
4. Wallets weighted by historical hit rate (>55% = 2x weight)

**Real-time Prediction:**
1. `whale_pattern_predictor.py` loads trained models
2. Fetches recent wallet data from cache
3. Computes flow features
4. Returns aggregated signal with confidence score

### 4. **Live Trading Pipeline**

**Location:** `live_trading_multi.py`

**Flow:**
```
1. Initialize MultiAssetTradingBot for each asset (BTC, ETH, SOL, XRP)
2. Load ultimate_agent.zip + vec_normalize.pkl
3. Initialize feature engines (UltimateFeatureEngine, WhaleTracker, etc.)
4. Loop every 5 minutes:
   a. Fetch latest OHLCV data
   b. Compute 150+ features
   c. Normalize observation with VecNormalize
   d. Get PPO model prediction (action 0/1/2)
   e. Compute confidence score (TFT forecast + whale signals + regime)
   f. Execute trade if confidence > 0.6
   g. Manage position (trailing SL, TP, min hold time)
   h. Update state to database
5. Self-improvement: Every 24h, fine-tune on high-reward trades
```

**Risk Management:**
- **Circuit Breaker:** Stops trading at 5% daily loss
- **Position Sizing:** 25% of balance per trade (was 50%)
- **Stop Loss:** 2.5% (adaptive based on regime)
- **Take Profit:** 5% (2:1 R:R ratio)
- **Trailing Stop:** 60% of max profit
- **Min Hold Time:** 4 hours
- **Cooldown:** 30 minutes after stop loss hit

### 5. **Backtesting System**

**Location:** `backtest_strategy.py`, `src/backtest/engine.py`

**Process:**
1. Load historical data (default: 1 year)
2. Replay EXACT live trading pipeline:
   - Same feature computation
   - Same VecNormalize scaling
   - Same PPO model
   - Same risk management rules
3. Track all trades, equity curve, Sharpe ratio
4. Save report to `data/backtest_report.json`

**Metrics:**
- Total Return (%)
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown (%)
- Win Rate (%)
- Average Trade Duration
- Total Trades

### 6. **Streamlit Dashboard**

**Location:** `src/ui/app.py` (2470 lines)

**Pages:**
1. **Bot Status**
   - Current position, PnL, equity
   - Model info (last trained, confidence)
   - Start/Stop bot controls

2. **Charts**
   - TradingView-style candlestick charts
   - Buy/Sell signal markers
   - Support/Resistance levels
   - Volume bars

3. **Whale Analytics**
   - Real-time whale flow signals
   - Top wallets by accuracy
   - Exchange dump ratio trends

4. **Market Analysis**
   - Funding rates
   - Order flow (CVD, OI)
   - Multi-timeframe alignment
   - Regime detection

5. **Trade History**
   - All executed trades
   - PnL breakdown
   - Performance metrics

**API Server:** `src/ui/api_server.py`
- Flask REST API for real-time data
- 60-second caching for external APIs
- Endpoints: `/whale`, `/funding`, `/order_flow`, `/trades`

---

## 🔄 Data Flow

### Training Flow
```
Historical Data (CSV)
  → DataLoader
  → UltimateTradingEnv
  → VecNormalize
  → PPO.learn()
  → Save model + VecNormalize stats
```

### Live Trading Flow
```
Binance API (real-time)
  → MultiAssetDataFetcher
  → Feature Engines (Whale, MTF, OrderFlow, etc.)
  → UltimateFeatureEngine (150+ features)
  → VecNormalize
  → PPO.predict()
  → Risk Manager
  → Order Executor
  → Database (trading.db)
```

### Whale Tracking Flow
```
Blockchain APIs (Etherscan, Solscan, XRPScan)
  → WhaleWalletCollector
  → JSON cache (data/whale_wallets/)
  → WhalePatternLearner (Random Forest)
  → WhalePatternPredictor
  → Signal aggregation
  → Live Trading Bot
```

---

## 🛠️ Technology Stack

### Core Frameworks
- **DRL:** `stable-baselines3` (PPO), `gymnasium` (env)
- **ML:** `scikit-learn` (Random Forest), `hmmlearn` (regime detection)
- **Neural Networks:** `torch` (TFT forecaster)
- **UI:** `streamlit`, `streamlit-lightweight-charts`, `plotly`
- **API:** `flask`, `flask-cors`

### Data & Exchange
- **Exchange:** `ccxt` (Binance API)
- **Data Processing:** `pandas`, `numpy`
- **Database:** SQLite (`sqlite3`), JSON, CSV

### Utilities
- **Config:** `pyyaml`, `python-dotenv`
- **Testing:** `pytest`, `pytest-asyncio`
- **Deployment:** Docker, Hugging Face Spaces

---

## 🔐 Security & Configuration

### Environment Variables (`.env`)
```bash
BINANCE_TESTNET_API_KEY=<key>
BINANCE_TESTNET_API_SECRET=<secret>
BINANCE_PROXY=<optional_proxy>
HF_TOKEN=<huggingface_token>
ETHERSCAN_API_KEY=<etherscan_key>
SOLSCAN_API_KEY=<solscan_key>
```

### Risk Parameters (`config/config.yaml`)
- Max Daily Loss: 5%
- Max Drawdown: 20%
- Stop Loss: 2.5%
- Take Profit: 5%
- Position Size: 25%

---

## 📊 Model Performance

### Ultimate Agent (Latest Training)
- **Training Data:** 1 year (2024-2025), 1h timeframe
- **Assets:** BTC, ETH, SOL, XRP
- **Timesteps:** 2M+ per asset
- **Validation Sharpe:** ~1.2-1.8 (asset-dependent)
- **Backtest Win Rate:** 55-65%

### Whale Pattern Models
- **ETH Whale Model:** 62% hit rate (top wallets)
- **SOL Whale Model:** 58% hit rate
- **XRP Whale Model:** 60% hit rate

---

## 🚀 Deployment

### Hugging Face Spaces
- **Space:** `chen470/drl-trading-bot`
- **Runtime:** Docker container
- **Auto-deploy:** Triggered by `git push origin main`
- **Build Time:** ~90 seconds
- **Logs:** Via HF API + `api_server.log`

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest
python backtest_strategy.py

# Start dashboard
streamlit run src/ui/app.py

# Run live trading (dry-run)
python live_trading_multi.py --assets BTCUSDT ETHUSDT --dry-run
```

---

## 📝 Key Files Reference

### Configuration
- `config/config.yaml` - All system parameters
- `.env` - API keys & secrets
- `requirements.txt` - Python dependencies

### Training
- `train_ultimate.py` - Main DRL training
- `train_whale_patterns.py` - Whale ML training
- `train_multi_asset.py` - Multi-asset transfer learning

### Live Trading
- `live_trading_multi.py` - Multi-asset trading bot (68KB)
- `src/api/executor.py` - Order execution
- `src/api/portfolio_manager.py` - Portfolio coordination

### Backtesting
- `backtest_strategy.py` - Strategy backtester (28KB)
- `src/backtest/engine.py` - Backtest engine

### UI
- `src/ui/app.py` - Streamlit dashboard (2470 lines)
- `src/ui/api_server.py` - Flask API server (21KB)

### Feature Engineering
- `src/features/ultimate_features.py` - Main feature engine
- `src/features/whale_tracker.py` - Whale tracking (46KB)
- `src/features/whale_pattern_predictor.py` - Whale ML predictor

---

## 🐛 Known Issues & Limitations

1. **scikit-learn 1.7.2 pinned** - Prevents pickle OOM crash on Hugging Face
2. **Proxy required for some APIs** - Binance Futures API, Etherscan (rate limits)
3. **TFT forecaster optional** - Falls back gracefully if model not trained
4. **Whale data collection blocking** - Should be async/background cron
5. **VecNormalize dependency** - Model predictions fail without proper normalization stats

---

## 🎯 Future Roadmap

1. **Async whale data collection** - Background scraper instead of blocking API calls
2. **Advanced ensemble methods** - Combine DRL + TFT + whale patterns more intelligently
3. **Real money trading** - Migrate from Testnet to production (with proper safeguards)
4. **More chains** - Add BTC-native whale tracking (currently uses ETH proxy)
5. **Improved UI** - Add more charts, alerts, mobile responsiveness
6. **Paper trading mode** - Simulate trades without Binance API

---

## 📚 Learning Resources

### Understanding the System
1. Start with `README.md` for high-level overview
2. Read `.agent/workflows/*.md` for development workflows
3. Study `config/config.yaml` for all parameters
4. Explore `src/env/ultimate_env.py` to understand the environment
5. Dive into `live_trading_multi.py` to see how it all connects

### Key Concepts
- **PPO (Proximal Policy Optimization):** DRL algorithm that balances exploration/exploitation
- **VecNormalize:** Critical for normalizing observations to prevent gradient explosion
- **Sharpe Ratio:** Risk-adjusted return metric (reward function)
- **Whale Tracking:** Monitor large holders to predict price movements
- **Wyckoff Analysis:** Market phase detection (accumulation, distribution)
- **Smart Money Concepts:** Institutional order flow analysis

---

## 🤝 Contributing

See `.agent/workflows/` for development workflows:
- `feature.md` - Adding new features
- `fix.md` - Bug fixes
- `train.md` - Model training
- `deploy.md` - Deployment process

---

## 📄 License

MIT License - See LICENSE file

---

**Document Version:** 1.0
**Last Reviewed:** March 12, 2026
**Maintainer:** DRL Trading System Team
