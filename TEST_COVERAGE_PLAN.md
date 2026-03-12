# DRL Trading System - Test Coverage Plan

**Created:** March 12, 2026
**QA Engineer:** Claude Sonnet 4.5
**Target Coverage:** >80% overall, >90% for critical modules

---

## Executive Summary

This document outlines a comprehensive test coverage plan for the DRL Trading System. Testing is prioritized by criticality: core logic first (feature engines, DRL brain, risk management), followed by data processing, API endpoints, and integrations.

---

## Priority 1: Core Logic (CRITICAL - 90%+ coverage target)

### 1.1 Feature Engineering (`src/features/`)

**Files to test:**
- `ultimate_features.py` (888 LOC) - Main feature engine
- `whale_pattern_predictor.py` (260 LOC) - Whale ML predictor
- `whale_tracker.py` (1172 LOC) - Real-time whale monitoring
- `regime_detector.py` - Market regime classification
- `order_flow.py` - CVD, OI, funding analysis
- `mtf_analyzer.py` - Multi-timeframe analysis
- `risk_manager.py` (features) - Adaptive risk

**Test cases:**
- ✓ Feature computation correctness (RSI, MACD, Bollinger Bands)
- ✓ Wyckoff phase detection accuracy
- ✓ SMC feature extraction (order blocks, FVG, BOS)
- ✓ Multi-timeframe feature aggregation
- ✓ Whale pattern signal generation
- ✓ No NaN/inf in feature outputs
- ✓ Feature dimensions match expected (150+)
- ✓ Edge cases: empty DataFrame, single row, all NaN columns
- ✓ Division by zero handling
- ✓ Whale model loading and prediction
- ✓ Regime detection (trending vs ranging)
- ✓ Correlation feature computation

### 1.2 DRL Brain (`src/env/`, `src/brain/`)

**Files to test:**
- `ultimate_env.py` (300+ LOC) - Main trading environment
- `rewards.py` - Reward function (Sharpe/Sortino)
- `agent.py` - PPO-LSTM wrapper
- `trainer.py` - Training loops

**Test cases:**
- ✓ Observation space dimensions (153)
- ✓ Action space validity (0, 1, 2)
- ✓ Environment reset functionality
- ✓ Step function correctness
- ✓ Reward calculation (Sharpe ratio)
- ✓ Model loading (ultimate_agent.zip)
- ✓ VecNormalize stats loading
- ✓ Prediction determinism (same input → same output)
- ✓ No NaN/inf in observations
- ✓ Episode termination conditions
- ✓ Position state tracking
- ✓ Balance updates after trades

### 1.3 Risk Management (`src/api/risk_manager.py`)

**Files to test:**
- `risk_manager.py` (300+ LOC) - Circuit breaker and controls

**Test cases:**
- ✓ Circuit breaker triggers at 5% daily loss
- ✓ Max drawdown enforcement (20%)
- ✓ Position sizing limits (25%)
- ✓ Daily metrics tracking
- ✓ Cooldown period enforcement
- ✓ Mode transitions (ACTIVE → CIRCUIT_BREAKER)
- ✓ Risk assessment scoring
- ✓ Trade rejection on breaker trip

---

## Priority 2: Data Processing & Validation (80%+ coverage)

### 2.1 Data Fetching (`src/data/`)

**Files to test:**
- `multi_asset_fetcher.py` - CCXT data fetching
- `storage.py` - Database abstraction
- `candle_stream.py` - Real-time streaming
- `whale_stream.py` - Whale data streaming

**Test cases:**
- ✓ Historical data fetching (mocked CCXT)
- ✓ CSV caching mechanism
- ✓ Cache invalidation (TTL)
- ✓ Empty response handling
- ✓ API rate limit handling
- ✓ Network timeout handling
- ✓ Data validation (required columns)
- ✓ SQLite storage operations
- ✓ JSON serialization/deserialization

### 2.2 Whale Data Collection (`src/features/whale_*.py`)

**Files to test:**
- `whale_wallet_collector.py` (666 LOC)
- `whale_wallet_registry.py` (436 LOC)

**Test cases:**
- ✓ Wallet registry loading (ETH/SOL/XRP)
- ✓ Transaction data collection (mocked APIs)
- ✓ JSON cache writing
- ✓ Flow feature computation
- ✓ Exchange dump ratio calculation
- ✓ Accumulator hoard ratio calculation
- ✓ Invalid wallet address handling
- ✓ API error handling (Etherscan, Solscan, XRPScan)

---

## Priority 3: Live Trading Logic (90%+ coverage)

### 3.1 Live Trading Bot (`live_trading_multi.py`)

**Files to test:**
- `live_trading_multi.py` (1700+ LOC)
- `src/api/portfolio_manager.py`

**Test cases:**
- ✓ Bot initialization
- ✓ Model + VecNormalize loading
- ✓ Position tracking (long/short/flat)
- ✓ Stop loss calculation
- ✓ Take profit calculation
- ✓ Trailing stop logic
- ✓ Min hold time enforcement (4 hours)
- ✓ Cooldown after loss (30 minutes)
- ✓ Position sizing (25% of balance)
- ✓ PnL calculation
- ✓ Trade logging
- ✓ Dry-run mode (no real orders)
- ✓ Multi-asset coordination

### 3.2 Order Execution (`src/api/`)

**Files to test:**
- `executor.py` (300+ LOC)
- `binance.py` - Exchange connector
- `portfolio_manager.py`

**Test cases:**
- ✓ Order creation (market/limit)
- ✓ Order validation
- ✓ Risk manager integration
- ✓ Order status tracking
- ✓ Filled order handling
- ✓ Rejected order handling
- ✓ Dry-run order simulation
- ✓ Balance validation before trade
- ✓ Portfolio allocation across assets

---

## Priority 4: Backtesting (80%+ coverage)

### 4.1 Backtest Engine (`src/backtest/`)

**Files to test:**
- `engine.py` (300+ LOC)
- `data_loader.py`

**Test cases:**
- ✓ Backtest execution
- ✓ Equity curve generation
- ✓ Performance metrics calculation
- ✓ Sharpe ratio computation
- ✓ Win rate calculation
- ✓ Max drawdown tracking
- ✓ Trade history logging
- ✓ Report JSON generation
- ✓ Insufficient data handling

---

## Priority 5: API & UI (70%+ coverage)

### 5.1 Flask API Server (`src/ui/api_server.py`)

**Files to test:**
- `api_server.py` (484 LOC)

**Test cases:**
- ✓ Server initialization
- ✓ `/whale` endpoint
- ✓ `/funding` endpoint
- ✓ `/order_flow` endpoint
- ✓ `/trades` endpoint
- ✓ Caching mechanism (60s TTL)
- ✓ CORS headers
- ✓ Error responses (500, 404)
- ✓ JSON serialization

### 5.2 Streamlit Dashboard (`src/ui/app.py`)

**Files to test:**
- `app.py` (2470 LOC) - Manual testing primarily
- `charts.py` (369 LOC)
- `components.py` (369 LOC)

**Test cases:**
- Manual: Dashboard loads without errors
- Manual: Charts render correctly
- Manual: Real-time updates work
- Unit: Chart data transformation
- Unit: Component rendering logic

---

## Priority 6: ML Models (80%+ coverage)

### 6.1 Whale Pattern Learner (`src/models/whale_pattern_learner.py`)

**Files to test:**
- `whale_pattern_learner.py` (795 LOC)
- `price_forecaster.py` (660 LOC)
- `confidence_engine.py` (104 LOC)

**Test cases:**
- ✓ Model training (mocked data)
- ✓ Model saving/loading
- ✓ Feature extraction for training
- ✓ Prediction generation
- ✓ Confidence scoring
- ✓ Wallet accuracy weighting
- ✓ TFT forecaster integration

---

## Edge Cases & Stress Tests

### Edge Cases (Must Test)
- [ ] Empty DataFrame (no market data)
- [ ] Single row DataFrame
- [ ] All NaN values in a column
- [ ] Division by zero in indicators
- [ ] API rate limit exceeded
- [ ] Network timeout
- [ ] Invalid API keys
- [ ] Market closed (no new data)
- [ ] Model file missing
- [ ] VecNormalize stats missing
- [ ] Corrupted whale data JSON
- [ ] Zero balance (no buying power)

### Stress Tests (Optional)
- [ ] 1M+ rows of data (memory usage)
- [ ] 24 hours continuous bot operation
- [ ] Multiple assets trading simultaneously
- [ ] High-frequency predictions (100+ per second)

---

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_features/
│   ├── __init__.py
│   ├── test_ultimate_features.py
│   ├── test_whale_predictor.py
│   ├── test_whale_tracker.py
│   ├── test_regime_detector.py
│   ├── test_order_flow.py
│   └── test_mtf_analyzer.py
├── test_env/
│   ├── __init__.py
│   ├── test_ultimate_env.py
│   ├── test_rewards.py
│   └── test_model_loading.py
├── test_risk/
│   ├── __init__.py
│   ├── test_risk_manager.py
│   └── test_circuit_breaker.py
├── test_data/
│   ├── __init__.py
│   ├── test_fetcher.py
│   ├── test_storage.py
│   └── test_whale_collector.py
├── test_api/
│   ├── __init__.py
│   ├── test_executor.py
│   ├── test_api_server.py
│   └── test_portfolio_manager.py
├── test_backtest/
│   ├── __init__.py
│   ├── test_engine.py
│   └── test_data_loader.py
├── test_live_trading/
│   ├── __init__.py
│   ├── test_bot_initialization.py
│   ├── test_position_management.py
│   └── test_trade_execution.py
└── test_models/
    ├── __init__.py
    ├── test_whale_learner.py
    ├── test_forecaster.py
    └── test_confidence_engine.py
```

---

## Testing Tools & Configuration

### Dependencies (already in requirements.txt)
- `pytest>=7.4.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov` (add if needed for coverage)
- `pytest-mock` (add if needed)

### Coverage Commands
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test module
pytest tests/test_features/test_ultimate_features.py -v

# Run with markers (unit, integration, slow)
pytest -m unit
pytest -m "not slow"

# Generate coverage report
open htmlcov/index.html
```

### Coverage Goals
- **Overall:** >80%
- **Critical modules:** >90%
  - `src/features/ultimate_features.py`
  - `src/env/ultimate_env.py`
  - `src/api/risk_manager.py`
  - `live_trading_multi.py`
- **Nice to have:** >70%
  - `src/ui/api_server.py`
  - `src/data/*`
  - `src/backtest/*`

---

## Bug Discovery Protocol

When bugs are discovered during testing:

1. **Document the bug** in a markdown file (`BUGS_FOUND.md`)
2. **Create a test case** that reproduces the bug
3. **Fix the bug** in the source code
4. **Verify the fix** by running the test
5. **Commit with message**: `Fix: [description] (found during QA testing)`

---

## Test Execution Order

1. **Unit tests first** - Isolated component testing
2. **Integration tests** - Component interaction testing
3. **End-to-end tests** - Full system flow testing
4. **Performance tests** - Latency and memory benchmarks

---

## Success Criteria

- [ ] All tests pass
- [ ] Coverage >80% overall
- [ ] Coverage >90% for critical modules
- [ ] All edge cases tested
- [ ] Bugs documented and fixed
- [ ] Test execution time <5 minutes
- [ ] No flaky tests (deterministic)

---

**Next Steps:**
1. Set up test infrastructure (conftest.py, fixtures)
2. Implement Priority 1 tests (core logic)
3. Run tests and fix bugs
4. Implement Priority 2-6 tests
5. Generate coverage report
6. Document findings

---

**Document Version:** 1.0
**Status:** Ready for implementation
