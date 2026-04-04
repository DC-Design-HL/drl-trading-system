---
name: market-analysis-audit
description: Deep audit and test all market analysis signal components in the DRL trading system. Use when asked to verify signal quality, test market analysis, review trading signals, or check if market data components produce real valuable signals. Covers whale tracking, order flow, regime detection, funding rates, and all signal providers.
---

# Market Analysis Audit

Comprehensive audit and testing framework for the DRL trading system's market analysis pipeline.

## When to Use

- User asks to test/verify market analysis signals
- After changes to any signal component in `src/features/` or `src/models/`
- Periodic health check of signal quality
- After adding new signal sources

## Project Location

`~/.openclaw/projects/drl-trading-system/repo`

## Signal Components to Audit

### Core Signals (src/features/)

| Component | File | Signal | Status |
|-----------|------|--------|--------|
| Whale Tracker | `whale_tracker.py` | Whale flow direction + score | Watch for key mapping bugs |
| Order Flow | `order_flow.py` | CVD, taker ratio, large orders | Usually reliable |
| Regime Detector | `regime_detector.py` | Bull/bear/range/breakout | HMM-based, needs trained model |
| MTF Analyzer | `mtf_analyzer.py` | Multi-timeframe trend | May show "Syncing..." if no fallback |
| Funding Rate | (via order_flow) | Funding rate + bias | Direct Binance API |
| Alternative Data | `alternative_data.py` | Fear & greed, BTC dominance | Cache-dependent |
| Whale Patterns | `whale_pattern_predictor.py` | ML whale predictions | Needs trained model files |
| On-chain Whales | `on_chain_whales.py` | On-chain flow analysis | Needs Etherscan/Helius API keys |
| Cross-chain Flow | `cross_chain_whale_flow.py` | Cross-chain whale flow | Needs multiple API keys |
| Correlation Engine | `correlation_engine.py` | Asset correlations | Not surfaced in API |
| Risk Manager | `risk_manager.py` | Risk signals | Not surfaced in API |

### Models (src/models/)

| Component | File | Purpose |
|-----------|------|---------|
| Regime Classifier | `regime_classifier.py` | HMM regime classification |
| Price Forecaster | `price_forecaster.py` | TFT price forecasting |
| Confidence Engine | `confidence_engine.py` | Signal confidence scoring |
| Ensemble Orchestrator | `ensemble_orchestrator.py` | Ensemble decision making |

## Audit Procedure

### Step 1: Test Live API

```bash
# Test all symbols
for sym in BTCUSDT ETHUSDT SOLUSDT XRPUSDT; do
  curl -s "http://127.0.0.1:5001/api/market?symbol=$sym" | python3 -m json.tool | head -5
done
```

Check each signal field: `whale`, `regime`, `funding`, `order_flow`, `forecast`, `news`

### Step 2: Run Existing Tests

```bash
cd ~/.openclaw/projects/drl-trading-system/repo
python3 -m pytest tests/test_market_analysis.py -v
```

### Step 3: Deep Dive Each Component

For each component, verify:
1. **Data source**: Real API calls, not hardcoded/mock
2. **Signal logic**: Financially sensible thresholds
3. **Error handling**: Graceful when data source unreachable
4. **Staleness**: Handles old cached values
5. **Output format**: Consistent structure
6. **Real value**: Useful for trading decisions

### Step 4: Check for Common Bugs

- Key mapping mismatches between component output and API server reader
- Uninitialized attributes (e.g., `oi_history` not set before use)
- `bias`/`signal`/`direction` field name inconsistencies
- Internal fields leaking to API response (`_fetched_at`)
- Division by zero in ratio calculations

## Delegation Prompt Template

When delegating to Claude Code / Builder agent:

```
Audit the market analysis pipeline in the DRL trading system.
Project: ~/.openclaw/projects/drl-trading-system/repo

1. Read MARKET_ANALYSIS_AUDIT.md for previous findings
2. Test live API: curl http://127.0.0.1:5001/api/market?symbol=BTCUSDT
3. Run existing tests: python3 -m pytest tests/test_market_analysis.py -v
4. For any failing tests, investigate and fix the root cause
5. For any signal returning null/stale data, investigate the component
6. Update MARKET_ANALYSIS_AUDIT.md with new findings
7. Commit fixes and push to dev branch
```

## Known Issues (as of 2026-03-21)

- `forecast` (TFT) intentionally disabled to prevent deadlock
- `news` disabled per user request
- `mtf` may show "Syncing..." when no bot process provides state
- Whale signals were fixed (key mapping: `recommendation` → `direction`, `combined_score` → `score`)
- `BinanceOITracker.oi_history` was uninitialized — fixed

## Test Coverage

- 110 tests in `tests/test_market_analysis.py`
- 76 tests in `tests/test_e2e_comprehensive.py`
- All use real API calls (no mocks)
