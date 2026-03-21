# DRL Trading System — Anti-Overfitting Retraining Plan

**Prepared:** 2026-03-20
**Analyst:** Quantitative ML Engineer
**Status:** Ready to execute

---

## Executive Summary

The current Ultimate Agent backtest (+5515% return, Sharpe 10.7) is **almost certainly overfit**. The basic agent's -5% return on the same period is a more realistic baseline. The TFT forecaster's 45–55% directional accuracy represents marginal edge, if any. This plan describes a production-grade retraining pipeline with walk-forward cross-validation, strict data splits, anti-overfitting measures, and realistic performance targets.

---

## 1. Root Cause Analysis of Overfitting

### Critical Bugs Found

| Severity | File | Issue | Impact |
|----------|------|--------|--------|
| CRITICAL | `retrain_ultimate_v3.py:143` | Regime detection reads `current_step + 100` — explicit 100-bar look-ahead | Agent sees future price action during training |
| HIGH | `train_ultimate.py:363` | VecNormalize stats not applied during evaluation | Train/eval distribution mismatch inflates backtest metrics |
| HIGH | `ultimate_env.py:271` | TP bonus (+0.10) is 2× the SL penalty (-0.05) | Agent learns to run winners / cut losers in training data only |
| HIGH | `retrain_ultimate_v3.py:92` | Regime multipliers (1.5×–2×) reward scaling | Overfits to specific regime patterns seen in training |
| HIGH | `retrain_ultimate_v3.py:114` | Holding bonus scales linearly with time (up to +50) | Agent learns to hold regardless of P&L |
| MEDIUM | `train_ultimate.py:110` | Early stopping monitors training reward, not validation | Stops on wrong signal; doesn't prevent generalization failure |
| MEDIUM | `ultimate_env.py:277` | Unrealized P&L reward during hold steps | Rewards mark-to-market drift, not actual edge |

### Structural Overfitting Causes

1. **Only 180 days of training data** — one crypto bull cycle, essentially one market regime
2. **No true out-of-sample test** — 15% holdout from same bull run is not independent
3. **150+ engineered features** — massive feature space relative to signal; many are derived from each other
4. **256→256→128 network** on 101 features — 250K parameters, severely over-parameterized for a noisy financial signal
5. **Regime specialists** trained on filtered regime subsets — tiny training sets, extremely prone to memorization
6. **No regularization** — no dropout, no weight decay, no L2

---

## 2. Data Requirements

### Minimum Viable Dataset

- **3 full years** of 1h data per asset (≈26,280 bars each)
- Covers multiple market regimes: 2023 bear/recovery, 2024 bull, 2025 correction/range
- All 4 assets: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT

### Download Command

```bash
python download_historical_data.py --years 3 --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT --interval 1h --output-dir data/historical
```

Expected output: 4 CSV files in `data/historical/`, ~26K rows each, total ~420MB.

### Data Quality Checks

- No gaps > 2 hours (Binance occasionally has brief outages)
- Completeness > 97%
- No zero/negative prices
- High ≥ Low ≥ Open/Close range consistency

---

## 3. Walk-Forward Validation Methodology

### Window Design

```
Total data: ~3 years (Jan 2023 → Mar 2026)
Train window: 12 months
Val window:   3 months (final quarter of train window, used for early stopping)
Test window:  3 months (strictly out-of-sample, never seen during training)
Slide step:   3 months (non-overlapping test periods)
```

### Fold Schedule (with 3-year dataset)

```
Fold  Train period          Val (in-window)       Test (OOS)
----  -------------------   -------------------   -------------------
  1   2023-01 → 2023-12     2023-10 → 2023-12     2024-01 → 2024-03
  2   2023-04 → 2024-03     2024-01 → 2024-03     2024-04 → 2024-06
  3   2023-07 → 2024-06     2024-04 → 2024-06     2024-07 → 2024-09
  4   2023-10 → 2024-09     2024-07 → 2024-09     2024-10 → 2024-12
  5   2024-01 → 2024-12     2024-10 → 2024-12     2025-01 → 2025-03
  6   2024-04 → 2025-03     2025-01 → 2025-03     2025-04 → 2025-06
  7   2024-07 → 2025-06     2025-04 → 2025-06     2025-07 → 2025-09
  8   2024-10 → 2025-09     2025-07 → 2025-09     2025-10 → 2025-12
  9   2025-01 → 2025-12     2025-10 → 2025-12     2026-01 → 2026-03
```

**Key invariant:** No fold's test period overlaps with any fold's training period. The test folds cover the full 2024–2026 span consecutively with no gaps and no leakage.

### Why This Design

- 12-month train window captures all four seasons / at least 1 full bull-bear cycle
- 3-month val window is large enough for reliable Sharpe estimation (~2000 hourly bars)
- 3-month test periods are fully non-overlapping → aggregate OOS metrics are not inflated
- Sliding by 3 months gives 9 folds → robust statistical estimate of expected live performance

---

## 4. Anti-Overfitting Measures

### 4.1 Reward Function Reform (Critical)

Replace the current complex reward with a **minimalist, unbiased reward**:

```python
# CLEAN REWARD FUNCTION
reward = 0.0

# 1. Realized P&L only (no unrealized credits)
if trade_closed:
    reward += realized_pnl_pct  # normalized by initial_balance

# 2. Symmetric cost for opening positions
if trade_opened:
    reward -= trading_fee_pct  # e.g., 0.0004

# 3. Drawdown penalty (proportional, not threshold-gated)
if drawdown > 0.10:
    reward -= drawdown * 0.05

# 4. No regime multipliers
# 5. No holding bonuses
# 6. No SL/TP asymmetry
# 7. No trend-entry bonuses
```

**Remove entirely:**
- `reward += 0.10` TP bonus / `reward -= 0.05` SL penalty asymmetry
- Regime multipliers (1.5×, 2.0×) in `retrain_ultimate_v3.py`
- Linear hold-time bonus (`position_hold_time * 2`)
- Profit-factor feedback loop in `advanced_rewards.py`

### 4.2 Model Architecture Reduction

| Component | Current | Recommended |
|-----------|---------|-------------|
| Network | 256→256→128 | 128→64 |
| Parameters | ~250K | ~30K |
| Features | 150+ | 50–60 (curated) |
| ent_coef | 0.01 | 0.05 (more exploration) |
| clip_range | 0.2 | 0.15 (more conservative updates) |

### 4.3 Observation Noise Injection

Add `ObservationNoiseWrapper` during training only:
- Gaussian noise: σ = 0.01 × feature std
- Applied to continuous features only (not position/balance state)
- NOT applied during validation or evaluation

This forces the network to learn robust features rather than memorizing specific patterns.

### 4.4 VecNormalize Discipline

```python
# CORRECT pattern (critical fix from bug #2):
train_env = VecNormalize(make_env(train_df), norm_obs=True, norm_reward=False)
# Train model on train_env

# Freeze stats for val and test:
val_env = VecNormalize(make_env(val_df), norm_obs=True, norm_reward=False, training=False)
val_env.obs_rms = train_env.obs_rms   # Use TRAIN stats
val_env.ret_rms = train_env.ret_rms

test_env = VecNormalize(make_env(test_df), norm_obs=True, norm_reward=False, training=False)
test_env.obs_rms = train_env.obs_rms  # Use TRAIN stats
```

### 4.5 Early Stopping on Validation Sharpe

Stop training when validation Sharpe shows no improvement for N consecutive checks:

```python
# Check every 25,000 steps
# Patience: 6 checks (150K steps without improvement)
# Minimum threshold: val_sharpe > 0.5 to save
# Save model at best val_sharpe, not at end of training
```

### 4.6 Train/Val Divergence Detection

Track and log the gap between training episode reward and validation Sharpe:
- If train reward is rising but val Sharpe is falling → overfitting → stop
- Log this gap every check interval
- Save a train_vs_val plot per fold

### 4.7 Ensemble of Walk-Forward Models

After all 9 folds complete:
- Load each fold's best model
- Ensemble prediction: average action log-probabilities across fold models
- Select action with highest average probability
- This naturally regularizes against fold-specific overfitting

---

## 5. Feature Engineering Fixes

### Bug Fix: Regime Look-Ahead

```python
# WRONG (retrain_ultimate_v3.py line 143):
regime_info = detector.detect_regime(self.df.iloc[:self.current_step + 100])

# CORRECT:
regime_info = detector.detect_regime(self.df.iloc[:self.current_step])
```

### Feature Set Reduction

Current: 150+ features with many correlated/redundant signals
Target: 50–60 features selected by:
1. Mutual information with 1h forward returns
2. Remove features with |correlation| > 0.95 with another feature
3. Prefer features with clear economic interpretation

**Recommended feature categories (keep):**
- Price momentum (returns over 4h, 12h, 24h, 72h)
- Volume-weighted signals (VWAP distance, volume ratio)
- Volatility (ATR/close ratio, rolling std of returns)
- Trend indicators (EMA ratios: 7/25, 25/100)
- Market microstructure (bid-ask spread proxy from OHLC)
- Cross-asset correlations (BTC vs ETH return correlation)

**Remove:**
- Simulated whale signals (heuristic with no ground truth)
- Wyckoff pattern labels (discretized rules, high variance)
- Constant cross-chain broadcast features (zero variance)
- Features derived from indicators computed on full dataset (potential leak)

---

## 6. Training Script Usage

### Step 1: Download Data

```bash
python download_historical_data.py \
    --years 3 \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --output-dir data/historical
```

### Step 2: Run Walk-Forward Training (Single Asset)

```bash
python train_walkforward_v2.py \
    --asset BTCUSDT \
    --data-dir data/historical \
    --output-dir data/models/wfv2 \
    --train-months 12 \
    --val-months 3 \
    --test-months 3 \
    --total-timesteps 300000 \
    --patience 6 \
    --noise-std 0.01
```

### Step 3: Run All 4 Assets

```bash
for asset in BTCUSDT ETHUSDT SOLUSDT XRPUSDT; do
    python train_walkforward_v2.py \
        --asset $asset \
        --data-dir data/historical \
        --output-dir data/models/wfv2 \
        --total-timesteps 300000
    echo "Done: $asset"
done
```

### Step 4: Evaluate Ensemble

```bash
python train_walkforward_v2.py \
    --asset BTCUSDT \
    --data-dir data/historical \
    --output-dir data/models/wfv2 \
    --eval-only \
    --ensemble
```

### Step 5: Review Overfitting Report

After each run, check:
```
data/models/wfv2/<ASSET>/fold_summary.json    — per-fold OOS metrics
data/models/wfv2/<ASSET>/overfit_report.json  — train vs val gap analysis
data/models/wfv2/<ASSET>/ensemble_metrics.json — final ensemble evaluation
```

---

## 7. Metrics That Matter

### Primary Metrics (OOS only)

| Metric | Target | Disqualifying |
|--------|--------|---------------|
| Out-of-Sample Sharpe | > 0.8 | < 0.3 |
| Max Drawdown | < 25% | > 40% |
| Win Rate | 45–60% | < 40% |
| Profit Factor | 1.1–2.0 | < 1.0 |
| Total Trades (per 3-month fold) | > 30 | < 10 (too few to assess) |
| Avg Trade Duration | 4–48h | < 1h (scalping noise) |

### Secondary Metrics

- Sortino Ratio (downside-only volatility)
- Calmar Ratio (annual return / max drawdown)
- Consistency: % of folds with positive OOS return

### Red Flags (likely overfitting)

- Train Sharpe > 5 while Val Sharpe < 1 → stop immediately
- OOS return varies wildly across folds (std > mean) → unstable strategy
- Zero trades in a fold → policy collapse
- 100% win rate in any fold → reward hacking

### Overfitting Detection Threshold

```
Train/Val Sharpe Ratio > 3.0 → overfitting
Example: Train=3.0, Val=1.0 → ratio=3.0 → marginal
Example: Train=10.0, Val=1.0 → ratio=10.0 → severe overfit (current state)
```

---

## 8. Realistic Performance Targets

Based on academic literature and live trading results for DRL on crypto:

| Metric | Conservative | Optimistic | Current (overfit) |
|--------|-------------|------------|-------------------|
| Annual Return | 15–30% | 30–60% | 5515% (fake) |
| Sharpe Ratio | 0.8–1.5 | 1.5–2.5 | 10.7 (fake) |
| Max Drawdown | 15–30% | 10–20% | unknown |
| Win Rate | 45–55% | 50–60% | unknown |
| Monthly Return | 1.2–2.2% | 2.2–4.1% | — |

**Any model showing > 100% annual return in walk-forward OOS should be treated with skepticism.**

### Per-Asset Expectations

| Asset | Expected OOS Sharpe | Notes |
|-------|--------------------|----|
| BTCUSDT | 0.8–1.8 | Most liquid, most studied — hardest to beat |
| ETHUSDT | 0.7–1.6 | Correlated with BTC, adds diversification |
| SOLUSDT | 0.9–2.0 | Higher volatility → more opportunity and risk |
| XRPUSDT | 0.6–1.4 | News-driven, harder to model with price alone |

---

## 9. VecNormalize Stats — Correct Deployment Pattern

This is a subtle but important bug in the existing codebase. When deploying a trained model:

```python
# At training time — save stats
vec_normalize.save("model_vecnorm.pkl")

# At deployment/evaluation time — MUST load and freeze stats
from stable_baselines3.common.vec_env import VecNormalize

live_env = DummyVecEnv([make_env_fn])
live_env = VecNormalize.load("model_vecnorm.pkl", live_env)
live_env.training = False      # Do NOT update running stats
live_env.norm_reward = False   # Do NOT normalize reward

model = PPO.load("model.zip")
obs = live_env.reset()
action, _ = model.predict(obs, deterministic=True)
```

Failure to do this means the model receives un-normalized observations during evaluation, producing garbage predictions that look correct in training but fail in evaluation and live trading.

---

## 10. Execution Checklist

- [ ] Download 3 years of data for all 4 assets
- [ ] Verify data quality (completeness > 97%, no large gaps)
- [ ] Fix `retrain_ultimate_v3.py` line 143 (regime look-ahead)
- [ ] Run `train_walkforward_v2.py` for BTCUSDT first (9 folds)
- [ ] Review `fold_summary.json` — flag any fold with Sharpe > 5 or return > 100%
- [ ] Review `overfit_report.json` — train/val gap < 3× is acceptable
- [ ] Repeat for ETHUSDT, SOLUSDT, XRPUSDT
- [ ] Build ensemble, run `ensemble_metrics.json`
- [ ] Paper trade for 30 days before live deployment
- [ ] Live deploy with 1–5% of capital, not 100%
