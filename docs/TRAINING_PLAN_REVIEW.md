# Training Plan v3 — Critical Review

**Reviewer:** DRL Trading Research Specialist
**Date:** 2026-04-12
**Scope:** Review of TRAINING_PLAN.md against codebase, backtest results, and current research

---

## 0. Overall Assessment

The plan correctly identifies the five root causes (reward misalignment, overfitting, feature gaps, env-reality gap, no per-symbol tuning) and proposes reasonable fixes for each. However, it has several critical blind spots and missed opportunities that could mean the difference between a marginal improvement and a genuinely deployable model. The sections below are ordered by expected impact.

---

## 1. Critical Issues in the Current Plan

### 1.1 The Reward Function Has a Fundamental Design Flaw

The proposed `WinRateAlignedReward` maintains a rolling window of 50 trade outcomes and adds a bonus/penalty proportional to `(current_win_rate - target)`. This is **non-stationary and non-Markovian** — the reward for closing a winning trade at step T depends on the outcomes of the previous 49 trades, which violates the assumptions underlying PPO's value function estimation.

**Why this matters for THIS system:** PPO's value network learns V(s) = expected discounted future reward from state s. If the reward for the same state-action pair changes depending on history the value network cannot see (past trade outcomes are not in the observation), the value estimates become noisy, the advantage estimates are biased, and training destabilizes. This is especially damaging with RecurrentPPO because the LSTM will try to memorize trade outcome sequences rather than learning market structure.

**Recommendation:** Remove the rolling-window win rate bonus entirely. Instead, use a **differential Sharpe ratio** reward (Moody & Saffell, 2001; updated by recent work from Deng et al. 2024):

```
At = Bt-1 * delta_t - 0.5 * At-1 * delta_t^2
     -----------------------------------------------
              (Bt-1 - At-1^2)^(3/2)

where:
  delta_t = realized return at step t (0 if no trade closes)
  At = exponential moving average of returns
  Bt = exponential moving average of squared returns
```

This gives a single-step, Markovian reward that inherently optimizes risk-adjusted returns. No rolling window, no non-stationarity. It rewards consistent small wins over erratic large wins — exactly what you want for a 50%+ win rate target.

For a simpler alternative: use `realized_pnl / max(running_std_of_returns, floor)` as the reward. This normalizes each trade's reward by recent volatility, creating an implicit Sharpe-like signal without non-Markovian state.

### 1.2 The Plan Confuses the Model's Role Given Structure-First Mode

The backtest results are unambiguous:
- **S1 (struct only, no model):** BTC +$3,806 (50% WR), ETH +$795, XRP +$1,659
- **S3 (struct + model):** BTC -$915 (29% WR), ETH +$322, XRP -$803
- **B (model first):** BTC -$3,913 (15% WR), devastating across all symbols

The model actively destroys the edge that structure signals provide. Yet the training plan proposes training the model to make entry decisions (action space = HOLD/LONG/SHORT). This is a fundamental mismatch with how the system is actually deployed (STRUCTURE_FIRST_MODE = True, model not used for entries).

**Recommendation:** Redefine the model's role. There are two viable architectures:

**Option A — Structure-Gated Entry Filter (recommended):**
- BOS/CHOCH generates candidate entries (as it does now in live)
- The model's action space becomes binary: ACCEPT or REJECT the candidate
- The model learns WHICH structure signals are high quality, not when to trade
- This is a much simpler learning problem (binary classification given market context) and directly leverages the proven edge of structure signals

**Option B — Exit Manager:**
- Entries come from structure signals
- The model manages the exit: HOLD, PARTIAL_TP, FULL_CLOSE, TIGHTEN_TRAIL
- This addresses the observation that live S1 hits TP only 50% of the time — there is alpha in better exit timing

Either option has a dramatically smaller action space and more targeted learning objective than the current HOLD/LONG/SHORT formulation.

### 1.3 The 140-Dim Observation is Still Too Large and Redundant

The plan proposes going from 117 to 140 features. Given the 19x overfit ratio with 117 features and ~500K training steps, adding 23 more features will make overfitting worse, not better. The plan's own diagnosis says "massively over-parameterized" — then it adds more parameters to the input.

**Recommendation:** Go in the opposite direction. Target 60-80 features maximum.

1. **Drop the 12 compact features repeated at 4 TFs** (48 features). Replace with a proper hierarchical representation: compute 6-8 features per TF that capture what is UNIQUE to that timeframe's perspective, not the same RSI at 4 scales.
2. **Use PCA or mutual information** on the current 117 features against next-bar returns to identify the top 40-50 features. The plan mentions "permutation importance" as a fallback but this should be step 1.
3. **Add the high-value new features** (BOS/CHOCH, funding rate, OI delta — maybe 10-12 total).
4. **Net result:** ~60 features that are informative rather than 140 that are mostly noise.

The information-theoretic justification: with ~105,000 15-minute bars over 3 years, and ~500K training steps per fold, you need the feature count low enough that the model can learn meaningful representations before the anti-overfitting gates trigger. The VC dimension of a 64-32 network with 140 inputs is approximately 140 * 64 + 64 * 32 + 32 * 3 = 11,200 parameters. With noisy financial labels (effective label noise > 40%), you need at least 10-20x the VC dimension in effective training samples to generalize. That means you need > 112,000 meaningful training transitions, but with random episode starts and a lot of HOLD steps, your effective sample size is much lower.

### 1.4 Walk-Forward Val/Test Overlap is a Data Leakage Risk

In the fold schedule (Section 8.2), the validation window is carved from the END of the training window:
```
Fold 0: Train 2023-04 to 2023-12, Val 2023-10 to 2023-12, Test 2023-12 to 2024-02
```

This means the val data (2023-10 to 2023-12) is ALSO part of the training data. The model literally trains on the validation data. The "val Sharpe" metric is meaningless — it is in-sample Sharpe on the last 2 months of the training window.

**Recommendation:** Use a strict 3-way split with NO overlap:
```
Fold 0: Train 2023-04 to 2023-10, Val 2023-10 to 2023-12, Test 2023-12 to 2024-02
```
This means 6 months train, 2 months val (held out from training), 2 months test (held out from everything). The val set is used for early stopping and hyperparameter selection; the test set is the OOS evaluation. This is standard practice in financial ML and the only way the val/test Sharpe ratio gate (Section 8.3) is meaningful.

### 1.5 No Purging or Embargo Between Folds

Even with non-overlapping splits, sequential financial data has autocorrelation. A training sample at 2023-09-30 23:45 is highly correlated with a validation sample at 2023-10-01 00:00. Without a gap (embargo) between train/val/test, information leaks across boundaries.

**Recommendation:** Add a 48-hour (192-bar) embargo between each split:
- Last training bar must be >= 48h before first val bar
- Last val bar must be >= 48h before first test bar
- This costs ~4 days of data per fold but eliminates autocorrelation leakage

This is from the "Advances in Financial Machine Learning" framework (de Prado) and is standard in any serious financial ML pipeline.

---

## 2. State-of-the-Art Techniques Missing from the Plan

### 2.1 Differential Sharpe Ratio as Reward (covered in 1.1 above)

This is the single highest-impact technique missing. It was introduced by Moody & Saffell in 2001 but has seen renewed use in recent DRL trading papers (2024-2025) because it directly optimizes risk-adjusted returns in a Markovian way.

### 2.2 Curriculum Learning for Trading

The plan mentions curriculum learning as a fallback (Section 13) but does not design it. For crypto on 15m candles, a proper curriculum would be:

**Stage 1 — Trending Regimes Only:** Filter training data to periods where ADX > 30 and a clear BOS sequence exists. The model learns to follow obvious trends. Train for 100K steps. This gives the model a "base policy" that knows trend-following is profitable.

**Stage 2 — Add Ranging Regimes:** Include ADX 15-30 periods. The model must learn to HOLD more often. Train for 200K steps from Stage 1 checkpoint.

**Stage 3 — Full Data (including choppy/news periods):** The model sees all regimes. Train for 200K steps. The curriculum progression means the model enters Stage 3 with a strong prior that "trend following works" rather than starting from scratch in noisy data.

**Why this matters:** The current model sees all market regimes from step 0. Early in training when the policy is random, choppy/ranging periods generate predominantly negative rewards (random entries + SL hits), which dominates the gradient signal and teaches the model that "all trades are bad" — leading to over-conservative HOLD-dominated policies or, conversely, random flailing.

### 2.3 Market Regime-Conditional Policies

Rather than one policy for all regimes, train a gating network that detects the regime (trending up, trending down, ranging, high volatility breakdown) and routes to a specialized sub-policy.

Concrete implementation for this system:
1. Use the existing ADX + BOS/CHOCH trend detection to classify bars into 4 regimes
2. Train 4 small sub-policies (32-16 each), one per regime
3. A gating function (can be rule-based using ADX thresholds you already have) selects which sub-policy to use

This is architecturally simpler than the plan's single large network and directly addresses the per-regime performance differences visible in the backtest data. The model's failure in ranging markets (ADX < 20) is a major contributor to the poor win rate — a regime-conditional approach lets the "ranging" sub-policy learn to HOLD almost always, while the "trending" sub-policy learns aggressive entries.

### 2.4 Multi-Objective Reward via Lexicographic PPO

Recent work (2024, Liu et al.) applies lexicographic multi-objective optimization to trading:
- **Primary objective:** Win rate above threshold (safety constraint)
- **Secondary objective:** Maximize risk-adjusted return (subject to primary being satisfied)

This avoids the reward engineering problem of balancing win rate bonus vs PnL reward via manual weights. The training alternates between two critics — first ensuring the win rate constraint is met, then optimizing return.

For your system, this directly targets the 50% win rate goal as a hard constraint rather than a soft bonus term.

### 2.5 Hindsight Relabeling for Missed Trades

Adapt Hindsight Experience Replay for trading: when the model HOLDs during a period where a perfect entry existed (a BOS that led to TP hit), relabel that trajectory with the reward the model WOULD have received if it had entered. Store these "hindsight" trajectories in a replay buffer and mix them into training at a low ratio (10-20%).

This is particularly relevant because the structure-first backtest shows that BOS/CHOCH signals identify real opportunities — the challenge is getting the model to recognize them. Hindsight relabeling gives explicit gradient signal pointing at "you should have entered here."

Implementation: After each episode, scan for BOS/CHOCH events where HOLD was chosen but a subsequent SL/TP analysis shows the trade would have been profitable. Create a synthetic experience tuple with the correct action and its realized reward.

---

## 3. Architecture Recommendations

### 3.1 PPO vs SAC vs TD3

The plan commits to RecurrentPPO. This is a reasonable choice for discrete action spaces but has known limitations:

**PPO strengths for this system:** On-policy, stable training, works well with discrete actions (HOLD/LONG/SHORT), mature SB3 implementation.

**PPO weaknesses:** Sample inefficient (discards data after each update), on-policy means no replay buffer (wastes rare but informative market events like liquidation cascades).

**Recommendation:** Stick with PPO for the initial retrain (it is the right conservative choice), but implement the model as a **Structure-Gated Filter** (Section 1.2) which reduces the action space to binary. Binary PPO with LSTM is extremely well-studied and stable.

If PPO underperforms, the next algorithm to try is **Discrete SAC** (Christodoulou, 2019), which:
- Is off-policy (uses a replay buffer — important for learning from rare events)
- Automatically tunes entropy via the dual temperature formulation
- Has shown strong results on financial trading in recent benchmarks
- The discrete variant works with your action space

Do NOT try TD3 — it is designed for continuous action spaces and would require rethinking the entire action formulation.

### 3.2 LSTM vs Transformer Attention

The plan uses LSTM (via RecurrentPPO). For 15m crypto candles, LSTM is actually the better choice over Transformers because:

1. **Sequence length is moderate** (~200-400 bars per episode). Transformers shine at very long sequences; LSTMs are competitive at this scale.
2. **Computational cost matters** — training on M3 Pro, Transformer attention is O(n^2) per episode vs LSTM's O(n). With 14 folds x 4 symbols, this difference matters.
3. **SB3 has mature RecurrentPPO** but no Transformer policy. Custom implementations risk bugs.

**However, there is an architectural improvement to consider:** Add a **temporal attention layer after the LSTM** that attends over the last 10 hidden states. This gives the model explicit "which of the recent timesteps is most relevant" capability. Implementation is straightforward: after the LSTM, take the last 10 hidden states, apply scaled dot-product attention with the current hidden state as the query, and concatenate the attended output with the current hidden state before the final MLP.

This is not worth implementing in v3 (keep it simple), but flag it for v4 if LSTM alone underperforms.

### 3.3 Network Size Recommendations

The plan proposes 64 LSTM hidden + [64] post-LSTM for BTC/ETH/XRP and 128 LSTM + [64, 32] for SOL. These sizes are reasonable IF the feature count is reduced to ~60-80. With 140 features, even 64 LSTM hidden is pushing it.

**Specific recommendation:** If you follow the feature reduction advice (Section 1.3), the network sizes should be:

| Symbol | Features | LSTM Hidden | Post-LSTM | Total Params | 
|--------|----------|-------------|-----------|-------------|
| BTC | ~65 | 48 | [32] | ~8K |
| ETH | ~65 | 48 | [32] | ~8K |
| SOL | ~65 | 64 | [48] | ~14K |
| XRP | ~65 | 48 | [32] | ~8K |

The key insight: **the current 500K parameter models have > 50x more parameters than the effective training signal can support.** Going to ~8K-14K parameters is aggressive but necessary given the noise level. If 8K is too small, double it — but never go above ~30K for financial RL with this data size.

### 3.4 Ensemble Strategy Beyond Top-3 Fold Averaging

The plan proposes averaging action log-probabilities from top-3 folds. This is better than a single model but misses two opportunities:

**Improvement 1 — Regime-Weighted Ensemble:** Instead of equal-weight averaging, weight each fold's vote by how well its training period matches the CURRENT market regime. If the current market is high-vol trending (like post-halving), upweight folds trained on similar periods and downweight folds trained during ranging periods. You already have ADX and vol-regime features to do this classification.

**Improvement 2 — Disagreement-Based Abstention:** If the 3 models disagree (e.g., 1 says LONG, 1 says SHORT, 1 says HOLD), output HOLD regardless. This is the "wisdom of crowds" applied to trading — consensus-only entries have dramatically higher win rates. The backtest data supports this: the model hurts performance primarily through spurious entries, not through missing good entries.

---

## 4. Feature Engineering Improvements Beyond the Plan

### 4.1 Cross-Asset Features (HIGH PRIORITY)

The plan mentions ETH/BTC ratio but does not systematically address cross-asset information. For crypto 15m candles, BTC leads altcoin moves by 1-5 candles. This is exploitable:

**Recommended cross-asset features (4 features per alt symbol):**
1. `btc_return_last_3bars` — BTC's 45-min return (leading indicator for alts)
2. `btc_eth_corr_rolling_20` — Rolling 20-bar correlation (when it breaks down, decorrelation opportunities exist)
3. `btc_dominance_momentum_5bar` — BTC.D 5-bar ROC (rising BTC.D = alts will underperform)
4. `alt_beta_to_btc_20bar` — Rolling beta of the alt to BTC (tells you if the alt is amplifying or dampening BTC moves)

For BTC itself, use ETH and SOL as "canary" indicators — if alts are breaking down while BTC holds, that is a bearish divergence signal.

### 4.2 Microstructure Features (MEDIUM PRIORITY)

The plan adds orderbook imbalance from existing code. Additional microstructure features worth computing:

1. **Spread dynamics:** `(ask - bid) / mid_price` rolling percentile (widening spread = uncertainty, narrowing = conviction). Available from existing orderbook code.
2. **Order flow persistence:** Autocorrelation of 1-bar volume delta over 10 bars. Persistent positive flow = genuine buying, not just noise.
3. **Volume clock:** Number of trades per 15m bar normalized against its own 20-bar average. High trade count with low volume = retail frenzy; low trade count with high volume = institutional block.

Note: Features 2 and 3 require tick-level data that may not be available historically. If not available, skip these for v3 and add when real-time data accumulates.

### 4.3 Temporal Features (HIGH PRIORITY, Zero Cost)

The plan adds sin/cos time-of-day and day-of-week (3 features). Expand to 5:

1. `time_of_day_sin` (captures Asia/Europe/US session effects)
2. `time_of_day_cos`
3. `day_of_week_sin` (weekend/weekday)
4. `day_of_week_cos` (the plan is missing the cosine component for day — you need both sin and cos to represent cyclical features without discontinuities)
5. `hours_to_funding` — Distance to next 8-hour funding interval, normalized 0-1. Crypto markets have a documented "funding rate front-running" effect where prices move in the opposite direction of the expected funding payment in the 1-2 hours before funding.

### 4.4 Volatility Regime Features (HIGH PRIORITY)

The plan includes ATR regime but does not capture volatility clustering, which is the strongest statistical regularity in financial data:

1. `garch_vol_ratio` — Current realized vol / EWMA vol (a simplified GARCH(1,1) proxy). When > 1, volatility is expanding; when < 1, contracting. This is computable from ATR with no external data.
2. `vol_of_vol` — Standard deviation of ATR over 20 bars. High vol-of-vol means regime uncertainty — the model should be more conservative.
3. `atr_percentile_100bar` — What percentile is the current ATR relative to the last 100 bars? This is better than the current ATR ratio because it captures the full distribution shape.

### 4.5 Features to DROP (as important as features to add)

Based on the plan's own analysis plus the code review:

| Feature | Why Drop |
|---------|----------|
| `wyckoff_phase` (1D slot 15) | Rule-based categorical with high false positive rate. The plan's own Section 4.2 flags this. |
| `doji_day` (1D slot 17) | Binary, extremely rare, negligible predictive power. |
| `vol_expansion` binary (1D slot 16) | Loses continuous information. Replace with continuous vol ratio if not already covered. |
| `ichimoku_cloud_pos` (1D slot 14) | 3-value categorical. The underlying Ichimoku lines ARE informative, but discretizing into {-1, -0.5, 0} destroys the signal. Either compute a continuous distance-from-cloud or drop it. |
| Redundant compact features across TFs | The 12-feature template at 4H/1H/15M gives 36 features that are near-duplicates. Keep 1H and 15M compact features, drop 4H compact features (they add negligible information beyond 1H), replace with 4H-specific structure features (BOS/CHOCH/OB — already there). |

---

## 5. Training Stability and Robustness

### 5.1 Reward Normalization

The plan says `norm_reward=False` in VecNormalize. This is correct IF the reward function produces well-scaled outputs. The proposed reward clips to [-2, 2] which is reasonable, but the trade closure reward (`realized_pnl_pct * 5.0`) can dominate when a trade hits a full 3% TP — that is `0.03 * 5 = 0.15`, which is fine. However, the scale of the entry quality reward (0.01-0.02) is ~10x smaller than the trade closure reward. This creates a gradient dominance problem where the model optimizes exclusively for trade outcomes and ignores entry quality signals.

**Recommendation:** Scale entry quality reward UP to be within 2-3x of the average trade closure reward. If average realized PnL reward is ~0.1, the entry quality reward should be ~0.03-0.05, not 0.01.

Alternatively, use **reward decomposition** with separate value heads (one for trade outcome, one for entry quality) — but this requires custom PPO modifications and is probably not worth the engineering effort for v3.

### 5.2 Action Masking (CRITICAL MISSING PIECE)

The plan does not mention action masking. The current environment allows nonsensical actions:
- LONG when already LONG (the env handles this by doing nothing, but the model wastes exploration on it)
- SHORT when already SHORT (same issue)

With 3 actions and no masking, the model needs to learn that 1/3 of its action space is invalid in any given state. This wastes significant training capacity.

**Recommendation:** Use `MaskablePPO` from `sb3-contrib` (already a dependency for RecurrentPPO). Define action masks:
- FLAT position: allow HOLD, LONG, SHORT (all valid)
- LONG position: allow HOLD, SHORT (close and reverse), mask LONG (redundant)
- SHORT position: allow HOLD, LONG (close and reverse), mask SHORT (redundant)

If using Structure-Gated Filter (Section 1.2), the action space is binary and masking is not needed.

### 5.3 Observation Noise Injection

The plan mentions per-symbol observation noise (0.01-0.02 std). This is a good idea but should be applied carefully:

- Apply noise to PRICE-DERIVED features only (RSI, momentum, MA distances). Do NOT apply noise to binary/categorical features (BOS signals, position state) or to features with natural noise already (volume, funding rate).
- Use **domain randomization** for slippage and fees in addition to observation noise: randomly perturb the fee from 0.03% to 0.05% and slippage from 0.5x to 2x base across episodes. This teaches the model to be robust to execution cost uncertainty.
- Consider **dropout on the observation** (0.05-0.10 probability of zeroing each feature per step) as an alternative to Gaussian noise. Dropout forces the model to not rely on any single feature and is more effective than additive noise for preventing co-adaptation.

### 5.4 Early Stopping Criteria

The plan's anti-overfitting gates (Section 8.3) are applied AFTER training completes. There is no early stopping DURING training. With 500K steps per fold (~20-30 min), this is a significant waste when overfit is detectable earlier.

**Recommendation:** Evaluate on the validation set every 50K steps. If val Sharpe decreases for 3 consecutive checkpoints while train Sharpe increases, stop training and use the checkpoint from 3 evaluations ago (the one before degradation began). This could save 30-50% of training time for overfit-prone folds.

The implementation is straightforward with SB3's `EvalCallback` — just ensure the eval env uses frozen VecNormalize stats from the training env.

### 5.5 Gradient Clipping and Learning Rate Schedule

The plan specifies fixed learning rates (2e-5 to 5e-5). For RecurrentPPO on noisy financial data, a **linear decay schedule** is strongly recommended:
- Start at the specified LR
- Linearly decay to 10% of the initial LR over the training horizon
- This allows initial exploration with large steps, then fine-tuning with small steps

SB3 supports this natively via the `learning_rate` parameter accepting a schedule function.

Also, the plan does not mention `max_grad_norm`. The default in SB3 is 0.5, which is appropriate. But for LSTM policies on financial data, consider reducing to 0.3 to prevent gradient explosions during regime transitions (e.g., a sudden 5% candle after hours of consolidation generates a large gradient spike in the LSTM).

---

## 6. Evaluation Methodology

### 6.1 Walk-Forward with 14 Folds — Assessment

14 folds with 2-month test windows is reasonable for 3 years of data. However:

**Issue 1:** The folds are NOT independent — they share training data. Fold 0 trains on 2023-04 to 2023-12; Fold 1 trains on 2023-06 to 2024-02. They overlap by 6 months. This means fold-level metrics are correlated, and averaging across folds overstates statistical significance.

**Recommendation:** Report metrics with a **block bootstrap** confidence interval rather than naive standard error. Sample folds in consecutive blocks (preserving temporal structure) and report 90% CI on OOS Sharpe and win rate. If the 90% CI for win rate includes values below 40%, the model should not be deployed even if the point estimate is 50%.

**Issue 2:** 14 folds x 4 symbols x multiple hyperparameter checks = significant multiple testing risk. The probability of at least one symbol appearing to "work" by chance increases with the number of evaluations.

**Recommendation:** Apply a **Bonferroni correction** or, better, use the Romano-Wolf stepwise procedure to control the family-wise error rate. In practice: require OOS Sharpe > 0.5 (not 0.3) to declare statistical significance after correcting for 4 symbols x 14 folds.

### 6.2 Statistical Significance Testing

The plan mentions no statistical tests at all. For a system making 15-25 trades per 2-month window, you need:

1. **Binomial test** for win rate: With 20 trades, a 55% observed win rate (11/20) has a p-value of 0.41 against the null of 50% — NOT significant. You need ~80 trades at 55% WR to get p < 0.05. This means a single 2-month fold CANNOT prove the model works. You need the aggregate across folds.

2. **Permutation test for Sharpe:** Shuffle the trade entry times randomly 1000 times, compute Sharpe for each shuffle. The model's Sharpe must exceed the 95th percentile of the shuffled distribution. This controls for the possibility that the model just happened to trade during favorable periods.

3. **Time-series bootstrap for drawdown:** Block bootstrap the equity curve to estimate the distribution of max drawdowns. Report the 95th percentile max drawdown, not just the observed one.

### 6.3 Minimum Number of Trades for Validation

The plan requires >= 15 OOS trades per fold. This is too few for statistical validity.

**Recommendation:** Require >= 30 trades per fold for any fold to count toward the aggregate. With 2-month test windows (approximately 5,760 bars of 15m data) and a 30-trade minimum, that is approximately one trade per 8 days — achievable with the current signal frequency.

For the AGGREGATE across all qualifying folds, require >= 200 total trades before declaring the model deployable. Below 200 trades, confidence intervals are too wide for any reliable conclusion.

### 6.4 Live-vs-Backtest Drift Detection

The plan proposes 30-day paper trading (Section 9, Phase 10) but gives no quantitative criteria for detecting drift.

**Recommendation:** Implement a **Page-Hinkley test** or **CUSUM detector** on the difference between live win rate and backtest win rate. Specifically:
- Compute the expected win rate from the backtest (e.g., 52%)
- After each live trade, update the cumulative sum of `(expected_WR - actual_outcome)`
- If the cumulative sum exceeds a threshold (calibrated to detect a 10% WR drop with < 50 trades), trigger an alert

This is much more principled than "if win rate drops below 40% for 7 consecutive days" (the plan's current criterion) because it accounts for trade frequency and detects gradual degradation.

### 6.5 Comparing Against Structure-First Baseline

The plan's baselines (Section 10.3) include random, buy-and-hold, RSI, and current production model. **The most important baseline is missing:** the structure-first system WITHOUT any model (the current S1 configuration).

Any retrained model must beat S1_struct_only for its symbol, or there is no justification for deploying the model at all. From the backtest:
- BTC S1: 50% WR, +$3,806
- ETH S1: 38% WR, +$795
- SOL S1: 33% WR, -$239
- XRP S1: 42% WR, +$1,659

The model must demonstrate ADDITIVE value on top of structure signals. If structure-only is better, ship structure-only.

---

## 7. Additional Recommendations

### 7.1 Consider Offline RL (Conservative Q-Learning)

Given that you have a proven structure-first system generating real trades, you have an increasingly large dataset of state-action-reward-next_state tuples from live trading. **Offline RL** methods like Conservative Q-Learning (CQL, Kumar et al. 2020) or Decision Transformer (Chen et al. 2021) can learn from this historical data without any environment simulation.

Benefits:
- No env-reality gap (you are learning from real data)
- No reward engineering (you use the actual realized PnL)
- The conservative penalty in CQL prevents the model from choosing actions outside the distribution of the historical policy, which naturally prevents the "model makes things worse" problem

This is a medium-term recommendation (not for v3) but could be the path to a model that actually helps rather than hurts.

### 7.2 VecNormalize Leakage

The plan correctly identifies that VecNormalize stats must be frozen for evaluation. But there is a subtlety: VecNormalize computes running mean/std across the ENTIRE training set. If the training set spans a regime change (e.g., low vol in 2023 H1 to high vol in 2023 H2), the normalization statistics represent neither regime accurately.

**Recommendation:** Use a **windowed normalization** that only considers the most recent N bars (e.g., 5000 bars = ~52 days). This is more adaptive to regime changes and produces normalization statistics that are representative of the current market condition. Implement by replacing VecNormalize with a custom wrapper that uses an exponential moving average for mean and variance estimation with a decay factor.

### 7.3 Position Sizing as a Learnable Action

The plan uses fixed position sizing (25% of balance). Consider expanding the action space to include position size as a separate dimension:
- Action 1: Direction (HOLD/ACCEPT/REJECT or HOLD/LONG/SHORT)
- Action 2: Size (SMALL/MEDIUM/LARGE = 10%/20%/30% of balance)

This lets the model express confidence through sizing — take larger positions on high-conviction setups. However, this increases the action space from 3 to 9 (3 directions x 3 sizes), which may hurt learning efficiency. Worth testing only after the base model is working.

### 7.4 Training Seed Diversity

The plan does not mention training with multiple random seeds per fold. Due to the stochastic nature of PPO training and the noisy financial labels, the same fold with different seeds can produce dramatically different models.

**Recommendation:** Train each fold with 3 different seeds. Select the seed that performs best on the validation set. This is cheap (3x compute per fold, but folds can run in parallel on M3 Pro) and significantly reduces seed-dependent variance.

---

## 8. Priority-Ordered Implementation Plan

If I had to rank every recommendation by expected impact per engineering hour:

| Priority | Recommendation | Expected Impact | Effort |
|----------|---------------|----------------|--------|
| 1 | Fix val/test data leakage (Section 1.4) | Eliminates false overfit detection | 1 hour |
| 2 | Redefine model role to Structure-Gated Filter (Section 1.2) | Leverages proven structure edge | 4 hours |
| 3 | Replace win-rate reward with differential Sharpe (Section 1.1) | Fixes non-Markovian reward | 2 hours |
| 4 | Reduce feature space to 60-80 (Section 1.3) | Reduces overfitting directly | 3 hours |
| 5 | Add embargo between splits (Section 1.5) | Eliminates autocorrelation leakage | 30 min |
| 6 | Action masking (Section 5.2) | Faster training, less wasted exploration | 1 hour |
| 7 | Add S1 baseline comparison (Section 6.5) | Prevents deploying model that hurts | 30 min |
| 8 | Curriculum learning (Section 2.2) | Better initial policy, faster convergence | 3 hours |
| 9 | Cross-asset features (Section 4.1) | BTC leading indicator is free alpha | 2 hours |
| 10 | Temporal + volatility regime features (Sections 4.3-4.4) | High information, zero cost | 1 hour |
| 11 | Early stopping during training (Section 5.4) | Saves 30-50% training time | 1 hour |
| 12 | Learning rate schedule (Section 5.5) | Standard practice, easy win | 15 min |
| 13 | Multi-seed training (Section 7.4) | Reduces variance | 0 (compute only) |
| 14 | Statistical significance tests (Section 6.2) | Prevents deploying lucky models | 2 hours |
| 15 | Regime-weighted ensemble (Section 3.4) | Better ensemble than equal weight | 2 hours |

Items 1-7 should be implemented before ANY training begins. Items 8-12 are strong-to-medium improvements. Items 13-15 are nice-to-haves for v3.

---

## 9. Summary of Key Findings

1. **The training plan's reward function is non-Markovian** — the rolling win rate bonus will destabilize PPO training. Use differential Sharpe ratio instead.

2. **The model should NOT make entry decisions** — structure signals (BOS/CHOCH) are the proven edge. Retrain the model as a structure signal filter or exit manager, not an independent trader.

3. **140 features is too many** — the plan correctly diagnoses overfitting then proposes adding more features. Reduce to 60-80 high-information features.

4. **The validation data overlaps with training data** — this makes the val/test Sharpe ratio gate meaningless. Strict 3-way split with embargo is required.

5. **No statistical significance testing** is planned — with 15-25 trades per fold, observed win rates have enormous confidence intervals. Require aggregate statistical tests before deployment.

6. **The most important baseline is missing** — the structure-first system (S1) must be the deployment bar, not random or RSI.

The plan is a strong foundation that identifies the right problems. Implementing the corrections above, especially items 1-5 in the priority list, should produce a model that genuinely adds value on top of the structure-first system rather than degrading it.
