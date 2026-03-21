# RL Algorithm Research: Upgrade Analysis for Crypto Trading

## Summary

This document analyses three reinforcement learning algorithms for the walk-forward crypto trading pipeline, justifies the selection of **RecurrentPPO** (primary) and **QRDQN** (secondary) over plain **PPO**, and provides implementation rationale with honest caveats.

---

## 1. Why Plain PPO Falls Short for Sequential Financial Data

PPO (`MlpPolicy`) processes each observation as a **stateless, independent sample**. At each step the policy sees only a fixed-length lookback window (48 bars) concatenated into a flat vector. This has two structural weaknesses:

1. **Markov assumption violated**: financial markets have long-range dependencies—regime changes, funding rate cycles, whale accumulation patterns—that span hundreds of bars. A 48-bar window misses these.
2. **No memory across steps**: the policy cannot distinguish "I've been holding this position for 3 hours" from "I just opened". It must re-infer context from raw prices every step.

Both RecurrentPPO and QRDQN address different aspects of these shortcomings.

---

## 2. RecurrentPPO (LSTM) — Primary Recommendation

### What it is

RecurrentPPO (from `sb3-contrib`) replaces the MLP policy with an LSTM-based policy (`MlpLstmPolicy`). The LSTM hidden state propagates **across timesteps within an episode**, giving the agent a memory of recent history beyond the fixed observation window.

### Why it fits financial time-series

- **Regime awareness**: LSTM can learn to detect bull/bear regimes and adjust risk exposure accordingly without explicit regime labels.
- **Position tracking**: the hidden state naturally encodes "how long have I been in this trade", enabling better exit timing.
- **Sequential inductive bias**: financial data *is* a sequence. Imposing a recurrent structure matches the domain.

### Academic references

- Moody et al. (1998) — "Performance Functions and Reinforcement Learning for Trading Systems and Portfolios" — early demonstration that recurrent networks outperform feedforward for trading.
- Lim & Zohren (2021) — "Time-series forecasting with deep learning: a survey" (Philosophical Transactions A) — survey of temporal architectures for financial prediction.
- Théate & Ernst (2021) — "An application of deep reinforcement learning to algorithmic trading" (Expert Systems with Applications) — shows LSTM-based DRL agents outperform MLP baselines on equity data.
- Carapuço et al. (2018) — "Reinforcement learning applied to Forex trading" — demonstrates recurrent policies learning multi-step strategies.

### Expected improvements (with honest caveats)

| Metric | Expectation | Caveat |
|--------|-------------|--------|
| Sharpe ratio | +5–20% vs PPO baseline | Highly data-dependent; noisy markets can hurt |
| Overfit resistance | Similar or slightly worse | More parameters → risk of memorisation; mitigated by smaller net_arch |
| Training stability | Slightly worse | LSTM gradients can explode; mitigated by lower lr=3e-5, clip_range=0.1 |
| Training speed | ~30% slower | LSTM forward pass is heavier than MLP |

**Key caveat**: RecurrentPPO *can* overfit more severely if the LSTM memorises specific historical sequences. The walk-forward validation Sharpe early stopping is critical here. Do not deploy if val/test Sharpe ratio exceeds 3×.

### Hyperparameter rationale

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lstm_hidden_size` | 64 | Enough memory capacity without exploding parameter count |
| `n_lstm_layers` | 1 | Single layer avoids vanishing gradients; multiple layers rarely help for RL |
| `learning_rate` | 3e-5 | 3× lower than PPO — LSTM loss landscape is steeper |
| `clip_range` | 0.1 | More conservative than PPO's 0.15 — LSTM policies are more sensitive to large updates |
| `ent_coef` | 0.02 | Lower than PPO's 0.05 — LSTM explores implicitly through memory |
| `n_steps` | 512 | Shorter rollouts — LSTM state carries information, so full 2048-step rollouts are not needed |
| `net_arch` | [64] | Small feedforward head after LSTM — total params remain ~50K |

---

## 3. QRDQN — Secondary Recommendation

### What it is

Quantile Regression DQN (Dabney et al., 2017) is a **distributional RL** algorithm. Instead of learning the expected Q-value `E[R]`, it learns the **full return distribution** via N quantile heads. The agent implicitly models risk alongside expected return.

### Why it fits risk-sensitive trading

- **Risk-aware decisions**: by knowing the full distribution of outcomes, the policy can prefer lower-variance actions — critical when managing drawdown.
- **No Gaussian assumption**: markets have fat tails. Distributional RL captures this without explicit modelling.
- **Off-policy efficiency**: QRDQN uses an experience replay buffer (100K steps), enabling data-efficient learning and stable updates. Useful when training data is limited.

### Academic references

- Dabney et al. (2017) — "Distributional Reinforcement Learning with Quantile Regression" (AAAI 2018) — original QRDQN paper.
- Lim et al. (2022) — "Distributional Reinforcement Learning for Risk-Aware Portfolio Optimization" — shows quantile-based agents achieve better Sharpe/CVaR tradeoffs than standard DRL.
- Yang et al. (2020) — "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" (ACM ICAIF) — ensemble of distributional agents outperforms single-agent baselines.

### Expected improvements (with honest caveats)

| Metric | Expectation | Caveat |
|--------|-------------|--------|
| Max drawdown | -10–25% reduction vs PPO | Depends on market regime; only holds if risk-aversion is rewarded |
| Sharpe ratio | Comparable or +5–15% | Distributional benefit requires sufficient exploration; learning_starts=5000 |
| Training stability | Better than PPO in volatile regimes | Replay buffer smooths out noisy reward signals |
| Sample efficiency | Better (off-policy) | Buffer size=100K means early training uses replayed experience |

**Key caveat**: QRDQN treats the problem as discrete action selection (HOLD / LONG / SHORT). It does not model continuous position sizing. The risk-awareness manifests in *which discrete action* to take, not in how much capital to allocate — which limits the practical risk management benefit.

### Hyperparameter rationale

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_quantiles` | 50 | 50 quantile heads give a detailed distribution estimate; diminishing returns beyond 100 |
| `buffer_size` | 100_000 | ~4 months of hourly data; large enough to decorrelate samples |
| `learning_starts` | 5_000 | Collect baseline experience before first gradient update |
| `batch_size` | 256 | Larger batches for stable quantile regression |
| `train_freq` | 4 | Update every 4 steps — balance between data efficiency and compute |
| `target_update_interval` | 1_000 | Slow target network updates for stability |
| `net_arch` | [128, 64] | Slightly larger than RecurrentPPO — no LSTM overhead |
| `learning_rate` | 1e-4 | Standard for DQN-family |

---

## 4. Algorithm Comparison Table

| Property | PPO (baseline) | RecurrentPPO | QRDQN |
|----------|---------------|--------------|-------|
| **Type** | On-policy | On-policy | Off-policy |
| **Memory** | None (flat obs) | LSTM hidden state | Replay buffer |
| **Distribution modelling** | No | No | Yes (quantile) |
| **Temporal memory** | Observation window only | Learned LSTM state | None |
| **Training parallelism** | 4 envs (used here) | 4 envs | 1 env (off-policy) |
| **Sample efficiency** | Low | Low | High |
| **Risk-awareness** | None | Implicit (via memory) | Explicit (quantile) |
| **Complexity** | Low (~30K params) | Medium (~50K params) | Medium (~40K params) |
| **Stability** | High | Medium | High |
| **Recommended for** | Baseline / fast iteration | Primary production | Risk-sensitive markets |
| **sb3-contrib required** | No | Yes | Yes |
| **Early stop mechanism** | Rollout-based (val Sharpe) | Rollout-based (val Sharpe) | Step-based (val Sharpe) |
| **Ensemble compatible** | Yes | Yes (LSTM state tracked) | Yes |

---

## 5. Algorithms Considered but Not Recommended

| Algorithm | Reason not selected |
|-----------|---------------------|
| **SAC** | Continuous action space only; discrete trading actions require discretisation hacks |
| **TD3** | Same as SAC — continuous-only |
| **A2C** | Synchronous updates, inferior to PPO in most benchmarks, no benefit here |
| **DQN (standard)** | QRDQN strictly dominates — same architecture, better by distributional learning |
| **Rainbow** | Overkill for this problem; adds n-step, prioritised replay, noisy nets — complex to tune |
| **Dreamer / MBRL** | Model-based RL requires accurate world model; financial markets too non-stationary |
| **TRPO** | Superceded by PPO; slower, harder to implement with SB3 |

---

## 6. Implementation Plan

### Phase 1 (complete): Algorithm layer integration
- [x] Add `--algorithm` flag: `ppo`, `recurrent_ppo`, `qrdqn`
- [x] Add `--compare` flag running all 3 with tabular output
- [x] LSTM state management in `ValidationSharpCallback` and `evaluate_model_on_env`
- [x] Off-policy compatibility (QRDQN uses same step-based callback, handles empty `ep_info_buffer` gracefully)
- [x] Ensemble majority voting (works across algorithm types; per-model LSTM state tracked)
- [x] Algorithm saved per fold (`algorithm.txt`) for correct loading

### Phase 2 (next steps)
- [ ] Hyperparameter search per algorithm using Optuna on a single fold
- [ ] Cross-algorithm ensemble: combine best PPO + best RecurrentPPO + best QRDQN models
- [ ] Attention-based policy (Transformer) as a future RecurrentPPO successor
- [ ] Per-asset algorithm selection based on walk-forward OOS Sharpe

### Phase 3 (research)
- [ ] Continuous action space (position sizing): switch to SAC/TD3 with a bounded continuous action
- [ ] Multi-asset recurrent policy with shared LSTM and asset-specific heads
- [ ] Distributional ensemble: combine QRDQN quantile outputs across folds before argmax

---

## 7. Usage Reference

```bash
# Train with RecurrentPPO (primary recommendation)
python train_walkforward_v2.py --asset BTCUSDT --algorithm recurrent_ppo

# Train with QRDQN (secondary — risk-aware)
python train_walkforward_v2.py --asset BTCUSDT --algorithm qrdqn

# PPO baseline (for comparison)
python train_walkforward_v2.py --asset BTCUSDT --algorithm ppo

# Run all 3 and get comparison table
python train_walkforward_v2.py --asset BTCUSDT --compare

# Quick test (1 fold, 50K steps)
python train_walkforward_v2.py --asset BTCUSDT --algorithm recurrent_ppo \
    --max-folds 1 --total-timesteps 50000

# Build ensemble after training
python train_walkforward_v2.py --asset BTCUSDT --algorithm recurrent_ppo \
    --eval-only --ensemble
```

---

## 8. Important Caveats

1. **No algorithm is guaranteed to outperform PPO** on any specific asset or time period. The walk-forward OOS Sharpe is the only honest measure. Run `--compare` and pick the winner empirically.

2. **RecurrentPPO's LSTM can memorise specific price sequences** if trained too long. The validation Sharpe early stopping (patience=6) is the primary defence. Monitor val/test Sharpe ratio in `overfit_report.json`.

3. **QRDQN's risk-awareness is limited** to discrete action selection. True risk management (position sizing, Kelly criterion) requires a continuous action space or a separate risk module.

4. **Training time increases significantly**: RecurrentPPO is ~30–40% slower than PPO per step. QRDQN is comparable to PPO per step but needs more total steps due to `learning_starts`.

5. **These algorithms do not change the reward function or environment**. All gains or losses are purely from the policy architecture. The environment's stop-loss and take-profit rules still apply.
