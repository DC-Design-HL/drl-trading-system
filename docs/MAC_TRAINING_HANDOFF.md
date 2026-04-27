# Mac M3 Training Handoff — 2026-04-27

This is the runnable spec for the model improvements that **must be
trained on Chen's Mac M3 Pro** (server has 2 CPUs / 3.7 GB RAM and is
production-only). All three projects below are independent — pick what
you can finish in a session.

Each has a **why**, a **concrete training command**, an **acceptance
gate**, and a **deploy procedure** that integrates with the existing
production cluster on `feature/bot-consolidation`.

---

## Project 1 — Structure-Gated Filter (highest priority)

**The problem.** The PPO model is currently trained to choose
{HOLD, LONG, SHORT} entry actions. The bot runs in `STRUCTURE_FIRST_MODE`
where BOS/CHOCH structure signals decide the direction, so the model's
`predict()` output is **never used for entry**. The 117-dim observation
gets computed every iteration and discarded. ETH model has walk-forward
Sharpe 12.42 and contributes nothing live.

This is documented in `docs/TRAINING_PLAN_REVIEW.md` Section 1.2 as the
"fundamental architectural mismatch."

**The fix.** Retrain the model as a **binary ACCEPT/REJECT classifier**
on structure signals. Action space: `{0=REJECT, 1=ACCEPT}`. Input: the
same observation vector as today. Label: did this structure signal lead
to a winning trade?

This is a 10× simpler learning problem than the current
HOLD/LONG/SHORT formulation. The reward function becomes:
- ACCEPT a signal that won → +1
- REJECT a signal that won → -0.5 (opportunity cost)
- ACCEPT a signal that lost → -1
- REJECT a signal that lost → +0.5

### Training data preparation

Run on Mac, in the repo:

```bash
# 1. Pull last 90 days of fills + signals from Binance + DB
python3 scripts/dump_training_dataset.py \
    --symbol BTCUSDT \
    --start 2026-01-27 \
    --end   2026-04-27 \
    --output data/training/sgfilter_btc.parquet

# Repeat for ETHUSDT, SOLUSDT, XRPUSDT.
```

The dataset will have one row per BOS/CHOCH detection in the historical
window with:
- All 117 observation features at detection time
- The structure direction (LONG/SHORT) chosen by the detector
- Whether the bot would have taken it (per current guards)
- The outcome if it had been taken (P&L, hold time)
- The label: 1 if win, 0 if loss

**Note:** I haven't built `dump_training_dataset.py` yet — file it under
"add to training repo if you want to run this." A 100-line script. If
you want me to write it, ask.

### Suggested training (once dataset exists)

```bash
# Per-symbol binary classifier with class balancing
python3 train_sgfilter.py \
    --dataset data/training/sgfilter_btc.parquet \
    --algorithm RecurrentPPO \
    --features 60 \
    --reward differential_sharpe \
    --hold_steps 0 \
    --val_split 0.2 \
    --test_split 0.2 \
    --embargo_hours 48 \
    --epochs 500 \
    --seeds 3 \
    --early_stop_sharpe 0.5 \
    --output data/models/sgfilter/btc/
```

Key choices, with rationale:

- **RecurrentPPO (LSTM head)** — `docs/TRAINING_PLAN.md` already specs
  this. The structure-first signal is sequence-dependent (regime
  history matters); LSTM head captures it.
- **Differential Sharpe reward** — `docs/TRAINING_PLAN_REVIEW.md`
  Section 1.3 flags the current PnL+penalty reward as non-Markovian.
  Fixes destabilization.
- **60 features (not 117)** — feature reduction per
  `TRAINING_PLAN_REVIEW.md` Section 2.1. Drop the 12 redundant compact
  features replicated across 4 timeframes; the resulting input has VC
  dimension ~2K (was ~11K), reducing overfit.
- **48h embargo** — strict 3-way val/test split with embargo prevents
  lookahead leakage. Critical for HTF data.
- **3 seeds, multi-seed averaging** — variance reduction in PPO
  training. Take the median of 3 seeds.
- **early_stop_sharpe 0.5** — abandon training run if validation Sharpe
  doesn't exceed 0.5 by epoch 200. Saves Mac time.

### Acceptance gate (don't deploy if these aren't met)

Before merging to `feature/bot-consolidation`:

1. Validation Sharpe ≥ 1.0 across all 3 seeds (median ≥ 1.5)
2. Test-set Sharpe ≥ 0.5 (out-of-sample reality check)
3. Test-set hit rate (fraction of structure signals correctly accepted)
   ≥ 60% AND fraction correctly rejected ≥ 50%. If the model is
   asymmetric (e.g. accepts everything), reject the run.
4. **Walk-forward backtest** (use `scripts/backtest_signal_filters.py`
   harness extended to call the trained model): Δ vs. live system
   ≥ +5% on 30-day window. If the trained model can't beat the simple
   per-symbol blocklist deployed today, don't deploy it.

### Deploy procedure (server side, after acceptance)

When you bring the trained model files back:

```bash
# 1. SCP the model artifacts to server
scp data/models/sgfilter/*.zip claude@server:~/packages/.../data/models/sgfilter/

# 2. On the server, switch the model loading to use SGFilter
git checkout -b feature/sgfilter-deploy
# (I'll add a wiring change to live_trading_htf.py:_load_model that picks
# up SGFilter outputs and applies them as ACCEPT/REJECT)

# 3. Run the canary on XRP first (lowest-volume symbol)
SGFILTER_CANARY_SYMBOLS="XRPUSDT" ./start_services.sh

# 4. Watch for 48h. If the canary firing-rate matches the validation
#    distribution and pnl is non-negative, expand canary to all 4 symbols.

# 5. After 1 week of clean SGFilter operation, deprecate STRUCTURE_FIRST_MODE
#    flag and remove the legacy HOLD/LONG/SHORT model loading code.
```

**I will write the wiring change** (step 2 above) once you have the
trained model artifacts. ~50 lines, one new method `_should_accept_signal`,
hooked into `_get_structure_direction`.

---

## Project 2 — ETH VecNormalize fix

**The problem.** Per `docs/eth-signal-investigation.md`, the deployed
ETH model was trained with `VecNormalize` (input normalization wrapper)
but receives **raw, unnormalized observations** at inference. The
normalization stats file (`vecnorm.pkl`) wasn't shipped or wasn't
loaded properly. Consequence: model output is essentially noise; it
reports false 0.95+ confidence; ETH LONG is -$23 / 39 trades historical.

**The fix.** Two options:

### Option A — Re-export with vecnorm (preserves training)

```bash
# On Mac, locate the training run that produced final_model_0.zip
cd training_runs/htf_walkforward_eth_final/

# Re-export with vecnorm explicitly bundled
python3 -c "
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import shutil

# Reload the trained agent
model = PPO.load('final_model_0.zip')
vec = VecNormalize.load('vecnorm.pkl', dummy_vec_env)
vec.training = False
vec.norm_reward = False

# Save side-by-side
shutil.copy('vecnorm.pkl', 'final_model_0_vecnorm.pkl')
print('Re-exported.')
"

# SCP back to server
scp final_model_0.zip final_model_0_vecnorm.pkl \
    claude@server:~/packages/.../data/models/htf_walkforward_eth/
```

### Option B — Retrain ETH model without vecnorm (cleaner)

Just retrain ETH using the same Project 1 pipeline but with
`--algorithm PPO --no_vecnorm` and the SGFilter formulation. This
replaces the broken model with a fresh one.

**Recommendation:** Option B if you do Project 1 anyway; Option A
otherwise.

### Acceptance gate

ETH model must produce diverse confidence outputs (not stuck at 0.95).
Verify with:

```bash
# On server after deploy
python3 -c "
from live_trading_htf import HTFLiveBot
bot = HTFLiveBot('ETHUSDT')
import numpy as np
fake_obs = np.random.randn(117)
confs = []
for _ in range(100):
    fake_obs = np.random.randn(117)
    _, conf = bot.get_action(fake_obs)
    confs.append(conf)
import statistics
print(f'mean={statistics.mean(confs):.3f}, std={statistics.stdev(confs):.3f}')
# Expected: std > 0.05. If std < 0.01, vecnorm is still broken.
"
```

---

## Project 3 — Whale model retraining

**The problem.** Per memory and the agent inventory, the deployed whale
LSTM has a `state_dict` shape mismatch (input dim 144→272, hidden
128→256 per a recent architecture change), so it never loads
successfully. The Robinhood wallet — primary training data source — has
been stale since Feb 14. The currently-live whale signal is mostly
NEUTRAL across all symbols, which is why the new WHALE_NEUTRAL_GUARD
deployed today is useful: it correctly identifies the "no signal" state.

**The fix.** Retrain on fresh wallet data with the new architecture.

```bash
# 1. Refresh whale-wallet labeled training data
cd ~/drl-trading-system  # local clone on Mac
python3 scripts/refresh_whale_dataset.py \
    --since 2026-02-15 \
    --output data/training/whale_v2_dataset.parquet
# This pulls fresh on-chain data from Etherscan/Alchemy and re-labels
# wallet activity. Memory has noted "1,750 new labeled sequences
# available" — those should be in the dataset.

# 2. Retrain
python3 train_whale_lstm.py \
    --dataset data/training/whale_v2_dataset.parquet \
    --architecture WhaleBehaviorLSTM \
    --input_dim 272 \
    --hidden_dim 256 \
    --num_layers 2 \
    --epochs 100 \
    --val_split 0.2 \
    --output data/models/whale_v2.pt

# 3. Validate predictions are non-trivial
python3 -c "
import torch
from src.whale_behavior.models.predictor import WhaleBehaviorLSTM
m = WhaleBehaviorLSTM(input_dim=272, hidden_dim=256, num_layers=2)
m.load_state_dict(torch.load('data/models/whale_v2.pt'))
# Run inference on recent wallet activity. Confidence should range
# 0-1 with std > 0.1 across 50 samples. If stuck near 0.5, model is
# uninformative.
"
```

### Acceptance gate

Whale model output must show:
- Confidence std > 0.1 across 50 recent samples
- Direction distribution NOT exclusively NEUTRAL (i.e. it sometimes
  predicts BULLISH / BEARISH)
- Walk-forward backtest: trades when whale.direction matches bot
  direction must have higher WR than baseline by ≥ +3 pp

If the whale model passes, **disable WHALE_NEUTRAL_GUARD on the server**
(`WHALE_NEUTRAL_GUARD_ENABLED = False` in `live_trading_htf.py`) — the
guard exists today only because the model is broken.

### Deploy procedure

```bash
scp data/models/whale_v2.pt claude@server:~/packages/.../data/models/
# Server-side: update src/whale_behavior/models/predictor.py to load
# whale_v2.pt with the correct architecture. ~10-line change.
./start_services.sh
```

I'll write the predictor.py loading change once you have the trained
model.

---

## Priority order (if Mac time is limited)

1. **Project 2 (ETH VecNormalize)** — ~2 hours on Mac. Highest
   leverage / effort ratio. Fixes a known broken model that's
   distorting the bot's confidence signals.
2. **Project 1 (Structure-Gated Filter)** — ~6-8 hours on Mac. Biggest
   strategic win, but requires the most infrastructure (dataset dump,
   training script, server-side wiring). I'll prep the
   `dump_training_dataset.py` script if you confirm interest.
3. **Project 3 (Whale retraining)** — ~4 hours on Mac. Useful but
   currently the WHALE_NEUTRAL_GUARD deployed today neutralizes the
   damage. Lower urgency.

---

## What I'll do server-side while you train

1. Continue monitoring Phase 3 deployment. If funding/whale guards fire
   correctly and pnl recovers, validates the backtest.
2. Instrument the trade DB to capture per-trade `fake_bos`,
   `fake_choch`, structure_quality, and signal-bucket data. This makes
   future filter backtests honest (today the DB doesn't store these
   fields per trade).
3. When you bring back trained model artifacts, wire them into the
   live bot and run the canary deploy procedures above.

---

## Honesty checkpoint

The training plan above is sound but **not a guarantee**. Realistic
outcomes:

- Project 2 (ETH fix): high probability of working, low magnitude
  improvement (~+$2-4/trade on ETH, $80-160/30d).
- Project 1 (SGFilter): high probability of *some* improvement (the
  current architecture is broken; a fixed one should outperform). But
  the magnitude is uncertain — could be +5%/mo, could be +20%/mo.
- Project 3 (Whale): high probability of producing a model. Whether
  it adds real signal vs the WHALE_NEUTRAL_GUARD baseline is
  empirical. Could be net-zero.

Total upside if all three work: estimated +10-25%/mo on top of current
performance, with proportional drawdown risk. Not 30%/mo. The
realistic ceiling for this strategy on testnet without further
architectural work is probably ~25%/mo.

---

## Restore points (in case anything goes wrong on the server)

- `v-pre-aggressive-sizing-20260425` — original 1× baseline, no Phase 2/3
  filters. The "go back to last week" tag.
- `v-3x-no-filters-20260427` — 3× sizing without filters. Only roll forward
  here if the filters prove harmful.
- `feature/bot-consolidation` HEAD — current live config (2× + blocklist +
  funding/whale guards + USDT.D + ADX>60 + stagnant band).

Tagged restore points are pushed to origin. Roll back with one git
command + restart.
