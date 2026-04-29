# Mac M3 Training Handoff — 2026-04-27 (scripts shipped 2026-04-29)

This is the runnable spec for the model improvements that **must be
trained on Chen's Mac M3 Pro** (server has 2 CPUs / 3.7 GB RAM and is
production-only). All three projects below are independent — pick what
you can finish in a session.

**All training uses ONLY real data** — no synthetic, no mock, no
augmentation. Every label comes from a real bot trade outcome on
Binance Futures testnet, every news event from the live RSS+sentiment
pipeline, every whale event from the on-chain transaction log.

Each project has a **runnable script**, an **acceptance gate**, and a
**deploy procedure**. Scripts live in `scripts/`. Install training
dependencies with:

```bash
python3 -m venv .venv-train && source .venv-train/bin/activate
pip install -r scripts/requirements-mac-training.txt
```

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

### Step 1 — Generate the training dataset (run on SERVER)

The server already has all the data. The dumper produces a single
self-contained parquet that you ship to Mac.

```bash
# On the server, in the repo
python3 scripts/dump_training_dataset.py
# Default produces data/training/sgfilter_dataset.parquet
# covering BTC/ETH/SOL/XRP since 2026-04-06.

# Verified on 2026-04-29: 292 closed trades joined with full signal
# context, news (8d window of 533 sentiment-scored events), and on-chain
# whale flows (26,868 events from labeled wallets). 62 features per row.
```

Then `scp` the parquet + metadata.json to the Mac repo at the same
path.

### Step 2 — Train the classifier (run on MAC)

The model is **LightGBM gradient-boosted trees**, not RecurrentPPO.
Reasons:
- Tabular features → trees outperform deep nets here. Pragmatic, not
  ideological.
- Trains in seconds on Mac M3 CPU. No GPU needed.
- Native categorical handling for `mtf_bias`, `regime_type`, etc.
- Native feature importance. You'll know exactly which signals the model
  uses.
- Direct probability output → simple ACCEPT/REJECT threshold.
- Reproducible (fixed seed).

```bash
# On Mac, in the repo, with the .venv-train activated
python3 scripts/train_sgfilter.py
# Default: 3 seeds, ACCEPT_THRESHOLD=0.55, walk-forward 60/20/20 split
# with 48h embargo. Output: data/models/sgfilter/sgfilter_seed{42,43,44}.txt
# plus feature_importance.txt and training_summary.json.
```

The script reports per-seed AUC, accuracy, precision, recall on
train/val/test; ensemble (median across seeds) results; top-15 feature
importance. Honest evaluation — no cheating with test data.

The training script implements:
- Walk-forward 60/20/20 split with 48-hour embargo (anti-leakage).
- 3-seed ensemble (median of probabilities).
- Early stopping on validation logloss (50 rounds patience).
- Feature importance per seed (gain-based).

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

### Step 0 — Validate the bug is real

```bash
# On Mac, in repo, with .venv-train activated
python3 scripts/train_eth_vecnorm_fix.py --mode validate
# Runs 200 random observations through the deployed ETH model. If
# confidence stdev < 0.01, VecNormalize is broken (the documented bug).
# If stdev > 0.05, the bug is NOT real and ETH's poor performance has
# a different root cause — stop and investigate elsewhere.
```

### Option A — Re-export with vecnorm (preserves training)

```bash
# On Mac, with the original training run dir on disk
python3 scripts/train_eth_vecnorm_fix.py --mode reexport \
    --training_run training_runs/htf_walkforward_eth_final/

# Then validate the fix:
python3 scripts/train_eth_vecnorm_fix.py --mode validate
# Confidence stdev should now be > 0.05.
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

**The fix.** Retrain the LSTM on the existing labeled data
(`data/whale_behavior/labeled_v2/*.jsonl`, 26,868 real on-chain events
across 3 wallets verified 2026-04-29). All training data is real
on-chain transactions with behavioral labels computed from observable
patterns; no synthetic generation.

```bash
# On Mac, with .venv-train activated
python3 scripts/train_whale_v2.py
# Default: seq_len=24, hidden=256, 2 LSTM layers + attention head,
# 50 epochs, early-stop on val accuracy. CrossEntropy loss over
# {BEARISH, NEUTRAL, BULLISH} classes.
# Output: data/models/whale_v2.pt
```

The script prints per-epoch train loss + val accuracy, then test-set
results. Acceptance gates checked automatically before save:
- Test confidence stdev > 0.10 (model is informative)
- No single class > 70% of predictions (model is balanced)

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
