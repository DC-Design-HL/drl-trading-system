# Whale Behavior Model — Retraining Plan

**Branch:** `research/whale-news-data-audit`
**Drafted:** 2026-05-11
**Status:** Plan only — no code yet
**Where training runs:** Local Mac M3 (per `feedback_training_local.md`). The Hetzner box has 3.7 GB RAM and consolidated-mode bots use ~2.2 GB. Training on it would OOM-kill live trading.

## 1. Context — Why we're not chasing alpha

The May 1 minimum-viable test (commit `06baaed`, output `data/training/whale_mvt_results.json`) Bonferroni-corrected three forward-return horizons against whale events:

| Horizon | n  | Effect      | p-value | Verdict |
|---------|----|-------------|---------|---------|
| 15m     | 87 | +4.99 bps   | 0.1444  | fail to reject |
| 1h      | 82 | +2.61 bps   | 0.6338  | fail to reject |
| 4h      | 58 | +11.72 bps  | 0.4724  | fail to reject |

All p-values >> α=0.0167 (Bonferroni-corrected). Largest observed effect (11.72 bps at 4h) is below the 15 bps trade-cost threshold even ignoring statistical significance. **Whale data has no detectable directional alpha at the resolutions we trade.**

So the retraining goal is **NOT** "predict price". It is:

1. **Restore the broken production predictor** (currently log-spamming with state_dict load errors)
2. **Surface whale-flow features for filtering** — e.g., the existing `WHALE-NEUTRAL` block on BTC LONG when whale.direction == NEUTRAL (37% WR regime)
3. **Stop wasting Groq calls + WebSocket bandwidth** on a downstream consumer that never loads

## 2. The architecture mismatch (root cause)

| Checkpoint (`whale_behavior_lstm.pt`) | Current code (`sequence_model.py`) |
|---|---|
| Trained: 2026-03-29 on M3 | Updated: 2026-04-03 (commit `a209fbc`) |
| Continuous features: 3 (time_gap, direction, gas_ratio) | Continuous features: 9 (added price_norm, price_roc_1h, price_roc_24h, price_volatility, value_usd_log, hour_sin) |
| LSTM input size: **19** | LSTM input size: **25** |
| LSTM hidden: 64 → 256-d output | LSTM hidden: 128 → 512-d output |
| Has `fc1` only | Has `fc1` + `fc2` |
| 9 epochs, val_loss 1.0789 | Cannot load |

`model.load_state_dict(checkpoint["model_state"])` fails because tensor shapes disagree. The predictor wraps this in a try/except so it fails-soft, but every refresh tick re-logs the error (we saw 27 of these in the last 34h).

## 3. Two paths — pick one

### Path A — Retrain (recommended)

**Why:** The new 6 features (price-context block added in `a209fbc`) are conceptually richer. Even if the model can't predict price direction profitably, the new features will produce more diverse intermediate representations for the downstream filtering use case. Also: re-running training validates the dataset + code on M3 end-to-end, which we'll need for the news model anyway.

**Effort:** ~2–4 hours wall time on M3 (was 9 epochs / under an hour at the March training).

### Path B — Revert code to match checkpoint

**Why:** Faster — no training run needed. Just delete the 6 price-context features from `sequence_model.py` and the dataset builder.

**Cost:** Throws away 6 features that arguably *do* help (price context is generically useful). Also commits us to the older 64-hidden architecture, which we'd want to undo eventually.

**Verdict:** Path A unless you want a same-day fix. Both work; A leaves us in a better long-term position.

## 4. Path A — Concrete execution on Mac M3

### Step 1: pull this branch + verify data
```
git fetch origin
git checkout research/whale-news-data-audit
git pull

# verify labeled data is present
ls -la data/whale_behavior/labeled/
# expect 11 .jsonl files, ~350K total rows
wc -l data/whale_behavior/labeled/*.jsonl
```

### Step 2: address class imbalance before training

Current label distribution (4h window):
- NEUTRAL: 254,562 (72.6%)
- SELL_SIGNAL: 49,509 (14.1%)
- BUY_SIGNAL: 46,726 (13.3%)

Imbalance is 5.4× — last training hit 24.5% intent accuracy (worse than random for 3-class).

Three mitigations to apply in `train_whale_behavior.py` (modify, don't replace):

1. **Weighted CrossEntropy:** weight = N / (3 × n_class), so the loss gradient is balanced. Easy 1-line change.
2. **Stratified sampling:** at minibatch construction, oversample BUY/SELL to 1:1:1 with NEUTRAL. Code lives in the Dataset class.
3. **Filter NEUTRAL to threshold:** only keep NEUTRAL examples with strong contextual signal (e.g., transaction value above per-wallet median). Reduces noise at cost of dataset size.

Recommend (1) + (3) together. Skip (2) for now — risk of overfitting to oversampled rare classes.

### Step 3: training run
```
# expected: ~1 hour on M3 with --mps, 50 epochs, batch 64
python train_whale_behavior.py \
  --window 4h \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --weighted-loss \
  --neutral-strong-filter \
  --device mps \
  --output-dir data/whale_behavior/models/v2/ \
  --val-split 0.15 \
  --test-split 0.15 \
  --patience 10
```

Output expected:
- `data/whale_behavior/models/v2/whale_behavior_lstm_v2.pt` — new checkpoint
- `data/whale_behavior/models/v2/training_results.json` — metrics
- `data/whale_behavior/models/v2/feature_config.json` — pinned feature list (for inference-side validation)

### Step 4: acceptance gates before deploy

Don't merge if any of these fail:

- [ ] Direction accuracy on test set > 60% (last run: 58.5%, marginally above chance — improve at least 2 pp)
- [ ] Intent F1-macro > 0.30 (last run: ~0.17 — at least double it)
- [ ] BUY-class precision > 35%, recall > 25% (currently 16.5% acc, mostly false negatives)
- [ ] Per-wallet evaluation: at least 7 of 11 wallets produce non-degenerate predictions (vs Kraken/ETH 2.0 which are >98% single-class garbage)

Failure on any of these → the predictor is still a fail-soft filter, but tighten the inference-side threshold so it only emits signals on smart_money_whale wallets where the data is more balanced.

### Step 5: deploy

The deploy is trivial once the checkpoint loads:

1. SCP `whale_behavior_lstm_v2.pt` and `feature_config.json` to the Hetzner box, into `data/whale_behavior/models/`
2. Update `predictor.py` to:
   - Read `feature_config.json` and assert it matches code expectations on load (avoids future drift bugs)
   - Point at `_v2.pt`
3. Use `restart_ui.sh` — wait no, this requires bot restart since the predictor is loaded in the bot process. Use `start_services.sh` (idempotent).
4. Tail `logs/bots_live.log` for absence of "Failed to load whale behavior model" lines (signal of success).
5. Optional: keep the old `whale_behavior_lstm.pt` around for 1 week as `whale_behavior_lstm_v1.pt.bak` for rollback.

## 5. Path B — Code revert (if same-day fix needed)

Faster path if you don't have a free M3 window today.

1. Revert `sequence_model.py` continuous features to 3:
   - Edit the feature list in `_compute_continuous_features()` to keep only `[time_gap_norm, direction, gas_ratio]`
   - Reduce LSTM input size: should auto-derive from feature count, but verify dim=19 at model build
   - Reduce LSTM hidden to 64 (matches old checkpoint)
   - Remove `fc2` from the FC head (old checkpoint only has fc1)
   - Cross-check against `git show 77e5808:src/whale_behavior/models/sequence_model.py` for the exact pre-`a209fbc` architecture

2. Verify load: `python -c "from src.whale_behavior.models.predictor import WhaleBehaviorPredictor; p = WhaleBehaviorPredictor(); print(p.model)"` should succeed

3. Restart bots: `./start_services.sh`

4. Confirm no load errors in logs after first refresh tick

Cost: throws away the price-context features. We'll want to redo Path A eventually anyway.

## 6. What this does NOT fix

- **No alpha generation.** Whale data has been falsified as a directional signal. Don't expect this to make money on its own.
- **Class imbalance is intrinsic.** Most whale transactions are routine (NEUTRAL). Even a perfect classifier on this dataset would have a low ceiling.
- **No new wallets.** This plan trains on the existing 11 ETH wallets only. Cross-chain expansion (BTC, SOL, XRP whale data) is a separate effort and the May 1 MVT said pause it.

## 7. Decision

| Question | Recommended answer |
|---|---|
| Path A or B? | A (retrain), unless you want same-day fix → B |
| Where to train? | Mac M3 only (hard rule) |
| Acceptance metric? | Direction acc > 60% AND F1-macro > 0.30 |
| If fails acceptance? | Deploy anyway as fail-soft filter, but restrict to smart_money_whale_1 inference only |
| Timeline? | A: ~1 day (training + deploy). B: ~1 hour. |
