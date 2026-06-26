# Mac Retrain Runbook — HTF Walk-Forward Model

**For:** Chen, on his Mac (training is Mac-only — never on the server).
**Goal:** retrain the HTF (high-timeframe) PPO model that drives live entries,
without re-creating the overfit model we have today.
**Date:** 2026-06-26.

> ⚠️ **Read this first.** Do NOT just run the training script as-is. The
> current training code has documented overfitting bugs (look-ahead leakage,
> asymmetric reward, no regularization) — see `RETRAINING_PLAN.md §1`. Training
> on top of those reproduces the fantasy backtest (+5515%, Sharpe 10.7) that
> does NOT generalize and is exactly why live is flat. **Claude prepares the
> anti-overfit fixes first** (reward reform + look-ahead removal per
> `RETRAINING_PLAN.md §4`); you train on the fixed pipeline. Confirm that prep
> is done before Step 3.

---

## Step 0 — One-time Mac setup

```bash
# Clone (or pull) the repo on your Mac
git clone https://github.com/DC-Design-HL/drl-trading-system.git
cd drl-trading-system
git checkout feature/autonomous-loop      # the live branch

# Python 3.12 + a clean venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt   # sb3, torch (CPU), gymnasium, pandas…
```

M-series note: torch installs the CPU wheel (the requirements pin
`download.pytorch.org/whl/cpu`). That's fine — training is CPU-bound here.

## Step 1 — Download 3 years of data (all 4 assets, 15m)

```bash
python download_historical_data.py \
  --years 3 --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT --interval 15m \
  --output-dir data/historical
```
Produces `data/historical/<ASSET>_15m.csv` (~100k+ rows each). Quality bar:
no >2h gaps, >97% complete, no zero/negative prices (`RETRAINING_PLAN.md §2`).

## Step 2 — Confirm Claude's anti-overfit prep is in

Before training, Claude will have committed the reward reform + look-ahead
fixes (clean reward = realized PnL − fees − proportional DD penalty; no regime
multipliers / hold bonuses / TP-SL asymmetry; smaller net + weight decay).
`git pull` so your Mac has them. Ask Claude to confirm "anti-overfit prep
merged" before you spend hours training.

## Step 3 — Train, one asset at a time (walk-forward)

```bash
# BTC (repeat for ETH/SOL/XRP, changing --data-path and --output-dir)
python train_htf_walkforward.py \
  --data-path data/historical/BTCUSDT_15m.csv \
  --output-dir data/models/htf_walkforward_btc \
  --phase1-steps 200000 --phase2-steps 400000 \
  --train-months 12 --val-months 3 --test-months 3 --slide-months 3
```
- Walk-forward = trains many folds, each tested on a 3-month period it never
  saw (the real out-of-sample check). Defaults are 6/2/2/2 months; the
  12/3/3/3 above matches `RETRAINING_PLAN.md §3` (fewer, more robust folds).
- **Time:** this is the big one — ~600k PPO steps × several folds per asset.
  On an M-series Mac expect **several hours per asset** (run overnight; do them
  one at a time or in sequence). `--max-folds 2` first to smoke-test the
  pipeline end-to-end before committing to the full run.

## Step 4 — Read the out-of-sample results (this is the whole point)

Each `data/models/htf_walkforward_<asset>/fold_XX/fold_result.json` has the OOS
test metrics. What we want to see (per `RETRAINING_PLAN.md §6` targets):
- **Aggregate OOS Sharpe positive and consistent across folds** (not one lucky
  fold). A model that's +10 Sharpe in train but ~0 OOS is overfit — reject it.
- Realistic returns (single-digit %/quarter), positive expectancy after fees.
- If OOS is junk across folds, the answer is "this approach doesn't have edge"
  — we stop, not deploy. That's a valid, honest outcome.

## Step 5 — Hand the models back to Claude

```bash
# zip the per-asset model dirs and send them (or push to a branch / drive)
tar czf htf_models_$(date +%Y%m%d).tgz data/models/htf_walkforward_*
# then upload however is easiest (Telegram file, scp to server, etc.)
```

## Step 6 — Claude validates + wires in (NOT you)

Claude will, on the server:
1. Forward-sim the new models over recent live data — **nothing goes live
   unless it beats the current baseline net PnL** (same gate we used for the
   ADX / short-bias changes).
2. Wire the model into the **live entry path** — the real gap today (entries
   currently run on structure signals, not the model). The model comes back as
   a confidence/sizing input first, then as a direction driver if OOS warrants.
3. Deploy as a canary with the circuit breaker watching, and report numbers.

---

### Division of labor
- **Claude (server, now):** anti-overfit fixes to the training code; the
  retrain "bundle" (ground-truth report + P5 signal-value attribution so the
  feature set targets what actually predicts); this runbook.
- **You (Mac):** Steps 0–5 — setup, data, training, send models back.
- **Claude (server, after):** Step 6 — validate, wire in, canary, report.

### Honest expectation
This is the real lever, but it is not a guaranteed win. A clean retrain might
find modest edge, or might confirm there isn't enough — either way we'll *know*
from the OOS numbers instead of guessing, and nothing risky goes live without
forward-sim proof. Reference methodology: `RETRAINING_PLAN.md`.
