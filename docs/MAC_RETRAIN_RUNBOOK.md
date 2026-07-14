# Mac Retrain Runbook — HTF Walk-Forward Model (READY TO RUN)

**For:** Chen, on his Mac (training is Mac-only — never on the server).
**Goal:** retrain the HTF PPO model that should drive live entries, WITHOUT
re-creating the overfit model we have today (the fake +5515% / Sharpe 10.7).
**Updated:** 2026-07-13 — added the SOL GATE section at the bottom (diagnostic
first, calibrated retrain only if needed). The 2026-07-03 anti-overfit fixes
remain in force.

---

## ✅ Read this first — what changed (why you can trust the number now)

The old pipeline reported fantasy out-of-sample (OOS) numbers because of 3 bugs.
All three are now fixed + unit-tested (`tests/test_retrain_antioverfit.py`, green):

1. **Evaluation fed the model RAW inputs.** The policy trains on normalized
   observations but was scored on un-normalized ones → every OOS Sharpe/return
   was noise. **Fixed:** eval now applies the trained normalization.
2. **Look-ahead leak in the higher-timeframe features.** The feature builder
   read the *still-forming* 1h/4h/1d bar (future price) at each 15m step.
   **Fixed:** it now uses only the last *fully-closed* higher-TF bar.
3. **The "best" model was never actually used.** Training saved the best-on-
   validation checkpoint but then scored the *final* (most-overfit) one.
   **Fixed:** it reloads the best checkpoint (+ its normalization) before OOS.

Net effect: **if the OOS numbers come back junk, that's the truth — the approach
doesn't have edge — not a bug.** That's the whole point of doing this.

> These are the must-fix items. There are a few optional extra hardening steps
> (observation noise, tighter SL/TP fill modeling, lower compute-vs-data ratio)
> we can add later if the first honest run looks promising — not needed tonight.

---

## Step 0 — One-time Mac setup (~10 min)

```bash
git clone https://github.com/DC-Design-HL/drl-trading-system.git   # or: git pull
cd drl-trading-system
git checkout feature/autonomous-loop      # the live branch (has the fixes)
git pull                                   # make sure you have today's commit

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt   # sb3, torch (MPS), gymnasium, pandas…
```
Notes:
- **Python 3.12 or 3.13 both work** (flexible `>=` pins). On Apple Silicon this
  installs an **MPS (Metal GPU) accelerated** torch — faster than CPU.
- **Handoff:** models train here but load on the server (Py 3.12). Right after
  install, send me `pip freeze | grep -Ei "torch|stable_baselines3|numpy|gymnasium"`
  so I can match the server versions before you send the models — avoids any
  cross-version load error after an overnight run.

**Sanity-check the fixes are present (10 seconds):**
```bash
python3 -m pytest tests/test_retrain_antioverfit.py -q
# expect: 6 passed
```

## Step 1 — Download historical data (~5 min, needs internet)

```bash
python download_historical_data.py \
  --years 3 --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT --interval 15m \
  --output-dir data/historical
```
Produces date-stamped files like `data/historical/BTCUSDT_15m_<start>_<end>.csv`.
The trainer's `--data-path data/historical/BTCUSDT_15m.csv` auto-resolves to the
date-stamped file, so you don't need to rename anything. Quality bar: no >2h
gaps, >97% complete, no zero/negative prices.

## Step 2 — Smoke-test the pipeline FIRST (~15–20 min, do NOT skip)

Run 2 short folds on one asset to confirm the whole thing works end-to-end
before you commit the machine overnight:

```bash
python train_htf_walkforward.py \
  --data-path data/historical/BTCUSDT_15m.csv \
  --output-dir data/models/htf_smoketest_btc \
  --phase1-steps 20000 --phase2-steps 20000 \
  --train-months 12 --val-months 3 --test-months 3 --slide-months 3 \
  --max-folds 2
```
In the log you should see, per fold: `Reloaded best-on-val checkpoint for OOS
eval` and both a `Val →` and `Test →` line with real (non-NaN) numbers. If that
works, delete `data/models/htf_smoketest_btc` and do the real run.

## Step 3 — The real run, one asset at a time (several hours each)

```bash
# BTC — then repeat for ETH / SOL / XRP (change --data-path and --output-dir)
python train_htf_walkforward.py \
  --data-path data/historical/BTCUSDT_15m.csv \
  --output-dir data/models/htf_walkforward_btc \
  --phase1-steps 200000 --phase2-steps 400000 \
  --train-months 12 --val-months 3 --test-months 3 --slide-months 3
```
```bash
python train_htf_walkforward.py --data-path data/historical/ETHUSDT_15m.csv --output-dir data/models/htf_walkforward_eth --phase1-steps 200000 --phase2-steps 400000 --train-months 12 --val-months 3 --test-months 3 --slide-months 3
python train_htf_walkforward.py --data-path data/historical/SOLUSDT_15m.csv --output-dir data/models/htf_walkforward_sol --phase1-steps 200000 --phase2-steps 400000 --train-months 12 --val-months 3 --test-months 3 --slide-months 3
python train_htf_walkforward.py --data-path data/historical/XRPUSDT_15m.csv --output-dir data/models/htf_walkforward_xrp --phase1-steps 200000 --phase2-steps 400000 --train-months 12 --val-months 3 --test-months 3 --slide-months 3
```
- **Time:** ~600k PPO steps × several folds/asset → **several hours per asset**
  on an M-series Mac. Run overnight; you can paste all 4 lines and let them run
  one after another.
- Walk-forward = many folds, each tested on a 3-month window it never saw.

## Step 4 — Read the OOS results (this is the whole point)

```bash
# per-fold OOS test metrics for one asset
cat data/models/htf_walkforward_btc/fold_*/fold_result.json
```
What good looks like:
- **OOS (test) Sharpe positive AND consistent across folds** — not one lucky
  fold. Train Sharpe hugely positive but OOS ≈ 0 = still overfit → reject.
- Realistic returns (single-digit %/quarter), positive expectancy after fees.
- **If OOS is junk across all folds, we STOP — that's a valid, honest answer,
  not a failure.** We never deploy a model that can't prove OOS edge.

## Step 5 — Send the models back

```bash
tar czf htf_models_$(date +%Y%m%d).tgz data/models/htf_walkforward_*
# send however's easiest — Telegram file, scp to server, or push to a branch
```
Also paste me the `Test →` lines / fold_result OOS numbers so I can sanity-check
before we risk anything.

## Step 6 — I take it from here (server side)

1. **Forward-sim the new models over recent live data — nothing goes live unless
   it beats the current baseline** (same gate we used for ADX / short-bias).
2. **Wire the model into the live entry path** — the real gap today: entries run
   on structure signals, the trained model is only shadow-logged. It comes in as
   an entry *gate* first (veto entries the model disagrees with), then as a
   driver if the OOS/forward numbers earn it.
3. Deploy as a canary with the circuit breaker watching; report numbers.

---

### Division of labor
- **You (Mac):** Steps 0–5 — setup, data, smoke-test, train, send models back.
- **Me (server):** the fixes (done), this guide, Step 6 (validate → gate → wire →
  canary → report).

### Honest expectation
This is the real profitability lever — but not a guaranteed win. A clean retrain
finds modest edge, or confirms there isn't enough. Either way we'll *know* from
trustworthy OOS numbers instead of guessing, and nothing risky goes live without
forward-sim proof. Methodology reference: `RETRAINING_PLAN.md`.

---

# SOL GATE — diagnostic first, retrain only if needed (2026-07-13)

**Background:** the SOL model gate failed because live "confidence" = max
softmax probability, which saturates at ~0.99 on nearly every entry — nothing
to gate on. But the **logit margin** (top-1 minus top-2 logit) underneath the
softmax still varies per decision. Your existing Jul-4 SOL fold models may
already contain gate signal we simply weren't reading. So: **run the 30-minute
diagnostic BEFORE burning an overnight retrain.**

## Step A — Diagnostic on the existing Jul-4 SOL models (~30 min, NO training)

```bash
cd drl-trading-system
git checkout feature/autonomous-loop && git pull    # today's commit has the tooling
source .venv/bin/activate
python3 -m pytest tests/test_gate_diagnostics.py -q  # expect: 10 passed

python scripts/sol_gate_diagnostic.py \
  --models-dir data/models/htf_walkforward_sol \
  --data-path data/historical/SOLUSDT_15m.csv
```
(If your July SOL CSV is gone, re-run the Step-1 downloader for SOLUSDT first.
The diagnostic auto-resolves date-stamped CSV names.)

It replays each fold's best-on-val checkpoint over that fold's out-of-sample
test window, joins every entry's logit margin to the trade outcome, and prints
per-fold + pooled **margin AUC** (does higher margin = higher win rate?).

**Read the verdict line:**
- **pooled margin_auc ≥ 0.55** → the model already discriminates. NO RETRAIN.
  Send me `data/models/htf_walkforward_sol/` (tar.gz, same as Step 5) +
  `gate_diagnostic.json`; I wire the gate on margin with a val-fitted
  threshold → forward-sim → canary.
- **~0.50–0.52** → certainty carries no outcome information → Step B.
- Also send me `gate_diagnostic.json` either way.

## Step B — Calibrated retrain, SOL only (overnight, only if Step A says NONE/WEAK)

Identical to the Step-3 SOL command plus `--ent-floor 0.02` (holds the entropy
coefficient at 0.02 instead of annealing to 0.005 — the anneal is what
collapses the policy to saturated one-hot outputs), and a fresh output dir:

```bash
python train_htf_walkforward.py \
  --data-path data/historical/SOLUSDT_15m.csv \
  --output-dir data/models/htf_walkforward_sol_cal \
  --phase1-steps 200000 --phase2-steps 400000 \
  --train-months 12 --val-months 3 --test-months 3 --slide-months 3 \
  --ent-floor 0.02
```

The summary now prints a **Gate Margin AUC** line and each
`fold_result.json` contains a `gate_diagnostics` block — the retrain's success
criterion is BOTH of:
1. OOS Sharpe stays positive with ≥50% positive folds (doesn't lose the Jul-4
   edge), and
2. gate_margin_auc_mean ≥ 0.55.

Send back the summary + `data/models/htf_walkforward_sol_cal/` as usual.
If (1) holds but (2) still fails after calibration, the honest conclusion is
the model's edge isn't expressible as a per-trade gate — we stop there.
