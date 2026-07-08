# Whale Behavior Model — Mac Retrain Runbook (READY TO RUN)

**For:** Chen, on his Mac (training is Mac-only — never on the server).
**Why now:** since **2026-07-04 06:14 UTC** the live bot throws on every iteration:
`Failed to load whale behavior model: Error(s) in loading state_dict for WhaleBehaviorLSTM`.
**Cause:** the code was upgraded to the v2 architecture (commit `a209fbc`:
LSTM hidden=128, input=25 = 9 continuous + 3 embeddings, plus an `fc2` layer),
but the checkpoint on disk (`data/whale_behavior/models/whale_behavior_lstm.pt`,
dated Apr 6) is the OLD v1 net (hidden=64, input=19, no `fc2`). Code and weights
don't match, so `_load()` fails.

**Impact today:** the predictor is **fail-open** — it returns `None`, so it does
NOT block trades. The only loss is that the whale signal (our strongest single
predictor historically) is dark. Fixing it re-arms that signal. No emergency.

**The fix:** retrain on the Mac with the *current* code — that regenerates the
`.pt` at the exact path the predictor loads, with the matching v2 architecture,
which clears the load error.

---

## Step 0 — Setup (once)

```bash
cd drl-trading-system
git checkout feature/autonomous-loop
git pull                         # gets the current v2 model code + committed training data
python3 -m venv .venv && source .venv/bin/activate   # if not already
pip install -r requirements.txt  # torch, etc. MPS build is fine on M-series
```

## Step 1 — (Optional) refresh the behavioral labels

The committed behavioral labels (`data/whale_behavior/labeled_v2/*_behavioral.jsonl`)
are from Apr 6. Retraining on them already fixes the load error. If you also want
FRESH labels from the raw whale data the server has been collecting, regenerate
them first:

```bash
python3 -c "from src.whale_behavior.data.behavioral_labeler import label_all_whale_wallets; print(label_all_whale_wallets())"
```
This re-labels every wallet in `data/whale_behavior/eth/*.jsonl` and rewrites
`data/whale_behavior/labeled_v2/*_behavioral.jsonl`. NOTE: your Mac will label
whatever raw data is in your checkout — to label the *freshest* server-collected
data, ask Claude to push a data snapshot first (the server's raw `eth/*.jsonl`
have uncommitted live updates). If you skip this whole step, training uses the
committed Apr-6 labels — which still fixes the load error.

## Step 2 — Train (~minutes on M-series, MPS)

```bash
# Behavioral mode = the v2 design (ACCUMULATING/DISTRIBUTING labels from labeled_v2/)
python3 train_whale_behavior.py --behavioral --window 4h --epochs 50 --patience 10
```
Output → `data/whale_behavior/models/whale_behavior_lstm.pt` (overwrites the stale one).
Watch the log for train/val accuracy; early-stopping kicks in on `--patience`.

## Step 3 — Ship the model back to the server

```bash
scp data/whale_behavior/models/whale_behavior_lstm.pt \
    claude@116.203.196.107:~/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/workspace/drl-trading-system/data/whale_behavior/models/
```

## Step 4 — Reload on the server (tell Claude, or run it)

```bash
cd ~/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/workspace/drl-trading-system
./start_services.sh              # bots reload the fresh checkpoint on restart
# verify the error is gone:
sleep 90 && grep -c "Failed to load whale behavior model" logs/bots_live.log   # should stop climbing
```

## Verify success
- `logs/bots_live.log` no longer prints the `Failed to load whale behavior model`
  error each iteration.
- A whale signal appears in `entry_signals.signals_json` (`"whale": {... "intent"
  != "unavailable" ...}`) once wallets are active.

## Rollback
If the new model behaves worse, the old `.pt` is recoverable from git history
(`git log --oneline -- data/whale_behavior/models/whale_behavior_lstm.pt`), but
since the current one doesn't even load, any trainable checkpoint is an
improvement.
