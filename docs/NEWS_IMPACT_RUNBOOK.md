# News-Impact Model — Runbook (v0, feasibility)

**Goal:** predict whether an asset moves UP after a news event, from the news
features the sentinel already extracts (sentiment, urgency, event_type, source,
confidence, asset, hour).

**Status (2026-07-08): NO EDGE YET — this is expected.** On the 405 events we
have, sentiment→direction AUC ≈ 0.51 and forward returns are dominated by overall
market drift (even *bearish* news is followed by up-moves because the whole market
rose). There isn't enough data yet, and the raw-return target is drift-contaminated.
The point of v0 is to have the pipeline ready so it improves as data accumulates.

**Why we only had 405:** the sentinel was auto-deleting events older than 7 days.
Fixed 2026-07-08 (`DB_CLEANUP_DAYS` 7 → 3650) — the corpus now grows ~45/day.
Re-run this in a few weeks with a real sample.

## Pipeline

### 1. Label (safe on server OR Mac — it's data processing, not training)
```bash
python3 scripts/news_impact/label_news.py
# -> data/news_impact/news_labeled.csv  + prints the signal check
```
Joins each `news_events` row to the mentioned asset's +1h/+4h forward return from
Binance 5m klines (ALL → BTC proxy). Also prints base rates, sentiment AUC, and
per-event-type mean returns so you can eyeball signal before training.

### 2. Train (MAC ONLY — training never runs on the server)
```bash
python3 scripts/news_impact/train_news_impact.py --horizon ret_4h
python3 scripts/news_impact/train_news_impact.py --horizon ret_1h
```
- Gradient-boosted classifier, **time-ordered** train/test split (no shuffle — news
  is a time series; shuffling would leak the future).
- Reports OOS AUC + accuracy vs the majority-class baseline. If it can't beat the
  baseline, there's no edge — don't wire it into the bot.
- Saves `data/news_impact/models/news_impact_gbm.joblib` + `news_impact_report.json`.

## When is it worth wiring into live trading?
Only when a Mac run shows **OOS AUC > ~0.57 AND accuracy > baseline + 3pts**, on a
sample of at least a few thousand events. Until then it stays offline.

## Next improvements (do these before expecting edge)
1. **Market-excess target** — subtract BTC's same-window return from the asset's
   return so the model learns news effect, not market drift. This is the single
   biggest fix. (v0 uses raw return; flagged in the trainer.)
2. **Dedup near-duplicate headlines** — the same story hits multiple RSS feeds;
   collapse by title similarity so one event isn't counted 3×.
3. **Surprise features** — novelty vs recent similar headlines; a repeated theme
   moves price less than a genuinely new one.
4. **Bigger sample** — the retention fix is the enabler; give it weeks.

## Reproduce the current "no edge" read
```bash
python3 scripts/news_impact/label_news.py   # look at the SIGNAL CHECK block
```
