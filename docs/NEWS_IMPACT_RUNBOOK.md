# News-Impact Model — Runbook (v0, feasibility)

**Goal:** predict whether an asset moves UP after a news event, from the news
features the sentinel already extracts (sentiment, urgency, event_type, source,
confidence, asset, hour).

**Status (2026-07-12): NO MEANINGFUL EDGE — now on a robust sample.** Chen
exported the Telegram news-alert channel (Luigi_News), recovering **3,017 unique
events over 3 months (Apr 9 → Jul 12)** — the DB only had 405 because the sentinel
purged >7d (fixed 2026-07-08, `DB_CLEANUP_DAYS` 7→3650). On the full sample:
- sentiment→direction AUC ≈ **0.49–0.54** across 1h/4h, raw AND market-excess.
- base rate ~50% (the earlier 58% was just a 9-day bull window; drift washes out
  over 3 months).
This is a trustworthy read, not a small-sample fluke: news sentiment does not
predict direction. Caveat: 2,770 of 3,017 events are BTC/market-wide; only ~247
are asset-specific (ETH/SOL/XRP), so per-asset excess signal is thin.

**Definitive test still to run (Mac):** the single-feature AUC is ~0.5, but the
full GBM (all features together) is the final word — run `train_news_impact.py`
on the Mac. Don't expect much; wire to live only if it clearly beats baseline.

**Data files:**
- `data/news_impact/news_labeled_full.csv` — 3,017 events (from the Telegram export)
- `data/news_impact/news_labeled.csv` — the 7-day DB slice (small)

## Pipeline

### 0. (One-time / periodic) Recover history from a Telegram export
```bash
# Telegram Desktop → news channel → Export chat history → JSON
python3 scripts/news_impact/parse_telegram_news.py path/to/result.json
# -> data/news_impact/news_labeled_full.csv (parses alerts, joins prices,
#    computes raw + market-excess returns, prints the signal check)
```

### 1. Label the live DB slice (safe on server OR Mac — data processing, not training)
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
