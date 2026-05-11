# News Signal Training Pipeline — Scope & Plan

**Branch:** `research/whale-news-data-audit`
**Drafted:** 2026-05-11
**Status:** Plan only — no code yet
**Where training runs:** Local Mac (per `feedback_training_local.md`). This server is for inference/scoring only.

## 1. Goal

Replace the heuristic `ext_pos_news` fade guard with a learned classifier that predicts whether a news event will produce a meaningful adverse forward-return at the trade-decision horizon. Production usage: gate LONG/SHORT entries when the model's predicted edge is significantly against the candidate side.

The existing guard already proves the signal exists:
- Extreme-positive sentiment (>+0.5) at 60m → −13.7 bps (p=2.5e-6, Bonferroni-OK)
- Extreme-positive sentiment at 240m → −35.0 bps (p=2.4e-7)

Goal of the model is to extend this beyond the single hand-tuned threshold to a calibrated probability per article.

## 2. Data

**Source 1 — durable labeled record:** `logs/news_pending_alerts.jsonl`
- 1,844 articles as of 2026-05-11, span Apr 8 → May 11 (33 days, ~56 articles/day)
- Fields: timestamp, source, title, body, sentiment_score, confidence, urgency, event_type, assets, scorer_method

**Source 2 — live stream:** `data/trading.db` table `news_events` (schema in `src/data/storage.py:138-162`)
- 7-day retention — for training, dump to durable file via the JSONL alerts as primary

**Source 3 — price truth:** kline cache (`data/kline_cache/`) for forward-return labels.

**Hold-out:** Last 7 days of data → test set. Days 8–14 from end → validation. Everything before → train. Strictly time-ordered, no shuffle.

## 3. Features

Per article:

**Sentiment block (raw inputs):**
- `sentiment_score` (float, -1..+1)
- `confidence` (float, 0..1)
- `urgency` (int, {1,2,3})
- `scorer_method` one-hot (Groq vs keyword-fallback) — confidence varies by source

**Categorical block (one-hot):**
- `event_type` ∈ {regulatory, macro, geopolitical, influencer, exchange, hack, adoption, other}
- `source` (top-20 sources, rest → "other")
- `assets_mentioned`: multi-hot over {BTC, ETH, SOL, XRP, ALL}

**Temporal block:**
- Hour-of-day sin/cos
- Day-of-week one-hot
- Minutes since previous Tier 2+ article (recency proxy)
- Count of articles in trailing 60m / 240m / 24h (news-flow density)

**Sentiment context block:**
- Rolling mean sentiment over trailing 60m / 240m (per-asset)
- Rolling article count in trailing 60m / 240m
- Time since last extreme-positive (>0.5) article
- Time since last extreme-negative (<-0.5) article

**Price-context block (joined from klines at article timestamp):**
- BTC trailing 60m return, 240m return, 24h return
- BTC realized vol over last 24h
- ETH trailing 60m, 240m returns
- Open interest delta over 60m (if available)

Total feature vector: ~40–55 dimensions.

## 4. Labels

Forward-return-based, per asset, multi-horizon:

For each article at time `t`, for each asset `a` ∈ {BTC, ETH, SOL, XRP}:
- `r_5m`  = (close(t+5m, a) − close(t, a)) / close(t, a)
- `r_15m` = (close(t+15m) − close(t)) / close(t)
- `r_60m` = (close(t+60m) − close(t)) / close(t)
- `r_240m` = (close(t+240m) − close(t)) / close(t)

For the LONG-fade use case (matching `ext_pos_news` guard):
- Primary target: `y_long_fade` = 1 if `r_60m < -0.10%` else 0
- Secondary: regression on `r_60m` directly for calibration

Per-asset model OR per-asset target columns in one shared model — start with shared model + asset one-hot.

## 5. Model

Three baselines in order. Don't skip to step 3 — start simple:

**Baseline A — Logistic regression** with the feature set above. Establishes a lower bound and tells you which features carry weight.

**Baseline B — Gradient boosting** (LightGBM or XGBoost). Strong on tabular + handles non-linearities. Expected to be the production model.

**Baseline C — Small MLP** (2 hidden layers, 64 units each, dropout 0.2). Only if B clearly hits a ceiling.

Skip transformers/LSTMs — the data volume (1,844 articles over 33d) is too small for sequence models to outperform tabular.

## 6. Training

- Train on Mac M3 (per the hard rule)
- Class imbalance: positive examples (fade-confirmed) will be ~10–20% of articles. Use class weights or SMOTE on training only.
- Cross-validate via expanding-window time-series CV (not k-fold — leaks future into past)
- 5 expanding folds: train on first N weeks, test on next, expand by 1 week each fold

**Hyperparameter tuning:** Optuna 50 trials per model, optimize PR-AUC on the validation fold.

## 7. Evaluation Metrics

Primary:
- **PR-AUC** (better than ROC-AUC for imbalanced classes)
- **Precision @ recall=0.5** — at the operating point we'd plausibly run live

Secondary:
- **Brier score** for probability calibration
- **Calibration curve** (binned by predicted probability, observed rate)
- **Per-event-type breakdown**: does the model only work on regulatory news, or also adoption / hack?
- **Per-asset breakdown**: BTC/ETH likely stronger than XRP given sample size

Acceptance criteria for production:
- PR-AUC > 0.65 (the current guard hits ~0.55 by analogy)
- Precision at recall=0.5 > 0.40
- Calibration error (ECE) < 0.05 in top decile
- Strictly improves over current `ext_pos_news` heuristic on the held-out 7d

## 8. Production Integration

The model lives behind the existing `ext_pos_news` guard interface in `live_trading_htf.py:~1630`:

1. Replace the hard-threshold check (`sentiment > 0.5`) with `model.predict_proba(article) > tau`
2. `tau` is tuned offline so the gate fires at the same overall frequency as today (drift-control)
3. Fail-open: if model load or prediction errors, fall back to the heuristic threshold
4. Log every gating decision with both the heuristic and model outputs for 2 weeks of A/B telemetry before fully switching

Model file location: `data/models/news/news_fade_v1.{pkl|onnx}`
Loader: new module `src/news/predictor.py` mirroring `src/whale_behavior/models/predictor.py` shape.

## 9. Risks & Open Questions

- **33-day window is short.** Model may overfit to current regime. Mitigation: shrink hyperparameter search space, prefer simpler models, accept lower ceiling. Re-train monthly.
- **Groq scorer drift.** If Groq updates its model, sentiment_score distribution may shift. Mitigation: monitor sentiment-score histograms weekly, retrain when KL-divergence vs training distribution > 0.1.
- **Asset-side coverage.** XRP has only 35 article mentions. The model will be near-random on XRP — accept that and gate only BTC/ETH/SOL initially.
- **Forward-return labels are noisy at 5m.** Spreads and slippage are non-trivial relative to expected effects. Prefer 60m+ horizon as primary; 5m as secondary diagnostic only.

## 10. Concrete Execution Steps (for Chen on Mac)

1. `git checkout research/whale-news-data-audit` and pull
2. `python scripts/build_news_dataset.py` (to be written) — joins `news_pending_alerts.jsonl` with klines, emits `data/training/news_labeled_v1.parquet`
3. `python scripts/train_news_classifier.py` (to be written) — trains LightGBM with the feature set above, writes `data/models/news/news_fade_v1.pkl` + `training_results.json`
4. `python scripts/eval_news_classifier.py` (to be written) — produces PR curves, calibration plots, per-category/per-asset breakdown
5. If acceptance criteria pass: open PR back to dev, integration follows the production-integration section
