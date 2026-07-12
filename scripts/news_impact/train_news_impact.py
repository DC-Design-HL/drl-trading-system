#!/usr/bin/env python3
"""
News-impact model trainer (v0) — RUN ON MAC (training is Mac-only).

Reads data/news_impact/news_labeled.csv (produced by label_news.py) and trains a
gradient-boosted classifier to predict whether the mentioned asset moves UP over
the horizon. Uses a TIME-ORDERED train/test split (no shuffling — news is a time
series, shuffling leaks the future) and reports OOS AUC/accuracy vs the majority-
class baseline. Saves model + a JSON report.

Honest-by-design:
- Time-ordered split so the OOS number is real, not leaked.
- Reports the majority-class baseline alongside — if the model can't beat "always
  predict the base rate", there is no edge (expected until the corpus is large).
- Prints feature importances so you can see what (if anything) carries signal.

KNOWN LIMITATION (v0): the target is raw asset return, which is dominated by
overall market drift. Once there's more data, switch the target to MARKET-EXCESS
return (asset return minus BTC return over the same window) to strip the drift —
that's where real news edge, if any, will show up. Flagged in the runbook.

Usage:
    python3 scripts/news_impact/train_news_impact.py --horizon ret_4h
    python3 scripts/news_impact/train_news_impact.py --horizon ret_1h --deadband 0.001
"""
import argparse, csv, json, os
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Prefer the full Telegram-recovered dataset (~3k events, 3 months) if present,
# else the 7-day DB slice. Override with --data.
DATA_FULL = "data/news_impact/news_labeled_full.csv"
DATA_DB = "data/news_impact/news_labeled.csv"
MODEL_DIR = "data/news_impact/models"
CAT = ["symbol", "source", "event_type"]
NUM = ["urgency", "sentiment", "abs_sentiment", "confidence", "hour"]

def load(horizon, deadband, data_path):
    rows = list(csv.DictReader(open(data_path)))
    rows.sort(key=lambda r: int(r["ts_ms"]))          # time order
    X_cat, X_num, y, keep = [], [], [], []
    for r in rows:
        ret = float(r[horizon])
        if abs(ret) < deadband:                        # drop flat (ambiguous) moves
            continue
        X_cat.append([r[c] for c in CAT])
        X_num.append([float(r[n]) for n in NUM])
        y.append(1 if ret > 0 else 0)
    return np.array(X_cat), np.array(X_num, dtype=float), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    # ret_* = raw return; exc_* = market-excess (asset minus BTC) — use exc to
    # strip drift. exc is only meaningful for non-BTC assets.
    ap.add_argument("--horizon", default="ret_4h",
                    choices=["ret_1h", "ret_4h", "exc_1h", "exc_4h"])
    ap.add_argument("--deadband", type=float, default=0.001)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--data", default=None, help="labeled CSV (default: full if present)")
    args = ap.parse_args()

    data_path = args.data or (DATA_FULL if os.path.exists(DATA_FULL) else DATA_DB)
    print(f"data: {data_path}")
    Xc, Xn, y = load(args.horizon, args.deadband, data_path)
    n = len(y)
    if n < 50:
        print(f"Only {n} usable rows — too few to train meaningfully. "
              f"Let the corpus grow (news retention is now 3650d) and re-run.")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xc_oh = enc.fit_transform(Xc)
    X = np.hstack([Xc_oh, Xn])
    feat_names = list(enc.get_feature_names_out(CAT)) + NUM

    split = int(n * (1 - args.test_frac))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    base = max(yte.mean(), 1 - yte.mean())             # majority-class accuracy OOS

    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else float("nan")
    acc = accuracy_score(yte, pred)

    imp = sorted(zip(feat_names, clf.feature_importances_), key=lambda t: -t[1])[:12]

    print("=" * 56)
    print(f"NEWS-IMPACT MODEL v0 — horizon={args.horizon}")
    print("=" * 56)
    print(f"rows={n}  train={split}  test={n-split}")
    print(f"OOS AUC       = {auc:.3f}   (0.5 = no signal)")
    print(f"OOS accuracy  = {acc:.3f}")
    print(f"baseline acc  = {base:.3f}   (always predict majority class)")
    verdict = ("EDGE" if (auc > 0.57 and acc > base + 0.03) else "NO EDGE YET")
    print(f"VERDICT       = {verdict}")
    print("\ntop features:")
    for f, w in imp:
        print(f"  {w:.3f}  {f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    joblib.dump({"clf": clf, "encoder": enc, "cat": CAT, "num": NUM,
                 "horizon": args.horizon}, os.path.join(MODEL_DIR, "news_impact_gbm.joblib"))
    report = dict(horizon=args.horizon, n=n, oos_auc=auc, oos_acc=acc,
                  baseline_acc=base, verdict=verdict,
                  top_features=[[f, float(w)] for f, w in imp],
                  trained_at_ts_ms=None)  # stamp on Mac; server clock intentionally not used
    json.dump(report, open(os.path.join(MODEL_DIR, "news_impact_report.json"), "w"), indent=2)
    print(f"\nsaved model + report -> {MODEL_DIR}/")

if __name__ == "__main__":
    main()
