#!/usr/bin/env python3
"""
eval_news_classifier.py — deep-dive evaluation of the news fade classifier.

Compares the model on the test split against:
  - random baseline (always predicts positive_rate)
  - existing heuristic (sentiment_score > 0.5)

Reports:
  - Confusion matrix at multiple thresholds
  - Calibration table
  - Per-event-type, per-source breakdown
  - Regime drift analysis between train/val/test
  - Comparison of label positive-rate over time

Writes: data/models/news/eval_v1.json
"""

from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "training" / "news_labeled_v1.parquet"
MODEL_PATH = REPO / "data" / "models" / "news" / "news_fade_v1.joblib"
FEATURE_CONFIG_PATH = REPO / "data" / "models" / "news" / "feature_config.json"
OUT_PATH = REPO / "data" / "models" / "news" / "eval_v1.json"

TARGET = "y_long_fade"


def time_split(df, test_frac=0.15, val_frac=0.15):
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def heuristic_proba(df):
    """Mimic the existing ext_pos_news guard: fade-likely when sentiment > +0.5."""
    return (df["sentiment_score"] > 0.5).astype(float).values


def regime_drift(splits):
    """Show positive rate, date range, sentiment distribution across splits."""
    out = {}
    for name, df in splits.items():
        out[name] = {
            "n": int(len(df)),
            "positive_rate": float(df[TARGET].mean()),
            "date_start": str(df["published_at"].min()),
            "date_end": str(df["published_at"].max()),
            "sentiment_mean": float(df["sentiment_score"].mean()),
            "sentiment_extreme_pos_rate": float((df["sentiment_score"] > 0.5).mean()),
            "sentiment_extreme_neg_rate": float((df["sentiment_score"] < -0.5).mean()),
            "btc_60m_return_mean": float(df["r_60m_BTC"].mean()) if "r_60m_BTC" in df else None,
        }
    return out


def calibration_table(y_true, y_proba, n_bins=10):
    """Reliability diagram values: predicted vs observed positive rate per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        out.append({
            "bin_low": float(bins[i]),
            "bin_high": float(bins[i + 1]),
            "n": int(mask.sum()),
            "predicted_mean": float(y_proba[mask].mean()),
            "observed_positive_rate": float(y_true[mask].mean()),
        })
    return out


def threshold_table(y_true, y_proba, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    out = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out.append({"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "precision": prec, "recall": rec, "f1": f1})
    return out


def per_group(df, y_true, y_proba, key, min_n=20):
    out = {}
    for g, sub in df.groupby(key):
        if len(sub) < min_n:
            continue
        idx = sub.index.values
        yt = y_true.iloc[df.index.get_indexer(idx)].values if hasattr(y_true, "iloc") else y_true[df.index.get_indexer(idx)]
        yp = y_proba[df.index.get_indexer(idx)]
        if len(np.unique(yt)) > 1:
            pr_auc = float(average_precision_score(yt, yp))
            roc_auc = float(roc_auc_score(yt, yp))
        else:
            pr_auc = roc_auc = float("nan")
        out[str(g)] = {
            "n": int(len(sub)),
            "positive_rate": float(yt.mean()),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        }
    return out


def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df.sort_values("published_at").reset_index(drop=True)

    feat_cfg = json.loads(FEATURE_CONFIG_PATH.read_text())
    feat_cols = feat_cfg["features"]

    train_df, val_df, test_df = time_split(df)
    splits_dict = {"train": train_df, "val": val_df, "test": test_df}

    model = joblib.load(MODEL_PATH)
    results = {"target": TARGET, "regime_drift": regime_drift(splits_dict)}

    for name, sub in splits_dict.items():
        X = sub[feat_cols].copy()
        y = sub[TARGET].astype(int)
        if len(np.unique(y)) < 2:
            print(f"[skip {name}] only one class present")
            continue
        proba = model.predict_proba(X)[:, 1]
        heur = heuristic_proba(sub)

        model_pr = float(average_precision_score(y, proba))
        heur_pr = float(average_precision_score(y, heur))
        model_roc = float(roc_auc_score(y, proba))
        heur_roc = float(roc_auc_score(y, heur))

        block = {
            "n": int(len(y)),
            "positive_rate": float(y.mean()),
            "model_pr_auc": model_pr,
            "heuristic_pr_auc": heur_pr,
            "model_roc_auc": model_roc,
            "heuristic_roc_auc": heur_roc,
            "model_brier": float(brier_score_loss(y, proba)),
            "heuristic_brier": float(brier_score_loss(y, heur)),
            "calibration": calibration_table(y.values, proba),
            "thresholds": threshold_table(y.values, proba),
            "per_event_type": per_group(sub.reset_index(drop=True), y.reset_index(drop=True), proba, "event_type", min_n=10),
            "per_scorer_method": per_group(sub.reset_index(drop=True), y.reset_index(drop=True), proba, "scorer_method", min_n=10),
        }
        results[name] = block

        print(f"\n=== {name} ({len(y)} rows, pos_rate {y.mean():.3f}) ===")
        print(f"  model    PR-AUC {model_pr:.4f}  ROC-AUC {model_roc:.4f}")
        print(f"  heuristic PR-AUC {heur_pr:.4f}  ROC-AUC {heur_roc:.4f}")
        print(f"  delta:   {model_pr - heur_pr:+.4f} PR-AUC ({'WIN' if model_pr > heur_pr else 'LOSS'} vs heuristic)")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n✅ wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
