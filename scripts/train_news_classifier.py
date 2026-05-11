#!/usr/bin/env python3
"""
train_news_classifier.py — train a news -> long-fade classifier.

Reads:  data/training/news_labeled_v1.parquet
Writes: data/models/news/news_fade_v1.joblib
        data/models/news/training_results.json
        data/models/news/feature_config.json

Model: sklearn HistGradientBoostingClassifier (LightGBM-equivalent, no extra deps).
Target: y_long_fade = 1 if r_60m_BTC < -0.10% else 0.
Split: time-ordered (last 15% test, prior 15% val, rest train).

Plan ref: docs/research/news_training_pipeline.md
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
import joblib

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "training" / "news_labeled_v1.parquet"
MODEL_DIR = REPO / "data" / "models" / "news"
MODEL_PATH = MODEL_DIR / "news_fade_v1.joblib"
RESULTS_PATH = MODEL_DIR / "training_results.json"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

TARGET = "y_long_fade"
META_COLS = {
    "type", "event_id", "urgency", "event_type", "source", "title",
    "sentiment_score", "confidence", "assets", "reasoning",
    "published_at", "scorer_method", "timestamp",
    TARGET, "y_r_60m_btc",
}
# Drop raw forward-return columns (would be label leakage)
LEAKAGE_PREFIXES = ("r_5m_", "r_15m_", "r_60m_", "r_240m_")


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df.sort_values("published_at").reset_index(drop=True)

    # Re-add raw sentiment / confidence / urgency / is_groq / is_keyword
    # as numeric features (event_type/source are already one-hot encoded).
    keep_raw = ["sentiment_score", "confidence", "urgency"]

    # Feature columns: everything that's a number, isn't a label/meta, isn't a forward return
    feat_cols = []
    for c in df.columns:
        if c in META_COLS and c not in keep_raw:
            continue
        if any(c.startswith(p) for p in LEAKAGE_PREFIXES):
            continue
        if df[c].dtype in (np.int64, np.int32, np.float64, np.float32, np.bool_):
            feat_cols.append(c)

    X = df[feat_cols].copy()
    y = df[TARGET].astype(int)

    # HistGradientBoosting can handle NaNs natively
    return X, y, feat_cols


def time_split(X: pd.DataFrame, y: pd.Series, test_frac=0.15, val_frac=0.15):
    n = len(X)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    return (
        X.iloc[:val_start], y.iloc[:val_start],
        X.iloc[val_start:test_start], y.iloc[val_start:test_start],
        X.iloc[test_start:], y.iloc[test_start:],
    )


def evaluate(y_true, y_proba, label):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_proba)
    # precision at recall ≈ 0.5
    idx = np.searchsorted(recall[::-1], 0.5)
    prec_at_r50 = float(precision[::-1][idx]) if idx < len(precision) else float("nan")
    return {
        "split": label,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier": float(brier),
        "precision_at_recall_0.5": prec_at_r50,
    }


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading {DATA_PATH}", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"      rows: {len(df)}", flush=True)

    print("[2/5] building X, y", flush=True)
    X, y, feat_cols = build_xy(df)
    print(f"      features: {len(feat_cols)}", flush=True)
    print(f"      positive rate: {y.mean():.3f}", flush=True)

    print("[3/5] time-ordered split", flush=True)
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)
    print(f"      train: {len(X_train)} ({y_train.mean():.3f} pos)", flush=True)
    print(f"      val:   {len(X_val)} ({y_val.mean():.3f} pos)", flush=True)
    print(f"      test:  {len(X_test)} ({y_test.mean():.3f} pos)", flush=True)

    print("[4/5] training HistGradientBoostingClassifier (single-thread, capped)", flush=True)
    # Lightweight grid: keep training under a minute on 1.5k rows
    best_val_pr = -1
    best_params = None
    best_model = None
    for max_depth in (3, 5, 7):
        for learning_rate in (0.05, 0.1):
            for max_iter in (100, 200):
                model = HistGradientBoostingClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                    class_weight="balanced",
                )
                model.fit(X_train, y_train)
                val_proba = model.predict_proba(X_val)[:, 1]
                val_pr = average_precision_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.0
                if val_pr > best_val_pr:
                    best_val_pr = val_pr
                    best_params = {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "max_iter": max_iter,
                    }
                    best_model = model
    print(f"      best val PR-AUC: {best_val_pr:.4f} @ {best_params}", flush=True)

    print("[5/5] evaluating on all splits", flush=True)
    splits = {}
    for X_, y_, label in [(X_train, y_train, "train"), (X_val, y_val, "val"), (X_test, y_test, "test")]:
        proba = best_model.predict_proba(X_)[:, 1]
        splits[label] = evaluate(y_, proba, label)
        print(f"      {label}: PR-AUC={splits[label]['pr_auc']:.4f} "
              f"ROC-AUC={splits[label]['roc_auc']:.4f} "
              f"Brier={splits[label]['brier']:.4f} "
              f"P@R50={splits[label]['precision_at_recall_0.5']:.4f}",
              flush=True)

    # Feature importance from the model (sklearn HGB exposes via permutation in this version)
    # Use feature_importances_ if available, else skip
    fi = {}
    try:
        if hasattr(best_model, "feature_importances_"):
            fi = dict(sorted(
                zip(feat_cols, best_model.feature_importances_.tolist()),
                key=lambda x: -x[1]
            )[:20])
    except Exception:
        pass

    results = {
        "model": "HistGradientBoostingClassifier",
        "target": TARGET,
        "n_features": len(feat_cols),
        "best_params": best_params,
        "best_val_pr_auc": best_val_pr,
        "splits": splits,
        "top_features": fi,
    }

    joblib.dump(best_model, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    FEATURE_CONFIG_PATH.write_text(json.dumps({"features": feat_cols, "target": TARGET}, indent=2))
    print(f"\n✅ saved model to {MODEL_PATH}", flush=True)
    print(f"   results: {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
