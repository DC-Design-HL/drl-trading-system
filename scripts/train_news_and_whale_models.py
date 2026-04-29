#!/usr/bin/env python3
"""
Train two separate supervised models:
  * News-only model — predicts trade win/loss using ONLY news features
                      (count, avg/min sentiment, max urgency at 1h/4h/24h)
  * Whale-only model — predicts trade win/loss using ONLY whale-flow features
                       (net flow, intent, in/out counts at 4h/12h/24h)

Both use LightGBM gradient-boosted trees. Walk-forward 60/20/20 split
with 8h embargo (smaller than SGFilter's 48h to avoid losing data).

Run on server, no Mac needed:
    python3 scripts/train_news_and_whale_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "training" / "sgfilter_dataset.parquet"
OUT_DIR = REPO / "data" / "models" / "news_whale"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_walkforward(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2,
                      embargo_hours: int = 8):
    df = df.sort_values("open_ts").reset_index(drop=True).copy()
    df["open_ts_dt"] = pd.to_datetime(df["open_ts"])
    n = len(df)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - test_n - val_n
    train = df.iloc[:train_n]
    val = df.iloc[train_n:train_n + val_n]
    test = df.iloc[train_n + val_n:]
    if not train.empty and not val.empty:
        end = train["open_ts_dt"].max()
        val = val[val["open_ts_dt"] > end + pd.Timedelta(hours=embargo_hours)]
    if not val.empty and not test.empty:
        end = val["open_ts_dt"].max()
        test = test[test["open_ts_dt"] > end + pd.Timedelta(hours=embargo_hours)]
    return train, val, test


def train_one(name: str, X_tr, y_tr, X_val, y_val, X_te, y_te, df_te, seed: int = 42):
    import lightgbm as lgb
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary", "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt", "num_leaves": 15, "learning_rate": 0.05,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 5,
        "min_data_in_leaf": 10, "verbose": -1, "seed": seed,
    }
    model = lgb.train(params, train_set, num_boost_round=500,
                      valid_sets=[train_set, val_set], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    # Eval on test
    proba_te = model.predict(X_te)
    accept_thresh = 0.5
    accepted = proba_te >= accept_thresh
    n_accept = int(accepted.sum())
    accept_pnl = float(df_te.loc[accepted, "pnl"].sum()) if n_accept else 0.0
    reject_pnl = float(df_te.loc[~accepted, "pnl"].sum())
    accept_wr = float(df_te.loc[accepted, "win"].mean()) if n_accept else 0.0
    base_pnl = float(df_te["pnl"].sum())
    base_wr = float(df_te["win"].mean())
    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(y_te, proba_te)
    except Exception:
        test_auc = float("nan")
    return {
        "name": name,
        "n_test": len(df_te),
        "test_auc": test_auc,
        "accepted": n_accept,
        "accepted_wr": accept_wr,
        "accepted_pnl": accept_pnl,
        "rejected_pnl": reject_pnl,
        "delta_vs_no_filter": -reject_pnl,
        "baseline_test_pnl": base_pnl,
        "baseline_test_wr": base_wr,
    }, model


def main():
    if not DATASET.exists():
        print(f"Dataset not found: {DATASET}", file=sys.stderr)
        return 1
    df = pd.read_parquet(DATASET)
    print(f"Loaded {len(df)} trades, {len(df.columns)} cols")
    news_cols = [c for c in df.columns if c.startswith("news_")]
    whale_cols = [c for c in df.columns if c.startswith("whale_")]
    print(f"News features ({len(news_cols)}): {news_cols}")
    print(f"Whale features ({len(whale_cols)}): {whale_cols}")
    train_df, val_df, test_df = split_walkforward(df)
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("\n=== Training news-only model ===")
    news_result, news_model = train_one(
        "news_only",
        train_df[news_cols].values, train_df["win"].astype(int).values,
        val_df[news_cols].values, val_df["win"].astype(int).values,
        test_df[news_cols].values, test_df["win"].astype(int).values, test_df,
    )
    for k, v in news_result.items():
        print(f"  {k}: {v}")
    news_model.save_model(str(OUT_DIR / "news_only.txt"))

    print("\n=== Training whale-only model ===")
    whale_result, whale_model = train_one(
        "whale_only",
        train_df[whale_cols].values, train_df["win"].astype(int).values,
        val_df[whale_cols].values, val_df["win"].astype(int).values,
        test_df[whale_cols].values, test_df["win"].astype(int).values, test_df,
    )
    for k, v in whale_result.items():
        print(f"  {k}: {v}")
    whale_model.save_model(str(OUT_DIR / "whale_only.txt"))

    # Acceptance gate
    def gate(r):
        # 3-class random baseline = 50% on binary win/loss problem
        if r["accepted"] == 0:
            return False, "model accepted zero trades"
        if r["test_auc"] < 0.55:
            return False, f"test AUC {r['test_auc']:.3f} < 0.55 (no signal vs random)"
        if r["delta_vs_no_filter"] <= 0:
            return False, f"filter delta_pnl ${r['delta_vs_no_filter']:+.2f} <= 0"
        if r["accepted_wr"] < r["baseline_test_wr"]:
            return False, f"accepted WR {r['accepted_wr']:.3f} < baseline test WR {r['baseline_test_wr']:.3f}"
        return True, "passes"

    print("\n=== Acceptance gates ===")
    for r in (news_result, whale_result):
        ok, why = gate(r)
        print(f"  {r['name']}: {'✅ PASS' if ok else '❌ FAIL'} — {why}")

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "news_model": news_result,
        "whale_model": whale_result,
    }
    (OUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten: {OUT_DIR}/training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
