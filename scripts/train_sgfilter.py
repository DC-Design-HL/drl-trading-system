#!/usr/bin/env python3
"""
Project 1 — Structure-Gated Filter (binary classifier).

Runs on Mac M3. Loads the parquet bundle produced by
scripts/dump_training_dataset.py and trains a model that predicts
whether a structure-first signal will win or lose, given:
  * MTF / regime / order-flow / orderbook / funding / whale signals
  * News context (count, sentiment, urgency in last 1h/4h/24h)
  * On-chain whale flow (net flow, intent score in last 4h/12h/24h)
  * Time-of-day features

Output: a binary classifier whose probability output replaces the
"trade everything structure says" heuristic. The bot accepts a signal
only if `predict_proba >= ACCEPT_THRESHOLD`.

We use LightGBM (gradient boosted trees). Reasons over RNN/PPO:
  * Tabular features are the right fit. Trees outperform deep nets on
    this type of input distribution.
  * Trains in seconds on Mac M3 CPU, no GPU needed.
  * Native handling of categorical features (mtf_bias, regime_type, etc.)
  * Native feature importance for interpretability — you can see WHICH
    features the model relies on, not just take it on faith.
  * Direct probability output → simple ACCEPT/REJECT threshold.
  * Reproducible (deterministic with fixed seed).

Walk-forward validation: 3-way time split with 48h embargo to prevent
lookahead leak.

Run on Mac:
    pip install pandas lightgbm scikit-learn pyarrow
    python3 scripts/train_sgfilter.py
        [--dataset data/training/sgfilter_dataset.parquet]
        [--output data/models/sgfilter/]
        [--seeds 3]
        [--accept_threshold 0.55]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

CATEGORICAL_FEATURES = [
    "symbol", "side",
    "sig_mtf_bias", "sig_mtf_15m_dir", "sig_mtf_1h_dir", "sig_mtf_4h_dir",
    "sig_of_bias", "sig_regime_type", "sig_regime_state",
    "sig_ob_bias", "sig_whale_dir",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_walkforward(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2,
                      embargo_hours: int = 48) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict time-ordered split with embargo gaps between train/val/test."""
    df = df.sort_values("open_ts").reset_index(drop=True)
    df["open_ts_dt"] = pd.to_datetime(df["open_ts"])
    n = len(df)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - test_n - val_n

    train = df.iloc[:train_n].copy()
    val = df.iloc[train_n:train_n + val_n].copy()
    test = df.iloc[train_n + val_n:].copy()

    # Embargo: drop rows in val that fall within `embargo_hours` of train end.
    if not train.empty and not val.empty:
        train_end = train["open_ts_dt"].max()
        val = val[val["open_ts_dt"] > train_end + pd.Timedelta(hours=embargo_hours)]
    if not val.empty and not test.empty:
        val_end = val["open_ts_dt"].max()
        test = test[test["open_ts_dt"] > val_end + pd.Timedelta(hours=embargo_hours)]

    return train.drop(columns=["open_ts_dt"]), val.drop(columns=["open_ts_dt"]), test.drop(columns=["open_ts_dt"])


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].copy()
    # LightGBM handles NaN natively but cast categoricals
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["win"].astype(int).values
    return X, y


def train_one_seed(X_train, y_train, X_val, y_val, feature_cols: list[str], seed: int):
    import lightgbm as lgb
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cats)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cats, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 8,
        "verbose": -1,
        "seed": seed,
    }

    model = lgb.train(
        params, train_set, num_boost_round=600,
        valid_sets=[train_set, val_set], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model


def evaluate(model, X, y, name: str, accept_threshold: float, df_meta: pd.DataFrame) -> dict:
    """Probability output → ACCEPT/REJECT decision → simulated PnL.
    The 'simulated' pnl here is the actual realized pnl of trades the bot
    historically took at these signals. So it's not synthetic — it's
    'what the historical bot would have done if our filter were applied'.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    proba = model.predict(X)
    accepted = proba >= accept_threshold
    auc = roc_auc_score(y, proba) if len(set(y)) > 1 else None
    acc = accuracy_score(y, accepted)
    prec = precision_score(y, accepted, zero_division=0)
    rec = recall_score(y, accepted, zero_division=0)
    base_pnl = df_meta["pnl"].sum()
    accepted_pnl = df_meta.loc[accepted, "pnl"].sum() if any(accepted) else 0.0
    rejected_pnl = df_meta.loc[~accepted, "pnl"].sum() if any(~accepted) else 0.0
    accept_count = int(accepted.sum())
    base_n = len(df_meta)
    accept_wr = float(df_meta.loc[accepted, "win"].mean()) if accept_count else 0.0
    return {
        "split": name,
        "n": int(base_n),
        "n_accepted": accept_count,
        "n_rejected": int(base_n - accept_count),
        "auc": auc,
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "baseline_pnl": float(base_pnl),
        "accepted_pnl": float(accepted_pnl),
        "rejected_pnl": float(rejected_pnl),
        "delta_pnl": float(-rejected_pnl),
        "accepted_wr": accept_wr,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/training/sgfilter_dataset.parquet")
    p.add_argument("--output", default="data/models/sgfilter/")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--accept_threshold", type=float, default=0.55)
    p.add_argument("--per_symbol", action="store_true",
                   help="Train separate models per (symbol, side) instead of one global model.")
    args = p.parse_args(argv)

    dataset = REPO / args.dataset
    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset} ...")
    df = load_dataset(dataset)
    print(f"  {len(df)} rows, {len(df.columns)} cols")
    if "win" not in df.columns:
        print("ERROR: dataset missing 'win' label column", file=sys.stderr)
        return 1

    # Feature selection
    meta_path = dataset.with_name(dataset.stem.replace("_dataset", "_metadata") + ".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_cols = list(meta.get("feature_cols", []))
    else:
        feature_cols = [c for c in df.columns if c.startswith(("sig_", "news_", "whale_",
                                                                "hour_", "day_", "is_", "session_"))]
    # symbol + side as features (categorical)
    if "symbol" not in feature_cols and not args.per_symbol:
        feature_cols.append("symbol")
    if "side" not in feature_cols and not args.per_symbol:
        feature_cols.append("side")
    print(f"  feature columns: {len(feature_cols)}")

    splits = split_walkforward(df)
    print(f"  train={len(splits[0])}  val={len(splits[1])}  test={len(splits[2])}")
    if min(len(s) for s in splits) < 20:
        print("WARNING: at least one split is small (<20 rows). "
              "Walk-forward results will be noisy.", file=sys.stderr)

    train_df, val_df, test_df = splits
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    seed_models = []
    seed_results = []
    for s in range(args.seeds):
        seed = 42 + s
        print(f"\n=== Seed {seed} ===")
        m = train_one_seed(X_train, y_train, X_val, y_val, feature_cols, seed)
        seed_models.append(m)
        for split_name, X, y, meta_df in [("train", X_train, y_train, train_df),
                                          ("val", X_val, y_val, val_df),
                                          ("test", X_test, y_test, test_df)]:
            r = evaluate(m, X, y, split_name, args.accept_threshold, meta_df)
            print(f"  {r}")
            seed_results.append({"seed": seed, **r})

    # Ensemble: average probabilities
    print("\n=== Ensemble (median across seeds) ===")
    for split_name, X, y, meta_df in [("train", X_train, y_train, train_df),
                                      ("val", X_val, y_val, val_df),
                                      ("test", X_test, y_test, test_df)]:
        probs = np.stack([m.predict(X) for m in seed_models])
        ensemble_proba = np.median(probs, axis=0)
        accepted = ensemble_proba >= args.accept_threshold
        accept_n = int(accepted.sum())
        accept_pnl = float(meta_df.loc[accepted, "pnl"].sum()) if accept_n else 0.0
        reject_pnl = float(meta_df.loc[~accepted, "pnl"].sum())
        accept_wr = float(meta_df.loc[accepted, "win"].mean()) if accept_n else 0.0
        print(f"  {split_name}: n={len(meta_df)} accepted={accept_n} accepted_WR={accept_wr*100:.1f}% "
              f"accepted_pnl=${accept_pnl:+.2f} delta_vs_no_filter=${-reject_pnl:+.2f}")

    # Save models
    for i, m in enumerate(seed_models):
        m.save_model(str(out_dir / f"sgfilter_seed{42 + i}.txt"))

    # Feature importance from seed 0
    fi = sorted(zip(feature_cols, seed_models[0].feature_importance(importance_type="gain")),
                key=lambda x: -x[1])
    fi_path = out_dir / "feature_importance.txt"
    with open(fi_path, "w") as fp:
        for name, gain in fi:
            fp.write(f"{name}\t{gain:.0f}\n")
    print(f"\nFeature importance: {fi_path}")
    print("Top 15 features:")
    for name, gain in fi[:15]:
        print(f"  {name:40s} {gain:>10.0f}")

    summary = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": str(dataset),
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "seeds": [42 + i for i in range(args.seeds)],
        "accept_threshold": args.accept_threshold,
        "results": seed_results,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary written to {out_dir / 'training_summary.json'}")
    print("\nAcceptance gate (recommended):")
    print("  - Median val AUC across seeds >= 0.60")
    print("  - Test-set delta_pnl > 0  AND  test accepted_WR >= 60%")
    print("  - If gates pass, ship the model. If not, the dataset/labels need work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
