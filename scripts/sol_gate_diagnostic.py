#!/usr/bin/env python3
"""
SOL gate diagnostic — run on Chen's Mac against the existing Jul-4 walk-forward
fold models. NO TRAINING. Decides whether a calibrated retrain is even needed.

Question answered: does the (honest, Jul-4) SOL model's LOGIT MARGIN rank
winning entries above losing ones on each fold's out-of-sample test window?
The live gate attempt failed because max-prob confidence saturates at ~0.99;
the margin survives saturation. If pooled margin AUC >= 0.55 the existing
model already contains gate signal (send it over, no retrain). If ~0.5, only
a calibrated retrain (train_htf_walkforward.py --ent-floor 0.02) can help.

Usage (from repo root, training venv):
  .venv/bin/python scripts/sol_gate_diagnostic.py \
      --models-dir data/models/htf_walkforward_sol \
      --data-path data/historical/SOLUSDT_15m.csv

Fold test windows are reconstructed from each fold_result.json (date-only
boundaries; the +/- few edge bars vs the original slice are immaterial here).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_htf_walkforward import (  # noqa: E402
    _create_env, build_htf_dataframes, load_15m_csv,
)
from src.brain.gate_diagnostics import (  # noqa: E402
    gate_stats, join_trades, policy_step_records,
)


def _slice(df, start, end):
    mask = (df["open_time"] >= start) & (df["open_time"] < end)
    return df[mask].reset_index(drop=True)


def run_fold(fold_dir: Path, df_15m, df_1h, df_4h, df_1d):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    result_file = fold_dir / "fold_result.json"
    if not result_file.exists():
        return None
    meta = json.load(open(result_file))
    test_start = pd.Timestamp(meta["test_start"], tz="UTC")
    test_end = pd.Timestamp(meta["test_end"], tz="UTC") + pd.Timedelta(days=1)

    model_zip = fold_dir / "best_model.zip"
    if not model_zip.exists():
        model_zip = fold_dir / "fold_model.zip"
        print(f"  {fold_dir.name}: no best_model.zip — using fold_model.zip "
              "(final weights; slightly more overfit)")
    if not model_zip.exists():
        return None

    env = _create_env(
        _slice(df_15m, test_start, test_end),
        _slice(df_1h, test_start, test_end),
        _slice(df_4h, test_start, test_end),
        _slice(df_1d, test_start, test_end),
        training=False,
    )

    model = PPO.load(str(model_zip))
    vecnorm = None
    vn_path = fold_dir / "best_vecnorm.pkl"
    if not vn_path.exists():
        vn_path = Path(str(model_zip).removesuffix(".zip") + "_vecnorm.pkl")
    if vn_path.exists():
        vecnorm = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: env]))
        vecnorm.training = False
    else:
        print(f"  {fold_dir.name}: WARNING no vecnorm stats — raw obs "
              "(diagnostic will be unreliable)")

    records = policy_step_records(model, vecnorm, env, deterministic=True)
    joined = join_trades(records, list(env.trades))
    stats = gate_stats(joined, all_records=records)
    stats["fold"] = fold_dir.name
    stats["test_window"] = f"{meta['test_start']} -> {meta['test_end']}"
    stats["oos_sharpe_from_retrain"] = meta.get("test_metrics", {}).get("sharpe_ratio")
    return stats, joined


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default="data/models/htf_walkforward_sol")
    ap.add_argument("--data-path", default="data/historical/SOLUSDT_15m.csv")
    ap.add_argument("--out", default=None,
                    help="JSON output path (default <models-dir>/gate_diagnostic.json)")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    fold_dirs = sorted(models_dir.glob("fold_*"))
    if not fold_dirs:
        raise SystemExit(f"no fold_* dirs under {models_dir}")

    df_15m = load_15m_csv(args.data_path)
    _, df_1h, df_4h, df_1d = build_htf_dataframes(df_15m)

    per_fold, pooled = [], []
    for fd in fold_dirs:
        out = run_fold(fd, df_15m, df_1h, df_4h, df_1d)
        if out is None:
            print(f"  {fd.name}: skipped (missing result/model)")
            continue
        stats, joined = out
        per_fold.append(stats)
        pooled.extend(joined)
        print(f"  {fd.name} [{stats['test_window']}]  trades={stats['n_trades']}  "
              f"margin_AUC={stats.get('margin_auc')}  "
              f"maxprob_AUC={stats.get('max_prob_auc')}  "
              f"sat%={stats.get('saturated_step_pct', float('nan')):.0f}  "
              f"-> {stats.get('verdict')}")

    print("\n" + "=" * 72)
    pooled_stats = gate_stats(pooled)
    print(f"POOLED over {len(per_fold)} folds: {pooled_stats['n_trades']} trades")
    for k in ("win_rate", "margin_auc", "max_prob_auc", "margin_pnl_pearson",
              "margin_median", "margin_iqr"):
        print(f"  {k}: {pooled_stats.get(k)}")
    print("  pnl by margin quartile (Q1 lowest margin -> Q4 highest):")
    for q in pooled_stats.get("pnl_by_margin_quartile", []):
        print(f"    Q{q['q']}: n={q['n']}  avg_pnl={q['avg_pnl_pct']:+.3f}%  "
              f"WR={q['win_rate']:.0%}")
    print(f"  VERDICT: {pooled_stats.get('verdict')}")
    print("=" * 72)
    print("Next step: margin_auc >= 0.55 -> send this model over, we wire the "
          "gate on logit margin (no retrain).\n0.5-ish -> calibrated retrain: "
          "train_htf_walkforward.py --ent-floor 0.02 (see runbook).")

    out_path = Path(args.out) if args.out else models_dir / "gate_diagnostic.json"
    json.dump({"per_fold": per_fold, "pooled": pooled_stats},
              open(out_path, "w"), indent=2, default=float)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
