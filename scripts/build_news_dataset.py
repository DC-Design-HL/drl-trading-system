#!/usr/bin/env python3
"""
build_news_dataset.py — assemble the labeled news dataset for training.

Reads:  logs/news_pending_alerts.jsonl  (1,853+ Tier 2+ articles)
        data/kline_cache/<SYMBOL>/<tf>/<date>.parquet
Writes: data/training/news_labeled_v1.parquet

Plan ref: docs/research/news_training_pipeline.md
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NEWS_JSONL = REPO / "logs" / "news_pending_alerts.jsonl"
KLINE_DIR = REPO / "data" / "kline_cache"
OUT_DIR = REPO / "data" / "training"
OUT_PATH = OUT_DIR / "news_labeled_v1.parquet"

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ASSET_TICKER = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP"}
HORIZONS_MIN = [5, 15, 60, 240]
KLINE_TF = "5m"

EVENT_TYPES = ["regulatory", "macro", "geopolitical", "influencer", "exchange", "hack", "adoption", "other"]
TOP_SOURCES = []  # populated after first scan


def load_news() -> pd.DataFrame:
    rows = []
    with NEWS_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at", "timestamp"]).reset_index(drop=True)
    df = df.sort_values("published_at").reset_index(drop=True)
    return df


def load_klines_for_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load all 5m klines covering [start, end+5h] from cache."""
    pad_end = end + pd.Timedelta(hours=5)
    dates = pd.date_range(start.date(), pad_end.date(), freq="D")
    frames = []
    sym_dir = KLINE_DIR / symbol / KLINE_TF
    for d in dates:
        p = sym_dir / f"{d.strftime('%Y-%m-%d')}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, columns=["open_time", "close"]))
    if not frames:
        return pd.DataFrame(columns=["open_time", "close"])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def forward_returns_for_symbol(symbol: str, news_df: pd.DataFrame) -> dict[str, list[float]]:
    """For each news article time, compute forward returns at HORIZONS_MIN minutes."""
    if news_df.empty:
        return {f"r_{h}m_{ASSET_TICKER[symbol]}": [] for h in HORIZONS_MIN}
    start = news_df["published_at"].min()
    end = news_df["published_at"].max()
    klines = load_klines_for_range(symbol, start, end)
    if klines.empty:
        return {f"r_{h}m_{ASSET_TICKER[symbol]}": [float("nan")] * len(news_df) for h in HORIZONS_MIN}

    k_ts = klines["dt"].values.astype("datetime64[ns]")
    k_close = klines["close"].values

    out = {f"r_{h}m_{ASSET_TICKER[symbol]}": [] for h in HORIZONS_MIN}
    for ts in news_df["published_at"].values.astype("datetime64[ns]"):
        idx = np.searchsorted(k_ts, ts, side="right") - 1
        if idx < 0 or idx >= len(k_close):
            for h in HORIZONS_MIN:
                out[f"r_{h}m_{ASSET_TICKER[symbol]}"].append(float("nan"))
            continue
        p0 = k_close[idx]
        for h in HORIZONS_MIN:
            future_ts = ts + np.timedelta64(h, "m")
            idx_future = np.searchsorted(k_ts, future_ts, side="right") - 1
            if idx_future < 0 or idx_future >= len(k_close):
                out[f"r_{h}m_{ASSET_TICKER[symbol]}"].append(float("nan"))
                continue
            p1 = k_close[idx_future]
            out[f"r_{h}m_{ASSET_TICKER[symbol]}"].append((p1 - p0) / p0)
    return out


def engineer_features(news: pd.DataFrame) -> pd.DataFrame:
    """Build feature columns per the plan."""
    df = news.copy()

    # Sentiment / scoring block
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0).astype(float)
    df["confidence"] = df["confidence"].fillna(0.0).astype(float)
    df["urgency"] = df["urgency"].fillna(2).astype(int)
    df["is_groq"] = (df["scorer_method"] == "groq").astype(int)
    df["is_keyword"] = (df["scorer_method"] == "keyword").astype(int)

    # Event-type one-hot
    for et in EVENT_TYPES:
        df[f"et_{et}"] = (df["event_type"] == et).astype(int)

    # Source one-hot (top-20)
    src_counts = df["source"].value_counts()
    top_sources = src_counts.head(20).index.tolist()
    for s in top_sources:
        col = f"src_{s.replace(':', '_').replace('/', '_')}"
        df[col] = (df["source"] == s).astype(int)
    df["src_other"] = (~df["source"].isin(top_sources)).astype(int)

    # Asset multi-hot
    def has_asset(assets_list, target):
        if not isinstance(assets_list, list):
            return 0
        return int(target in assets_list)
    for t in ["BTC", "ETH", "SOL", "XRP", "ALL"]:
        df[f"asset_{t}"] = df["assets"].apply(lambda a: has_asset(a, t))

    # Temporal
    pub = df["published_at"].dt
    df["hour_sin"] = np.sin(2 * np.pi * pub.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * pub.hour / 24)
    for d in range(7):
        df[f"dow_{d}"] = (pub.dayofweek == d).astype(int)

    # Recency / news-flow density (per-row look-back over all articles)
    ts_arr = df["published_at"].values.astype("datetime64[ns]")
    df["minutes_since_prev_article"] = np.append(
        [9999.0], (np.diff(ts_arr).astype("timedelta64[s]").astype(int) / 60.0)
    )
    for win_min in (60, 240, 1440):
        col = f"n_articles_trailing_{win_min}m"
        counts = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            cutoff = ts_arr[i] - np.timedelta64(win_min, "m")
            counts[i] = np.searchsorted(ts_arr, ts_arr[i], side="right") - np.searchsorted(ts_arr, cutoff, side="left")
        df[col] = counts

    # Rolling-sentiment context (per the plan: trailing 60m / 240m)
    sent_arr = df["sentiment_score"].values
    for win_min in (60, 240):
        rolling_mean = np.zeros(len(df))
        for i in range(len(df)):
            cutoff = ts_arr[i] - np.timedelta64(win_min, "m")
            lo = np.searchsorted(ts_arr, cutoff, side="left")
            hi = np.searchsorted(ts_arr, ts_arr[i], side="right")
            window = sent_arr[lo:hi]
            rolling_mean[i] = float(window.mean()) if window.size else 0.0
        df[f"sent_mean_trailing_{win_min}m"] = rolling_mean

    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Primary target: y_long_fade = 1 if r_60m_BTC < -0.10% else 0.

    Why BTC: it has 582 article mentions, the broadest coverage. ETH/SOL/XRP get
    weaker per-asset targets which we leave as columns too for later analysis.
    """
    threshold = -0.0010  # -0.10%
    df["y_long_fade"] = (df["r_60m_BTC"] < threshold).astype(float)
    df["y_long_fade"] = df["y_long_fade"].mask(df["r_60m_BTC"].isna())

    # Regression target for calibration analysis
    df["y_r_60m_btc"] = df["r_60m_BTC"]
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading news from {NEWS_JSONL}", flush=True)
    news = load_news()
    print(f"      {len(news)} articles, {news['published_at'].min()} -> {news['published_at'].max()}", flush=True)

    print(f"[2/4] computing forward returns for {len(ASSETS)} symbols at {HORIZONS_MIN} min", flush=True)
    feature_df = news.copy()
    for sym in ASSETS:
        rets = forward_returns_for_symbol(sym, news)
        for k, v in rets.items():
            feature_df[k] = v
        kept = sum(1 for x in rets[f"r_60m_{ASSET_TICKER[sym]}"] if not (isinstance(x, float) and np.isnan(x)))
        print(f"      {sym}: {kept}/{len(news)} articles got 60m return", flush=True)

    print("[3/4] engineering features", flush=True)
    feature_df = engineer_features(feature_df)

    print("[4/4] building labels + writing parquet", flush=True)
    feature_df = make_labels(feature_df)

    # Drop the human-readable columns from features but keep them for traceability
    feature_df["title"] = news["title"]
    feature_df["reasoning"] = news.get("reasoning", "")
    feature_df["event_id"] = news.get("event_id", -1)

    feature_df.to_parquet(OUT_PATH, index=False)
    print(f"\n✅ wrote {OUT_PATH}", flush=True)
    print(f"   rows: {len(feature_df)}", flush=True)
    print(f"   cols: {len(feature_df.columns)}", flush=True)

    # Summary stats for sanity check
    valid_label = feature_df["y_long_fade"].notna().sum()
    pos_rate = float(feature_df["y_long_fade"].sum()) / max(valid_label, 1)
    print(f"   y_long_fade valid: {valid_label}/{len(feature_df)}", flush=True)
    print(f"   y_long_fade positive rate: {pos_rate:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
