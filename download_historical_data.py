#!/usr/bin/env python3
"""
Download 3 years of 1h historical OHLCV data from Binance public API.

Usage:
    python download_historical_data.py [--years 3] [--assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT]

Saves CSV files to data/historical/<SYMBOL>_1h_<start>_<end>.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = os.environ.get(
    "BINANCE_API_URL", "https://data-api.binance.vision/api/v3/klines"
)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

DEFAULT_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def fetch_klines_batch(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    for attempt in range(max_retries):
        try:
            resp = session.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(f"Request failed ({attempt + 1}/{max_retries}): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {symbol} after {max_retries} attempts")


def fetch_full_history(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    session = requests.Session()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    batch_num = 0

    logger.info(f"  Fetching {symbol} {interval} from {start_dt.date()} to {end_dt.date()}")

    while current_start < end_ms:
        batch = fetch_klines_batch(session, symbol, interval, current_start, end_ms)
        if not batch:
            break

        all_klines.extend(batch)
        batch_num += 1
        # Next batch starts after the last candle's close_time
        current_start = int(batch[-1][6]) + 1

        if batch_num % 10 == 0:
            n = len(all_klines)
            pct = min(100, (current_start - start_ms) / (end_ms - start_ms) * 100)
            logger.info(f"    ... {n:,} candles ({pct:.1f}%)")

        # Polite rate limiting: Binance allows 1200 reqs/min on public endpoints
        time.sleep(0.08)

    if not all_klines:
        logger.error(f"  No data returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    logger.info(f"  Done: {len(df):,} candles")
    return df


def validate_data(df: pd.DataFrame, symbol: str, interval: str) -> dict:
    """Check for gaps, NaNs, zero prices, and report quality stats."""
    issues = []
    stats = {}

    if df.empty:
        return {"ok": False, "issues": ["Empty dataframe"], "stats": {}}

    # Expected candle gap in minutes
    interval_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(interval, 60)
    expected_gap = pd.Timedelta(minutes=interval_minutes)

    # Timestamp diffs
    diffs = df["timestamp"].diff().dropna()
    gap_threshold = expected_gap * 1.5  # Allow up to 1.5x before flagging
    large_gaps = diffs[diffs > gap_threshold]
    if len(large_gaps) > 0:
        issues.append(f"{len(large_gaps)} gaps > {gap_threshold} (largest: {large_gaps.max()})")

    # Expected rows
    total_hours = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
    expected_rows = int(total_hours / interval_minutes * 60)
    actual_rows = len(df)
    completeness = actual_rows / max(expected_rows, 1) * 100
    if completeness < 95:
        issues.append(f"Only {completeness:.1f}% complete ({actual_rows} / ~{expected_rows} expected rows)")

    # NaNs
    nan_counts = df[["open", "high", "low", "close", "volume"]].isna().sum()
    if nan_counts.any():
        issues.append(f"NaN values: {nan_counts[nan_counts > 0].to_dict()}")

    # Zero or negative prices
    for col in ["open", "high", "low", "close"]:
        zeros = (df[col] <= 0).sum()
        if zeros > 0:
            issues.append(f"{zeros} zero/negative values in {col}")

    # High/Low integrity
    hl_issues = (df["high"] < df["low"]).sum()
    if hl_issues > 0:
        issues.append(f"{hl_issues} rows where high < low")

    stats = {
        "rows": actual_rows,
        "start": str(df["timestamp"].iloc[0].date()),
        "end": str(df["timestamp"].iloc[-1].date()),
        "completeness_pct": round(completeness, 1),
        "gap_count": len(large_gaps),
        "price_range": f"${df['close'].min():,.0f} – ${df['close'].max():,.0f}",
    }

    return {"ok": len(issues) == 0, "issues": issues, "stats": stats}


def download_asset(
    symbol: str,
    years: int,
    output_dir: Path,
    interval: str = "1h",
) -> Path:
    end_dt = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt.replace(year=end_dt.year - years)

    csv_path = output_dir / f"{symbol}_{interval}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logger.info(f"  Cache hit: {csv_path.name} ({len(existing):,} rows). Skipping download.")
        return csv_path

    df = fetch_full_history(symbol, interval, start_dt, end_dt)

    if df.empty:
        logger.error(f"  Skipping {symbol} — no data returned")
        return None

    result = validate_data(df, symbol, interval)
    logger.info(f"  Quality check for {symbol}: {'PASS' if result['ok'] else 'WARN'}")
    for k, v in result["stats"].items():
        logger.info(f"    {k}: {v}")
    if result["issues"]:
        for issue in result["issues"]:
            logger.warning(f"    [!] {issue}")

    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path} ({len(df):,} rows)")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data from Binance")
    parser.add_argument("--years", type=int, default=3, help="Years of history to download (default: 3)")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS,
                        help="Asset symbols to download")
    parser.add_argument("--interval", type=str, default="1h",
                        help="Candle interval (default: 1h)")
    parser.add_argument("--output-dir", type=str, default="data/historical",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.resolve()}")

    results = {}
    for symbol in args.assets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Downloading {symbol} ({args.years}y, {args.interval})...")
        logger.info(f"{'='*50}")
        try:
            path = download_asset(symbol, args.years, output_dir, args.interval)
            results[symbol] = {"status": "ok", "path": str(path)}
        except Exception as exc:
            logger.error(f"  FAILED for {symbol}: {exc}")
            results[symbol] = {"status": "error", "error": str(exc)}

    logger.info("\n" + "="*50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*50)
    for symbol, res in results.items():
        status = "OK" if res["status"] == "ok" else "FAILED"
        detail = res.get("path", res.get("error", ""))
        logger.info(f"  {symbol}: {status} — {detail}")

    # Final validation pass — re-read all CSVs and confirm
    logger.info("\nFINAL VALIDATION:")
    all_ok = True
    for symbol in args.assets:
        if results.get(symbol, {}).get("status") != "ok":
            continue
        path = results[symbol]["path"]
        df = pd.read_csv(path, parse_dates=["timestamp"])
        result = validate_data(df, symbol, args.interval)
        status_str = "PASS" if result["ok"] else "WARN"
        logger.info(f"  {symbol}: {status_str} | {result['stats']}")
        if not result["ok"]:
            all_ok = False

    if all_ok:
        logger.info("\nAll assets downloaded and validated successfully.")
    else:
        logger.warning("\nSome assets have data quality warnings. Review before training.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
