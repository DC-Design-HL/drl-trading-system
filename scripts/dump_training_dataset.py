#!/usr/bin/env python3
"""
Dump training dataset for the Mac M3 retraining projects.

Runs on the SERVER (where all data lives). Produces a single
self-contained parquet bundle that Chen ships to Mac for training.

Inputs (all real, no synthetic):
  * data/trading.db
      - trades table  (588 paired OPEN/CLOSE with realized pnl)
      - news_events   (533 news items with sentiment + asset tags + urgency)
  * logs/htf_pending_alerts.jsonl
      - per-trade signal context: MTF, regime, order flow, orderbook,
        funding, whale-direction, confidence
  * data/whale_behavior/labeled_v2/*.jsonl
      - real on-chain wallet activity with behavioral labels
        (net_flow_4h, intent_score_4h, etc.)

Output (parquet bundle):
  data/training/sgfilter_dataset.parquet  — one row per OPEN signal
  data/training/sgfilter_metadata.json    — schema + feature list

Each row contains:
  * Identifiers:    open_ts, symbol, side, open_id
  * Bot decision:   confidence (model output), entry_price, units, sl, tp
  * Outcome:        pnl, pnl_pct, hold_hours, win (label, 0/1), reason
  * Signal context: 30+ features (mtf_*, regime_*, of_*, ob_*, funding_*,
                                  whale_*)
  * News context:   news_recent_24h_count, news_recent_24h_sentiment_avg,
                    news_recent_4h_*, news_recent_1h_*, news_max_urgency
  * Whale context:  whale_eth_net_flow_4h, whale_eth_intent_4h, etc.
                    (aggregated across the labeled wallets)

Run:
    python3 scripts/dump_training_dataset.py
        [--start 2026-04-06]   # default: full history
        [--end 2026-04-29]
        [--symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT]
        [--output data/training/sgfilter_dataset.parquet]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "trading.db"
ALERTS_PATH = REPO / "logs" / "htf_pending_alerts.jsonl"
# Use raw transaction logs from data/whale_behavior/eth/ (10 wallets, current
# through Apr 2026) instead of labeled_v2/ which only had 3 wallets and ends
# 2026-04-03 — BEFORE any bot trades. The May 2 filter analysis already uses
# eth/ — this aligns the training data source with the live signal.
WHALE_DIR = REPO / "data" / "whale_behavior" / "eth"

EXCHANGE_WALLETS = {
    "binance_hot_wallet", "binance_cold_wallet", "binance_cold_2",
    "binance_reserve", "coinbase_institutional", "kraken_deposit",
}


def parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    if "+" not in s and "-" not in s[-6:]:
        s += "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Trades + signal join
# ---------------------------------------------------------------------------

def load_trade_outcomes(start: datetime, end: datetime, symbols: list[str]) -> list[dict]:
    """Pair OPEN/CLOSE rows from data/trading.db; return one dict per closed trade."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    sym_in = ",".join(f"'{s}'" for s in symbols)
    cur.execute(
        f"""SELECT id, timestamp, symbol, action, price, pnl, reason, data, confidence
            FROM trades WHERE is_testnet=1 AND symbol IN ({sym_in})
              AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp""",
        (start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    conn.close()

    open_pos: dict[str, dict] = {}
    out: list[dict] = []
    for tid, ts, sym, action, price, pnl, reason, data, conf in rows:
        d = json.loads(data) if data else {}
        if "OPEN" in action and "PARTIAL" not in action:
            open_pos[sym] = {
                "open_id": tid,
                "open_ts": ts[:19],
                "symbol": sym,
                "side": "LONG" if "LONG" in action else "SHORT",
                "entry": price,
                "units": d.get("units"),
                "sl": d.get("sl"),
                "tp": d.get("tp"),
                "model_confidence_at_open": conf or 0.0,
            }
        elif "CLOSE" in action and "PARTIAL" not in action and sym in open_pos:
            o = open_pos.pop(sym)
            try:
                hold_h = (parse_iso(ts) - parse_iso(o["open_ts"])).total_seconds() / 3600
            except Exception:
                hold_h = None
            entry = o["entry"] or 0
            pnl_pct = ((price / entry) - 1) * 100 * (1 if o["side"] == "LONG" else -1) if entry > 0 else None
            out.append({
                **o,
                "close_ts": ts[:19],
                "exit_price": price,
                "pnl": pnl or 0.0,
                "pnl_pct": pnl_pct,
                "hold_hours": hold_h,
                "close_reason": reason or "",
                "win": 1 if (pnl or 0) > 0 else 0,
            })
    return out


def load_signal_context() -> dict[tuple, dict]:
    """Index alerts by (ts, symbol, side) for join."""
    by_key: dict[tuple, dict] = {}
    if not ALERTS_PATH.exists():
        return by_key
    with open(ALERTS_PATH) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("trade", {})
            action = t.get("action", "")
            if not action.startswith("OPEN") or "PARTIAL" in action:
                continue
            sigs = d.get("signals", {})
            mtf = sigs.get("mtf") or {}
            of = sigs.get("order_flow") or {}
            regime = sigs.get("regime") or {}
            ob = sigs.get("orderbook") or {}
            funding = sigs.get("funding") or {}
            whale = sigs.get("whale") or {}
            mtf_signals = mtf.get("signals") or {}
            mtf_15m = mtf_signals.get("15m", {}) or {}
            mtf_1h = mtf_signals.get("1h", {}) or {}
            mtf_4h = mtf_signals.get("4h", {}) or {}
            key = (d["timestamp"][:19], t.get("symbol"), "LONG" if "LONG" in action else "SHORT")
            by_key[key] = {
                # MTF
                "sig_mtf_bias": (mtf.get("bias") or "NEUTRAL").upper(),
                "sig_mtf_aligned": bool(mtf.get("aligned")),
                "sig_mtf_strength": float(mtf.get("strength") or 0),
                "sig_mtf_15m_dir": (mtf_15m.get("direction") or "neutral"),
                "sig_mtf_15m_rsi": float(mtf_15m.get("rsi") or 50),
                "sig_mtf_15m_strength": float(mtf_15m.get("strength") or 0),
                "sig_mtf_1h_dir": (mtf_1h.get("direction") or "neutral"),
                "sig_mtf_1h_rsi": float(mtf_1h.get("rsi") or 50),
                "sig_mtf_1h_strength": float(mtf_1h.get("strength") or 0),
                "sig_mtf_4h_dir": (mtf_4h.get("direction") or "neutral"),
                "sig_mtf_4h_rsi": float(mtf_4h.get("rsi") or 50),
                "sig_mtf_4h_strength": float(mtf_4h.get("strength") or 0),
                # Order flow
                "sig_of_bias": (of.get("bias") or "neutral").lower(),
                "sig_of_score": float(of.get("score") or 0),
                "sig_of_large_buys": int(of.get("large_buys") or 0),
                "sig_of_large_sells": int(of.get("large_sells") or 0),
                # Regime
                "sig_regime_type": regime.get("type") or "UNKNOWN",
                "sig_regime_state": regime.get("state") or "UNKNOWN",
                "sig_regime_adx": float(regime.get("adx") or 0),
                # Orderbook
                "sig_ob_bias": (ob.get("bias") or "neutral").lower(),
                "sig_ob_imbalance_10": float(ob.get("imbalance_10") or 0),
                # Funding
                "sig_funding_rate": float(funding.get("rate") or 0),
                # Whale (the per-iteration signal at entry)
                "sig_whale_dir": (whale.get("direction") or "NEUTRAL").upper(),
                "sig_whale_score": float(whale.get("score") or 0),
                "sig_whale_confidence": float(whale.get("confidence") or 0),
            }
    return by_key


# ---------------------------------------------------------------------------
# News features
# ---------------------------------------------------------------------------

def load_news_events() -> list[tuple[datetime, str, float, int, str, list[str]]]:
    """Return all news events as (ts, source, sentiment, urgency, event_type, asset_tags)."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """SELECT published_at, source, sentiment_score, urgency, event_type, assets
           FROM news_events WHERE sentiment_score IS NOT NULL
           ORDER BY published_at"""
    )
    out = []
    for ts, source, score, urgency, event_type, assets in cur.fetchall():
        try:
            dt = parse_iso(ts)
        except Exception:
            continue
        try:
            tags = json.loads(assets) if assets else []
        except Exception:
            tags = []
        out.append((dt, source or "", float(score or 0), int(urgency or 1),
                    event_type or "unknown", tags or []))
    conn.close()
    return out


def news_features_at(open_ts: datetime, symbol: str,
                     news_events: list, windows_h: tuple = (1, 4, 24)) -> dict:
    """Aggregate news within the last N hours of an OPEN event.
    Filters to news that mentions the symbol's asset (BTC/ETH/SOL/XRP) OR is unfiltered global.
    """
    asset_match = symbol.replace("USDT", "").upper()  # BTC/ETH/SOL/XRP
    out: dict = {}
    for hours in windows_h:
        window_start = open_ts - timedelta(hours=hours)
        relevant = [
            (ts, src, sent, urg, et, tags)
            for ts, src, sent, urg, et, tags in news_events
            if window_start <= ts <= open_ts
            and (
                not tags  # global news with no asset tag
                or any(asset_match in (tag or "").upper() for tag in tags)
            )
        ]
        n = len(relevant)
        sentiments = [t[2] for t in relevant]
        urgencies = [t[3] for t in relevant]
        out[f"news_{hours}h_count"] = n
        out[f"news_{hours}h_sentiment_avg"] = sum(sentiments) / n if n else 0.0
        out[f"news_{hours}h_sentiment_min"] = min(sentiments) if n else 0.0
        out[f"news_{hours}h_max_urgency"] = max(urgencies) if n else 0
    return out


# ---------------------------------------------------------------------------
# Whale features (real on-chain ETH activity, used for all symbols as macro signal)
# ---------------------------------------------------------------------------

def load_whale_events() -> list[dict]:
    """Load raw whale txs from data/whale_behavior/eth/ (10 wallets,
    825K events, current through Apr 2026). Each tx has direction (in/out)
    and value_eth. The aggregation in whale_features_at() computes net
    exchange flow over trailing windows — same convention as the May 2
    filter analysis (scripts/backtest_whale_flow_news_filters.py).
    """
    if not WHALE_DIR.exists():
        return []
    out = []
    MIN_FLOW_ETH = 100.0  # filter dust txs to keep aggregations meaningful
    for f in sorted(WHALE_DIR.glob("*.jsonl")):
        wallet_name = f.stem
        is_exchange = wallet_name in EXCHANGE_WALLETS
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts_raw = d.get("timestamp")
                if ts_raw is None:
                    continue
                try:
                    dt = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                except Exception:
                    continue
                value = float(d.get("value_eth") or 0)
                if value < MIN_FLOW_ETH:
                    continue
                out.append({
                    "ts": dt, "wallet": wallet_name, "is_exchange": is_exchange,
                    "value_eth": value,
                    "direction": (d.get("direction") or "").lower(),
                })
    out.sort(key=lambda x: x["ts"])
    return out


def whale_features_at(open_ts: datetime, whale_events: list[dict]) -> dict:
    """Aggregate exchange-wallet net flow + per-direction counts in trailing windows.

    Net flow IN = distribution (whales depositing → about to sell) → bearish.
    Net flow OUT = accumulation (whales withdrawing → holding) → bullish.
    Same convention as scripts/backtest_whale_flow_news_filters.py.
    """
    out: dict = {}
    for window_h in (4, 12, 24):
        window_start = open_ts - timedelta(hours=window_h)
        relevant = [w for w in whale_events if window_start <= w["ts"] <= open_ts]
        ex_in = sum(w["value_eth"] for w in relevant if w["is_exchange"] and w["direction"] == "in")
        ex_out = sum(w["value_eth"] for w in relevant if w["is_exchange"] and w["direction"] == "out")
        out[f"whale_{window_h}h_exchange_in_eth"] = ex_in
        out[f"whale_{window_h}h_exchange_out_eth"] = ex_out
        out[f"whale_{window_h}h_exchange_net_in_eth"] = ex_in - ex_out
        out[f"whale_{window_h}h_n_events"] = len(relevant)
        out[f"whale_{window_h}h_n_exchange_events"] = sum(1 for w in relevant if w["is_exchange"])
        # Regime label encoding: 1 = ACCUMULATION (net out), -1 = DISTRIBUTION (net in), 0 = balanced
        if ex_in > ex_out:
            regime_code = -1
        elif ex_out > ex_in:
            regime_code = 1
        else:
            regime_code = 0
        out[f"whale_{window_h}h_regime_code"] = regime_code
    return out


# ---------------------------------------------------------------------------
# Time-of-day features
# ---------------------------------------------------------------------------

def time_features(open_ts: datetime) -> dict:
    return {
        "hour_utc": open_ts.hour,
        "day_of_week": open_ts.weekday(),
        "is_weekend": int(open_ts.weekday() >= 5),
        "session_asia": int(0 <= open_ts.hour < 8),
        "session_london": int(8 <= open_ts.hour < 16),
        "session_ny": int(13 <= open_ts.hour < 21),
        "session_off": int(open_ts.hour >= 21 or open_ts.hour < 8),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-04-06", help="ISO start (inclusive)")
    p.add_argument("--end",   default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT")
    p.add_argument("--output", default="data/training/sgfilter_dataset.parquet")
    args = p.parse_args(argv)

    start = parse_iso(args.start + "T00:00:00")
    end = parse_iso(args.end + "T00:00:00") + timedelta(days=1)
    symbols = args.symbols.split(",")
    out_path = REPO / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Window: {start.date()} → {end.date()} | Symbols: {symbols}")

    print("Loading trade outcomes from DB ...")
    trades = load_trade_outcomes(start, end, symbols)
    print(f"  {len(trades)} closed round-trips")

    print("Loading signal context from pending_alerts ...")
    sig_index = load_signal_context()
    print(f"  {len(sig_index)} indexed OPEN events")

    print("Loading news_events ...")
    news_events = load_news_events()
    print(f"  {len(news_events)} news with sentiment")
    if news_events:
        print(f"  range: {news_events[0][0].date()} → {news_events[-1][0].date()}")

    print("Loading whale labeled_v2 ...")
    whale_events = load_whale_events()
    print(f"  {len(whale_events)} whale events from {len({w['wallet'] for w in whale_events})} wallets")
    if whale_events:
        print(f"  range: {whale_events[0]['ts'].date()} → {whale_events[-1]['ts'].date()}")

    print("Joining ...")
    rows: list[dict] = []
    skipped_no_signal = 0
    for t in trades:
        try:
            open_dt = parse_iso(t["open_ts"])
        except Exception:
            continue
        # Find matching alert (fuzzy ±60s)
        key = (t["open_ts"], t["symbol"], t["side"])
        sig = sig_index.get(key)
        if sig is None:
            cands = [
                v for k, v in sig_index.items()
                if k[1] == t["symbol"] and k[2] == t["side"]
                and abs((parse_iso(k[0]) - open_dt).total_seconds()) < 60
            ]
            sig = cands[0] if cands else None
        if sig is None:
            skipped_no_signal += 1
            continue
        row = {**t, **sig,
               **news_features_at(open_dt, t["symbol"], news_events),
               **whale_features_at(open_dt, whale_events),
               **time_features(open_dt)}
        rows.append(row)
    print(f"  joined: {len(rows)}, skipped (no signal): {skipped_no_signal}")

    if not rows:
        print("No rows produced — aborting.", file=sys.stderr)
        return 1

    # Write parquet (preferred) or CSV fallback
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False)
        print(f"Wrote parquet: {out_path}  ({len(df)} rows × {len(df.columns)} cols)")
    except ImportError:
        import csv
        out_csv = out_path.with_suffix(".csv")
        cols = list({c for r in rows for c in r.keys()})
        with open(out_csv, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"pandas not available — wrote CSV: {out_csv}")
        out_path = out_csv

    # Metadata
    meta_path = out_path.with_name(out_path.stem.replace("_dataset", "_metadata") + ".json")
    feature_cols = [c for c in rows[0].keys() if c.startswith(("sig_", "news_", "whale_", "hour_", "day_", "is_", "session_"))]
    win_rate = sum(r["win"] for r in rows) / len(rows)
    pnl_total = sum(r["pnl"] for r in rows)
    by_sym = {}
    for r in rows:
        k = f"{r['symbol']}_{r['side']}"
        d = by_sym.setdefault(k, {"n": 0, "wins": 0, "pnl": 0.0})
        d["n"] += 1
        d["wins"] += r["win"]
        d["pnl"] += r["pnl"]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols": symbols,
        "n_rows": len(rows),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "label_col": "win",
        "regression_label_col": "pnl",
        "label_balance": {"win=1": int(win_rate * len(rows)), "win=0": int((1 - win_rate) * len(rows))},
        "baseline_win_rate": round(win_rate, 4),
        "baseline_pnl_total": round(pnl_total, 2),
        "per_symbol_side": {k: {**v, "wr": round(v["wins"] / v["n"], 3)} for k, v in by_sym.items()},
        "data_sources": {
            "trades_db": str(DB_PATH),
            "alerts_jsonl": str(ALERTS_PATH),
            "whale_dir": str(WHALE_DIR),
            "news_table": "trading.db.news_events",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote metadata: {meta_path}")
    print(f"\nBaseline WR: {win_rate * 100:.1f}%, total pnl ${pnl_total:+.2f}")
    print(f"Features: {len(feature_cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
