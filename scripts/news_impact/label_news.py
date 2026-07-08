#!/usr/bin/env python3
"""
News-impact labeler + signal check (v0).

Joins each news_events row to the forward price return of the asset it mentions
(ALL -> BTC as market proxy), at +1h and +4h horizons, and writes a labeled
tabular dataset for the Mac trainer. Also prints an honest signal read
(base rates, sentiment->direction correlation, single-feature AUCs) WITHOUT
fitting a model — so it's safe to run on the server (data processing + stats,
not training).

Usage:  python3 scripts/news_impact/label_news.py
Output: data/news_impact/news_labeled.csv
"""
import sqlite3, json, urllib.request, time, os, math
from datetime import datetime, timezone
from collections import defaultdict

DB = "data/trading.db"
OUT_DIR = "data/news_impact"
OUT = os.path.join(OUT_DIR, "news_labeled.csv")
PROXY = "BTCUSDT"
ASSET_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
             "ALL": PROXY}
HORIZONS = {"ret_1h": 3600, "ret_4h": 4 * 3600}
DEADBAND = 0.001  # 0.1% -> below this magnitude is "flat" (excluded from binary dir)

def parse_ts(s):
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)

def primary_asset(assets_field):
    if not assets_field:
        return None
    try:
        v = json.loads(assets_field) if assets_field.strip().startswith("[") else \
            [x.strip() for x in assets_field.split(",")]
    except Exception:
        v = [assets_field]
    for a in v:
        a = str(a).upper().strip().strip('"')
        if a in ASSET_MAP:
            return a
    return None

def fetch_klines(sym, start_ms, end_ms):
    out = []
    cur = start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={sym}"
               f"&interval=5m&startTime={cur}&limit=1000")
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        if not data:
            break
        for k in data:
            out.append((k[0], float(k[4])))   # (openTime_ms, close)
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.15)
    return out

def price_at(bars, ts_ms):
    """close of the first 5m bar whose openTime >= ts_ms (nearest forward bar)."""
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid][0] < ts_ms:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(bars):
        return None
    return bars[lo][1]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    c = sqlite3.connect(DB); cur = c.cursor()
    rows = cur.execute(
        "SELECT published_at, source, event_type, urgency, sentiment_score, "
        "confidence, assets FROM news_events WHERE sentiment_score IS NOT NULL "
        "ORDER BY published_at ASC").fetchall()
    c.close()

    # figure out needed time span per symbol
    need = defaultdict(lambda: [math.inf, -math.inf])
    parsed = []
    for pub, src, et, urg, sent, conf, assets in rows:
        pa = primary_asset(assets)
        if pa is None:
            continue
        sym = ASSET_MAP[pa]
        ts_ms = int(parse_ts(pub).timestamp() * 1000)
        parsed.append((ts_ms, sym, src, et, urg, sent, conf, pa))
        need[sym][0] = min(need[sym][0], ts_ms - 3600_000)
        need[sym][1] = max(need[sym][1], ts_ms + max(HORIZONS.values()) * 1000 + 3600_000)

    klines = {}
    for sym, (a, b) in need.items():
        klines[sym] = fetch_klines(sym, int(a), int(b))
        print(f"fetched {len(klines[sym])} 5m bars for {sym}")

    labeled = []
    for ts_ms, sym, src, et, urg, sent, conf, pa in parsed:
        bars = klines[sym]
        p0 = price_at(bars, ts_ms)
        if not p0:
            continue
        row = dict(ts_ms=ts_ms, symbol=sym, asset=pa, source=src, event_type=et,
                   urgency=urg, sentiment=sent, confidence=conf,
                   abs_sentiment=abs(sent),
                   hour=datetime.fromtimestamp(ts_ms/1000, timezone.utc).hour)
        ok = True
        for name, secs in HORIZONS.items():
            p1 = price_at(bars, ts_ms + secs * 1000)
            if not p1:
                ok = False; break
            row[name] = p1 / p0 - 1
        if ok:
            labeled.append(row)

    # write CSV
    cols = ["ts_ms","symbol","asset","source","event_type","urgency","sentiment",
            "abs_sentiment","confidence","hour"] + list(HORIZONS.keys())
    with open(OUT, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in labeled:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {len(labeled)} labeled rows -> {OUT}")

    # ── signal check (stats only, no model.fit) ──
    def auc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return float("nan")
        wins = sum(1 for p in pos for n in neg if p > n) + \
               0.5 * sum(1 for p in pos for n in neg if p == n)
        return wins / (len(pos) * len(neg))

    print("\n" + "=" * 56)
    print("SIGNAL CHECK (feasibility — is there any news->price edge?)")
    print("=" * 56)
    for h in HORIZONS:
        rets = [r[h] for r in labeled]
        dirs = [1 if r[h] > DEADBAND else (0 if r[h] < -DEADBAND else None) for r in labeled]
        pairs = [(r, d) for r, d in zip(labeled, dirs) if d is not None]
        n = len(pairs)
        up = sum(1 for _, d in pairs if d == 1)
        print(f"\n[{h}] n={n} (|move|>{DEADBAND*100:.1f}%)  base rate UP={up/n*100:.1f}%")
        sent_scores = [r["sentiment"] for r, d in pairs]
        labs = [d for _, d in pairs]
        a = auc(sent_scores, labs)
        print(f"  sentiment -> UP  AUC={a:.3f}  (0.5=no signal)")
        # mean forward return by sentiment sign
        posn = [r[h] for r in labeled if r["sentiment"] > 0.1]
        negn = [r[h] for r in labeled if r["sentiment"] < -0.1]
        if posn and negn:
            print(f"  mean {h}: bullish-news={sum(posn)/len(posn)*100:+.3f}%  "
                  f"bearish-news={sum(negn)/len(negn)*100:+.3f}%  "
                  f"(n+={len(posn)}, n-={len(negn)})")
        # by event_type mean return
        byet = defaultdict(list)
        for r in labeled:
            byet[r["event_type"]].append(r[h])
        rank = sorted(((sum(v)/len(v), k, len(v)) for k, v in byet.items()), reverse=True)
        print(f"  event_type mean {h}: " +
              " ".join(f"{k}={m*100:+.2f}%(n{n_})" for m, k, n_ in rank))

if __name__ == "__main__":
    main()
