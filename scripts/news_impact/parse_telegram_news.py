#!/usr/bin/env python3
"""
Parse a Telegram export of the news-alert channel into a labeled dataset.

Recovers ~3 months of history that the DB purged (the sentinel only kept 7d).
Each alert message looks like:

    📰 [NEWS ALERT T2]
    GEOPOLITICAL | urgency: 2/3
    Sentiment: +1.00 | conf: 0.30
    Assets: BTC
    Source: decrypt
    "title..."

Produces data/news_impact/news_labeled_full.csv with the SAME schema as
label_news.py PLUS market-excess returns (asset return minus BTC return over
the same window) to strip market drift — the key modeling fix.

Usage: python3 scripts/news_impact/parse_telegram_news.py <export.json>
Note: this is data processing (parsing + price join), not model training —
safe to run on the server. Model training stays on the Mac.
"""
import sys, os, re, json, urllib.request, time, math
from datetime import datetime, timezone
from collections import defaultdict

OUT_DIR = "data/news_impact"
OUT = os.path.join(OUT_DIR, "news_labeled_full.csv")
ASSET_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
             "ALL": "BTCUSDT"}
HORIZONS = {"1h": 3600, "4h": 4 * 3600}
DEADBAND = 0.001

RE_URG = re.compile(r"urgency:\s*(\d)\s*/\s*3", re.I)
RE_SENT = re.compile(r"Sentiment:\s*([+-]?\d+\.?\d*)", re.I)
RE_CONF = re.compile(r"conf:\s*(\d+\.?\d*)", re.I)
RE_ASSETS = re.compile(r"Assets:\s*([A-Za-z0-9, ]+)", re.I)
RE_SRC = re.compile(r"Source:\s*([A-Za-z0-9_:\-]+)", re.I)
RE_ETYPE = re.compile(r"^([A-Z]+)\s*\|\s*urgency", re.M)

def flat(t):
    if isinstance(t, str):
        return t
    return "".join(e["text"] if isinstance(e, dict) else e for e in t)

def parse_msg(text, date_str):
    et = RE_ETYPE.search(text)
    urg = RE_URG.search(text)
    sent = RE_SENT.search(text)
    conf = RE_CONF.search(text)
    assets = RE_ASSETS.search(text)
    src = RE_SRC.search(text)
    if not (urg and sent and assets):
        return None
    # primary asset
    pa = None
    for a in assets.group(1).split(","):
        a = a.strip().upper()
        if a in ASSET_MAP:
            pa = a; break
    if pa is None:
        return None
    ts = datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return dict(
        ts_ms=int(ts.timestamp() * 1000),
        symbol=ASSET_MAP[pa], asset=pa,
        source=(src.group(1).lower() if src else "unknown"),
        event_type=(et.group(1).lower() if et else "other"),
        urgency=int(urg.group(1)),
        sentiment=float(sent.group(1)),
        confidence=float(conf.group(1)) if conf else 0.0,
    )

def fetch_klines(sym, start_ms, end_ms):
    out, cur = [], start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol={sym}"
               f"&interval=5m&startTime={cur}&limit=1000")
        with urllib.request.urlopen(url, timeout=25) as r:
            data = json.load(r)
        if not data:
            break
        out += [(k[0], float(k[4])) for k in data]
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.12)
    return out

def price_at(bars, ts_ms):
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid][0] < ts_ms:
            lo = mid + 1
        else:
            hi = mid
    return bars[lo][1] if lo < len(bars) else None

def main():
    if len(sys.argv) < 2:
        print("usage: parse_telegram_news.py <export.json>"); sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    d = json.load(open(sys.argv[1]))
    msgs = d.get("messages", [])

    parsed, seen = [], set()
    for m in msgs:
        if m.get("type") != "message":
            continue
        text = flat(m.get("text", ""))
        if "urgency:" not in text.lower():
            continue
        rec = parse_msg(text, m.get("date", ""))
        if not rec:
            continue
        # dedup exact same title+asset appearing repeatedly (re-alerts)
        title = text.split('"')[1] if '"' in text else text[:40]
        key = (title.strip().lower(), rec["asset"])
        if key in seen:
            continue
        seen.add(key)
        parsed.append(rec)
    print(f"parsed {len(parsed)} unique alerts from {len(msgs)} messages")

    # price span per symbol (+BTC always, for excess)
    need = defaultdict(lambda: [math.inf, -math.inf])
    for r in parsed:
        for sym in (r["symbol"], "BTCUSDT"):
            need[sym][0] = min(need[sym][0], r["ts_ms"] - 3600_000)
            need[sym][1] = max(need[sym][1], r["ts_ms"] + max(HORIZONS.values()) * 1000 + 3600_000)
    klines = {}
    for sym, (a, b) in need.items():
        klines[sym] = fetch_klines(sym, int(a), int(b))
        print(f"  {sym}: {len(klines[sym])} bars")

    rows = []
    for r in parsed:
        bars = klines[r["symbol"]]; btc = klines["BTCUSDT"]
        p0 = price_at(bars, r["ts_ms"]); b0 = price_at(btc, r["ts_ms"])
        if not p0 or not b0:
            continue
        r = dict(r, abs_sentiment=abs(r["sentiment"]),
                 hour=datetime.fromtimestamp(r["ts_ms"]/1000, timezone.utc).hour)
        ok = True
        for name, secs in HORIZONS.items():
            p1 = price_at(bars, r["ts_ms"] + secs*1000)
            b1 = price_at(btc, r["ts_ms"] + secs*1000)
            if not p1 or not b1:
                ok = False; break
            r[f"ret_{name}"] = p1/p0 - 1
            r[f"exc_{name}"] = (p1/p0 - 1) - (b1/b0 - 1)   # market-excess (0 for BTC)
        if ok:
            rows.append(r)

    cols = ["ts_ms","symbol","asset","source","event_type","urgency","sentiment",
            "abs_sentiment","confidence","hour",
            "ret_1h","ret_4h","exc_1h","exc_4h"]
    with open(OUT, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {len(rows)} labeled rows -> {OUT}")

    # ── signal check: RAW vs EXCESS ──
    def auc(scores, labels):
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return float("nan")
        w = sum(1 for p in pos for n in neg if p > n) + 0.5*sum(1 for p in pos for n in neg if p == n)
        return w/(len(pos)*len(neg))

    print("\n" + "="*60)
    print(f"SIGNAL CHECK — {len(rows)} events (Apr 9 → Jul 12)")
    print("="*60)
    for tgt in ("ret_4h", "exc_4h", "ret_1h", "exc_1h"):
        pairs = [(r, 1 if r[tgt] > DEADBAND else (0 if r[tgt] < -DEADBAND else None)) for r in rows]
        pairs = [(r, d) for r, d in pairs if d is not None]
        if not pairs:
            continue
        n = len(pairs); up = sum(1 for _, d in pairs if d == 1)
        a = auc([r["sentiment"] for r, _ in pairs], [d for _, d in pairs])
        tag = "EXCESS(drift-stripped)" if tgt.startswith("exc") else "raw"
        print(f"\n[{tgt}] {tag}  n={n}  base UP={up/n*100:.1f}%  sentiment->UP AUC={a:.3f}")
        # non-BTC only for excess (BTC excess is ~0)
        if tgt.startswith("exc"):
            nb = [(r, d) for r, d in pairs if r["asset"] not in ("BTC", "ALL")]
            if nb:
                a2 = auc([r["sentiment"] for r, _ in nb], [d for _, d in nb])
                print(f"     non-BTC only: n={len(nb)} sentiment->UP AUC={a2:.3f}")

if __name__ == "__main__":
    main()
