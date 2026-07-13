#!/usr/bin/env python3
"""
Exchange-netflow signal validation (2026-07-13, foundation rebuild).

Builds a daily exchange-netflow series from the abundant fresh exchange-wallet
data (binance hot/cold, coinbase) and validates whether it predicts forward ETH
returns. Net INFLOW to exchanges = coins moving to sell = bearish (expect
negative forward return); net OUTFLOW = accumulation = bullish.

Data processing + stats only (no model training) — safe on server. Prints an
honest read: correlation, direction-AUC, and quantile-bucketed forward returns.
"""
import json, datetime, urllib.request, statistics
from collections import defaultdict

WALLETS = ["binance_hot_wallet", "binance_cold_wallet", "binance_reserve",
           "coinbase_institutional"]
START_DAY = "2026-04-06"   # data availability start for price join

def build_netflow():
    daily = defaultdict(lambda: [0.0, 0.0])  # day -> [inflow, outflow] (native ETH)
    for w in WALLETS:
        try:
            f = open(f"data/whale_behavior/eth/{w}.jsonl")
        except FileNotFoundError:
            continue
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            v = d.get("value_eth", 0) or 0
            if v <= 0:
                continue
            day = datetime.datetime.utcfromtimestamp(int(d.get("timestamp", 0))).strftime("%Y-%m-%d")
            if day < START_DAY:
                continue
            if d.get("direction") == "in":
                daily[day][0] += v      # into exchange = sell pressure
            else:
                daily[day][1] += v      # out of exchange = accumulation
    # netflow = inflow - outflow (positive = net deposits = bearish)
    return {day: io[0] - io[1] for day, io in daily.items()}

def eth_daily_closes(start_day, end_day):
    start_ms = int(datetime.datetime.strptime(start_day, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.datetime.strptime(end_day, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000) + 86400_000
    closes = {}
    cur = start_ms
    while cur < end_ms:
        url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1d&startTime={cur}&limit=1000"
        data = json.load(urllib.request.urlopen(url, timeout=20))
        if not data:
            break
        for k in data:
            day = datetime.datetime.utcfromtimestamp(k[0] / 1000).strftime("%Y-%m-%d")
            closes[day] = float(k[4])
        cur = data[-1][0] + 86400_000
        if len(data) < 1000:
            break
    return closes

def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs) ** 0.5
    dy = sum((y-my)**2 for y in ys) ** 0.5
    return num/(dx*dy) if dx and dy else float("nan")

def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for n in neg if p > n) + 0.5*sum(1 for p in pos for n in neg if p == n)
    return w/(len(pos)*len(neg))

def main():
    nf = build_netflow()
    days = sorted(nf)
    closes = eth_daily_closes(days[0], days[-1])
    # build (netflow_day_d, forward return d->d+1) pairs
    rows = []
    sdays = sorted(closes)
    idx = {d: i for i, d in enumerate(sdays)}
    for d in days:
        if d not in idx:
            continue
        i = idx[d]
        if i+1 >= len(sdays):
            continue
        c0, c1 = closes[sdays[i]], closes[sdays[i+1]]
        fwd = c1/c0 - 1
        rows.append((d, nf[d], fwd))
    print(f"netflow days: {len(nf)}, joined to fwd returns: {len(rows)}")
    print(f"date range: {rows[0][0]} -> {rows[-1][0]}")
    netflows = [r[1] for r in rows]
    fwds = [r[2] for r in rows]

    print("\n=== SIGNAL: daily net exchange flow (ETH) -> next-day ETH return ===")
    # expect NEGATIVE corr: net inflow (deposits, +) -> price down
    print(f"Pearson(netflow, next-day return) = {pearson(netflows, fwds):+.3f}  (expect NEGATIVE if signal real)")
    # AUC: does net-OUTFLOW (negative netflow) predict UP day? use -netflow as bullish score
    labels = [1 if f > 0 else 0 for f in fwds]
    print(f"AUC(-netflow -> UP day) = {auc([-n for n in netflows], labels):.3f}  (0.5 = no signal)")
    base = sum(labels)/len(labels)
    print(f"base rate UP = {base*100:.1f}%")

    # quantile buckets by netflow
    order = sorted(rows, key=lambda r: r[1])
    q = len(order)//4 or 1
    print("\nforward return by netflow quartile (Q1=most outflow/bullish → Q4=most inflow/bearish):")
    for name, chunk in [("Q1 (outflow)", order[:q]), ("Q2", order[q:2*q]),
                        ("Q3", order[2*q:3*q]), ("Q4 (inflow)", order[3*q:])]:
        fr = [r[2] for r in chunk]
        wr = sum(1 for x in fr if x > 0)/len(fr)*100
        print(f"  {name:14} n={len(fr):3} avg_fwd={statistics.mean(fr)*100:+.3f}%  UP%={wr:.0f}  avg_netflow={statistics.mean([r[1] for r in chunk]):>+11.0f}")

if __name__ == "__main__":
    main()
