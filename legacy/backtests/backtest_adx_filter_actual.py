"""
Compare actual trade results: current logic vs current logic + ADX>=20 filter.
Retroactively computes ADX at each trade's entry time to see which trades
would have been blocked.
"""
import sqlite3, json, sys, time
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from collections import defaultdict

BINANCE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ADX_PERIOD = 14
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

def fetch_klines(symbol, interval, days):
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86400 * 1000 - 200 * 15 * 60 * 1000  # extra for ADX warmup
    all_k = []
    while start < end:
        url = f"{BINANCE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&startTime={start}&limit=1500"
        r = requests.get(url, timeout=30)
        data = r.json()
        if not data:
            break
        all_k.extend(data)
        start = data[-1][0] + 1
        time.sleep(0.1)
    out = []
    for k in all_k:
        out.append({
            'ts': k[0],
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
        })
    return out

def compute_adx_series(klines):
    """Returns dict mapping timestamp_ms -> ADX value"""
    h = np.array([k['high'] for k in klines])
    l = np.array([k['low'] for k in klines])
    c = np.array([k['close'] for k in klines])

    n = len(h)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0

    # Smoothed with Wilder's method
    p = ADX_PERIOD
    atr = np.zeros(n)
    sp = np.zeros(n)
    sm = np.zeros(n)

    atr[p] = np.mean(tr[1:p+1])
    sp[p] = np.mean(plus_dm[1:p+1])
    sm[p] = np.mean(minus_dm[1:p+1])

    for i in range(p+1, n):
        atr[i] = (atr[i-1] * (p-1) + tr[i]) / p
        sp[i] = (sp[i-1] * (p-1) + plus_dm[i]) / p
        sm[i] = (sm[i-1] * (p-1) + minus_dm[i]) / p

    pdi = np.where(atr > 0, 100 * sp / atr, 0)
    mdi = np.where(atr > 0, 100 * sm / atr, 0)
    ds = pdi + mdi
    dx = np.where(ds > 0, 100 * np.abs(pdi - mdi) / ds, 0)

    adx = np.zeros(n)
    # First ADX = mean of first p DX values after warmup
    start_idx = 2 * p
    if start_idx < n:
        adx[start_idx] = np.mean(dx[p+1:start_idx+1])
        for i in range(start_idx + 1, n):
            adx[i] = (adx[i-1] * (p-1) + dx[i]) / p

    result = {}
    for i in range(n):
        result[klines[i]['ts']] = adx[i]
    return result

def find_adx_at_time(adx_series, timestamps_sorted, entry_ts_ms):
    """Find the ADX value at or just before entry_ts_ms"""
    # Binary search for the closest timestamp <= entry_ts_ms
    idx = np.searchsorted(timestamps_sorted, entry_ts_ms, side='right') - 1
    if idx < 0:
        return 0
    ts = timestamps_sorted[idx]
    return adx_series[ts]

def main():
    # 1. Get closed trades from DB
    db = sqlite3.connect('data/trading.db')
    db.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat()
    rows = db.execute('''
        SELECT * FROM trades WHERE is_testnet=1 AND timestamp >= ? ORDER BY timestamp
    ''', (cutoff,)).fetchall()

    # Parse into open/close pairs
    opens = {}
    closed_trades = []
    for r in rows:
        d = dict(r)
        action = d['action']
        symbol = d['symbol']
        data = json.loads(d['data']) if d['data'] else {}

        if 'OPEN' in action:
            opens[symbol] = {
                'action': action, 'symbol': symbol,
                'timestamp': d['timestamp'],
                'price': data.get('price', 0),
                'data': data,
            }
        elif 'CLOSE' in action and symbol in opens:
            entry = opens.pop(symbol)
            pnl = data.get('pnl', data.get('realized_pnl', 0))
            if pnl is None:
                pnl = 0
            closed_trades.append({
                'symbol': symbol,
                'direction': 'LONG' if 'LONG' in entry['action'] else 'SHORT',
                'entry_time': entry['timestamp'],
                'exit_time': d['timestamp'],
                'entry_price': entry['price'],
                'pnl': float(pnl),
            })

    print(f"Found {len(closed_trades)} closed trades in last {DAYS} days\n")

    # 2. Fetch klines and compute ADX for each symbol
    adx_data = {}  # symbol -> (adx_series dict, sorted timestamps array)
    for sym in SYMBOLS:
        print(f"  Fetching {sym} klines...")
        klines = fetch_klines(sym, '15m', DAYS + 5)
        print(f"    {len(klines)} bars")
        adx_series = compute_adx_series(klines)
        ts_sorted = np.array(sorted(adx_series.keys()))
        adx_data[sym] = (adx_series, ts_sorted)

    # 3. For each trade, look up ADX at entry time
    for t in closed_trades:
        sym = t['symbol']
        entry_dt = datetime.fromisoformat(t['entry_time']).replace(tzinfo=timezone.utc)
        entry_ms = int(entry_dt.timestamp() * 1000)
        adx_series, ts_sorted = adx_data[sym]
        t['adx'] = find_adx_at_time(adx_series, ts_sorted, entry_ms)

    # 4. Compare scenarios
    print()
    print("=" * 90)
    print(f"  ACTUAL TRADES — LAST {DAYS} DAYS: Current Logic vs ADX>=20 Filter")
    print("=" * 90)

    scenarios = [
        ("Current logic (no ADX filter)", lambda t: True),
        ("With ADX >= 15 filter", lambda t: t['adx'] >= 15),
        ("With ADX >= 20 filter", lambda t: t['adx'] >= 20),
        ("With ADX >= 25 filter", lambda t: t['adx'] >= 25),
    ]

    print(f"\n{'Scenario':<40} {'Tr':>4} {'WR':>7} {'$/tr':>8} {'Total':>10} {'Blocked':>8}")
    print("-" * 80)

    for name, filt in scenarios:
        kept = [t for t in closed_trades if filt(t)]
        blocked = len(closed_trades) - len(kept)
        if not kept:
            print(f"  {name:<38} {'0':>4} {'N/A':>7} {'N/A':>8} {'$0.00':>10} {blocked:>8}")
            continue
        total = sum(t['pnl'] for t in kept)
        wins = sum(1 for t in kept if t['pnl'] > 0)
        wr = wins / len(kept) * 100
        avg = total / len(kept)
        print(f"  {name:<38} {len(kept):>4} {wr:>6.1f}% ${avg:>7.2f} ${total:>9.2f} {blocked:>8}")

    # 5. Per-symbol breakdown for each scenario
    print(f"\n{'=' * 90}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"{'=' * 90}")

    for name, filt in scenarios:
        print(f"\n  {name}:")
        print(f"  {'Symbol':<12} {'Tr':>4} {'WR':>7} {'$/tr':>8} {'Total':>10} {'Blocked':>8}")
        print(f"  {'-' * 55}")
        for sym in SYMBOLS:
            all_sym = [t for t in closed_trades if t['symbol'] == sym]
            kept = [t for t in all_sym if filt(t)]
            blocked = len(all_sym) - len(kept)
            if not kept:
                print(f"  {sym:<12} {'0':>4} {'N/A':>7} {'N/A':>8} {'$0.00':>10} {blocked:>8}")
                continue
            total = sum(t['pnl'] for t in kept)
            wins = sum(1 for t in kept if t['pnl'] > 0)
            wr = wins / len(kept) * 100
            avg = total / len(kept)
            print(f"  {sym:<12} {len(kept):>4} {wr:>6.1f}% ${avg:>7.2f} ${total:>9.2f} {blocked:>8}")

    # 6. Show blocked trades analysis
    print(f"\n{'=' * 90}")
    print(f"  BLOCKED TRADES ANALYSIS (ADX >= 20)")
    print(f"{'=' * 90}")

    blocked = [t for t in closed_trades if t['adx'] < 20]
    kept = [t for t in closed_trades if t['adx'] >= 20]

    if blocked:
        b_total = sum(t['pnl'] for t in blocked)
        b_wins = sum(1 for t in blocked if t['pnl'] > 0)
        b_wr = b_wins / len(blocked) * 100
        print(f"\n  Trades that WOULD be blocked (ADX < 20): {len(blocked)}")
        print(f"  Their combined PnL: ${b_total:.2f}")
        print(f"  Their win rate: {b_wr:.1f}%")
        print(f"  Avg PnL per blocked trade: ${b_total/len(blocked):.2f}")

        if b_total < 0:
            print(f"\n  >> Blocking these saves ${abs(b_total):.2f} in losses!")
        else:
            print(f"\n  >> WARNING: Blocking these would LOSE ${b_total:.2f} in profits!")

    # 7. ADX distribution of trades
    print(f"\n{'=' * 90}")
    print(f"  ADX DISTRIBUTION AT ENTRY")
    print(f"{'=' * 90}")

    ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 100)]
    print(f"\n  {'ADX Range':<15} {'Count':>6} {'WR':>7} {'Total PnL':>12} {'Avg PnL':>10}")
    print(f"  {'-' * 55}")
    for lo, hi in ranges:
        group = [t for t in closed_trades if lo <= t['adx'] < hi]
        if not group:
            print(f"  {f'{lo}-{hi}':<15} {'0':>6}")
            continue
        total = sum(t['pnl'] for t in group)
        wins = sum(1 for t in group if t['pnl'] > 0)
        wr = wins / len(group) * 100
        avg = total / len(group)
        print(f"  {f'{lo}-{hi}':<15} {len(group):>6} {wr:>6.1f}% ${total:>11.2f} ${avg:>9.2f}")

if __name__ == "__main__":
    main()
