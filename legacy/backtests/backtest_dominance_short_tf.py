"""
Test USDT.D and BTC.D with shorter lookbacks: 15m, 1h, 4h, 12h
on actual trades from last 30 days.
"""
import sqlite3, json, sys, time
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

BINANCE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
ADX_PERIOD = 14
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

def fetch_klines(symbol, interval, days, extra_bars=200):
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - days * 86400 * 1000 - extra_bars * 15 * 60 * 1000
    all_k = []
    while start < end:
        url = f"{BINANCE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&startTime={start}&limit=1500"
        r = requests.get(url, timeout=30)
        data = r.json()
        if not data: break
        all_k.extend(data)
        start = data[-1][0] + 1
        time.sleep(0.1)
    return [{'ts': k[0], 'open': float(k[1]), 'high': float(k[2]),
             'low': float(k[3]), 'close': float(k[4])} for k in all_k]

def compute_adx_series(klines):
    h = np.array([k['high'] for k in klines])
    l = np.array([k['low'] for k in klines])
    c = np.array([k['close'] for k in klines])
    n = len(h)
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0
    p = ADX_PERIOD
    atr = np.zeros(n); sp = np.zeros(n); sm = np.zeros(n)
    if p < n:
        atr[p] = np.mean(tr[1:p+1]); sp[p] = np.mean(plus_dm[1:p+1]); sm[p] = np.mean(minus_dm[1:p+1])
    for i in range(p+1, n):
        atr[i] = (atr[i-1]*(p-1)+tr[i])/p
        sp[i] = (sp[i-1]*(p-1)+plus_dm[i])/p
        sm[i] = (sm[i-1]*(p-1)+minus_dm[i])/p
    pdi = np.where(atr>0, 100*sp/atr, 0)
    mdi = np.where(atr>0, 100*sm/atr, 0)
    ds = pdi+mdi
    dx = np.where(ds>0, 100*np.abs(pdi-mdi)/ds, 0)
    adx = np.zeros(n)
    si = 2*p
    if si < n:
        adx[si] = np.mean(dx[p+1:si+1])
        for i in range(si+1, n): adx[i] = (adx[i-1]*(p-1)+dx[i])/p
    return {klines[i]['ts']: adx[i] for i in range(n)}

def find_at_time(series, ts_sorted, entry_ms):
    idx = np.searchsorted(ts_sorted, entry_ms, side='right') - 1
    if idx < 0: return None
    return series.get(ts_sorted[idx])

def main():
    db = sqlite3.connect('data/trading.db')
    db.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat()
    rows = db.execute('SELECT * FROM trades WHERE is_testnet=1 AND timestamp >= ? ORDER BY timestamp', (cutoff,)).fetchall()

    opens = {}
    trades = []
    for r in rows:
        d = dict(r)
        action = d['action']; symbol = d['symbol']
        data = json.loads(d['data']) if d['data'] else {}
        if 'OPEN' in action:
            opens[symbol] = {'action': action, 'symbol': symbol, 'timestamp': d['timestamp'], 'data': data}
        elif 'CLOSE' in action and symbol in opens:
            entry = opens.pop(symbol)
            pnl = data.get('pnl', data.get('realized_pnl', 0)) or 0
            trades.append({
                'symbol': symbol,
                'direction': 'LONG' if 'LONG' in entry['action'] else 'SHORT',
                'entry_time': entry['timestamp'],
                'pnl': float(pnl),
                'confidence': entry['data'].get('confidence', 0),
                'is_alt': symbol != 'BTCUSDT',
            })

    print(f"Loaded {len(trades)} closed trades\n")

    # Fetch data
    print("  Fetching BTCDOMUSDT...")
    btcdom_klines = fetch_klines('BTCDOMUSDT', '15m', DAYS + 5)
    btcdom_close = {k['ts']: k['close'] for k in btcdom_klines}
    btcdom_ts = sorted(btcdom_close.keys())

    print("  Fetching asset klines + ADX...")
    sym_close = {}
    adx_data = {}
    first_prices = {}
    for sym in SYMBOLS:
        klines = fetch_klines(sym, '15m', DAYS + 5)
        close_map = {k['ts']: k['close'] for k in klines}
        sym_close[sym] = close_map
        adx_series = compute_adx_series(klines)
        adx_data[sym] = (adx_series, np.array(sorted(adx_series.keys())))

    # Common timestamps for crypto index
    ts_sets = [set(sym_close[s].keys()) for s in SYMBOLS]
    common_ts = sorted(ts_sets[0].intersection(*ts_sets[1:]))
    for sym in SYMBOLS:
        first_prices[sym] = sym_close[sym][common_ts[0]]

    crypto_index = {}
    for ts in common_ts:
        crypto_index[ts] = sum(sym_close[sym][ts] / first_prices[sym] * 100 for sym in SYMBOLS) / len(SYMBOLS)

    # Lookback periods in 15m bars
    lookbacks = {
        '15m': 1,
        '30m': 2,
        '1h': 4,
        '2h': 8,
        '4h': 16,
        '8h': 32,
        '12h': 48,
    }

    # Compute trends for each lookback
    def compute_trends(values, ts_list, bars, threshold_pct):
        result = {}
        vals = [values[t] for t in ts_list]
        for i in range(bars, len(vals)):
            if vals[i - bars] == 0: continue
            change = (vals[i] - vals[i - bars]) / vals[i - bars] * 100
            if change > threshold_pct:
                result[ts_list[i]] = 'rising'
            elif change < -threshold_pct:
                result[ts_list[i]] = 'falling'
            else:
                result[ts_list[i]] = 'flat'
        return result

    # Different thresholds for different timeframes
    thresholds = {'15m': 0.02, '30m': 0.03, '1h': 0.05, '2h': 0.08, '4h': 0.1, '8h': 0.15, '12h': 0.2}

    btcdom_trends = {}
    usdtd_trends = {}
    for tf, bars in lookbacks.items():
        th = thresholds[tf]
        btcdom_trends[tf] = compute_trends(btcdom_close, btcdom_ts, bars, th)
        usdtd_trends_raw = compute_trends(crypto_index, common_ts, bars, th)
        # Invert for USDT.D: crypto rising = USDT.D falling
        usdtd_trends[tf] = {}
        for ts, trend in usdtd_trends_raw.items():
            if trend == 'rising': usdtd_trends[tf][ts] = 'falling'
            elif trend == 'falling': usdtd_trends[tf][ts] = 'rising'
            else: usdtd_trends[tf][ts] = 'flat'

    btcdom_ts_arr = np.array(btcdom_ts)
    common_ts_arr = np.array(common_ts)

    # Enrich trades
    for t in trades:
        entry_dt = datetime.fromisoformat(t['entry_time']).replace(tzinfo=timezone.utc)
        entry_ms = int(entry_dt.timestamp() * 1000)
        adx_s, ts_s = adx_data[t['symbol']]
        t['adx'] = find_at_time(adx_s, ts_s, entry_ms) or 0
        for tf in lookbacks:
            t[f'btcdom_{tf}'] = find_at_time(btcdom_trends[tf], btcdom_ts_arr, entry_ms)
            t[f'usdtd_{tf}'] = find_at_time(usdtd_trends[tf], common_ts_arr, entry_ms)

    def report(name, group):
        if not group:
            return f"  {name:<55} {'0':>4} {'---':>7} {'---':>8} {'---':>10}"
        total = sum(t['pnl'] for t in group)
        wins = sum(1 for t in group if t['pnl'] > 0)
        wr = wins/len(group)*100
        avg = total/len(group)
        return f"  {name:<55} {len(group):>4} {wr:>6.1f}% ${avg:>7.2f} ${total:>9.2f}"

    hdr = f"  {'Scenario':<55} {'Tr':>4} {'WR':>7} {'$/tr':>8} {'Total':>10}"
    sep = "  " + "-" * 85

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  1. USDT.D PROXY — ALL TIMEFRAMES")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    for tf in lookbacks:
        print()
        for trend in ['rising', 'falling', 'flat']:
            label = {'rising': f'USDT.D {tf} rising (bearish)', 'falling': f'USDT.D {tf} falling (bullish)', 'flat': f'USDT.D {tf} flat'}[trend]
            g = [t for t in trades if t[f'usdtd_{tf}'] == trend]
            print(report(label, g))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  2. BTC.D — ALL TIMEFRAMES")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    for tf in lookbacks:
        print()
        for trend in ['rising', 'falling', 'flat']:
            g = [t for t in trades if t[f'btcdom_{tf}'] == trend]
            print(report(f"BTC.D {tf} {trend}", g))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  3. BLOCK LONG WHEN USDT.D RISING — BY TIMEFRAME")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("15<=ADX<40 (ADX only)", [t for t in trades if 15 <= t['adx'] < 40]))
    for tf in lookbacks:
        kept = [t for t in trades if 15 <= t['adx'] < 40 and not (t['direction'] == 'LONG' and t[f'usdtd_{tf}'] == 'rising')]
        print(report(f"15<=ADX<40 + block LONG when USDT.D {tf} rising", kept))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  4. BLOCK ALT LONG WHEN BTC.D RISING — BY TIMEFRAME")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("15<=ADX<40 (ADX only)", [t for t in trades if 15 <= t['adx'] < 40]))
    for tf in lookbacks:
        kept = [t for t in trades if 15 <= t['adx'] < 40 and not (t['is_alt'] and t['direction'] == 'LONG' and t[f'btcdom_{tf}'] == 'rising')]
        print(report(f"15<=ADX<40 + block ALT LONG when BTC.D {tf} rising", kept))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  5. ALIGN DIRECTION WITH USDT.D — BY TIMEFRAME")
    print("    (LONG only when USDT.D falling, SHORT only when USDT.D rising)")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("15<=ADX<40 (ADX only)", [t for t in trades if 15 <= t['adx'] < 40]))
    for tf in lookbacks:
        kept = [t for t in trades if 15 <= t['adx'] < 40 and (
            (t['direction'] == 'LONG' and t[f'usdtd_{tf}'] == 'falling') or
            (t['direction'] == 'SHORT' and t[f'usdtd_{tf}'] == 'rising') or
            t[f'usdtd_{tf}'] == 'flat')]
        print(report(f"15<=ADX<40 + align w/ USDT.D {tf}", kept))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  6. BEST COMBOS ACROSS ALL TIMEFRAMES")
    print("=" * 90)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("15<=ADX<40 (ADX only)", [t for t in trades if 15 <= t['adx'] < 40]))

    # Test: block LONG when USDT.D rising at ANY of multiple timeframes
    for tf1 in ['1h', '4h']:
        for tf2 in ['4h', '12h']:
            if tf1 == tf2: continue
            kept = [t for t in trades if 15 <= t['adx'] < 40 and not (
                t['direction'] == 'LONG' and (t[f'usdtd_{tf1}'] == 'rising' or t[f'usdtd_{tf2}'] == 'rising'))]
            print(report(f"ADX + block LONG when USDT.D {tf1} OR {tf2} rising", kept))

    # Block short when USDT.D falling
    for tf in ['1h', '4h', '12h']:
        kept = [t for t in trades if 15 <= t['adx'] < 40 and not (
            (t['direction'] == 'LONG' and t[f'usdtd_{tf}'] == 'rising') or
            (t['direction'] == 'SHORT' and t[f'usdtd_{tf}'] == 'falling'))]
        print(report(f"ADX + block counter-trend trades (USDT.D {tf})", kept))

if __name__ == "__main__":
    main()
