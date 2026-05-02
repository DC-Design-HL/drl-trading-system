"""
Test BTC dominance and USDT dominance (proxy) as trade filters on actual trades.

BTC.D from Binance Futures BTCDOMUSDT pair.
USDT.D proxy: inverse of combined BTC+ETH+SOL+XRP price momentum
  (when crypto drops, money flows to stablecoins → USDT.D rises)

Filters tested:
- BTC.D trend (rising/falling over N bars)
- BTC.D level (high/low dominance)
- USDT.D proxy trend (rising = bearish, falling = bullish)
- Combined with ADX
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

def compute_sma(values_by_ts, ts_list, period):
    """Compute SMA over a sorted time series. Returns dict ts -> sma."""
    result = {}
    vals = [values_by_ts[t] for t in ts_list]
    for i in range(period - 1, len(vals)):
        result[ts_list[i]] = np.mean(vals[i - period + 1:i + 1])
    return result

def compute_trend(values_by_ts, ts_list, lookback):
    """Returns dict ts -> 'rising'/'falling'/'flat' based on change over lookback bars."""
    result = {}
    vals = [values_by_ts[t] for t in ts_list]
    for i in range(lookback, len(vals)):
        change_pct = (vals[i] - vals[i - lookback]) / vals[i - lookback] * 100 if vals[i - lookback] != 0 else 0
        if change_pct > 0.1:
            result[ts_list[i]] = 'rising'
        elif change_pct < -0.1:
            result[ts_list[i]] = 'falling'
        else:
            result[ts_list[i]] = 'flat'
    return result

def find_at_time(series, ts_sorted, entry_ms):
    idx = np.searchsorted(ts_sorted, entry_ms, side='right') - 1
    if idx < 0: return None
    ts = ts_sorted[idx]
    return series.get(ts)

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
            })

    print(f"Loaded {len(trades)} closed trades\n")

    # ── Fetch BTCDOMUSDT ──
    print("  Fetching BTCDOMUSDT (BTC dominance)...")
    btcdom_klines = fetch_klines('BTCDOMUSDT', '15m', DAYS + 5)
    print(f"    {len(btcdom_klines)} bars")
    # Values are in basis points-like format (e.g., 5065 = ~50.65%)
    btcdom_close = {k['ts']: k['close'] for k in btcdom_klines}
    btcdom_ts = sorted(btcdom_close.keys())

    # BTC.D trend over various lookback periods
    btcdom_trend_4h = compute_trend(btcdom_close, btcdom_ts, 16)   # 16 x 15m = 4h
    btcdom_trend_12h = compute_trend(btcdom_close, btcdom_ts, 48)  # 48 x 15m = 12h
    btcdom_trend_24h = compute_trend(btcdom_close, btcdom_ts, 96)  # 96 x 15m = 24h

    # BTC.D SMA for level-based filter
    btcdom_sma50 = compute_sma(btcdom_close, btcdom_ts, 50)

    # ── Compute USDT.D proxy ──
    # Use inverse of average crypto price change as USDT.D proxy
    print("  Computing USDT.D proxy from crypto basket...")
    basket_klines = {}
    for sym in SYMBOLS:
        print(f"    Fetching {sym}...")
        basket_klines[sym] = fetch_klines(sym, '15m', DAYS + 5)

    # Create a combined "crypto index" from all 4 symbols (normalized returns)
    # Find common timestamps
    ts_sets = [set(k['ts'] for k in basket_klines[sym]) for sym in SYMBOLS]
    common_ts = sorted(ts_sets[0].intersection(*ts_sets[1:]))

    # Build price index: for each common timestamp, average the normalized price
    # Normalize each symbol to start at 100
    first_prices = {}
    sym_close = {}
    for sym in SYMBOLS:
        close_map = {k['ts']: k['close'] for k in basket_klines[sym]}
        sym_close[sym] = close_map
        first_prices[sym] = close_map[common_ts[0]]

    crypto_index = {}
    for ts in common_ts:
        norm_sum = 0
        for sym in SYMBOLS:
            norm_sum += sym_close[sym][ts] / first_prices[sym] * 100
        crypto_index[ts] = norm_sum / len(SYMBOLS)

    # USDT.D proxy = inverse of crypto index trend
    # When crypto drops → USDT.D rises → bearish
    usdtd_trend_4h = {}
    usdtd_trend_12h = {}
    usdtd_trend_24h = {}
    ci_list = [crypto_index[t] for t in common_ts]
    for i in range(16, len(common_ts)):
        change = (ci_list[i] - ci_list[i-16]) / ci_list[i-16] * 100
        # Invert: crypto rising = USDT.D falling
        if change > 0.1:
            usdtd_trend_4h[common_ts[i]] = 'falling'  # bullish
        elif change < -0.1:
            usdtd_trend_4h[common_ts[i]] = 'rising'   # bearish
        else:
            usdtd_trend_4h[common_ts[i]] = 'flat'
    for i in range(48, len(common_ts)):
        change = (ci_list[i] - ci_list[i-48]) / ci_list[i-48] * 100
        if change > 0.2:
            usdtd_trend_12h[common_ts[i]] = 'falling'
        elif change < -0.2:
            usdtd_trend_12h[common_ts[i]] = 'rising'
        else:
            usdtd_trend_12h[common_ts[i]] = 'flat'
    for i in range(96, len(common_ts)):
        change = (ci_list[i] - ci_list[i-96]) / ci_list[i-96] * 100
        if change > 0.3:
            usdtd_trend_24h[common_ts[i]] = 'falling'
        elif change < -0.3:
            usdtd_trend_24h[common_ts[i]] = 'rising'
        else:
            usdtd_trend_24h[common_ts[i]] = 'flat'

    # ── ADX data ──
    print("  Computing ADX...")
    adx_data = {}
    for sym in SYMBOLS:
        adx_series = compute_adx_series(basket_klines[sym])
        ts_sorted = np.array(sorted(adx_series.keys()))
        adx_data[sym] = (adx_series, ts_sorted)

    # ── Enrich trades ──
    btcdom_ts_arr = np.array(btcdom_ts)
    common_ts_arr = np.array(common_ts)

    for t in trades:
        entry_dt = datetime.fromisoformat(t['entry_time']).replace(tzinfo=timezone.utc)
        entry_ms = int(entry_dt.timestamp() * 1000)

        # ADX
        adx_s, ts_s = adx_data[t['symbol']]
        t['adx'] = find_at_time(adx_s, ts_s, entry_ms) or 0

        # BTC.D level
        t['btcdom'] = find_at_time(btcdom_close, btcdom_ts_arr, entry_ms) or 0
        t['btcdom_sma50'] = find_at_time(btcdom_sma50, btcdom_ts_arr, entry_ms) or 0
        t['btcdom_above_sma'] = t['btcdom'] > t['btcdom_sma50'] if t['btcdom_sma50'] else None

        # BTC.D trends
        t['btcdom_trend_4h'] = find_at_time(btcdom_trend_4h, btcdom_ts_arr, entry_ms)
        t['btcdom_trend_12h'] = find_at_time(btcdom_trend_12h, btcdom_ts_arr, entry_ms)
        t['btcdom_trend_24h'] = find_at_time(btcdom_trend_24h, btcdom_ts_arr, entry_ms)

        # USDT.D proxy trends
        t['usdtd_trend_4h'] = find_at_time(usdtd_trend_4h, common_ts_arr, entry_ms)
        t['usdtd_trend_12h'] = find_at_time(usdtd_trend_12h, common_ts_arr, entry_ms)
        t['usdtd_trend_24h'] = find_at_time(usdtd_trend_24h, common_ts_arr, entry_ms)

        # Is this an alt trade?
        t['is_alt'] = t['symbol'] != 'BTCUSDT'

    def report(name, group):
        if not group:
            return f"  {name:<50} {'0':>4} {'---':>7} {'---':>8} {'---':>10}"
        total = sum(t['pnl'] for t in group)
        wins = sum(1 for t in group if t['pnl'] > 0)
        wr = wins/len(group)*100
        avg = total/len(group)
        return f"  {name:<50} {len(group):>4} {wr:>6.1f}% ${avg:>7.2f} ${total:>9.2f}"

    hdr = f"  {'Scenario':<50} {'Tr':>4} {'WR':>7} {'$/tr':>8} {'Total':>10}"
    sep = "  " + "-" * 80

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  1. BTC DOMINANCE TREND (BTC.D from Binance BTCDOMUSDT)")
    print("=" * 85)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))

    print(f"\n  -- BTC.D 4h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['btcdom_trend_4h'] == trend]
        print(report(f"BTC.D 4h {trend}", g))

    print(f"\n  -- BTC.D 12h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['btcdom_trend_12h'] == trend]
        print(report(f"BTC.D 12h {trend}", g))

    print(f"\n  -- BTC.D 24h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['btcdom_trend_24h'] == trend]
        print(report(f"BTC.D 24h {trend}", g))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  2. BTC.D LEVEL (above/below 50-bar SMA)")
    print("=" * 85)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("BTC.D above SMA50 (BTC strengthening)", [t for t in trades if t['btcdom_above_sma'] == True]))
    print(report("BTC.D below SMA50 (alts strengthening)", [t for t in trades if t['btcdom_above_sma'] == False]))

    # For alts only
    alts = [t for t in trades if t['is_alt']]
    btc_only = [t for t in trades if not t['is_alt']]
    print(f"\n  -- ALTS only (ETH/SOL/XRP) --")
    print(report("Alts: BTC.D above SMA (BTC strong = alts weak)", [t for t in alts if t['btcdom_above_sma'] == True]))
    print(report("Alts: BTC.D below SMA (alts strong)", [t for t in alts if t['btcdom_above_sma'] == False]))
    print(f"\n  -- BTC only --")
    print(report("BTC: BTC.D above SMA (BTC strong)", [t for t in btc_only if t['btcdom_above_sma'] == True]))
    print(report("BTC: BTC.D below SMA (BTC weak)", [t for t in btc_only if t['btcdom_above_sma'] == False]))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  3. USDT DOMINANCE PROXY (inverse crypto momentum)")
    print("=" * 85)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))

    print(f"\n  -- USDT.D 4h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['usdtd_trend_4h'] == trend]
        label = {'rising': 'rising (bearish)', 'falling': 'falling (bullish)', 'flat': 'flat'}[trend]
        print(report(f"USDT.D 4h {label}", g))

    print(f"\n  -- USDT.D 12h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['usdtd_trend_12h'] == trend]
        label = {'rising': 'rising (bearish)', 'falling': 'falling (bullish)', 'flat': 'flat'}[trend]
        print(report(f"USDT.D 12h {label}", g))

    print(f"\n  -- USDT.D 24h trend --")
    for trend in ['rising', 'falling', 'flat']:
        g = [t for t in trades if t['usdtd_trend_24h'] == trend]
        label = {'rising': 'rising (bearish)', 'falling': 'falling (bullish)', 'flat': 'flat'}[trend]
        print(report(f"USDT.D 24h {label}", g))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  4. DIRECTIONAL FILTERS (dominance + trade direction)")
    print("=" * 85)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))

    # When BTC.D rising + trading LONG on alts → bad (money leaving alts)
    print(report("ALT LONG when BTC.D 12h rising (danger)",
        [t for t in trades if t['is_alt'] and t['direction'] == 'LONG' and t['btcdom_trend_12h'] == 'rising']))
    print(report("ALT LONG when BTC.D 12h falling (good)",
        [t for t in trades if t['is_alt'] and t['direction'] == 'LONG' and t['btcdom_trend_12h'] == 'falling']))
    print(report("ALT SHORT when BTC.D 12h rising (good)",
        [t for t in trades if t['is_alt'] and t['direction'] == 'SHORT' and t['btcdom_trend_12h'] == 'rising']))

    # When USDT.D rising → market selling → LONG is risky
    print(report("LONG when USDT.D 12h rising (bearish mkt)",
        [t for t in trades if t['direction'] == 'LONG' and t['usdtd_trend_12h'] == 'rising']))
    print(report("LONG when USDT.D 12h falling (bullish mkt)",
        [t for t in trades if t['direction'] == 'LONG' and t['usdtd_trend_12h'] == 'falling']))
    print(report("SHORT when USDT.D 12h rising (bearish mkt)",
        [t for t in trades if t['direction'] == 'SHORT' and t['usdtd_trend_12h'] == 'rising']))
    print(report("SHORT when USDT.D 12h falling (bullish mkt)",
        [t for t in trades if t['direction'] == 'SHORT' and t['usdtd_trend_12h'] == 'falling']))

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  5. BEST COMBINATIONS (dominance + ADX)")
    print("=" * 85)
    print(hdr); print(sep)
    print(report("All trades (baseline)", trades))
    print(report("15<=ADX<40 (best ADX filter from before)",
        [t for t in trades if 15 <= t['adx'] < 40]))

    combos = [
        ("15<=ADX<40 + block LONG when USDT.D rising",
         lambda t: 15 <= t['adx'] < 40 and not (t['direction'] == 'LONG' and t['usdtd_trend_12h'] == 'rising')),
        ("15<=ADX<40 + block ALT LONG when BTC.D rising",
         lambda t: 15 <= t['adx'] < 40 and not (t['is_alt'] and t['direction'] == 'LONG' and t['btcdom_trend_12h'] == 'rising')),
        ("15<=ADX<40 + only trade when USDT.D falling",
         lambda t: 15 <= t['adx'] < 40 and t['usdtd_trend_12h'] == 'falling'),
        ("15<=ADX<40 + BTC.D below SMA",
         lambda t: 15 <= t['adx'] < 40 and t['btcdom_above_sma'] == False),
        ("15<=ADX<40 + BTC.D above SMA",
         lambda t: 15 <= t['adx'] < 40 and t['btcdom_above_sma'] == True),
        ("15<=ADX<40 + align w/ USDT.D (LONG=falling, SHORT=rising)",
         lambda t: 15 <= t['adx'] < 40 and (
             (t['direction'] == 'LONG' and t['usdtd_trend_12h'] == 'falling') or
             (t['direction'] == 'SHORT' and t['usdtd_trend_12h'] == 'rising'))),
        ("15<=ADX<40 + align w/ USDT.D 4h",
         lambda t: 15 <= t['adx'] < 40 and (
             (t['direction'] == 'LONG' and t['usdtd_trend_4h'] == 'falling') or
             (t['direction'] == 'SHORT' and t['usdtd_trend_4h'] == 'rising'))),
        ("ADX>=15 + align w/ USDT.D 12h",
         lambda t: t['adx'] >= 15 and (
             (t['direction'] == 'LONG' and t['usdtd_trend_12h'] == 'falling') or
             (t['direction'] == 'SHORT' and t['usdtd_trend_12h'] == 'rising'))),
    ]

    for name, filt in combos:
        kept = [t for t in trades if filt(t)]
        print(report(name, kept))

if __name__ == "__main__":
    main()
