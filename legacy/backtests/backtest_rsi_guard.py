"""
Backtest RSI guard alternatives on actual trades.
Goal: avoid buying at tops but still catch trend starts.
"""
import sqlite3, json, sys, time
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

BINANCE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ADX_PERIOD = 14

def fetch_klines(symbol, interval, days, extra_bars=300):
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
             'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])} for k in all_k]

def compute_rsi(closes, period=14):
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1: return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi[period] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi

def compute_stoch_rsi(rsi_vals, period=14, k_period=3):
    n = len(rsi_vals)
    stoch_rsi = np.full(n, 50.0)
    for i in range(period, n):
        window = rsi_vals[i - period + 1:i + 1]
        mn, mx = np.min(window), np.max(window)
        stoch_rsi[i] = (rsi_vals[i] - mn) / (mx - mn) * 100 if mx - mn > 0 else 50
    k = np.full(n, 50.0)
    for i in range(k_period - 1, n):
        k[i] = np.mean(stoch_rsi[i - k_period + 1:i + 1])
    return k

def compute_cci(highs, lows, closes, period=20):
    n = len(closes); cci = np.zeros(n)
    tp = (highs + lows + closes) / 3
    for i in range(period - 1, n):
        window = tp[i - period + 1:i + 1]
        sma = np.mean(window); mad = np.mean(np.abs(window - sma))
        if mad > 0: cci[i] = (tp[i] - sma) / (0.015 * mad)
    return cci

def compute_williams_r(highs, lows, closes, period=14):
    n = len(closes); wr = np.full(n, -50.0)
    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1:i + 1])
        ll = np.min(lows[i - period + 1:i + 1])
        if hh - ll > 0: wr[i] = (hh - closes[i]) / (hh - ll) * -100
    return wr

def compute_adx(highs, lows, closes):
    n = len(highs)
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up = highs[i]-highs[i-1]; dn = lows[i-1]-lows[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0
    p = ADX_PERIOD
    atr = np.zeros(n); sp = np.zeros(n); sm = np.zeros(n)
    if p < n:
        atr[p] = np.mean(tr[1:p+1]); sp[p] = np.mean(plus_dm[1:p+1]); sm[p] = np.mean(minus_dm[1:p+1])
    for i in range(p+1, n):
        atr[i] = (atr[i-1]*(p-1)+tr[i])/p; sp[i] = (sp[i-1]*(p-1)+plus_dm[i])/p; sm[i] = (sm[i-1]*(p-1)+minus_dm[i])/p
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    ds = pdi+mdi; dx = np.where(ds>0, 100*np.abs(pdi-mdi)/ds, 0)
    adx = np.zeros(n); si = 2*p
    if si < n:
        adx[si] = np.mean(dx[p+1:si+1])
        for i in range(si+1, n): adx[i] = (adx[i-1]*(p-1)+dx[i])/p
    return adx

def find_at(series_arr, ts_arr, entry_ms):
    idx = np.searchsorted(ts_arr, entry_ms, side='right') - 1
    return series_arr[idx] if idx >= 0 else None

def main():
    db = sqlite3.connect('data/trading.db')
    db.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(days=DAYS)).isoformat()
    rows = db.execute('SELECT * FROM trades WHERE is_testnet=1 AND timestamp >= ? ORDER BY timestamp', (cutoff,)).fetchall()
    opens = {}; trades = []
    for r in rows:
        d = dict(r); action = d['action']; symbol = d['symbol']
        data = json.loads(d['data']) if d['data'] else {}
        if 'OPEN' in action:
            opens[symbol] = {'action': action, 'symbol': symbol, 'timestamp': d['timestamp'], 'data': data}
        elif 'CLOSE' in action and symbol in opens:
            entry = opens.pop(symbol)
            pnl = data.get('pnl', data.get('realized_pnl', 0)) or 0
            trades.append({'symbol': symbol, 'direction': 'LONG' if 'LONG' in entry['action'] else 'SHORT',
                           'entry_time': entry['timestamp'], 'pnl': float(pnl), 'confidence': entry['data'].get('confidence', 0)})
    print(f"Loaded {len(trades)} closed trades\n")

    indicators = {}
    for sym in SYMBOLS:
        print(f"  Fetching {sym} 15m...")
        k15 = fetch_klines(sym, '15m', DAYS + 5)
        print(f"  Fetching {sym} 1h...")
        k1h = fetch_klines(sym, '1h', DAYS + 5, extra_bars=100)
        ts_15 = np.array([k['ts'] for k in k15])
        c15 = np.array([k['close'] for k in k15])
        h15 = np.array([k['high'] for k in k15])
        l15 = np.array([k['low'] for k in k15])
        ts_1h = np.array([k['ts'] for k in k1h])
        c1h = np.array([k['close'] for k in k1h])
        h1h = np.array([k['high'] for k in k1h])
        l1h = np.array([k['low'] for k in k1h])
        rsi_15m = compute_rsi(c15); rsi_1h = compute_rsi(c1h)
        stoch_rsi_15m = compute_stoch_rsi(rsi_15m)
        cci_15m = compute_cci(h15, l15, c15); cci_1h = compute_cci(h1h, l1h, c1h)
        wr_15m = compute_williams_r(h15, l15, c15)
        adx_15m = compute_adx(h15, l15, c15)
        rsi_slope = np.zeros(len(rsi_15m))
        for i in range(3, len(rsi_15m)): rsi_slope[i] = rsi_15m[i] - rsi_15m[i - 3]
        rsi_recent_low = np.full(len(rsi_15m), 50.0)
        for i in range(8, len(rsi_15m)): rsi_recent_low[i] = np.min(rsi_15m[i-8:i])
        indicators[sym] = {'ts_15': ts_15, 'rsi_15m': rsi_15m, 'stoch_rsi': stoch_rsi_15m,
            'cci_15m': cci_15m, 'wr_15m': wr_15m, 'adx_15m': adx_15m, 'rsi_slope': rsi_slope,
            'rsi_recent_low': rsi_recent_low, 'ts_1h': ts_1h, 'rsi_1h': rsi_1h, 'cci_1h': cci_1h}

    for t in trades:
        ind = indicators[t['symbol']]
        entry_ms = int(datetime.fromisoformat(t['entry_time']).replace(tzinfo=timezone.utc).timestamp() * 1000)
        t['rsi_15m'] = find_at(ind['rsi_15m'], ind['ts_15'], entry_ms) or 50
        t['rsi_1h'] = find_at(ind['rsi_1h'], ind['ts_1h'], entry_ms) or 50
        t['stoch_rsi'] = find_at(ind['stoch_rsi'], ind['ts_15'], entry_ms) or 50
        t['cci_15m'] = find_at(ind['cci_15m'], ind['ts_15'], entry_ms) or 0
        t['cci_1h'] = find_at(ind['cci_1h'], ind['ts_1h'], entry_ms) or 0
        t['wr_15m'] = find_at(ind['wr_15m'], ind['ts_15'], entry_ms) or -50
        t['adx'] = find_at(ind['adx_15m'], ind['ts_15'], entry_ms) or 0
        t['rsi_slope'] = find_at(ind['rsi_slope'], ind['ts_15'], entry_ms) or 0
        t['rsi_recent_low'] = find_at(ind['rsi_recent_low'], ind['ts_15'], entry_ms) or 50

    def is_ob(t):
        if t['direction'] == 'LONG': return t['rsi_15m'] > 70
        else: return t['rsi_15m'] < 30

    def report(name, group):
        if not group: return f"  {name:<60} {'0':>4} {'---':>7} {'---':>8} {'---':>10}"
        total = sum(t['pnl'] for t in group); wins = sum(1 for t in group if t['pnl'] > 0)
        wr = wins/len(group)*100; avg = total/len(group)
        return f"  {name:<60} {len(group):>4} {wr:>6.1f}% ${avg:>7.2f} ${total:>9.2f}"

    hdr = f"  {'Scenario':<60} {'Tr':>4} {'WR':>7} {'$/tr':>8} {'Total':>10}"
    sep = "  " + "-" * 90

    print("\n" + "=" * 95)
    print("  1. RSI THRESHOLD COMPARISON (block LONG when RSI > X)")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No RSI guard (all trades)", trades))
    for thresh in [65, 70, 75, 80, 85, 90]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>thresh) or (t['direction']=='SHORT' and t['rsi_15m']<(100-thresh)))]
        print(report(f"RSI 15m threshold = {thresh}", kept))

    print("\n" + "=" * 95)
    print("  2. RSI + ADX COMBO (allow high RSI when ADX confirms strong trend)")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    print(report("Current: RSI>70 blocks all", [t for t in trades if not is_ob(t)]))
    for adx_th in [20, 25, 30, 35]:
        kept = [t for t in trades if (t['adx'] >= adx_th) or (not is_ob(t))]
        print(report(f"RSI>70 blocked only when ADX < {adx_th} (trending=allow)", kept))

    print("\n" + "=" * 95)
    print("  3. RSI SLOPE (block only when RSI is FADING from OB)")
    print("  Rising RSI in OB = trend strength. Falling RSI = momentum dying.")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_slope']<0) or
            (t['direction']=='SHORT' and t['rsi_15m']<30 and t['rsi_slope']>0))]
    print(report("Block OB only when RSI slope negative (fading)", kept))
    kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_slope']<-2) or
            (t['direction']=='SHORT' and t['rsi_15m']<30 and t['rsi_slope']>2))]
    print(report("Block OB only when RSI slope < -2 (steep fade)", kept))
    kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>75 and t['rsi_slope']<0) or
            (t['direction']=='SHORT' and t['rsi_15m']<25 and t['rsi_slope']>0))]
    print(report("Block only RSI>75 + slope negative", kept))

    print("\n" + "=" * 95)
    print("  4. STOCHASTIC RSI")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    for th in [70, 75, 80, 85, 90]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['stoch_rsi']>th) or (t['direction']=='SHORT' and t['stoch_rsi']<(100-th)))]
        print(report(f"StochRSI threshold = {th}", kept))

    print("\n" + "=" * 95)
    print("  5. CCI (Commodity Channel Index)")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    for th in [100, 150, 200, 250]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['cci_15m']>th) or (t['direction']=='SHORT' and t['cci_15m']<-th))]
        print(report(f"CCI 15m > {th}", kept))
    print()
    for th in [100, 150, 200]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['cci_1h']>th) or (t['direction']=='SHORT' and t['cci_1h']<-th))]
        print(report(f"CCI 1h > {th}", kept))

    print("\n" + "=" * 95)
    print("  6. WILLIAMS %R")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    for th in [-10, -15, -20, -25, -30]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['wr_15m']>th) or (t['direction']=='SHORT' and t['wr_15m']<-(100+th)))]
        print(report(f"Williams %R threshold = {th}", kept))

    print("\n" + "=" * 95)
    print("  7. MULTI-TIMEFRAME RSI (1h)")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    for th in [65, 70, 75, 80]:
        kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_1h']>th) or (t['direction']=='SHORT' and t['rsi_1h']<(100-th)))]
        print(report(f"RSI 1h threshold = {th}", kept))
    kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_1h']>65) or
            (t['direction']=='SHORT' and t['rsi_15m']<30 and t['rsi_1h']<35))]
    print(report("Block only when BOTH 15m>70 AND 1h>65", kept))
    kept = [t for t in trades if not ((t['direction']=='LONG' and t['rsi_15m']>75 and t['rsi_1h']>70) or
            (t['direction']=='SHORT' and t['rsi_15m']<25 and t['rsi_1h']<30))]
    print(report("Block only when BOTH 15m>75 AND 1h>70", kept))

    print("\n" + "=" * 95)
    print("  8. RSI PULLBACK (allow OB if RSI recently dipped = trend continuation)")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    for dip_th in [50, 55, 60, 65]:
        kept = [t for t in trades if not (
            (t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_recent_low']>=dip_th) or
            (t['direction']=='SHORT' and t['rsi_15m']<30 and (100-t['rsi_recent_low'])>=dip_th))]
        print(report(f"Allow OB if recent 8-bar RSI dip below {dip_th}", kept))

    print("\n" + "=" * 95)
    print("  9. BEST COMBINATIONS")
    print("=" * 95)
    print(hdr); print(sep)
    print(report("No guard (all trades)", trades))
    print(report("Current RSI>70 guard", [t for t in trades if not is_ob(t)]))

    combos = [
        ("RSI>80 (looser threshold)", lambda t: not ((t['direction']=='LONG' and t['rsi_15m']>80) or (t['direction']=='SHORT' and t['rsi_15m']<20))),
        ("RSI>70 only when ADX<25", lambda t: t['adx']>=25 or not is_ob(t)),
        ("RSI>70 only when ADX<30", lambda t: t['adx']>=30 or not is_ob(t)),
        ("RSI>75 + slope<0", lambda t: not ((t['direction']=='LONG' and t['rsi_15m']>75 and t['rsi_slope']<0) or (t['direction']=='SHORT' and t['rsi_15m']<25 and t['rsi_slope']>0))),
        ("RSI>70 + slope<-2 (steep fade)", lambda t: not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_slope']<-2) or (t['direction']=='SHORT' and t['rsi_15m']<30 and t['rsi_slope']>2))),
        ("CCI>200 (extreme only)", lambda t: not ((t['direction']=='LONG' and t['cci_15m']>200) or (t['direction']=='SHORT' and t['cci_15m']<-200))),
        ("Both 15m>70 AND 1h>65", lambda t: not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['rsi_1h']>65) or (t['direction']=='SHORT' and t['rsi_15m']<30 and t['rsi_1h']<35))),
        ("RSI>70 + ADX<25 + slope<0 (triple)", lambda t: not ((t['direction']=='LONG' and t['rsi_15m']>70 and t['adx']<25 and t['rsi_slope']<0) or (t['direction']=='SHORT' and t['rsi_15m']<30 and t['adx']<25 and t['rsi_slope']>0))),
        ("StochRSI>85", lambda t: not ((t['direction']=='LONG' and t['stoch_rsi']>85) or (t['direction']=='SHORT' and t['stoch_rsi']<15))),
        ("RSI 1h>75", lambda t: not ((t['direction']=='LONG' and t['rsi_1h']>75) or (t['direction']=='SHORT' and t['rsi_1h']<25))),
    ]
    for name, filt in combos:
        print(report(name, [t for t in trades if filt(t)]))

    print("\n" + "=" * 95)
    print("  10. RSI DISTRIBUTION OF ACTUAL TRADES")
    print("=" * 95)
    ranges = [(0,30),(30,40),(40,50),(50,60),(60,65),(65,70),(70,75),(75,80),(80,100)]
    for dir_name, group in [("LONG", [t for t in trades if t['direction']=='LONG']), ("SHORT", [t for t in trades if t['direction']=='SHORT'])]:
        print(f"\n  {dir_name} trades RSI 15m distribution:")
        print(f"  {'RSI Range':<12} {'Count':>6} {'WR':>7} {'Total PnL':>12} {'Avg PnL':>10}")
        print(f"  {'-'*50}")
        for lo, hi in ranges:
            g = [t for t in group if lo <= t['rsi_15m'] < hi]
            if not g: print(f"  {f'{lo}-{hi}':<12} {'0':>6}"); continue
            total = sum(t['pnl'] for t in g); w = sum(1 for t in g if t['pnl']>0)
            print(f"  {f'{lo}-{hi}':<12} {len(g):>6} {w/len(g)*100:>6.1f}% ${total:>11.2f} ${total/len(g):>9.2f}")

if __name__ == "__main__":
    main()
