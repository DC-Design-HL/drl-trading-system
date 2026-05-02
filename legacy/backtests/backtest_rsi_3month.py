"""
3-month simulated backtest: current RSI guard vs proposed RSI>75+slope guard.
Simulates structure-first entries with all guards, realistic execution.
"""
import sys, time, json
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

BINANCE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 90
ADX_PERIOD = 14

# Live system params
BASE_SL_PCT = 0.015
BASE_TP_PCT = 0.030
LEVERAGE = 40
SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
MAX_NOTIONAL = 3000
PARTIAL_TP1_R = 0.40
PARTIAL_TP2_R = 0.35
PARTIAL_TP3_R = 0.25
STAGNANT_BARS = 24  # 6h at 15m
STAGNANT_MIN = -0.003
STAGNANT_MAX = 0.005

def fetch_klines(symbol, interval, days, extra_bars=500):
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

def compute_rsi(closes, period=14):
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1: return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period]); avg_loss = np.mean(losses[:period])
    rsi[period] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi

def compute_adx(highs, lows, closes):
    n = len(highs); p = ADX_PERIOD
    tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up = highs[i]-highs[i-1]; dn = lows[i-1]-lows[i]
        pdm[i] = up if (up > dn and up > 0) else 0
        mdm[i] = dn if (dn > up and dn > 0) else 0
    atr = np.zeros(n); sp = np.zeros(n); sm = np.zeros(n)
    if p < n:
        atr[p] = np.mean(tr[1:p+1]); sp[p] = np.mean(pdm[1:p+1]); sm[p] = np.mean(mdm[1:p+1])
    for i in range(p+1, n):
        atr[i] = (atr[i-1]*(p-1)+tr[i])/p; sp[i] = (sp[i-1]*(p-1)+pdm[i])/p; sm[i] = (sm[i-1]*(p-1)+mdm[i])/p
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    ds = pdi+mdi; dx = np.where(ds>0, 100*np.abs(pdi-mdi)/ds, 0)
    adx = np.zeros(n); si = 2*p
    if si < n:
        adx[si] = np.mean(dx[p+1:si+1])
        for i in range(si+1, n): adx[i] = (adx[i-1]*(p-1)+dx[i])/p
    return adx, atr

def compute_atr(highs, lows, closes, period=14):
    n = len(highs); atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    if period < n:
        atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, n):
        atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return atr

def simple_structure_signal(closes, highs, lows, i, lookback=20):
    """Simplified structure signal: trend based on swing highs/lows."""
    if i < lookback + 5: return None
    window_h = highs[i-lookback:i]
    window_l = lows[i-lookback:i]
    window_c = closes[i-lookback:i]
    # Higher highs + higher lows = bullish
    mid = lookback // 2
    hh1 = np.max(window_h[:mid]); hh2 = np.max(window_h[mid:])
    ll1 = np.min(window_l[:mid]); ll2 = np.min(window_l[mid:])
    if hh2 > hh1 and ll2 > ll1:
        return 'LONG'
    elif hh2 < hh1 and ll2 < ll1:
        return 'SHORT'
    return None

class Position:
    def __init__(self, direction, entry_price, units, sl, tp, tp1, tp2, bar_idx):
        self.direction = direction
        self.entry = entry_price
        self.units = units
        self.remaining = units
        self.sl = sl; self.tp = tp; self.tp1 = tp1; self.tp2 = tp2
        self.entry_bar = bar_idx
        self.tp1_hit = False; self.tp2_hit = False
        self.peak = entry_price
        self.trailing_active = False
        self.pnl = 0.0

    def check_exit(self, bar_idx, high, low, close):
        d = 1 if self.direction == 'LONG' else -1
        # Update peak
        if d == 1: self.peak = max(self.peak, high)
        else: self.peak = min(self.peak, low)
        # SL
        if (d == 1 and low <= self.sl) or (d == -1 and high >= self.sl):
            exit_p = self.sl * (1 - SLIPPAGE_PCT * d)
            self.pnl = (exit_p - self.entry) * d * self.remaining - abs(exit_p * self.remaining) * FEE_PCT
            return 'SL', self.pnl
        # TP1
        if not self.tp1_hit:
            if (d == 1 and high >= self.tp1) or (d == -1 and low <= self.tp1):
                portion = self.remaining * PARTIAL_TP1_R
                exit_p = self.tp1 * (1 - SLIPPAGE_PCT * d)
                partial = (exit_p - self.entry) * d * portion - abs(exit_p * portion) * FEE_PCT
                self.pnl += partial
                self.remaining -= portion
                self.tp1_hit = True
                # Move SL to breakeven
                self.sl = self.entry
        # TP2
        if self.tp1_hit and not self.tp2_hit:
            if (d == 1 and high >= self.tp2) or (d == -1 and low <= self.tp2):
                portion = self.remaining * (PARTIAL_TP2_R / (PARTIAL_TP2_R + PARTIAL_TP3_R))
                exit_p = self.tp2 * (1 - SLIPPAGE_PCT * d)
                partial = (exit_p - self.entry) * d * portion - abs(exit_p * portion) * FEE_PCT
                self.pnl += partial
                self.remaining -= portion
                self.tp2_hit = True
        # Full TP
        if (d == 1 and high >= self.tp) or (d == -1 and low <= self.tp):
            exit_p = self.tp * (1 - SLIPPAGE_PCT * d)
            self.pnl += (exit_p - self.entry) * d * self.remaining - abs(exit_p * self.remaining) * FEE_PCT
            return 'TP', self.pnl
        # Trailing
        if self.tp1_hit:
            move = (self.peak - self.entry) * d / self.entry
            if move > 0.008:
                trail_dist = 0.005 if not self.tp2_hit else 0.008
                new_sl = self.peak - d * self.entry * trail_dist if d == 1 else self.peak + self.entry * trail_dist
                if d == 1: self.sl = max(self.sl, new_sl)
                else: self.sl = min(self.sl, new_sl)
        # Stagnant
        bars_held = bar_idx - self.entry_bar
        if bars_held >= STAGNANT_BARS:
            unrealized = (close - self.entry) * d / self.entry
            if STAGNANT_MIN <= unrealized <= STAGNANT_MAX:
                exit_p = close * (1 - SLIPPAGE_PCT * d)
                self.pnl += (exit_p - self.entry) * d * self.remaining - abs(exit_p * self.remaining) * FEE_PCT
                return 'STAGNANT', self.pnl
        return None, 0

def run_scenario(sym_data, scenario_name, rsi_guard_fn):
    """Run one scenario across all symbols."""
    results = {'trades': 0, 'wins': 0, 'total_pnl': 0, 'by_exit': {}, 'by_symbol': {}}
    
    for sym in SYMBOLS:
        d = sym_data[sym]
        closes = d['close']; highs = d['high']; lows = d['low']; opens_ = d['open']
        rsi = d['rsi']; adx = d['adx']; atr = d['atr']; rsi_slope = d['rsi_slope']
        n = len(closes)
        start_bar = 500  # warmup
        
        pos = None
        sym_pnl = 0; sym_trades = 0; sym_wins = 0
        
        for i in range(start_bar, n - 1):
            if pos is not None:
                reason, pnl = pos.check_exit(i, highs[i], lows[i], closes[i])
                if reason:
                    sym_pnl += pnl; sym_trades += 1
                    if pnl > 0: sym_wins += 1
                    results['by_exit'][reason] = results['by_exit'].get(reason, 0) + 1
                    pos = None
                continue
            
            # Entry logic: simplified structure signal
            signal = simple_structure_signal(closes, highs, lows, i)
            if signal is None: continue
            
            # ADX guard (>= 20)
            if adx[i] < 20: continue
            
            # RSI guard — varies by scenario
            if not rsi_guard_fn(signal, rsi[i], rsi_slope[i], adx[i]): continue
            
            # Entry on next bar open
            entry_price = opens_[i + 1]
            if entry_price <= 0: continue
            
            # ATR-based SL/TP
            cur_atr = atr[i]
            if cur_atr <= 0: continue
            sl_dist = max(BASE_SL_PCT, 1.5 * cur_atr / entry_price)
            tp_dist = max(BASE_TP_PCT, 3.0 * cur_atr / entry_price)
            
            d_sign = 1 if signal == 'LONG' else -1
            sl_price = entry_price * (1 - d_sign * sl_dist)
            tp_price = entry_price * (1 + d_sign * tp_dist)
            tp1_price = entry_price * (1 + d_sign * tp_dist * 0.33)
            tp2_price = entry_price * (1 + d_sign * tp_dist * 0.66)
            
            # Slippage on entry
            entry_price *= (1 + SLIPPAGE_PCT * d_sign)
            units = MAX_NOTIONAL / entry_price
            # Entry fee
            entry_fee = abs(entry_price * units) * FEE_PCT
            
            pos = Position(signal, entry_price, units, sl_price, tp_price, tp1_price, tp2_price, i)
            pos.pnl = -entry_fee
        
        # Close any open position at end
        if pos is not None:
            exit_p = closes[-1]
            d_sign = 1 if pos.direction == 'LONG' else -1
            pos.pnl += (exit_p - pos.entry) * d_sign * pos.remaining - abs(exit_p * pos.remaining) * FEE_PCT
            sym_pnl += pos.pnl; sym_trades += 1
            if pos.pnl > 0: sym_wins += 1
        
        results['by_symbol'][sym] = {'trades': sym_trades, 'wins': sym_wins, 'pnl': sym_pnl}
        results['trades'] += sym_trades
        results['wins'] += sym_wins
        results['total_pnl'] += sym_pnl
    
    return results

def main():
    print(f"Fetching {DAYS}-day data for all symbols...\n")
    
    sym_data = {}
    for sym in SYMBOLS:
        print(f"  {sym} 15m...")
        k15 = fetch_klines(sym, '15m', DAYS + 10)
        print(f"    {len(k15)} bars")
        
        closes = np.array([k['close'] for k in k15])
        highs = np.array([k['high'] for k in k15])
        lows = np.array([k['low'] for k in k15])
        opens = np.array([k['open'] for k in k15])
        
        rsi = compute_rsi(closes)
        adx, _ = compute_adx(highs, lows, closes)
        atr = compute_atr(highs, lows, closes)
        
        rsi_slope = np.zeros(len(rsi))
        for i in range(3, len(rsi)): rsi_slope[i] = rsi[i] - rsi[i-3]
        
        sym_data[sym] = {'close': closes, 'high': highs, 'low': lows, 'open': opens,
                         'rsi': rsi, 'adx': adx, 'atr': atr, 'rsi_slope': rsi_slope}
    
    # Define RSI guard scenarios
    scenarios = [
        ("No RSI guard",
         lambda sig, rsi, slope, adx: True),
        
        ("Current: RSI > 70 blocks (LONG>70, SHORT<30)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>70) or (sig=='SHORT' and rsi<30))),
        
        ("Proposed: RSI>75 + slope<0 (fading momentum)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>75 and slope<0) or (sig=='SHORT' and rsi<25 and slope>0))),
        
        ("RSI > 75 (simple higher threshold)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>75) or (sig=='SHORT' and rsi<25))),
        
        ("RSI > 80 (very loose)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>80) or (sig=='SHORT' and rsi<20))),
        
        ("RSI>70 + slope<0 (current thresh + slope)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>70 and slope<0) or (sig=='SHORT' and rsi<30 and slope>0))),
        
        ("RSI>70 only when ADX<25 (trend-aware)",
         lambda sig, rsi, slope, adx: True if adx >= 25 else not ((sig=='LONG' and rsi>70) or (sig=='SHORT' and rsi<30))),
        
        ("RSI>75 + slope<0 + ADX<25 (full combo)",
         lambda sig, rsi, slope, adx: True if adx >= 25 else not ((sig=='LONG' and rsi>75 and slope<0) or (sig=='SHORT' and rsi<25 and slope>0))),
        
        ("RSI>70 + slope<-2 (steep fade only)",
         lambda sig, rsi, slope, adx: not ((sig=='LONG' and rsi>70 and slope<-2) or (sig=='SHORT' and rsi<30 and slope>2))),
    ]
    
    print(f"\n{'='*95}")
    print(f"  {DAYS}-DAY BACKTEST: RSI GUARD COMPARISON")
    print(f"  Structure signals + ADX>=20 + RSI guard variant")
    print(f"  Realistic: slippage, fees, partial TP, trailing, stagnant exit")
    print(f"{'='*95}")
    
    hdr = f"  {'Scenario':<50} {'Tr':>5} {'WR':>7} {'$/tr':>8} {'Total':>11}"
    sep = "  " + "-" * 85
    print(f"\n{hdr}\n{sep}")
    
    all_results = {}
    for name, guard_fn in scenarios:
        res = run_scenario(sym_data, name, guard_fn)
        all_results[name] = res
        tr = res['trades']
        if tr == 0:
            print(f"  {name:<50} {'0':>5}")
            continue
        wr = res['wins']/tr*100; avg = res['total_pnl']/tr
        print(f"  {name:<50} {tr:>5} {wr:>6.1f}% ${avg:>7.2f} ${res['total_pnl']:>10.2f}")
    
    # Per-symbol for top scenarios
    top_names = ["Current: RSI > 70 blocks (LONG>70, SHORT<30)",
                 "Proposed: RSI>75 + slope<0 (fading momentum)",
                 "No RSI guard"]
    
    print(f"\n{'='*95}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"{'='*95}")
    
    for name in top_names:
        res = all_results.get(name)
        if not res: continue
        print(f"\n  {name}:")
        print(f"  {'Symbol':<12} {'Tr':>5} {'WR':>7} {'PnL':>10}")
        print(f"  {'-'*40}")
        for sym in SYMBOLS:
            s = res['by_symbol'].get(sym, {})
            tr = s.get('trades', 0)
            if tr == 0: print(f"  {sym:<12} {'0':>5}"); continue
            wr = s['wins']/tr*100
            print(f"  {sym:<12} {tr:>5} {wr:>6.1f}% ${s['pnl']:>9.2f}")
    
    # Exit reason comparison
    print(f"\n{'='*95}")
    print(f"  EXIT REASON DISTRIBUTION")
    print(f"{'='*95}")
    for name in top_names:
        res = all_results.get(name)
        if not res: continue
        print(f"\n  {name}:")
        for reason, count in sorted(res['by_exit'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

if __name__ == "__main__":
    main()
