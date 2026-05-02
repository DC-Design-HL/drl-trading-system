#!/usr/bin/env python3
"""
3-month backtest with ACTUAL MarketStructure (BOS/ChoCH) logic.
Compares RSI guard variants. Full live-trading mechanics.
"""
import sys, os, json, glob, logging, time as _time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("bt_rsi")
logger.setLevel(logging.INFO)

from src.signals.bos_choch import MarketStructure

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

BASE_SL_PCT = 0.015; BASE_TP_PCT = 0.030; LEVERAGE = 40
TRADE_SIZE_PCT = 0.10; MAX_NOTIONAL = 3000; INITIAL_BALANCE = 5000.0
MIN_BARS_BETWEEN_TRADES = 4
SLIPPAGE_PCT = 0.0005; FEE_PCT = 0.0004
PARTIAL_TP1_RATIO = 0.40; PARTIAL_TP2_RATIO = 0.35; PARTIAL_TP3_RATIO = 0.25
TRAILING_ACTIVATE_PCT = 0.008; TRAILING_DISTANCE_PCT = 0.005; TRAILING_DISTANCE_POST_TP1 = 0.008
STAGNANT_BARS = 24; STAGNANT_MIN_PNL = -0.003; STAGNANT_MAX_PNL = 0.005

def fetch_candles(symbol, interval, total_bars):
    import urllib.request
    all_data = []; end_time = None; remaining = total_bars
    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    while remaining > 0:
        batch = min(remaining, 1000)
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={batch}"
        if end_time: url += f"&endTime={end_time}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        if not data: break
        all_data = data + all_data
        end_time = data[0][0] - 1
        remaining -= len(data)
        if len(data) < batch: break
        _time.sleep(0.15)
    df = pd.DataFrame(all_data, columns=cols)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df

def compute_adx(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(h); tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        pdm[i] = up if up > dn and up > 0 else 0
        mdm[i] = dn if dn > up and dn > 0 else 0
    atr = np.zeros(n); sp = np.zeros(n); sm = np.zeros(n)
    if n > period:
        atr[period] = np.mean(tr[1:period+1]); sp[period] = np.mean(pdm[1:period+1]); sm[period] = np.mean(mdm[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1]*(period-1)+tr[i])/period; sp[i] = (sp[i-1]*(period-1)+pdm[i])/period; sm[i] = (sm[i-1]*(period-1)+mdm[i])/period
    pdi = np.where(atr>0, 100*sp/atr, 0); mdi = np.where(atr>0, 100*sm/atr, 0)
    ds = pdi+mdi; dx = np.where(ds>0, 100*np.abs(pdi-mdi)/ds, 0)
    adx = np.zeros(n); s = 2*period
    if s < n:
        adx[s] = np.mean(dx[period+1:s+1])
        for i in range(s+1, n): adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    return pd.Series(adx, index=df.index), pd.Series(atr, index=df.index)

def compute_atr(df, period=20):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(h); tr = np.zeros(n)
    for i in range(1, n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.zeros(n)
    if n > period:
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n): atr[i] = (atr[i-1]*(period-1)+tr[i])/period
    return pd.Series(atr, index=df.index)

def compute_rsi(closes, period=14):
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1: return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0); losses = np.where(deltas < 0, -deltas, 0)
    ag = np.mean(gains[:period]); al = np.mean(losses[:period])
    rsi[period] = 100 if al == 0 else 100 - 100/(1+ag/al)
    for i in range(period, len(deltas)):
        ag = (ag*(period-1)+gains[i])/period; al = (al*(period-1)+losses[i])/period
        rsi[i+1] = 100 if al == 0 else 100 - 100/(1+ag/al)
    return rsi

def get_structure_signal(window_15m, window_1h=None):
    ms = MarketStructure()
    sig = ms.get_signals(window_15m, window_1h)
    trend = sig.get("trend", "ranging")
    last_dir = sig.get("last_signal_direction", "none")
    if trend == "bullish" and last_dir == "bullish": return "LONG", sig
    elif trend == "bearish" and last_dir == "bearish": return "SHORT", sig
    return None, sig

def apply_slippage(price, direction, is_entry):
    if (direction == "LONG" and is_entry) or (direction == "SHORT" and not is_entry):
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)

class Position:
    def __init__(self, direction, entry_price, sl, tp, bar_idx, units, sl_pct):
        self.direction = direction; self.entry_price = entry_price
        self.sl = sl; self.tp = tp; self.bar_idx = bar_idx
        self.units = units; self.remaining_units = units; self.sl_pct = sl_pct
        d = 1 if direction == "LONG" else -1
        self.tp1 = entry_price * (1 + d * sl_pct)
        self.tp2 = entry_price * (1 + d * 2 * sl_pct)
        self.tp_level = 0; self.peak_price = entry_price
        self.trailing_active = False; self.realized_pnl = 0.0

    def update_peak(self, high, low):
        if self.direction == "LONG": self.peak_price = max(self.peak_price, high)
        else: self.peak_price = min(self.peak_price, low)

    def get_trailing_sl(self):
        if not self.trailing_active: return None
        dist = TRAILING_DISTANCE_POST_TP1 if self.tp_level >= 1 else TRAILING_DISTANCE_PCT
        if self.direction == "LONG": return self.peak_price * (1 - dist)
        return self.peak_price * (1 + dist)

def run_symbol(symbol, df_15m, df_1h, bar_signals, rsi_guard_fn, adx_min=20):
    position = None; balance = INITIAL_BALANCE; trades_list = []; last_trade_bar = -999
    start_bar = min(bar_signals.keys()) if bar_signals else 200

    for i in range(start_bar, len(df_15m)):
        if i not in bar_signals: continue
        bar = df_15m.iloc[i]
        price = float(bar["close"]); high_val = float(bar["high"]); low_val = float(bar["low"])
        sig = bar_signals[i]

        if position is not None:
            position.update_peak(high_val, low_val)
            bars_held = i - position.bar_idx
            if position.direction == "LONG":
                pp = (high_val - position.entry_price) / position.entry_price
            else:
                pp = (position.entry_price - low_val) / position.entry_price
            if pp >= TRAILING_ACTIVATE_PCT: position.trailing_active = True

            eff_sl = position.sl
            tsl = position.get_trailing_sl()
            if tsl:
                if position.direction == "LONG": eff_sl = max(eff_sl, tsl)
                else: eff_sl = min(eff_sl, tsl)

            if position.tp_level == 0:
                tp1_hit = (position.direction=="LONG" and high_val>=position.tp1) or (position.direction=="SHORT" and low_val<=position.tp1)
                if tp1_hit:
                    cu = position.remaining_units * PARTIAL_TP1_RATIO
                    ep = apply_slippage(position.tp1, position.direction, False)
                    pnl = cu * (ep - position.entry_price if position.direction=="LONG" else position.entry_price - ep) * LEVERAGE
                    pnl -= cu * position.entry_price * FEE_PCT * LEVERAGE
                    position.realized_pnl += pnl; position.remaining_units -= cu; position.tp_level = 1

            if position.tp_level == 1:
                tp2_hit = (position.direction=="LONG" and high_val>=position.tp2) or (position.direction=="SHORT" and low_val<=position.tp2)
                if tp2_hit:
                    cu = position.remaining_units * (PARTIAL_TP2_RATIO/(PARTIAL_TP2_RATIO+PARTIAL_TP3_RATIO))
                    ep = apply_slippage(position.tp2, position.direction, False)
                    pnl = cu * (ep - position.entry_price if position.direction=="LONG" else position.entry_price - ep) * LEVERAGE
                    pnl -= cu * position.entry_price * FEE_PCT * LEVERAGE
                    position.realized_pnl += pnl; position.remaining_units -= cu; position.tp_level = 2

            sl_hit = (position.direction=="LONG" and low_val<=eff_sl) or (position.direction=="SHORT" and high_val>=eff_sl)
            tp_hit = (position.direction=="LONG" and high_val>=position.tp) or (position.direction=="SHORT" and low_val<=position.tp)
            stagnant = False
            if bars_held >= STAGNANT_BARS:
                cp = (price-position.entry_price)/position.entry_price if position.direction=="LONG" else (position.entry_price-price)/position.entry_price
                if STAGNANT_MIN_PNL <= cp <= STAGNANT_MAX_PNL: stagnant = True

            if sl_hit or tp_hit or stagnant:
                if sl_hit: ep = apply_slippage(eff_sl, position.direction, False); reason = "SL"
                elif tp_hit: ep = apply_slippage(position.tp, position.direction, False); reason = "TP"
                else: ep = apply_slippage(price, position.direction, False); reason = "STAGNANT"
                pnl = position.remaining_units * (ep-position.entry_price if position.direction=="LONG" else position.entry_price-ep) * LEVERAGE
                pnl -= position.remaining_units * position.entry_price * FEE_PCT * LEVERAGE
                total_pnl = position.realized_pnl + pnl; balance += total_pnl
                trades_list.append({"pnl": total_pnl, "reason": reason, "tp_level": position.tp_level})
                position = None; continue

        if position is not None: continue

        if i - last_trade_bar < MIN_BARS_BETWEEN_TRADES: continue
        direction = sig["direction"]
        if direction is None: continue
        if sig["adx"] < adx_min: continue

        # RSI guard check
        if not rsi_guard_fn(direction, sig["rsi"], sig["rsi_slope"], sig["adx"]): continue

        if i + 1 >= len(df_15m): continue
        next_bar = df_15m.iloc[i + 1]
        entry_price = apply_slippage(float(next_bar["open"]), direction, True)
        atr_val = sig["atr20"]
        sl_pct = BASE_SL_PCT; tp_pct = BASE_TP_PCT
        if atr_val > 0 and entry_price > 0:
            atr_pct = atr_val / entry_price
            sl_pct = max(sl_pct, 1.5 * atr_pct); tp_pct = max(tp_pct, 3.0 * atr_pct)
        if sig["adx"] < 20: sl_pct *= 1.2; tp_pct *= 0.8
        notional = min(balance * TRADE_SIZE_PCT * LEVERAGE, MAX_NOTIONAL * LEVERAGE)
        units = notional / (entry_price * LEVERAGE) if entry_price > 0 else 0
        entry_fee = units * entry_price * FEE_PCT * LEVERAGE
        d = 1 if direction == "LONG" else -1
        sl = entry_price * (1 - d * sl_pct); tp = entry_price * (1 + d * tp_pct)
        position = Position(direction, entry_price, sl, tp, i, units, sl_pct)
        position.realized_pnl -= entry_fee; last_trade_bar = i

    if position is not None:
        lp = apply_slippage(float(df_15m.iloc[-1]["close"]), position.direction, False)
        pnl = position.remaining_units * (lp-position.entry_price if position.direction=="LONG" else position.entry_price-lp) * LEVERAGE
        pnl -= position.remaining_units * position.entry_price * FEE_PCT * LEVERAGE
        balance += position.realized_pnl + pnl
        trades_list.append({"pnl": position.realized_pnl + pnl, "reason": "END", "tp_level": position.tp_level})

    wins = sum(1 for t in trades_list if t["pnl"] > 0)
    total = sum(t["pnl"] for t in trades_list)
    return {"trades": len(trades_list), "wins": wins, "pnl": total, "balance": balance,
            "by_reason": {r: sum(1 for t in trades_list if t["reason"]==r) for r in set(t["reason"] for t in trades_list)}}

def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    bars_15m = days * 96 + 200; bars_1h = days * 24 + 50

    # Precompute signals per symbol (shared across scenarios)
    sym_signals = {}
    sym_data = {}

    for symbol in SYMBOLS:
        logger.info(f"\n{'='*60}\n  Fetching {symbol} ({days} days)\n{'='*60}")
        df_15m = fetch_candles(symbol, "15m", bars_15m)
        df_1h = fetch_candles(symbol, "1h", bars_1h)
        logger.info(f"  {len(df_15m)} 15m bars, {len(df_1h)} 1h bars")

        adx_s, atr14_s = compute_adx(df_15m)
        atr20_s = compute_atr(df_15m, 20)
        df_15m["adx"] = adx_s; df_15m["atr20"] = atr20_s

        # RSI
        rsi = compute_rsi(df_15m["close"].values)
        rsi_slope = np.zeros(len(rsi))
        for i in range(3, len(rsi)): rsi_slope[i] = rsi[i] - rsi[i-3]

        start_bar = 200
        bar_signals = {}
        logger.info(f"  Precomputing structure signals for {symbol}...")

        for i in range(start_bar, len(df_15m)):
            bar = df_15m.iloc[i]
            bos_window = df_15m.iloc[max(0, i-100):i+1].copy()
            bar_ts = bar["timestamp"]
            h1_window = df_1h[df_1h["timestamp"] <= bar_ts].tail(50)
            direction, sig = get_structure_signal(bos_window, h1_window if len(h1_window) > 10 else None)

            bar_signals[i] = {
                "direction": direction,
                "adx": float(bar["adx"]),
                "atr20": float(bar["atr20"]),
                "rsi": rsi[i],
                "rsi_slope": rsi_slope[i],
            }

            if (i - start_bar) % 1000 == 0 and i > start_bar:
                logger.info(f"    {symbol} bar {i-start_bar}/{len(df_15m)-start_bar}")

        logger.info(f"  {symbol} signals done.")
        sym_signals[symbol] = bar_signals
        sym_data[symbol] = (df_15m, df_1h)

    # RSI guard scenarios
    scenarios = {
        "No RSI guard": lambda d, r, s, a: True,
        "Current: RSI>70": lambda d, r, s, a: not ((d=='LONG' and r>70) or (d=='SHORT' and r<30)),
        "RSI>75": lambda d, r, s, a: not ((d=='LONG' and r>75) or (d=='SHORT' and r<25)),
        "RSI>80": lambda d, r, s, a: not ((d=='LONG' and r>80) or (d=='SHORT' and r<20)),
        "RSI>70 + slope<0": lambda d, r, s, a: not ((d=='LONG' and r>70 and s<0) or (d=='SHORT' and r<30 and s>0)),
        "RSI>75 + slope<0": lambda d, r, s, a: not ((d=='LONG' and r>75 and s<0) or (d=='SHORT' and r<25 and s>0)),
        "RSI>70 only when ADX<25": lambda d, r, s, a: True if a>=25 else not ((d=='LONG' and r>70) or (d=='SHORT' and r<30)),
        "RSI>70 + slope<-2": lambda d, r, s, a: not ((d=='LONG' and r>70 and s<-2) or (d=='SHORT' and r<30 and s>2)),
    }

    print(f"\n{'='*100}")
    print(f"  {days}-DAY BACKTEST — ACTUAL BOS/ChoCH STRUCTURE + ADX>=20 + RSI GUARD VARIANTS")
    print(f"  Full mechanics: slippage, fees, partial TP, trailing, stagnant exit, ATR SL/TP")
    print(f"{'='*100}")

    hdr = f"  {'Scenario':<40} {'Tr':>5} {'WR':>7} {'$/tr':>8} {'Total':>11} {'Final$':>8}"
    sep = "  " + "-" * 85
    print(f"\n{hdr}\n{sep}")

    all_results = {}
    for scen_name, guard_fn in scenarios.items():
        total_tr = 0; total_wins = 0; total_pnl = 0.0; total_bal = 0.0
        sym_res = {}
        for symbol in SYMBOLS:
            df_15m, df_1h = sym_data[symbol]
            res = run_symbol(symbol, df_15m, df_1h, sym_signals[symbol], guard_fn)
            sym_res[symbol] = res
            total_tr += res["trades"]; total_wins += res["wins"]
            total_pnl += res["pnl"]; total_bal += res["balance"]

        wr = total_wins/total_tr*100 if total_tr > 0 else 0
        avg = total_pnl/total_tr if total_tr > 0 else 0
        print(f"  {scen_name:<40} {total_tr:>5} {wr:>6.1f}% ${avg:>7.2f} ${total_pnl:>10.2f} ${total_bal:>7.0f}")
        all_results[scen_name] = {"total": {"tr": total_tr, "wins": total_wins, "pnl": total_pnl, "bal": total_bal}, "by_sym": sym_res}

    # Per-symbol for top scenarios
    print(f"\n{'='*100}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"{'='*100}")
    for scen_name in ["No RSI guard", "Current: RSI>70", "RSI>70 + slope<0", "RSI>75 + slope<0"]:
        if scen_name not in all_results: continue
        print(f"\n  {scen_name}:")
        print(f"  {'Symbol':<12} {'Tr':>5} {'WR':>7} {'PnL':>11} {'Balance':>9}")
        print(f"  {'-'*50}")
        for sym in SYMBOLS:
            r = all_results[scen_name]["by_sym"][sym]
            wr = r["wins"]/r["trades"]*100 if r["trades"] > 0 else 0
            print(f"  {sym:<12} {r['trades']:>5} {wr:>6.1f}% ${r['pnl']:>10.2f} ${r['balance']:>8.0f}")

if __name__ == "__main__":
    main()
