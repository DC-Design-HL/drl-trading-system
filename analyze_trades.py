#!/usr/bin/env python3
"""Quick trade analysis script."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.env.ultimate_env import UltimateTradingEnv
from src.backtest.data_loader import DataLoader
from datetime import datetime

# Load data
loader = DataLoader()
df = loader.load('BTC/USDT', '1h', 
    start_date=(datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
    end_date=datetime.now().strftime('%Y-%m-%d'))
test_df = df.iloc[int(len(df) * 0.85):]
print(f"Test data: {len(test_df)} candles")

# Create env
env = UltimateTradingEnv(test_df, position_size=0.50, stop_loss_pct=0.02, take_profit_pct=0.04)
model = PPO.load('./data/models/ultimate_agent.zip')

# Run evaluation
obs, _ = env.reset(options={'random_start': False})
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

# Analyze
trades = env.trades
longs = [t for t in trades if t['direction'] == 'long']
shorts = [t for t in trades if t['direction'] == 'short']

print("\n" + "="*60)
print("TRADE ANALYSIS")
print("="*60)
print(f"Total trades: {len(trades)}")
print(f"LONG trades: {len(longs)}")
print(f"SHORT trades: {len(shorts)}")
print(f"\nLong win rate: {100*sum(1 for t in longs if t['pnl']>0)/max(1,len(longs)):.1f}%")
print(f"Short win rate: {100*sum(1 for t in shorts if t['pnl']>0)/max(1,len(shorts)):.1f}%")

# SL/TP analysis
sl = sum(1 for t in trades if t['pnl_pct'] <= -0.018)
tp = sum(1 for t in trades if t['pnl_pct'] >= 0.035)
print(f"\nExit types:")
print(f"  Stop Loss hits: ~{sl}")
print(f"  Take Profit hits: ~{tp}")
print(f"  Model exits: ~{len(trades)-sl-tp}")

print("\nSample trades:")
for i, t in enumerate(trades[:8]):
    direction = t['direction'].upper()
    exit_type = "TP" if t['pnl_pct'] >= 0.035 else ("SL" if t['pnl_pct'] <= -0.018 else "EXIT")
    print(f"  {i+1}. {direction} -> {exit_type}: Entry=${t['entry']:.0f}, Exit=${t['exit']:.0f}, PnL={t['pnl_pct']*100:+.2f}%")
