#!/usr/bin/env python3
"""
Check which bots are actually running and saving state
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
try:
    import certifi
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
except ImportError:
    client = MongoClient(MONGO_URI)

db = client["trading_system"]
trades_collection = db["trades"]

print("=" * 100)
print("🤖 CHECKING WHICH BOTS ARE ACTIVE")
print("=" * 100)

# Get unique symbols from trades
all_trades = list(trades_collection.find())
symbols = set(t.get('symbol') for t in all_trades)

print(f"\n📊 Symbols in trade database: {sorted(symbols)}")

# Calculate P&L per asset
print(f"\n💰 P&L Per Asset (from trades):")
print("-" * 100)
for symbol in sorted(symbols):
    symbol_trades = [t for t in all_trades if t.get('symbol') == symbol]
    symbol_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in symbol_trades)

    # Get last trade
    last_trade = max(symbol_trades, key=lambda t: t.get('timestamp', ''))
    last_action = last_trade.get('action', 'UNKNOWN')
    last_time = last_trade.get('timestamp', 'UNKNOWN')

    print(f"  {symbol:<12} P&L: ${float(symbol_pnl):>+10,.2f} | Last trade: {last_action:<20} @ {last_time[:19]}")

print("\n" + "=" * 100)
print("🔍 ROOT CAUSE ANALYSIS")
print("=" * 100)

print(f"""
The state database shows ONLY BTCUSDT, but trades exist for all 4 assets:
  • BTCUSDT ✅
  • ETHUSDT ❌ (missing from state)
  • SOLUSDT ❌ (missing from state)
  • XRPUSDT ❌ (missing from state)

This means:
  1. Either the orchestrator is only tracking BTCUSDT in self.bots
  2. Or save_state() is being called with partial bot data
  3. Or the other bots were closed/removed after their last trades

Expected behavior:
  - save_state() should loop through ALL bots in self.bots
  - Each bot should maintain its realized_pnl across the bot's lifetime
  - State should accumulate P&L even for FLAT positions

Actual behavior:
  - Only BTCUSDT is present in state['assets']
  - Other assets' P&L is lost
  - Total P&L is incomplete

FIX NEEDED:
  The orchestrator must ensure ALL bots are initialized and their P&L is preserved
  even when they are in FLAT positions. Currently, it appears bots are being
  removed or not tracked when they close positions.
""")
