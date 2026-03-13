#!/usr/bin/env python3
"""
Investigate BTC P&L mismatch
State shows -$59.60 but trades show -$40.92
Difference: -$18.68
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
print("🔍 INVESTIGATING BTCUSDT P&L MISMATCH")
print("=" * 100)

# Get all BTC trades
btc_trades = list(trades_collection.find({"symbol": "BTCUSDT"}).sort("timestamp", 1))

print(f"\n📊 Total BTC trades: {len(btc_trades)}")
print("\nDetailed Trade History:")
print("-" * 100)

total_pnl = Decimal('0')
running_balance = Decimal('5000')  # Starting balance

print(f"{'#':<4} {'Timestamp':<26} {'Action':<20} {'Price':<15} {'PnL':<15} {'Balance (Trade)':<18} {'Running Balance':<18}")
print("-" * 100)

for i, trade in enumerate(btc_trades, 1):
    action = trade.get('action', 'UNKNOWN')
    pnl = Decimal(str(trade.get('pnl', 0)))
    price = trade.get('price', 0)
    balance_in_trade = trade.get('balance', 0)
    timestamp = trade.get('timestamp', 'UNKNOWN')

    total_pnl += pnl
    if pnl != 0:  # Only update running balance on closed trades
        running_balance += pnl

    pnl_str = f"${float(pnl):+.2f}" if pnl != 0 else "$0.00"
    print(f"{i:<4} {timestamp[:26]:<26} {action:<20} ${price:<14,.2f} {pnl_str:<15} ${balance_in_trade:<17,.2f} ${float(running_balance):<17,.2f}")

print("-" * 100)
print(f"\n💰 SUMMARY:")
print(f"   Total P&L from trades: ${float(total_pnl):+.2f}")
print(f"   State shows:           $-59.60")
print(f"   Discrepancy:           ${-59.60 - float(total_pnl):+.2f}")

print("\n" + "=" * 100)
print("🔍 HYPOTHESIS:")
print("=" * 100)

print("""
Looking at the trade history, I notice:
  - Trade #17 (OPEN_LONG) has balance=$19,961.70 (very high!)
  - Trade #18 (OPEN_LONG) has balance=$3,532.90 (normal)
  - Trade #21 (CLOSE_LONG) has balance=$4,965.35 and P&L=$-22.12

The issue might be:
  1. The bot's balance field is not the realized P&L tracker
  2. The balance field in trades is the bot's cash balance at trade time
  3. The realized_pnl is accumulated separately in the bot

The state's 'pnl' field should be bot.realized_pnl, which is:
  - Initialized to 0
  - Updated every time a position closes: self.realized_pnl += pnl

If the bot was restarted or recreated, realized_pnl might have been reset,
losing historical P&L.

ACTUAL BUG:
  When the system was restarted with --assets BTCUSDT only, the bot was
  re-created with realized_pnl = 0, even though there were previous trades.

  The bot then accumulated NEW P&L from trades after the restart, but
  the OLD P&L from trades before restart was lost.

  This is why State P&L != Trades P&L.
""")

# Calculate P&L before and after likely restart point
print("\n" + "=" * 100)
print("🔍 DETECTING RESTART POINT:")
print("=" * 100)

# Look for the restart point - probably when balance jumped to $5000
restart_idx = None
for i, trade in enumerate(btc_trades):
    balance = trade.get('balance', 0)
    if i > 0:
        prev_balance = btc_trades[i-1].get('balance', 0)
        # Check if balance reset to starting value
        if abs(balance - 5000) < 100 and abs(prev_balance - 5000) > 500:
            restart_idx = i
            print(f"\n⚠️  POTENTIAL RESTART DETECTED AT TRADE #{i+1}")
            print(f"    Previous balance: ${prev_balance:,.2f}")
            print(f"    New balance: ${balance:,.2f}")
            break

if restart_idx:
    pre_restart_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in btc_trades[:restart_idx])
    post_restart_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in btc_trades[restart_idx:])

    print(f"\n📊 P&L BREAKDOWN:")
    print(f"   Before restart (trades 1-{restart_idx}): ${float(pre_restart_pnl):+.2f}")
    print(f"   After restart (trades {restart_idx+1}-{len(btc_trades)}): ${float(post_restart_pnl):+.2f}")
    print(f"   Total: ${float(pre_restart_pnl + post_restart_pnl):+.2f}")
    print(f"\n   State shows: $-59.60")
    print(f"   This suggests the bot is only tracking post-restart P&L")
else:
    print("\n✅ No obvious restart point detected based on balance resets")
