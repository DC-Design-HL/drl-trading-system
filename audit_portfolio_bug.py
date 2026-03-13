#!/usr/bin/env python3
"""
CRITICAL BUG INVESTIGATION: Portfolio P&L Mismatch Audit
=========================================================

Expected:
- State shows -$102.89 total P&L
- Trades DB shows -$92.32 total P&L
- Discrepancy: $10.57

Mission:
1. Query all 46 trades from MongoDB
2. Manually recalculate total P&L
3. Find the missing/extra $10.57
4. Identify which calculation is wrong
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import json
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("❌ MONGO_URI not found in .env")
    sys.exit(1)

try:
    import certifi
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
except ImportError:
    client = MongoClient(MONGO_URI)

db = client["trading_system"]
state_collection = db["state"]
trades_collection = db["trades"]

print("=" * 100)
print("🔍 CRITICAL BUG INVESTIGATION: Portfolio P&L Mismatch")
print("=" * 100)

# 1. Get all trades
all_trades = list(trades_collection.find().sort("timestamp", 1))
print(f"\n📊 Total trades in database: {len(all_trades)}")

# 2. Calculate P&L manually
print("\n" + "=" * 100)
print("💰 MANUAL P&L CALCULATION - ALL TRADES")
print("=" * 100)

total_pnl_manual = Decimal('0')
trades_with_pnl = []
trades_without_pnl = []

print(f"\n{'#':<4} {'Timestamp':<26} {'Symbol':<12} {'Action':<20} {'Price':<12} {'PnL':<12} {'Balance':<12}")
print("-" * 100)

for i, trade in enumerate(all_trades, 1):
    symbol = trade.get('symbol', 'UNKNOWN')
    action = trade.get('action', 'UNKNOWN')
    pnl = trade.get('pnl', 0)
    price = trade.get('price', 0)
    balance = trade.get('balance', 0)
    timestamp = trade.get('timestamp', 'UNKNOWN')

    # Track trades
    if pnl != 0:
        trades_with_pnl.append(trade)
        total_pnl_manual += Decimal(str(pnl))
    else:
        trades_without_pnl.append(trade)

    pnl_str = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
    print(f"{i:<4} {timestamp:<26} {symbol:<12} {action:<20} ${price:<11,.2f} {pnl_str:<12} ${balance:<11,.2f}")

print("-" * 100)
print(f"\n✅ Trades with P&L: {len(trades_with_pnl)}")
print(f"⚪ Trades without P&L (entries): {len(trades_without_pnl)}")
print(f"\n💎 MANUALLY CALCULATED TOTAL P&L: ${float(total_pnl_manual):+.2f}")

# 3. Get current state
print("\n" + "=" * 100)
print("💼 PORTFOLIO STATE FROM DATABASE")
print("=" * 100)

current_state = state_collection.find_one({"_id": "current_state"})

if current_state:
    timestamp = current_state.get('timestamp', 'UNKNOWN')
    print(f"\n🕐 Last Updated: {timestamp}")

    assets = current_state.get('assets', {})
    total_pnl_state = Decimal('0')
    total_balance_state = Decimal('0')

    print(f"\n{'Asset':<12} {'Position':<10} {'Balance':<15} {'Realized P&L':<15} {'Entry Price':<15}")
    print("-" * 100)

    for asset_name, asset_data in sorted(assets.items()):
        position = asset_data.get('position', 0)
        balance = asset_data.get('balance', 0)
        pnl = asset_data.get('pnl', 0)
        entry_price = asset_data.get('entry_price', 0)

        total_pnl_state += Decimal(str(pnl))
        total_balance_state += Decimal(str(balance))

        pos_str = "LONG" if position == 1 else ("SHORT" if position == -1 else "FLAT")
        print(f"{asset_name:<12} {pos_str:<10} ${balance:<14,.2f} ${pnl:>+14,.2f} ${entry_price:<14,.2f}")

    print("-" * 100)
    print(f"\n📊 STATE TOTALS:")
    print(f"   Total Balance: ${float(total_balance_state):,.2f}")
    print(f"   Total Realized P&L: ${float(total_pnl_state):+.2f}")
else:
    print("\n❌ No state found in database!")
    total_pnl_state = Decimal('0')

# 4. COMPARISON
print("\n" + "=" * 100)
print("⚖️  DISCREPANCY ANALYSIS")
print("=" * 100)

discrepancy = float(total_pnl_state - total_pnl_manual)

print(f"\n📋 Summary:")
print(f"   P&L from State:  ${float(total_pnl_state):+.2f}")
print(f"   P&L from Trades: ${float(total_pnl_manual):+.2f}")
print(f"   Discrepancy:     ${discrepancy:+.2f}")

if abs(discrepancy) < 0.01:
    print(f"\n✅ MATCH! No significant discrepancy.")
else:
    print(f"\n❌ MISMATCH DETECTED!")
    print(f"\n🔍 DISCREPANCY BREAKDOWN:")

    # Check per-asset
    print(f"\n{'Asset':<12} {'State P&L':<15} {'Trades P&L':<15} {'Diff':<15}")
    print("-" * 60)

    for asset_name in assets.keys():
        state_pnl = Decimal(str(assets[asset_name].get('pnl', 0)))

        # Calculate from trades
        trades_pnl = Decimal('0')
        for trade in trades_with_pnl:
            if trade.get('symbol') == asset_name:
                trades_pnl += Decimal(str(trade.get('pnl', 0)))

        diff = float(state_pnl - trades_pnl)

        print(f"{asset_name:<12} ${float(state_pnl):>+14,.2f} ${float(trades_pnl):>+14,.2f} ${diff:>+14,.2f}")

# 5. DETAILED TRADE BREAKDOWN BY ASSET
print("\n" + "=" * 100)
print("📊 PER-ASSET TRADE HISTORY (Detailed)")
print("=" * 100)

for symbol in sorted(set(t.get('symbol') for t in all_trades)):
    symbol_trades = [t for t in all_trades if t.get('symbol') == symbol]
    symbol_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in symbol_trades)

    print(f"\n{symbol}:")
    print(f"  Total trades: {len(symbol_trades)}")
    print(f"  Total P&L: ${float(symbol_pnl):+.2f}")

    # Show each trade
    for i, trade in enumerate(symbol_trades, 1):
        action = trade.get('action', 'UNKNOWN')
        pnl = trade.get('pnl', 0)
        price = trade.get('price', 0)
        timestamp = trade.get('timestamp', 'UNKNOWN')

        pnl_str = f"${pnl:+.2f}" if pnl != 0 else "$0.00"
        print(f"    {i}. {timestamp[:19]} | {action:<20} @ ${price:,.2f} | P&L: {pnl_str}")

# 6. CHECK FOR DUPLICATE TRADES
print("\n" + "=" * 100)
print("🔍 DUPLICATE TRADE DETECTION")
print("=" * 100)

# Group by timestamp and action
trade_signatures = {}
duplicates_found = False

for trade in all_trades:
    signature = (
        trade.get('timestamp'),
        trade.get('symbol'),
        trade.get('action'),
        trade.get('price')
    )

    if signature in trade_signatures:
        trade_signatures[signature].append(trade)
        duplicates_found = True
    else:
        trade_signatures[signature] = [trade]

if duplicates_found:
    print("\n⚠️ DUPLICATE TRADES FOUND:")
    for sig, trades in trade_signatures.items():
        if len(trades) > 1:
            print(f"\n  {sig[0]} | {sig[1]} | {sig[2]} @ ${sig[3]:,.2f}")
            print(f"  Found {len(trades)} identical trades:")
            for t in trades:
                print(f"    - PnL: ${t.get('pnl', 0):+.2f}")
else:
    print("\n✅ No duplicate trades detected")

# 7. FINAL DIAGNOSIS
print("\n" + "=" * 100)
print("🩺 ROOT CAUSE DIAGNOSIS")
print("=" * 100)

print(f"\n🔍 Investigation Results:")
print(f"   • Total trades analyzed: {len(all_trades)}")
print(f"   • Trades with P&L: {len(trades_with_pnl)}")
print(f"   • Trades without P&L: {len(trades_without_pnl)}")
print(f"   • Manually calculated total: ${float(total_pnl_manual):+.2f}")
print(f"   • State total: ${float(total_pnl_state):+.2f}")
print(f"   • Discrepancy: ${discrepancy:+.2f}")

if abs(discrepancy) > 0.01:
    print(f"\n❌ BUG CONFIRMED: Portfolio state calculation is incorrect")
    print(f"\nPossible causes:")
    print(f"   1. State is adding/subtracting P&L incorrectly")
    print(f"   2. Some trades are missing from the database")
    print(f"   3. P&L is being double-counted somewhere")
    print(f"   4. State is not being updated after trades")
    print(f"\n💡 Next steps:")
    print(f"   → Review save_state() in live_trading_multi.py")
    print(f"   → Check execute_trade() P&L calculations")
    print(f"   → Verify log_trade() is saving correct values")
else:
    print(f"\n✅ No discrepancy found - portfolio calculations are correct")

print("\n" + "=" * 100)
print("✅ AUDIT COMPLETE")
print("=" * 100)
print()
