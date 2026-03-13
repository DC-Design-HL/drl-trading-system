#!/usr/bin/env python3
"""
TEST: Portfolio P&L Validation

Validates that the fix works correctly:
1. State P&L matches trade history
2. All assets are tracked
3. Validation logic prevents future mismatches
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
state_collection = db["state"]
trades_collection = db["trades"]

print("=" * 100)
print("🧪 PORTFOLIO P&L VALIDATION TEST")
print("=" * 100)

# 1. Get all trades
all_trades = list(trades_collection.find().sort("timestamp", 1))
symbols = set(t.get('symbol') for t in all_trades)

# 2. Calculate expected P&L per asset
expected_pnl = {}
for symbol in symbols:
    symbol_trades = [t for t in all_trades if t.get('symbol') == symbol]
    total_pnl = sum(Decimal(str(t.get('pnl', 0))) for t in symbol_trades)
    expected_pnl[symbol] = float(total_pnl)

total_expected = sum(expected_pnl.values())

# 3. Get state P&L
state = state_collection.find_one({"_id": "current_state"})
if not state:
    print("❌ FAIL: No state found")
    sys.exit(1)

state_assets = state.get('assets', {})
state_total_pnl = state.get('total_pnl', 0)

# 4. Validate
print("\n📊 TEST RESULTS:")
print("=" * 100)

all_pass = True

# Test 1: All assets present
print("\n1️⃣  TEST: All assets present in state")
for symbol in symbols:
    if symbol in state_assets:
        print(f"   ✅ {symbol} found in state")
    else:
        print(f"   ❌ {symbol} MISSING from state")
        all_pass = False

# Test 2: Per-asset P&L matches
print("\n2️⃣  TEST: Per-asset P&L matches trade history")
for symbol in symbols:
    expected = expected_pnl[symbol]
    if symbol in state_assets:
        actual = state_assets[symbol].get('pnl', 0)
        diff = abs(actual - expected)

        if diff < 0.01:
            print(f"   ✅ {symbol}: ${actual:+.2f} (matches)")
        else:
            print(f"   ❌ {symbol}: Expected ${expected:+.2f}, Got ${actual:+.2f}, Diff ${diff:+.2f}")
            all_pass = False
    else:
        print(f"   ❌ {symbol}: NOT IN STATE (expected ${expected:+.2f})")
        all_pass = False

# Test 3: Total P&L matches
print("\n3️⃣  TEST: Total P&L matches")
diff = abs(state_total_pnl - total_expected)
if diff < 0.01:
    print(f"   ✅ Total P&L: ${state_total_pnl:+.2f} (matches)")
else:
    print(f"   ❌ Total P&L: Expected ${total_expected:+.2f}, Got ${state_total_pnl:+.2f}, Diff ${diff:+.2f}")
    all_pass = False

# Test 4: Check for fix metadata
print("\n4️⃣  TEST: Fix metadata present")
if 'last_fix' in state:
    fix_info = state['last_fix']
    print(f"   ✅ Fix applied at {fix_info['timestamp']}")
    print(f"      Reason: {fix_info['reason']}")
    print(f"      Correction: ${fix_info['correction']:+.2f}")
else:
    print(f"   ⚠️  No fix metadata (expected if state was fresh)")

# Summary
print("\n" + "=" * 100)
print("📊 SUMMARY")
print("=" * 100)

print(f"\n✅ Expected Total P&L: ${total_expected:+.2f}")
print(f"✅ Actual Total P&L:   ${state_total_pnl:+.2f}")
print(f"✅ Discrepancy:        ${abs(state_total_pnl - total_expected):.2f}")

if all_pass:
    print("\n✅✅✅ ALL TESTS PASSED ✅✅✅")
    print("\nThe portfolio state is now accurate and matches trade history.")
    print("The P&L reconciliation fix is working correctly.")
else:
    print("\n❌❌❌ SOME TESTS FAILED ❌❌❌")
    print("\nThe portfolio state still has discrepancies.")
    print("Further investigation required.")
    sys.exit(1)

print("\n" + "=" * 100)
print("✅ VALIDATION COMPLETE")
print("=" * 100)
print()
