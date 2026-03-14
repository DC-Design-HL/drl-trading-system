#!/usr/bin/env python3
"""
Verify Quick Wins Implementation - Code Inspection

Checks that all 3 fixes are properly implemented in the code.
"""

import re

print("=" * 80)
print("VERIFYING QUICK WINS IMPLEMENTATION")
print("=" * 80)

# Read the live_trading_multi.py file
with open('live_trading_multi.py', 'r') as f:
    code = f.read()

# ──────────────────────────────────────────────────────────────────────────────
# Test 1: XRP Trading Block
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("TEST 1: XRP Trading Block (+$282.54 projected)")
print("─" * 80)

xrp_block_pattern = r"if\s+'XRP'\s+in\s+self\.symbol"
if re.search(xrp_block_pattern, code):
    print("✅ XRP trading block FOUND in code")
    # Extract the block
    match = re.search(r"# Fix #2:.*?reason = .*disabled.*\n", code, re.DOTALL)
    if match:
        print("\n📝 Implementation:")
        for line in match.group(0).split('\n')[:5]:
            print(f"   {line}")
else:
    print("❌ XRP trading block NOT FOUND")

# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Time-Based SL Relaxation
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("TEST 2: Time-Based SL Relaxation (+$111.58 projected)")
print("─" * 80)

# Check for position_entry_time variable
if 'self.position_entry_time' in code:
    print("✅ position_entry_time tracking FOUND")
    entry_time_count = code.count('position_entry_time')
    print(f"   Used {entry_time_count} times in code")
else:
    print("❌ position_entry_time tracking NOT FOUND")

# Check for time-based relaxation logic
time_relax_pattern = r"# Fix #3: Time-Based SL Relaxation"
if re.search(time_relax_pattern, code):
    print("✅ Time-based SL relaxation logic FOUND")
    # Check for 12 hour (43200 seconds) threshold
    if '43200' in code:
        print("   ✅ 12-hour threshold (43200s) configured")
    # Check for 25% relaxation (0.75 multiplier)
    if '0.75' in code and 'Move 25% closer' in code:
        print("   ✅ 25% relaxation (0.75 multiplier) configured")
else:
    print("❌ Time-based SL relaxation logic NOT FOUND")

# Check for entry_time in state persistence
if "'entry_time': bot.position_entry_time" in code:
    print("✅ State save includes entry_time")
else:
    print("❌ State save missing entry_time")

if "self.position_entry_time = state.get('entry_time'" in code:
    print("✅ State restore includes entry_time")
else:
    print("❌ State restore missing entry_time")

# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Enhanced Regime-Adaptive SL
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("TEST 3: Enhanced Regime-Adaptive SL (+$223.85 projected)")
print("─" * 80)

# Check for Fix #1 comment
regime_fix_pattern = r"# Enhanced Regime-adaptive adjustments \(Fix #1"
if re.search(regime_fix_pattern, code):
    print("✅ Enhanced regime-adaptive logic FOUND")

    # Check for specific multipliers
    if "sl_pct *= 2.0  # UPDATED: Wider stops in high vol (was 1.5x)" in code:
        print("   ✅ HIGH_VOLATILITY: 2.0x SL (was 1.5x)")
    else:
        print("   ❌ HIGH_VOLATILITY multiplier incorrect")

    if "sl_pct *= 1.8  # UPDATED: Wider stops to avoid chop (was 1.5x)" in code:
        print("   ✅ RANGING: 1.8x SL (was 1.5x)")
    else:
        print("   ❌ RANGING multiplier incorrect")

    if "sl_pct *= 1.3  # Slightly wider for counter-trend" in code:
        print("   ✅ COUNTER-TREND: 1.3x SL")
    else:
        print("   ❌ COUNTER-TREND multiplier missing")

    # Check if trend-following keeps tight SL
    if "tight SL for trend-following" in code:
        print("   ✅ TRENDING: Tight SL (1.0x) for trend-following")
    else:
        print("   ❌ TRENDING logic incomplete")

else:
    print("❌ Enhanced regime-adaptive logic NOT FOUND")

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

# Count successes
checks = []

# Fix 1: Regime-adaptive
checks.append(re.search(regime_fix_pattern, code) is not None)
checks.append("sl_pct *= 2.0" in code)
checks.append("sl_pct *= 1.8" in code)
checks.append("sl_pct *= 1.3" in code)

# Fix 2: XRP block
checks.append("'XRP' in self.symbol" in code)
checks.append("XRP trading disabled" in code)

# Fix 3: Time-based SL
checks.append("position_entry_time" in code)
checks.append("43200" in code)
checks.append("0.75" in code and "Move 25% closer" in code)
checks.append("'entry_time': bot.position_entry_time" in code)
checks.append("self.position_entry_time = state.get('entry_time'" in code)

passed = sum(checks)
total = len(checks)

print(f"\n📊 Verification Results: {passed}/{total} checks passed")

if passed == total:
    print("\n✅ ALL CHECKS PASSED - Ready for deployment!")
    print("\n🚀 Next Steps:")
    print("   1. Commit changes to dev branch")
    print("   2. Deploy to dev Hugging Face Space")
    print("   3. Monitor for 24-48 hours")
    print("   4. Validate P&L improvement")
elif passed >= total * 0.8:
    print(f"\n⚠️ MOSTLY COMPLETE - {total - passed} checks failed")
    print("   Review failed checks above")
else:
    print(f"\n❌ INCOMPLETE - {total - passed} checks failed")
    print("   Implementation needs work")

print("\n" + "=" * 80)
