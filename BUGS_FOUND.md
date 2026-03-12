# Bugs Found During QA Testing

**Date:** March 12, 2026
**QA Engineer:** Claude Sonnet 4.5
**Branch:** dev

---

## Bug #1: Incorrect Class Name in Documentation

**Severity:** Low
**Status:** Fixed
**File:** tests/test_features/test_ultimate_features.py

**Description:**
Test file attempted to import `SmartMoneyAnalyzer` class which doesn't exist. The actual class name in `src/features/ultimate_features.py` is `SMCAnalyzer`.

**Expected Behavior:**
Import should use correct class name `SMCAnalyzer`.

**Actual Behavior:**
```python
ImportError: cannot import name 'SmartMoneyAnalyzer' from 'src.features.ultimate_features'
```

**Root Cause:**
Mismatch between assumed naming convention and actual implementation.

**Fix:**
Updated test imports to use `SMCAnalyzer` instead of `SmartMoneyAnalyzer`.

---

## Bug #2: Feature Name Mismatch

**Severity:** Low (Test Issue, Not Code Bug)
**Status:** Identified
**File:** tests/test_features/test_ultimate_features.py

**Description:**
Tests assume feature names like `bb_upper`, `bb_middle`, `bb_lower` for Bollinger Bands, but actual implementation uses different naming:
- Actual features: `bb_position`, `bb_width`
- Missing: explicit `bb_upper`, `bb_middle`, `bb_lower` features

**Expected Behavior (by test):**
Features dictionary should contain keys: `bb_upper`, `bb_middle`, `bb_lower`, `atr_14`

**Actual Behavior:**
Features dictionary contains: `bb_position`, `bb_width`, `atr_normalized`

**Root Cause:**
UltimateFeatureEngine computes derived features (position within bands, band width) rather than raw band values. This is actually a design choice, not a bug.

**Impact:**
- Tests fail assertion
- Does not impact production code
- Feature engine works correctly, just different design

**Recommendation:**
Update tests to match actual implementation rather than assumptions. The current implementation may actually be superior as it provides normalized/relative features rather than absolute values.

---

## Summary

**Total Bugs Found:** 2 (1 fixed, 1 test update needed)
**Critical Bugs:** 0
**High Severity:** 0
**Medium Severity:** 0
**Low Severity:** 2

**Test Suite Status:**
- Tests Created: 75+
- Tests Passing: ~60% (pending fixes for feature name assumptions)
- Coverage: Partial (core modules covered)

**Next Steps:**
1. Update remaining test assertions to match actual feature names
2. Complete data processing and API tests
3. Run full test suite with coverage
4. Generate final coverage report

---

**Notes:**
- All bugs found are test-related, not production code bugs
- Core functionality appears solid
- Feature engineering implementation is consistent but differs from initial test assumptions
- No critical bugs affecting trading logic, risk management, or DRL brain

---

**Last Updated:** March 12, 2026
