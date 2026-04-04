---
name: ui-type-safety
description: Enforce type-safe formatting in Streamlit UI code. Use when reviewing or writing app.py code that displays data from API responses. Every f-string format operation must handle None, missing keys, and wrong types. Prevents TypeError crashes like "unsupported format string passed to NoneType.__format__". Apply to all UI code changes.
---

# UI Type Safety

## Rule
Every numeric format string in app.py must be safe against None/missing values.

## Common crash pattern
```python
# CRASHES when pnl is None
f"${pnl:,.2f}"

# SAFE
f"${(pnl or 0):,.2f}"
f"${pnl:,.2f}" if pnl is not None and isinstance(pnl, (int, float)) else "—"
```

## Checklist (run on every UI code change)
1. Search for `:{pattern}f` format specs (`:,.2f`, `:+.2f`, `:+,.2f`, `:.1f`)
2. For each: verify the variable CANNOT be None at that point
3. If it comes from `.get()` or API response → add `or 0` or type guard
4. If displaying to user → prefer "—" or "N/A" over 0 when data is truly missing

## Safe patterns
```python
# API data with fallback
value = data.get('price') or 0
f"${value:,.2f}"

# Conditional display
f"${value:,.2f}" if isinstance(value, (int, float)) else "—"

# Metric with delta
delta=f"{pnl:+,.2f}" if isinstance(pnl, (int, float)) and pnl != 0 else None
```

## Integration test requirements
- Test every API endpoint with empty/null field responses
- Test UI rendering functions with None values for all numeric fields
- Test that no f-string formatting crashes when API returns partial data
