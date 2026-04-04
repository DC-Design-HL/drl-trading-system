---
name: no-mock-data
description: Enforce zero mock/fake/hardcoded data in the DRL trading system. Use when reviewing code changes, building features, or auditing the system. Every data point displayed in the UI must come from real API calls (Binance testnet, MongoDB, server endpoints). No placeholder values, no hardcoded prices, no simulated balances, no fake trades. Apply to all projects by default.
---

# No Mock Data Policy

## Rule
**Every piece of data shown in the UI must come from a real source.** No exceptions.

## What counts as mock/fake data
- Hardcoded prices (e.g. `price = 70000.0`)
- Default balances (e.g. `balance = 10000.0` as initial value when API fails)
- Placeholder text pretending to be real data (e.g. "BTC $70,493.70" in HTML)
- Simulated trades or positions not from the exchange
- Random/generated data used in place of real API responses
- Fallback values that look like real data (use "N/A" or "—" or error messages instead)

## What to do when data is unavailable
- Show clear error state: "Unable to load data" with reason
- Show "N/A" or "—" for individual fields
- Show "No data" for empty lists
- Never show a number that could be mistaken for real data

## Audit checklist (run on every code change)
1. Search for hardcoded numeric values that look like prices/balances
2. Search for `= 10000`, `= 0.0` used as display defaults
3. Check that every metric/chart/table traces back to an API call
4. Verify error states show clear "unavailable" messages, not fake values
5. Check testnet data comes from testnet API only, not from main bot state

## Testnet-specific rules
- Testnet tab must ONLY show data from Binance testnet API (`/api/testnet/*` endpoints)
- Never mix testnet data with mainnet/bot data
- Portfolio value, balances, positions, trades — all from testnet API
- The $10,000 USDT balance shown should be the REAL testnet USDT balance, not a default
