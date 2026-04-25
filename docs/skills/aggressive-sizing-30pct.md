---
name: aggressive-sizing-30pct
description: The 30%/month aggressive-sizing rollout for the DRL trading system. Use when reviewing, deploying, or debugging the 3× position-size scaling, the USDT.D filter, or the ADX>60 exhaustion guard. Covers what changed vs. the v-pre-aggressive-sizing-20260425 baseline, why each lever was chosen, the backtest evidence, and the rollback path.
---

# Aggressive Sizing — 30%/Month Target

## What this is

A two-lever change to the live HTF bot designed to lift monthly returns
from the recent ~10.8%/mo baseline to ~30%/mo, on the same strategy and
the same model. The branch is `feature/aggressive-sizing-30pct`. The
restore-point tag is `v-pre-aggressive-sizing-20260425`.

This skill is the canonical reference for the rollout. Any change to
position sizing, USDT.D filter, or ADX exhaustion behavior must be
checked against this doc.

## The two changes (everything else identical to baseline)

### 1. Triple per-trade dollar_risk

```
RISK_POOL_PCT      0.10  →  0.30   # 10% of balance → 30%
FIXED_MAX_NOTIONAL 3000  →  6000   # raise the notional ceiling
```

`live_trading_htf.py:88, 222`. Every other risk constant is unchanged
(`RISK_BUDGET_PARTS=20`, `STOP_LOSS_PCT=0.015`, `LIQ_BUFFER_PCT=0.01`,
`MAX_LEVERAGE=50`).

Per-trade economics at $5K balance, before vs. after:

| | Baseline | Aggressive |
|---|---:|---:|
| dollar_risk per trade | $25 | $75 |
| notional per trade | $1,667 | $5,000 |
| leverage | 33-40× | 33-40× (unchanged) |
| margin per trade | ~$50 | ~$150 |
| risk-per-trade as % of $5K | 0.5% | 1.5% |

The model and structure signals fire identically; the only change is the
dollars in and out.

### 2. New entry filters (LONG-side hardening)

**USDT.D rising → block LONGs** (`_check_usdt_d_guard`,
`live_trading_htf.py`). Synthetic proxy: average % change of the
4-symbol crypto basket (BTC/ETH/SOL/XRP) over the trailing 2h. If the
basket dropped > 0.5%, treat USDT.D as rising and refuse new LONGs.
SHORTs are not affected — they benefit from this regime, blocking them
would invert the signal.

Module helper `_is_usdt_dominance_rising(fetcher)` is cached for 10
minutes and shared across all 4 symbol threads in the consolidated
process. Returns `None` only on hard fetch failure; the guard fails
open in that case.

**ADX > 60 → block all entries** (`_check_rsi_adx_guard`, lines updated
to add the upper bound). Existing `ADX_GUARD_MIN=20` already blocked
ranging markets; this adds the symmetric upper bound for trend
exhaustion. Constant: `ADX_GUARD_MAX = 60`.

## Backtest evidence

35-day reconstruction from Binance Futures userTrades (456 round trips,
Mar 22 → Apr 25, the maximum testnet retention).

**Sizing sweep on the recent 20-day post-fix regime** (276 closes, +$360
baseline pnl, 58.3% WR):

| Scale | Monthly return | Max DD | Worst day |
|------:|--------------:|-------:|----------:|
| 1× (baseline) | 10.8% | 3.5% | -1.9% |
| 2× | 21.6% | 6.8% | -3.9% |
| **3× (deployed)** | **32.4%** | **10.0%** | **-5.8%** |
| 4× | 43.2% | 13.0% | -7.7% |
| 6× | 64.7% | 18.7% | -11.6% |

3× lands in the target range with a 10% drawdown ceiling on the recent
regime. The 35-day window includes a money-losing period (Mar 22 → Apr
6 lost ~$484 baseline) — at 3× scale, that period would have drawn down
~$1,452 = -28% of balance. **The 10% DD figure is the good-regime
floor; bad-regime exposure is ~3× higher.**

**USDT.D filter** — `docs/backtest-signal-combinations-2026-04-21.md`,
`backtest_dominance_filter.py`. Variant "ADX 15-40 + USDT.D 2h rising
blocks LONG" lifted WR 58.5% → 65.8% and pnl +$265 → +$431 on 248
trades (+62% pnl improvement). Synthetic proxy used here matches the
same formulation.

**ADX>60 exhaustion** — `docs/adx-exhaustion-guard-proposal.md`. ADX>60
cluster: 8 trades, 25% WR, -$67 pnl. Blocking saves +$81 net (+$93
losses avoided, -$12 winners missed). 53-trade sample.

Backtest scripts that produced these numbers:
- `scripts/backtest_max_hold_30d.py` — round-trip reconstruction +
  hold-time + stagnant-band rules
- `scripts/backtest_30pct_target.py` — combined rules + sizing sweep

## Decision flow (where the new logic enters)

Trade execution path in `execute_trade()`:

```
1. Direction signal (structure-first: BOS/CHOCH)
2. Reverse-close gates (existing)
3. Orderbook guard            ← unchanged
4. RSI extreme + ADX range/exhaustion guard   ← ADX>60 added here
5. USDT.D filter              ← NEW (LONG-only)
6. Signal gate (skipped in structure-first mode)
7. _open_position(...)        ← uses new RISK_POOL_PCT, FIXED_MAX_NOTIONAL
8. SL/TP placed on exchange
```

## What NOT to change without re-doing the math

These constants are interlocked. If any of them moves, the 30%/mo
projection is invalid until re-backtested:

- `RISK_POOL_PCT`, `RISK_BUDGET_PARTS` — dollar_risk per trade
- `STOP_LOSS_PCT`, `LIQ_BUFFER_PCT` — derived notional and leverage
- `FIXED_MAX_NOTIONAL` — caps the position cap at sizing
- Stagnant band `STAGNANT_PCT_MIN`, `STAGNANT_PCT_MAX` — exit timing
- `ADX_GUARD_MIN`, `ADX_GUARD_MAX` — admit-the-trade thresholds
- `USDT_D_THRESHOLD_PCT`, `USDT_D_LOOKBACK_HOURS` — filter sensitivity

## Rollout plan

**Staged, not big-bang.** Recommend running on a single symbol or in
half-size for the first week even at "3×" — this is a 3× change in
risk, not a 3× change in confidence.

1. **Week 1, 2× scale** — change `RISK_POOL_PCT` to `0.20` first. Watch
   for ~21%/mo materializing, max DD < 7%. If it diverges hard from the
   backtest (either way), pause.
2. **Week 2, 3× scale** — flip `RISK_POOL_PCT` to `0.30`. Watch DD
   ceiling 10%, daily worst-case ~6%.
3. **Don't go past 3×** until the model has been retrained with the
   architectural fixes in `docs/TRAINING_PLAN_REVIEW.md` (Differential
   Sharpe reward, structure-gated filter). Those are real WR
   improvements, but they require Mac-side training.

## Rollback

```
# Quick: flip the two constants in live_trading_htf.py
RISK_POOL_PCT = 0.10
FIXED_MAX_NOTIONAL = 3000.0
# Restart cluster — the change picks up on next iteration.
./start_services.sh
```

For a full revert (including the new filters):

```
git checkout v-pre-aggressive-sizing-20260425 -- live_trading_htf.py
./start_services.sh
```

## Live signatures to watch

In `logs/bots_live.log`:

- `🛡️ USDT.D FILTER BLOCK: <SYMBOL> LONG blocked — crypto basket -X.YZ%
  over 2h ...` — guard firing as expected.
- `ADX=N > 60 (exhaustion / overextended trend)` in the
  `_check_rsi_adx_guard` block reason — exhaustion guard firing.
- Iteration `notional` field in `Iteration complete` should be ~$5,000
  on filled trades (was ~$1,667). If you still see ~$1,667 the constants
  didn't propagate — restart didn't pick up the file change.

In `logs/htf_pending_alerts.jsonl`:
- New OPEN alerts will report `dollar_risk=75`, `margin=~$150`,
  `trade_value=~$5000`. Anything else means the change didn't take.

## When to use this skill

Invoke when:
- Reviewing PnL or drawdown that looks unexpectedly large/small
- Debugging a `LONG blocked` log entry
- Asked to tune position sizing or filters
- Reverting from the aggressive rollout
- Preparing the next sizing step (4× / 6×) — re-derive max-DD ceiling
  before doing it
