# Retired backtest scripts

Moved here 2026-05-02 as part of the btengine framework rollout
(see `docs/backtest_redesign_proposal.md`). These scripts are kept for
historical reference but should NOT be edited or used for new analysis.

## Why they were retired

The backtest audit (May 2 2026) found these 8 scripts were either:

* **Stale**: tested rules that have been superseded by current
  production filters (Apr 27 SYMBOL_BLOCKLIST, FUNDING_LONG, etc.)
* **Duplicated**: near-identical to a more recent script
* **Dead code**: tested an always-on feature with no remaining variant
* **Pre-Apr-22**: predate the Apr 27 → May 1 filter cascade and would
  never match current production behavior again without rewrites

| Script | Reason |
|---|---|
| `backtest_strategy.py` | Apr 5 — uses old UltimateFeatureEngine, predates structure-first |
| `backtest_rsi_3month.py` | Apr 22 — RSI rules superseded by ADX guard |
| `backtest_rsi_3month_real.py` | Apr 22 — twin of above |
| `backtest_rsi_guard.py` | Replaced by `scripts/backtest_phase3_filters.py` |
| `backtest_dominance_filter.py` | Apr 21 — USDT.D guard is now event-based, not modeled |
| `backtest_dominance_short_tf.py` | Apr 21 — same |
| `backtest_whale_news.py` | Apr 20 — superseded by `scripts/backtest_whale_flow_news_filters.py` |
| `backtest_adx_filter_actual.py` | Apr 21 — ADX is always on, no remaining variant |

## What replaces them

`src/btengine/` — the unified backtest framework. To express the same
test as a YAML config, see `configs/sweeps/`. New investigations should
go through btengine, not new copy-pasted scripts.

## Future cleanup

Phase 2 of the redesign will retire ~14 more scripts (the larger
sweep-style tests). Phase 3 ports the remaining ~8 walk-forward /
counterfactual scripts.
