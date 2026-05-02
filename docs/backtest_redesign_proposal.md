# Backtest Framework — Audit + Redesign Proposal (2026-05-02)

Two parallel agents audited the existing backtest infrastructure and
designed a unified replacement. Verdict: redesign is STRONGLY justified.

## TL;DR

- **35 backtest scripts** in repo (`backtest_*.py` + `scripts/backtest_*.py`)
- **~60% are stale** (predate the Apr 27 filter cascade — symbol-blocklist,
  FUNDING_LONG_GUARD, WHALE_NEUTRAL_GUARD, EXT_POS_NEWS_GUARD,
  REVERSE_CLOSE_LONG canary)
- **Silent bugs found**: fee double-counting in 4 scripts, 9 different
  ADX implementations (results drift ±5% WR), fuzzy timestamp matching
  silently drops 5–10% of trades in some scripts
- **No code reuse**: 18 copies of `fetch_klines()`, 15 copies of trade
  pair reconstruction, 9 ADX implementations
- **Cannot scale**: adding one production filter requires editing 5–10
  scripts manually; no parameter sweep, walk-forward, or parallelization
- **Only `backtest_current_production_logic.py` (Apr 30)** is reasonably
  faithful to current production. Others diverge silently.

## Key Audit Findings

### Filter coverage matrix

| Filter (live) | Backtest coverage | Status |
|---|---|---|
| SYMBOL_SIDE_BLOCKLIST | 2/35 files | CRITICAL GAP |
| FUNDING_LONG_GUARD | 1/35 files | CRITICAL GAP |
| WHALE_NEUTRAL_GUARD | 2/35 files | CRITICAL GAP |
| EXT_POS_NEWS_GUARD | 4/35 files | partial |
| ORDERBOOK_GUARD | 3/35 files | partial |
| ADX_GUARD | 18/35 files | OK (but inconsistent thresholds) |
| RSI_GUARD | 8/35 files | OK (but inconsistent thresholds) |
| REVERSE_CLOSE_LONG canary | 2/35 files | CRITICAL GAP |

### Top 5 specific bugs / issues

1. **Inconsistent ADX**: 9 reimplementations, ±5% WR drift
2. **Fee double-counting**: 4 files compute exit fee on entry price
   instead of exit price (~$50–$200/month error per backtest)
3. **SYMBOL_SIDE_BLOCKLIST not backported**: 17 mid-April scripts will
   produce false positives on current production
4. **Signal context fuzzy match**: 60s tolerance silently fails 5–10%
   of trades, no warning if match rate < 95%
5. **No centralized config**: each script hard-codes constants → drift

## Proposed Architecture

```
src/backtest/
├── runner.py        BacktestRunner: one config → one result set
├── sweep.py         SweepRunner: param grid expansion + parallel fan-out
├── config.py        YAML loader, schema validation, defaults merge
├── data/
│   ├── kline_cache.py   parquet-backed disk cache, day-keyed
│   └── replay.py        bar-by-bar multi-symbol/multi-interval iterator
├── strategy/
│   ├── base.py          Strategy ABC: on_bar(ctx) → Intent
│   ├── components.py    EntryRule | GuardChain | ExitPolicy primitives
│   └── library/         built-in strategies (structure_first_v3, …)
├── guards/          one file per guard (adx, usdtd, funding, whale, …)
├── exits/           partial_tp, trailing, stagnant, reverse_close
├── sim/
│   ├── broker.py        fills, fees, slippage, SL/TP order book
│   └── portfolio.py     multi-symbol equity, margin, max-notional cap
├── results/         parquet trades + json summary + csv equity
└── walkforward.py   train/test fold splitter
```

Reused unchanged:
- `src/signals/bos_choch.py::MarketStructure`
- `src/features/htf_features.py::HTFFeatureEngine`
- `src/features/regime_detector.py`
- `src/data/multi_asset_fetcher.py`

## Strategy Abstraction (one example)

```python
@register_strategy("structure_first_v3")
class StructureFirstV3(Strategy):
    entry = AnyOf(
        BOSEntry(htf_confirm=True, min_confidence=0.65),
        OBEntry(min_imbalance=0.7),
    )
    guards = GuardChain([
        ADXGuard(min_adx=20),
        USDTDGuard(threshold=0.7),
        FundingLongGuard(max_funding=0.05),
        WhaleNeutralGuard(),
        ExtPosNewsGuard(),
    ])
    exits = ExitPolicy(
        partials=[(1.0, 0.40), (2.0, 0.35)],
        trail=TrailingStop(activation_r=2.0, distance_pct=0.3),
        stagnant=StagnantExit(hours=6, pnl_band=(-0.3, 0.5)),
        reverse_close=ReverseClose(side="long", canary=True),
    )
    sizing = FixedNotional(usd=3000)
```

## Config Schema (excerpt)

```yaml
run_id: usdtd_sweep_2026_05
window: { start: 2026-02-01, end: 2026-04-30 }
symbols: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]
intervals: { primary: 5m, htf: [15m, 1h, 4h], context: 1m }
seed: 42

strategy: structure_first_v3
guards:
  enabled: [adx, usdtd, funding_long, whale_neutral, ext_pos_news]

sweep:
  mode: grid    # grid | random | walkforward
  axes:
    - { path: guards.params.usdtd.threshold,
        values: [0.50,0.55,...,1.00] }
  parallel: 1   # server has 2 CPU
```

CLI: `python3 backtest.py --config configs/sweeps/usdtd_threshold.yaml`

## Migration Plan (3 weeks, ~30 scripts retired)

**Phase 1 (week 1) — establish parity.**
Build core harness. Port `backtest_realistic.py` into `library/structure_first_v3`. Add the 4 missing guards + REVERSE_CLOSE_LONG canary. Validate parity within 0.5% of legacy output. Retire 8 trivial filter-ablation scripts.

**Phase 2 (week 2) — sweep migration.**
Express ~14 sweep-style scripts as YAML configs (USDT.D, news permutations, OB guards, TP variants, conditional TP, phase3 filters, max-hold, extreme-pos-news, whale-flow-news, OI filter, etc.). Retire 14 scripts.

**Phase 3 (week 3) — walk-forward + counterfactuals.**
Add fold splitter. Port the 8 remaining scripts that share folds. Preserve counterfactual scripts (`audit_portfolio_bug`, `analyze_trades`, etc.) and add `src/backtest/counterfactual.py` for future MFE/MAE counterfactuals.

## Critical Risks (and mitigations)

1. **Live divergence** → extract guard constants into shared `live_constants.py`; CI test asserts replay matches live SQLite log within tolerance
2. **Lookahead bias** → `WindowedReplay` exposes only `bars[:cursor]`; CI test shifts a feature one bar, asserts identical output
3. **Partial TP simulation bugs** → explicit broker order book; unit-test against hand-computed 40/35/25 ladder fixture
4. **OOM on 3.7 GB server** → `--max-jobs 1` default, lazy windowed loading, per-cell subprocess recycle, peak-RSS assertion
5. **Non-determinism** → explicit seed, canonical guard ordering, frozen dtypes, `config.resolved.yaml` with git SHA

## Non-Goals (explicit)

- No model training (training stays on Chen's local Mac)
- No order-book microstructure beyond static slippage_bps
- No latency simulation (next-bar-open fills)
- No automatic retirement of legacy scripts — phase migrations are
  manual reviews
- No live UI (output is files; viz stays in notebooks)

## Cost / Effort Estimate

- **Phase 1 (parity harness + structure_first_v3)**: ~2-3 days focused work
- **Phase 2 (port sweeps)**: ~3-4 days
- **Phase 3 (walk-forward + counterfactual API)**: ~2-3 days
- **Total**: 1.5-2 weeks of dedicated effort to get the existing 30+
  scripts onto the new framework

## Decision Required

Do we (a) green-light the 1.5-2 week redesign sprint, (b) do a smaller
"Phase 1 only" sprint to get current-production parity (3 days, fixes
the most painful divergence), or (c) keep firefighting per-experiment
in the current sprawl?

Recommendation: **Phase 1 only** is the highest-leverage start. After
that, decide whether to continue based on actual usage friction.
