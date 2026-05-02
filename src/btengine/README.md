# btengine — Unified Backtest Framework

Composable, reproducible, multi-symbol backtest engine for the DRL
trading system. Replaces the sprawl of one-off `backtest_*.py` scripts
with a single YAML-configurable runner.

See `docs/backtest_redesign_proposal.md` for the full audit findings
and migration plan.

## Quick start

```bash
# Dry-run validates config + probes data without simulating
python3 backtest.py --config configs/sweeps/example_dry_run.yaml --dry-run

# Full run produces runs/<run_id>/{trades,blocked}.parquet +
#                                  summary.json + equity.csv
python3 backtest.py --config configs/sweeps/last_14d_parity.yaml
```

## Architecture

```
src/btengine/
├── backtest.py (CLI)        ← entry point at repo root
├── config.py                ← YAML loader + dataclass schema
├── live_constants.py        ← ONE source of truth for production constants
├── runner.py                ← BacktestRunner orchestration
├── data/kline_cache.py      ← Parquet day-keyed cache (today's bar live)
├── strategy/
│   ├── base.py              ← Strategy ABC + EntryRule/Guard/ExitPolicy
│   ├── indicators.py        ← Canonical Wilder ADX/RSI/ATR
│   └── library/
│       └── structure_first.py   ← BOS/CHOCH-driven entry, mirrors live
├── guards/                  ← 7 production guards
│   ├── symbol_blocklist.py
│   ├── adx.py
│   ├── usdt_d.py
│   ├── funding_long.py
│   ├── whale_neutral.py
│   ├── ext_pos_news.py
│   └── reverse_close_long.py    ← asymmetric May-1 canary
├── sim/
│   ├── context.py           ← Per-bar Ctx with strict htf_up_to_now()
│   ├── replay.py            ← Multi-symbol bar interleaving
│   ├── position.py          ← Position lifecycle (mirrors live state)
│   └── broker.py            ← Slippage + fees + SL/partial-TP/trail/stagnant
└── results/writer.py        ← parquet/json/csv output
```

## Config schema (excerpt)

```yaml
run_id: my_run
window: { start: 2026-04-15, end: 2026-05-01 }
symbols: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT]
intervals:
  primary: 15m
  htf: [1h, 4h]
strategy: structure_first_v3
guards:
  enabled: [symbol_blocklist, adx, usdtd, funding_long, whale_neutral, ext_pos_news]
  params:
    adx: { min_adx: 20, max_adx: 60 }
sizing: { type: fixed_notional, usd: 1500, max_concurrent: 4 }
fees: { taker: 0.0004, slippage_pct: 0.0005 }
output: { dir: 'runs/${run_id}' }
```

## Live parity

`tests/test_btengine_m5_parity.py` replays the last 14 days through
the framework and asserts the result matches the SQLite trade log
within tolerance.

Latest measurement (2026-05-02, Apr 18 → May 1):

| metric | live (SQLite) | btengine | delta |
|---|---|---|---|
| closed trades | 114 | 76 | -33% |
| win rate | 48.2% | 47.4% | -0.8pp |
| total PnL | $-429.43 | $-380.41 | +11% |
| STAGNANT exits | 24 | 24 | exact |

Trade-count gap is dominated by Phase 1 simplification: btengine uses
the primary timeframe (15m) for structure detection, while live uses
5m. Phase 2 may add multi-resolution structure if needed.

## Adding a new strategy

```python
# src/btengine/strategy/library/my_strategy.py
from ..base import Strategy, EntryRule, Intent, register_strategy

class MyEntry(EntryRule):
    def __call__(self, ctx) -> Intent:
        if some_condition(ctx):
            return Intent(action="OPEN_LONG", confidence=0.8, reason="my_rule")
        return Intent(action="HOLD")

@register_strategy("my_strategy")
class MyStrategy(Strategy):
    def __init__(self, **overrides):
        self.entry = MyEntry()
```

Then in your YAML config: `strategy: my_strategy`.

## Adding a new guard

```python
# src/btengine/guards/my_guard.py
from ..strategy.base import Guard, GuardResult, Intent

class MyGuard(Guard):
    name = "my_guard"
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    def __call__(self, intent: Intent, ctx) -> GuardResult:
        if intent.action != "OPEN_LONG":
            return GuardResult.allow()
        if some_check(ctx) > self.threshold:
            return GuardResult.block(f"my_guard: bad")
        return GuardResult.allow()
```

Register in `guards/__init__.py::GUARD_CLASSES`.
