# Position Management Rules

Rules for managing trading positions in the DRL trading system. These are non-negotiable.

---

## Rule 1: One Position Per Asset — ALWAYS

Each asset (BTCUSDT, ETHUSDT, etc.) can have **at most ONE** active position at any time.

### What this means:
- If the bot is LONG BTCUSDT and the model says LONG → **do nothing** (hold). Do NOT open a new position or log a new trade.
- If the bot is SHORT ETHUSDT and the model says SHORT → **do nothing** (hold).
- Only execute when there is an actual state CHANGE:
  - FLAT → LONG (open)
  - FLAT → SHORT (open)
  - LONG → FLAT (close)
  - SHORT → FLAT (close)
  - LONG → SHORT (close then open)
  - SHORT → LONG (close then open)

### How to verify:
```bash
# Check exchange — each symbol should have at most 1 position
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import os, sys; sys.path.insert(0, '.')
from src.api.binance_futures import BinanceFuturesConnector
c = BinanceFuturesConnector(
    api_key=os.environ['BINANCE_FUTURES_API_KEY'],
    api_secret=os.environ['BINANCE_FUTURES_API_SECRET'],
    base_url=os.environ.get('BINANCE_FUTURES_BASE_URL', 'https://demo-fapi.binance.com'))
positions = [p for p in c.get_positions() if float(p.get('positionAmt', 0)) != 0]
for p in positions:
    print(f'{p[\"symbol\"]}: amt={p[\"positionAmt\"]}')
print(f'Total: {len(positions)} positions')
"
```

### Common violations:
- `_mirror_testnet()` called on every iteration instead of only on state changes
- Trade logged to `htf_trades.json` when holding (no actual trade happened)
- Bot doesn't check `state['position']` before executing

---

## Rule 2: Per-Symbol State Files

Each asset MUST have its own state file. No sharing.

| Asset | State file |
|-------|-----------|
| BTCUSDT | `logs/htf_trading_state.json` (backward compat default) |
| ETHUSDT | `logs/htf_trading_state_ETHUSDT.json` |
| Other | `logs/htf_trading_state_{SYMBOL}.json` |

The state file path MUST be derived from the `--symbol` argument in `live_trading_htf.py`.

### What goes wrong without this:
- BTC bot writes position=1 (LONG)
- ETH bot overwrites with position=-1 (SHORT)
- BTC bot reads SHORT state, thinks it needs to close, chaos ensues

---

## Rule 3: Never Open Without SL and TP

Every position MUST have:
1. **TP order** on exchange (LIMIT reduceOnly) — verified via `get_open_orders()`
2. **SL monitoring** via WebSocket (demo-fapi doesn't support STOP_MARKET)

If TP placement fails → **immediately close the position**. No unprotected positions ever.

See also: `skills/testnet-validation-checklist/SKILL.md`

---

## Rule 4: Paper Trade and Testnet Must Be In Sync

When the bot opens a position:
1. Paper trade state updates (local state file)
2. Testnet mirrors the SAME position on Binance Futures

Both should show the SAME position direction and roughly the same entry price. If they diverge, something is broken.

### How to verify sync:
```bash
# Compare state file vs exchange
cat logs/htf_trading_state_BTCUSDT.json | python3 -c "import sys,json; s=json.load(sys.stdin); print(f'Paper: pos={s[\"position\"]} price={s[\"position_price\"]}')"
# Then check exchange position for same symbol
```

---

## Validation Checklist

Before restarting bots after any code change:

- [ ] Each asset has its own state file
- [ ] State files are FLAT (position=0) if starting fresh
- [ ] Exchange has 0 positions and 0 open orders
- [ ] Code only executes trades on state CHANGES (not holds)
- [ ] TP order is verified after every position open
- [ ] Trade logs are clean (no stale entries)
