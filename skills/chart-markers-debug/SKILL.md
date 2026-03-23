# Chart Markers Debug Skill

## Overview

The Live Chart uses **TradingView lightweight-charts v4** to render candlestick
data with trade markers (entry/exit arrows, squares) overlaid on candles.

## Data Flow

```
/api/testnet/trades (Binance fills)
  → Python normalisation (action: OPEN_LONG | CLOSE_LONG | OPEN_SHORT | CLOSE_SHORT)
  → create_tradingview_chart_with_websocket()
    → _snap_to_candle() — snaps trade timestamps to nearest candle open-time
    → Per-candle deduplication (aggregate many trades → max 2 markers/candle)
    → JSON-encode into HTML <script>
  → JavaScript: candlestickSeries.setMarkers(markers)
```

## Key Files

| File | Lines | What |
|------|-------|------|
| `src/ui/app.py` | ~408-600 | `create_tradingview_chart_with_websocket()` — marker generation |
| `src/ui/app.py` | ~344-405 | `_classify_exit_reason()` — SL/TP classification |
| `src/ui/api_server.py` | ~807-870 | `/api/testnet/trades` — normalises Binance fills |
| `tests/test_chart_markers.py` | — | Unit + integration tests |

## Common Failure Modes

### 1. Missing CLOSE actions
**Symptom**: No exit markers at all.
**Check**: `curl -s http://127.0.0.1:5001/api/testnet/trades?symbol=BTCUSDT&limit=10 | python3 -m json.tool`
**Look for**: `action` field must contain `CLOSE_LONG` or `CLOSE_SHORT`.
The API derives this from `side + realizedPnl != 0`.

### 2. Marker times don't match candle times
**Symptom**: Markers silently dropped (chart shows nothing).
**Cause**: lightweight-charts v4 requires every marker `time` to exactly match a
candle `time`. The `_snap_to_candle()` function handles this via bisect.
**Check**: If trades are far outside the chart's candle range, they'll snap to
edge candles or be dropped.

### 3. Markers not sorted ascending
**Symptom**: ALL markers silently dropped (not just unsorted ones).
**Cause**: lightweight-charts v4 drops the entire marker array if any marker has
a time less than a previous marker's time.
**Fix**: `markers.sort(key=lambda m: m['time'])` — done in Python AND JavaScript.

### 4. Too many markers at same candle (stacking overflow)
**Symptom**: Entries show but exits are invisible (pushed off visible area).
**Cause**: When 20+ scalp trades happen within one candle, stacked markers
overflow the chart viewport.
**Fix**: Per-candle deduplication — aggregate into `LONG×N`, `EXIT×N` labels.
Max 2 markers per candle (1 entry + 1 exit).

### 5. `reason` field is None
**Symptom**: CLOSE trades silently skipped due to `None.lower()` → AttributeError
caught by bare `except`.
**Fix**: Use `str(trade.get('reason') or '').lower()` instead of
`trade.get('reason', '').lower()`.

### 6. Circle shape too small
**Symptom**: EXIT markers technically render but are barely visible.
**Fix**: Use `shape: 'square'` for all exit markers (more visible than `circle`).

## Debug Commands

```bash
# Check API returns CLOSE trades
curl -s http://127.0.0.1:5001/api/testnet/trades?symbol=BTCUSDT&limit=10 | \
  python3 -c "import sys,json; trades=json.load(sys.stdin)['trades']; \
  print('Actions:', set(t['action'] for t in trades))"

# Run marker tests
cd /root/.openclaw/projects/drl-trading-system/repo
python3 -m pytest tests/test_chart_markers.py -v

# Generate and inspect HTML markers offline
python3 -c "
import requests, json, re, sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.ui.app import create_tradingview_chart_with_websocket
resp = requests.get('http://127.0.0.1:5001/api/ohlcv', params={'symbol':'BTCUSDT','interval':'1h','limit':500}, timeout=10)
df = pd.DataFrame(resp.json()); df.index = pd.to_datetime(df['time'], unit='s'); df = df[['open','high','low','close','volume']]
trades = requests.get('http://127.0.0.1:5001/api/testnet/trades?symbol=BTCUSDT&limit=100', timeout=10).json()['trades']
html = create_tradingview_chart_with_websocket(df, trades, '1h', 'BTC/USDT')
markers = json.loads(re.search(r'const markers = (\[.*?\]);\s*\n', html, re.DOTALL).group(1))
for m in markers: print(f\"  {m['text']:12s} time={m['time']} pos={m['position']} shape={m['shape']}\")
print(f'Total: {len(markers)} markers ({sum(1 for m in markers if \"EXIT\" in m[\"text\"])} exits)')
"
```

## Marker Visual Reference

| Trade | Shape | Color | Position |
|-------|-------|-------|----------|
| LONG entry | arrowUp ▲ | #26a69a (green) | belowBar |
| SHORT entry | arrowDown ▼ | #ef5350 (red) | aboveBar |
| EXIT(SL) | square ■ | #ff4444 (bright red) | aboveBar |
| EXIT(TP) | square ■ | #00e676 (bright green) | belowBar |
| EXIT (generic) | square ■ | #ffc107 (amber) | aboveBar |
| Aggregated EXIT | square ■ | varies | aboveBar |
