# Trade Alert System

Direct Telegram trade alerts with **zero AI token usage**.

## Architecture

```
Trading Bots (htf/partial/hybrid)
        │
        ▼
logs/htf_pending_alerts.jsonl  ← append-only JSONL file
        │
        ▼
trade_alerter.py (systemd service)
        │
        ▼
Telegram Bot API → Your Telegram group
```

The trading bots write trade events as JSON lines to a shared file.
A lightweight Python daemon (`trade_alerter.py`) watches the file for new
lines, formats them into Telegram messages, and sends them via the Bot API.

No AI models, no LLM calls, no token usage. Pure Python → HTTP POST.

## Alert Types

### Trade Opens
```
🟢 OPEN LONG — BTCUSDT

💰 Entry: $70,349.99
🎯 TP: $72,460.49 (+3.0%)
🛑 SL: $69,294.74 (-1.5%)
📊 Confidence: 96%
📈 Strategy: Hybrid

🐋 Whale: BULLISH (50%)
💱 Funding: Short-favored (0.0055)
📊 Order Flow: Bullish (5B/0S)

💼 Trade Value: $37,575.03
🕐 2026-03-23 12:31 UTC
```

### Trade Closes
```
✅ CLOSE LONG — BTCUSDT (PROFIT)

💰 Entry: $68,349.40
🏁 Exit: $70,399.98 (+3.0%)
📈 PnL: +$246.56
📋 Reason: TP
📈 Strategy: Hybrid

🕐 2026-03-23 11:06 UTC
```

### Partial Closes
```
🔶 PARTIAL CLOSE 1 — BTCUSDT LONG

💰 Entry: $70,349.99
🏁 Exit: $71,405.35 (+1.5%)
📈 PnL: +$274.10
📦 Remaining: 50% of position
🛑 SL moved: $69,294.74 → $70,349.99
📍 Trailing stop: ACTIVE
📈 Strategy: Hybrid

🕐 2026-03-23 13:44 UTC
```

### SL/TP Updates
```
🔄 SL Updated — BTCUSDT LONG
$69,294.74 → $70,349.99 (+1.5%)
📋 Reason: Trailing (peak=$71,405.35)
📈 Strategy: Hybrid
🕐 2026-03-23 13:44 UTC
```

## Configuration

Environment variables (set in `.env`):
- `TELEGRAM_BOT_TOKEN` — Bot token from @BotFather
- `TELEGRAM_CHAT_ID` — Target chat/group ID (negative for groups)

## Service Management

```bash
# Check status
systemctl status drl-trade-alerter

# View logs (live)
journalctl -u drl-trade-alerter -f

# Restart
systemctl restart drl-trade-alerter

# Stop
systemctl stop drl-trade-alerter
```

## Files

| File | Purpose |
|------|---------|
| `trade_alerter.py` | Main alerter daemon |
| `drl-trade-alerter.service` | systemd unit file |
| `logs/htf_pending_alerts.jsonl` | Shared alert queue (append-only) |
| `logs/.alerter_offset` | File offset tracker (don't delete) |

## Adding New Alert Types

1. In the trading bot, write a JSON line to `logs/htf_pending_alerts.jsonl`:
```python
import json
from pathlib import Path
from datetime import datetime

alert_file = Path("logs/htf_pending_alerts.jsonl")
with open(alert_file, "a") as f:
    f.write(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "strategy": "your_strategy",
        "trade": {
            "action": "YOUR_ACTION",
            "symbol": "BTCUSDT",
            "price": 70000.0,
            # ... your fields
        },
        "signals": {},
        "position": { ... },
    }) + "\n")
```

2. In `trade_alerter.py`, add a formatter function and register it in `format_alert()`.

## Troubleshooting

**Service won't start:**
```bash
journalctl -u drl-trade-alerter -n 50
# Check for missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID
```

**Alerts not sending:**
```bash
# Verify the bot token
curl https://api.telegram.org/bot<TOKEN>/getMe

# Verify the chat ID
curl https://api.telegram.org/bot<TOKEN>/getChat?chat_id=<CHAT_ID>

# Check the offset file hasn't gotten ahead of the actual file
wc -c logs/htf_pending_alerts.jsonl
cat logs/.alerter_offset
```

**Duplicate alerts after restart:**
- The offset file tracks position. If deleted, all existing alerts will re-send.
- If you WANT to re-send: `echo 0 > logs/.alerter_offset && systemctl restart drl-trade-alerter`

**File permissions:**
```bash
ls -la logs/htf_pending_alerts.jsonl
ls -la logs/.alerter_offset
# Both should be readable/writable by root (service runs as root)
```
