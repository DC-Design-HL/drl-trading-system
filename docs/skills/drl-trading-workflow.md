---
name: drl-trading-workflow
description: Workflow rules for the DRL Trading System project. Use when making any changes, deployments, or fixes to the drl-trading-system project. Covers deployment flow (dev vs prod), HuggingFace log monitoring, Binance testnet connectivity, local server requirements, and integration testing standards.
---

# DRL Trading System Workflow

## Server & Access

- **Server**: `116.203.196.107` (Hetzner VPS)
- **Repo**: `/root/.openclaw/projects/drl-trading-system/repo`
- **Git**: `https://github.com/DC-Design-HL/drl-trading-system.git`, branch `dev`
- **UI**: `http://116.203.196.107` (port 80 via Caddy → Streamlit 8501)
- **API**: port `5001` (Flask, bound to 0.0.0.0)

## Network / Ports

Hetzner Cloud Firewall only allows ports **80, 443, 22**. Do NOT expose services on custom ports (8501, 5001, etc.) directly — they won't be reachable externally.

**Routing through Caddy:**
- Port 80 → Streamlit UI (localhost:8501)
- Port 443 → OpenClaw gateway (localhost:37677)
- API port 5001 is internal only (UI connects via localhost)

If a new service needs external access, add a Caddy reverse proxy route on port 80/443 with path-based routing.

## Services (systemd)

| Service | Purpose | Port |
|---------|---------|------|
| `drl-htf-agent` | BTC trading bot | — |
| `drl-htf-eth` | ETH trading bot | — |
| `drl-trade-alerter` | Telegram alert service | — |
| `drl-trading-server` | Flask API server | 5001 |
| `drl-trading-ui` | Streamlit UI | 8501 |

Restart: `systemctl restart <service>`
Logs: `journalctl -u <service> --since "X min ago" --no-pager`

## Git Workflow

- Work on `dev` branch only
- NEVER push to `main`/`prod` unless Chen explicitly says so
- Commit after every fix, push to `origin dev`

## State Files

| File | Description |
|------|-------------|
| `logs/htf_trading_state.json` | BTC bot state |
| `logs/htf_trading_state_ETHUSDT.json` | ETH bot state |
| `logs/htf_pending_alerts.jsonl` | Alert queue |
| `logs/.alerter_offset` | Alerter position in queue |

## Testnet Rules (CRITICAL)

1. Internal bot triggers → opens REAL futures testnet position (not simulated)
2. Every position MUST have TP and SL configured on the exchange
3. ALL data in testnet UI tab = real data from Binance futures API, zero local calculations
4. **NEVER open a position without SL and TP** — if TP placement fails, immediately close the position
5. Initial balance baseline: **$5,000 USDT** (hardcoded for PnL calculation)

## Binance Testnet Quirks

These errors are testnet limitations, NOT real failures:
- `-4509`: TIF GTE can only be used with open positions
- `-2021`: Order would immediately trigger
- `-2022`: ReduceOnly Order is rejected

Bot-side WebSocket monitoring handles SL/TP when exchange algo orders fail.

## Models

| Symbol | Model Path |
|--------|-----------|
| BTC | `data/models/htf_walkforward_50pct_v2/fold_06/best_model.zip` |
| ETH | `data/models/htf_walkforward_eth/fold_08/best_model.zip` |

VecNormalize files: `fold_model_vecnorm.pkl` in same directory.

## Trading Config

```
MIN_CONFIDENCE=0.45
RANGING_MIN_CONFIDENCE=0.80
RANGING_ADX_THRESHOLD=20.0
EXHAUSTION_ATR_THRESHOLD=3.0
MIN_HOLD_SECONDS=3600 (with ≥0.70 reversal override)
STOP_LOSS_PCT=0.015
TAKE_PROFIT_PCT=0.030
```
