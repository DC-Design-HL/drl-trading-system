# DRL Trading System — Full Handoff Document

**Date**: April 4, 2026  
**From**: OpenClaw CEO Agent → Claude Code (direct)  
**Owner**: Chen Chen (@Chen4700)

---

## 1. Server Info

- **Host**: `oc-luigi-a9c232bb` (Hetzner VPS)
- **IP**: `116.203.196.107`
- **OS**: Ubuntu, Linux 6.8.0-106-generic (x86_64)
- **Resources**: 2 CPUs, 3.7GB RAM, 38GB disk (28GB used, 8.2GB free)
- **No swap** — be careful with memory-hungry operations
- **⚠️ NEVER run model training on this server** — OOM kills everything. Training happens on Chen's Mac M3 Pro only.

## 2. Project Location

```
~/.openclaw/projects/drl-trading-system/repo/
```

- **Git branch**: `dev` (always work on dev, NEVER push to main unless Chen says so)
- **Remote**: `origin` → `github.com/DC-Design-HL/drl-trading-system.git`
- **HuggingFace remote**: `hf-dev2` (no longer used — UI is self-hosted)

## 3. Running Services (systemd)

All 9 services are currently **active and running**:

| Service | Description | Command |
|---------|-------------|---------|
| `drl-htf-agent` | BTC HTF trading bot | `live_trading_htf.py --live --interval 15` |
| `drl-htf-eth` | ETH HTF trading bot | `live_trading_htf.py --live --interval 15 --symbol ETHUSDT` |
| `drl-htf-sol` | SOL HTF trading bot | `live_trading_htf.py --live --interval 15 --symbol SOLUSDT` |
| `drl-htf-xrp` | XRP HTF trading bot | `live_trading_htf.py --live --interval 15 --symbol XRPUSDT` |
| `drl-trade-alerter` | Telegram trade alerts | `trade_alerter.py` |
| `drl-trading-server` | Flask API server (port 5001) | `start_local_server.py` |
| `drl-trading-ui` | Streamlit UI (port 8501) | `streamlit run src/ui/app.py` |
| `drl-whale-ws` | Whale wallet WebSocket collector | `whale_behavior_ws.py` |
| `drl-championship-shadow` | Championship model shadow comparator | `championship_shadow.py --symbol BTCUSDT --interval 15` |

### Service Management
```bash
# Check status
systemctl status drl-htf-agent

# Restart a bot
systemctl restart drl-htf-eth

# View logs
journalctl -u drl-htf-agent -f --no-pager

# Service files location
/etc/systemd/system/drl-*.service
```

## 4. Web / Reverse Proxy

- **Caddy** reverse proxy at `/etc/caddy/Caddyfile`
- `http://116.203.196.107` → Streamlit UI (port 8501)
- API routes: `http://116.203.196.107/api/*` → Flask server (port 5001)
- OpenClaw domain: `luigi-a9c232bb.botitout.com` → port 37677 (OpenClaw gateway)

## 5. Environment Variables (.env)

Located at `~/.openclaw/projects/drl-trading-system/repo/.env`

Key variables:
- `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` — Spot testnet
- `BINANCE_FUTURES_API_KEY` / `BINANCE_FUTURES_API_SECRET` — Futures testnet
- `BINANCE_FUTURES_BASE_URL` — demo-fapi.binance.com
- `STORAGE_TYPE=sqlite` — Primary storage (MongoDB broken)
- `MONGO_URI` — MongoDB Atlas (currently broken, DNS unresolvable)
- `HF_TOKEN` — HuggingFace token
- `ETHERSCAN_API_KEY`, `SOLSCAN_API_KEY`, `HELIUS_API_KEY` — Blockchain APIs
- `NEWS_API_KEY`, `CRYPTOPANIC_TOKEN`, `CRYPTOCOMPARE_API_KEY` — News sentiment
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — AI bot Telegram
- `TELEGRAM_ALERT_BOT_TOKEN` — Separate alert bot (not the AI bot)
- `ALCHEMY_API_KEY` — Whale WebSocket tracking
- `TESTNET_MIRROR=true` — Bot decisions mirror to Binance Futures testnet
- `ENVIRONMENT=testnet`

## 6. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   VPS Server                     │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Flask API     │  │ 4× HTF Trading Bots      │ │
│  │ (port 5001)  │  │ BTC, ETH, SOL, XRP       │ │
│  └──────┬───────┘  │ (live_trading_htf.py)     │ │
│         │          └──────────────────────────┘ │
│  ┌──────┴───────┐                               │
│  │ Streamlit UI  │  ┌──────────────────────────┐ │
│  │ (port 8501)  │  │ Support Services          │ │
│  └──────────────┘  │ - Trade Alerter           │ │
│                    │ - Whale WS Collector       │ │
│  ┌──────────────┐  │ - Championship Shadow     │ │
│  │ Caddy Proxy  │  └──────────────────────────┘ │
│  │ (port 80)    │                               │
│  └──────────────┘  ┌──────────────────────────┐ │
│                    │ Storage: SQLite            │ │
│                    │ (data/trading.db)          │ │
│                    └──────────────────────────┘ │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐  ┌──────────────────┐
│ Binance Futures  │  │ Telegram Alerts   │
│ Testnet          │  │ (Luigi_Trader)    │
└─────────────────┘  └──────────────────┘
```

## 7. Trading System Logic

### HTF Bot (`live_trading_htf.py`)
- Hierarchical multi-timeframe: 1D → 4H → 1H → 15M
- QRDQN model trained via walk-forward validation
- 15-minute decision intervals
- 50% position sizing

### Guards & Filters (in order of execution)
1. **ADX Guard** (`ADX_GUARD_MIN = 20`) — blocks trades when ADX < 20 (ranging market)
2. **Signal Gate** — Tier 1 (conf ≥ 0.80): autonomous. Tier 2 (conf < 0.80): needs 2/4 signal confirmations
3. **Orderbook Guard** (`ORDERBOOK_GUARD_ENABLED = True`) — blocks trades when orderbook bias contradicts direction
4. **Rescue Override**: DISABLED (`RESCUE_ENABLED = False`)

### Risk Management
- Partial TP: 40% at 1R → 35% at 2R → 25% trails
- ATR floor: SL ≥ 1.5×ATR, TP ≥ 3.0×ATR
- Trailing: activation at +0.5%, distance 0.3%
- Max $3,000 USDT notional per trade
- Stagnant exit: close after 6h if PnL in [-0.3%, +0.5%]
- **CRITICAL**: Every position MUST have SL and TP. If TP placement fails, close position immediately.

### Signal Sources
- MTF (Multi-Timeframe alignment)
- Order Flow analysis
- Regime detection (trending/ranging/volatile)
- Orderbook depth/bias
- Whale behavior (DISPLAY ONLY — does not affect decisions)
- BOS/CHOCH structure breaks (dynamic SL/TP adjustment)

### BOS/CHOCH Clean-Line Validation
- Structure break: Candle A's wick → Candle B's body
- Line must be clean (no intermediate wick intersections)
- If any wick crosses → break is invalid
- Implemented in `src/signals/bos_choch.py` → `_is_clean_break()`

## 8. Key Files

| File | Purpose |
|------|---------|
| `live_trading_htf.py` | Main trading bot (all 4 symbols) |
| `start_local_server.py` | Flask API server |
| `src/ui/app.py` | Streamlit dashboard |
| `trade_alerter.py` | Telegram trade alerts |
| `whale_behavior_ws.py` | Real-time whale wallet monitoring |
| `championship_shadow.py` | Shadow comparison of championship model |
| `src/signals/bos_choch.py` | BOS/CHOCH structure break detection |
| `src/features/htf_features.py` | 117-dim feature engine across 4 timeframes |
| `src/env/htf_env.py` | Multi-TF gymnasium environment |
| `src/brain/htf_agent.py` | Hierarchical agent with curriculum training |
| `src/data/storage.py` | Storage backend (SQLite/MongoDB/JSON) |
| `src/risk/risk_manager.py` | Risk management with ATR floors |
| `src/whale_behavior/models/predictor.py` | Whale behavior prediction (display only) |
| `src/features/wyckoff_detector.py` | Wyckoff detector (ON HOLD) |

## 9. Pending / In-Progress Items

### ADX Exhaustion Guard — Decision Due Apr 4
- Proposal: block trades when ADX > 60 (saves ~$93, loses ~$12)
- ADX 30-50 sweet spot (67-71% WR), ADX > 60 terrible (29% WR)
- Doc: `docs/adx-exhaustion-guard-proposal.md`

### Orderbook Guard Review — Apr 7
- Review 1 week of orderbook guard performance
- If good → deploy tiered sizing in shadow mode
- Apr 14: review shadow → enable actual tiered sizing

### Wyckoff — ON HOLD
- Detector built and tested but not useful as guard yet
- Need 500+ labeled trades (2-3 months)
- Full analysis: `docs/wyckoff-analysis-summary.md`

### Whale Model — Needs Retraining
- Current model not predictive (SELL never reaches threshold)
- 1,750 new labeled sequences available in `dev`
- Training must happen on Mac M3 Pro
- Robinhood wallet dead since Feb 14

### MongoDB — BROKEN
- Atlas DNS unresolvable (free tier likely paused)
- System uses SQLite as default (`STORAGE_TYPE=sqlite`)
- To restore: fix Atlas cluster + IP whitelist + set `STORAGE_TYPE=mongo`

### Championship Shadow — COLLECTING DATA
- Running comparison between live model and championship model
- Need ~50+ data points before analysis

## 10. Python Dependencies

Key packages installed:
- `torch 2.10.0+cpu` (CPU only — no GPU on this server)
- `stable_baselines3 2.7.1`
- `gymnasium 1.2.3`
- `ccxt 4.5.44`
- `Flask 3.1.3`
- `streamlit 1.55.0`
- `pandas 2.3.3`, `numpy 2.4.3`
- `websocket-client 1.9.0`

Full requirements: `requirements.txt` in repo root

## 11. Critical Rules

1. **NEVER run training on the server** — 2 CPUs + 3.7GB RAM = OOM death
2. **NEVER push to main** — always work on `dev` branch
3. **NEVER open a position without SL and TP** — close immediately if TP fails
4. **ALL testnet UI data must come from real Binance API** — zero local calculations
5. **NO mock/fake/hardcoded data** anywhere in the system
6. **Storage is SQLite** (`data/trading.db`) — MongoDB is broken
7. **After any code change**: restart the affected systemd service(s)
8. **Git workflow**: edit → commit → `git push origin dev`

## 12. Trade Analysis Summary (as of Apr 2)

- **Period**: Mar 29 - Apr 2 (53 trades)
- **Win Rate**: 53%
- **PnL**: -$218
- **Best ADX band**: 30-50 (67-71% WR, profitable)
- **Worst ADX bands**: <20 (29% WR) and >60 (29% WR)
- **Average MFE on winners**: ~1.4%, capturing ~60%
- **New trailing params** should improve capture to ~70-75%

## 13. Skills / Documentation

All skill docs are in `docs/skills/` (being copied from OpenClaw workspace):

- Trading workflow, position management, risk models
- Signal gate architecture, orderbook guard
- BOS/CHOCH, Wyckoff implementation
- Testnet validation, position sync
- Storage backend rules
- UI type safety, no-mock-data policy

---

*This document was generated automatically from the OpenClaw CEO agent's full memory, tasks, skills, and server state. It contains everything needed to continue development of the DRL trading system.*
