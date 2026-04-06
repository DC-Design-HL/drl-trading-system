# CLAUDE.md — Instructions for Claude Code

## Project: DRL Trading System

A Deep Reinforcement Learning crypto trading system running on Binance Futures testnet.
4 trading bots (BTC, ETH, SOL, XRP) running 24/7 as systemd services on a Hetzner VPS.

## Quick Start

```bash
# Project root
REPO=/home/claude/packages/327adce6-6ec4-4402-890c-9d12c6e8a471/workspace/drl-trading-system
cd $REPO

# Start / restart ALL services (bots, alerter, API, UI, tunnel) — session-independent
./start_services.sh

# Check running services
cat logs/running_services.json

# Check if processes are alive
ps aux | grep -E "live_trading_htf|trade_alerter|start_local_server|streamlit|localtunnel" | grep -v grep

# View bot logs
tail -f logs/btc_live.log   # or eth/sol/xrp_live.log
tail -f logs/alerter.log
tail -f logs/api_server.log
tail -f logs/dashboard.log
tail -f logs/tunnel.log     # contains public dashboard URL

# Push changes
git add -A && git commit -m "description" && git push origin dev
```

## Process Management

All services run via `setsid` — fully detached from any shell/SSH/Claude session.
On reboot, they auto-restart via `@reboot` crontab entry.

**To restart after a code change:**
```bash
./start_services.sh   # kills old instances, relaunches with setsid
```

**Services managed by start_services.sh:**
| Service | Script | Log |
|---------|--------|-----|
| API Server | `start_local_server.py` | `logs/api_server.log` |
| Bot BTC | `live_trading_htf.py --symbol BTCUSDT` | `logs/btc_live.log` |
| Bot ETH | `live_trading_htf.py --symbol ETHUSDT` | `logs/eth_live.log` |
| Bot SOL | `live_trading_htf.py --symbol SOLUSDT` | `logs/sol_live.log` |
| Bot XRP | `live_trading_htf.py --symbol XRPUSDT` | `logs/xrp_live.log` |
| Trade Alerter | `trade_alerter.py` | `logs/alerter.log` |
| Streamlit UI | `src/ui/app.py` (port 8501) | `logs/dashboard.log` |
| Tunnel | `localtunnel --port 8501` | `logs/tunnel.log` |

Note: localtunnel assigns a random subdomain each run — check `logs/tunnel.log` for current URL.

## Critical Rules (MUST READ)

1. **NEVER run model training on this server** — 2 CPUs, 3.7GB RAM, no swap. Training = OOM = all bots die. Training only on Chen's Mac M3 Pro.
2. **NEVER push to main** — always `dev` branch only.
3. **NEVER open a position without SL and TP** — if TP placement fails, immediately close the position. This is the #1 rule.
4. **ALL UI/testnet data = real Binance API data** — zero local calculations, zero mock data, zero hardcoded values.
5. **Storage = SQLite** (`data/trading.db`). MongoDB Atlas is broken (DNS unreachable). Don't try to fix it unless asked.
6. **After ANY code change**: run `./start_services.sh` to restart all services.
7. **Git workflow**: always work on `dev`, push to `origin dev`.

## Documentation

- **Full handoff doc**: `docs/HANDOFF.md` — architecture, services, guards, risk management, everything
- **Memory / history**: `docs/MEMORY.md` — all accumulated knowledge, decisions, preferences
- **Task tracker**: `docs/TASKS.md` — all tasks with status
- **Skills/rules**: `docs/skills/*.md` — detailed rules for each subsystem
- **Session logs**: `docs/memory-logs/*.md` — daily work logs
- **Systemd services**: `deploy/systemd/*.service` — all service unit files
- **Caddy config**: `deploy/Caddyfile` — reverse proxy setup

## Architecture

- **4 Trading Bots**: BTC, ETH, SOL, XRP — each runs `live_trading_htf.py` with `--symbol` flag
- **Flask API** (port 5001): `start_local_server.py`
- **Streamlit UI** (port 8501): `src/ui/app.py`
- **Caddy** (port 80): reverse proxies to Streamlit, API under `/api/*`
- **Trade Alerter**: `trade_alerter.py` → sends alerts to Telegram via separate bot token
- **Whale WS**: `whale_behavior_ws.py` → real-time whale wallet tracking via Alchemy
- **Championship Shadow**: `championship_shadow.py` → compares live vs championship model

## Trading Guards (execution order)

1. **ADX Guard** (ADX_GUARD_MIN=20) — blocks ranging markets
2. **Signal Gate** — Tier 1 (conf≥0.80): autonomous. Tier 2: needs 2/4 signals
3. **Orderbook Guard** — blocks when orderbook contradicts direction
4. Rescue Override: DISABLED

## Risk Management

- Partial TP: 40% at 1R → 35% at 2R → 25% trails
- ATR floor: SL ≥ 1.5×ATR, TP ≥ 3.0×ATR
- Trailing: activation +0.5%, distance 0.3%
- Max $3,000 USDT notional per trade
- Stagnant exit: 6h if PnL in [-0.3%, +0.5%]

## Env Vars

All in `.env` file. Key ones:
- `BINANCE_FUTURES_API_KEY/SECRET` — Futures testnet
- `STORAGE_TYPE=sqlite`
- `TESTNET_MIRROR=true`
- `TELEGRAM_ALERT_BOT_TOKEN` — for trade alerts (separate from AI bot)

## Pending Decisions

- **ADX Exhaustion Guard** (ADX>60 block) — see `docs/adx-exhaustion-guard-proposal.md`
- **Orderbook Guard review** — due Apr 7, tiered sizing if good
- **Wyckoff** — ON HOLD, need more labeled data
- **Whale model** — needs retraining on Mac M3

## Server

- IP: `116.203.196.107`
- Hetzner VPS, Ubuntu, 2 CPU, 3.7GB RAM
- Python 3.x, PyTorch 2.10 (CPU), stable-baselines3 2.7.1
