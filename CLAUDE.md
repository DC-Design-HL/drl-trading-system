# CLAUDE.md — Instructions for Claude Code

## Project: DRL Trading System

A Deep Reinforcement Learning crypto trading system running on Binance Futures testnet.
4 trading bots (BTC, ETH, SOL, XRP) running 24/7 as systemd services on a Hetzner VPS.

## Quick Start

```bash
# Project root
cd ~/.openclaw/projects/drl-trading-system/repo

# Check all services
systemctl status drl-htf-agent drl-htf-eth drl-htf-sol drl-htf-xrp drl-trade-alerter drl-trading-server drl-trading-ui drl-whale-ws

# View bot logs
journalctl -u drl-htf-agent -f --no-pager

# Restart after code changes
systemctl restart drl-htf-agent drl-htf-eth drl-htf-sol drl-htf-xrp

# Push changes
git add -A && git commit -m "description" && git push origin dev
```

## Critical Rules (MUST READ)

1. **NEVER run model training on this server** — 2 CPUs, 3.7GB RAM, no swap. Training = OOM = all bots die. Training only on Chen's Mac M3 Pro.
2. **NEVER push to main** — always `dev` branch only.
3. **NEVER open a position without SL and TP** — if TP placement fails, immediately close the position. This is the #1 rule.
4. **ALL UI/testnet data = real Binance API data** — zero local calculations, zero mock data, zero hardcoded values.
5. **Storage = SQLite** (`data/trading.db`). MongoDB Atlas is broken (DNS unreachable). Don't try to fix it unless asked.
6. **After ANY code change**: restart affected systemd service(s).
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
