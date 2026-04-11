#!/bin/bash
# start_consolidated.sh — Start DRL services in CONSOLIDATED mode.
#
# Consolidated mode runs all 4 trading bots (BTC/ETH/SOL/XRP) as threads
# inside a single Python process (live_trading_all.py), saving ~1.2 GB RAM
# vs the default 4-process layout. Each symbol has its own thread-level
# watchdog — a crash in one symbol does not take down the others.
#
# This is a thin wrapper around start_services.sh — all the heavy lifting
# (kill existing, Caddy config, API/UI/alerter startup, PID file, Telegram
# notify) lives there. We just flip the CONSOLIDATED_BOTS flag and delegate.
#
# Usage:
#   ./start_consolidated.sh         # start in consolidated mode
#
# To revert to the default 4-process layout:
#   ./start_services.sh             # CONSOLIDATED_BOTS defaults to 0
#
# The host-level watchdog (watchdog.sh, cron */2) reads running_services.json
# and kill -0's every PID regardless of keys, so no changes are needed there.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONSOLIDATED_BOTS=1
exec "$REPO/start_services.sh" "$@"
