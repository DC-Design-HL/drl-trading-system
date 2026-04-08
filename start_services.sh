#!/bin/bash
# start_services.sh — Start all DRL trading system processes detached from any session.
# Uses setsid so processes survive shell/SSH/Claude Code session termination.
# Safe to run multiple times: kills existing instances before relaunching.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO/logs"
PIDS="$REPO/logs/running_services.json"

mkdir -p "$LOG"

# Load env
set -a
source "$REPO/.env"
export TESTNET_MIRROR=true
# Limit PyTorch/OpenBLAS thread pools — 4 bots on 2 CPUs, 1 thread each saves ~150-200MB
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
set +a

echo "[start_services] Starting DRL Trading System services..."

# Kill existing instances gracefully
for name in live_trading_htf trade_alerter start_local_server streamlit localtunnel news_sentinel news_alerter; do
    pids=$(pgrep -f "$name" 2>/dev/null) && echo "[start_services] Stopping $name (PIDs: $pids)" && kill $pids 2>/dev/null || true
done
sleep 3

# --- API Server ---
echo "[start_services] Starting API server..."
setsid python3 "$REPO/start_local_server.py" > "$LOG/api_server.log" 2>&1 &
API_PID=$!
echo "[start_services]   API server PID: $API_PID"
sleep 3

# --- Trading Bots ---
echo "[start_services] Starting trading bots..."
setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol BTCUSDT > "$LOG/btc_live.log" 2>&1 &
BTC_PID=$!

setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol ETHUSDT > "$LOG/eth_live.log" 2>&1 &
ETH_PID=$!

setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol SOLUSDT > "$LOG/sol_live.log" 2>&1 &
SOL_PID=$!

setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol XRPUSDT > "$LOG/xrp_live.log" 2>&1 &
XRP_PID=$!

echo "[start_services]   BTC=$BTC_PID ETH=$ETH_PID SOL=$SOL_PID XRP=$XRP_PID"

# --- Trade Alerter ---
echo "[start_services] Starting trade alerter..."
setsid python3 "$REPO/trade_alerter.py" > "$LOG/alerter.log" 2>&1 &
ALERTER_PID=$!
echo "[start_services]   Alerter PID: $ALERTER_PID"

# --- News Sentinel ---
echo "[start_services] Starting news sentinel..."
setsid python3 "$REPO/news_sentinel.py" > "$LOG/news_sentinel.log" 2>&1 &
NEWS_SENTINEL_PID=$!
echo "[start_services]   News Sentinel PID: $NEWS_SENTINEL_PID"

# --- News Alerter ---
echo "[start_services] Starting news alerter..."
setsid python3 "$REPO/news_alerter.py" > "$LOG/news_alerter.log" 2>&1 &
NEWS_ALERTER_PID=$!
echo "[start_services]   News Alerter PID: $NEWS_ALERTER_PID"

# --- Streamlit Dashboard ---
echo "[start_services] Starting Streamlit dashboard..."
setsid python3 -m streamlit run "$REPO/src/ui/app.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    > "$LOG/dashboard.log" 2>&1 &
UI_PID=$!
echo "[start_services]   Streamlit PID: $UI_PID"

# Give Streamlit time to start before tunnel
sleep 8

# --- Localtunnel ---
echo "[start_services] Starting localtunnel..."
setsid /home/claude/.npm/_npx/75ac80b86e83d4a2/node_modules/.bin/lt --port 8501 -s drl-trading-chen > "$LOG/tunnel.log" 2>&1 &
TUNNEL_PID=$!
sleep 4
TUNNEL_URL=$(grep -m1 "your url is:" "$LOG/tunnel.log" 2>/dev/null | awk '{print $NF}' || echo "check logs/tunnel.log")
echo "[start_services]   Tunnel PID: $TUNNEL_PID — $TUNNEL_URL"

# --- Save PIDs ---
cat > "$PIDS" << JSON
{
  "btc":     {"pid": $BTC_PID,     "log": "logs/btc_live.log",   "symbol": "BTCUSDT", "sharpe": 7.92},
  "eth":     {"pid": $ETH_PID,     "log": "logs/eth_live.log",   "symbol": "ETHUSDT", "sharpe": 9.90},
  "sol":     {"pid": $SOL_PID,     "log": "logs/sol_live.log",   "symbol": "SOLUSDT", "sharpe": 6.79},
  "xrp":     {"pid": $XRP_PID,     "log": "logs/xrp_live.log",   "symbol": "XRPUSDT", "sharpe": 12.42},
  "alerter":       {"pid": $ALERTER_PID,       "log": "logs/alerter.log"},
  "api":           {"pid": $API_PID,           "log": "logs/api_server.log"},
  "ui":            {"pid": $UI_PID,            "log": "logs/dashboard.log"},
  "tunnel":        {"pid": $TUNNEL_PID,        "log": "logs/tunnel.log", "url": "$TUNNEL_URL"},
  "news_sentinel": {"pid": $NEWS_SENTINEL_PID, "log": "logs/news_sentinel.log"},
  "news_alerter":  {"pid": $NEWS_ALERTER_PID,  "log": "logs/news_alerter.log"}
}
JSON

echo ""
echo "[start_services] ✅ All services started"
echo "[start_services] Dashboard: $TUNNEL_URL"
echo "[start_services] PIDs saved to: $PIDS"
