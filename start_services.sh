#!/bin/bash
# start_services.sh — Start all DRL trading system processes detached from any session.
# Uses setsid so processes survive shell/SSH/Claude Code session termination.
# Safe to run multiple times: kills existing instances before relaunching.
# Dashboard available at http://116.203.196.107 (Caddy proxies port 80 → Streamlit).

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

# --- Apply Caddy reverse proxy config (port 80 → Streamlit) via admin API ---
python3 - << 'PYEOF'
import urllib.request, json, time
new_srv1 = {
    "listen": [":80"],
    "routes": [
        {"match": [{"path": ["/health"]}], "handle": [{"handler": "static_response", "body": "OK", "status_code": 200}]},
        {"match": [{"path": ["/api/*"]}], "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "localhost:5001"}]}]},
        {"handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "localhost:8501"}],
            "transport": {"protocol": "http", "versions": ["1.1", "2"]}}]}
    ]
}
req = urllib.request.Request("http://localhost:2019/config/apps/http/servers/srv1",
    data=json.dumps(new_srv1).encode(), headers={"Content-Type": "application/json"}, method="PATCH")
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        print("[start_services] Caddy port 80 → Streamlit configured")
except Exception as e:
    print(f"[start_services] Caddy config warning: {e}")
PYEOF

# Kill existing instances gracefully.
# Includes BOTH per-symbol (live_trading_htf) and consolidated (live_trading_all)
# so cutover between modes is always clean regardless of which one was running.
for name in live_trading_all live_trading_htf trade_alerter start_local_server streamlit news_sentinel news_alerter; do
    pids=$(pgrep -f "$name" 2>/dev/null) && echo "[start_services] Stopping $name (PIDs: $pids)" && kill $pids 2>/dev/null || true
done
sleep 3

# Consolidated mode: one python process runs all 4 bots as threads, saving ~1.2 GB RAM.
# Triggered by CONSOLIDATED_BOTS=1 env var (set by start_consolidated.sh wrapper).
CONSOLIDATED_BOTS="${CONSOLIDATED_BOTS:-0}"

# --- API Server ---
echo "[start_services] Starting API server..."
setsid python3 "$REPO/start_local_server.py" > "$LOG/api_server.log" 2>&1 &
API_PID=$!
echo "[start_services]   API server PID: $API_PID"
sleep 3

# --- Trading Bots ---
if [ "$CONSOLIDATED_BOTS" = "1" ]; then
    echo "[start_services] Starting CONSOLIDATED trading bot (1 process, all 4 symbols as threads)..."
    setsid python3 "$REPO/live_trading_all.py" --live --interval 15 > "$LOG/bots_live.log" 2>&1 &
    BOTS_PID=$!
    echo "[start_services]   Consolidated bots PID: $BOTS_PID"
else
    echo "[start_services] Starting trading bots (4 separate processes)..."
    setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol BTCUSDT > "$LOG/btc_live.log" 2>&1 &
    BTC_PID=$!

    setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol ETHUSDT > "$LOG/eth_live.log" 2>&1 &
    ETH_PID=$!

    setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol SOLUSDT > "$LOG/sol_live.log" 2>&1 &
    SOL_PID=$!

    setsid python3 "$REPO/live_trading_htf.py" --live --interval 15 --symbol XRPUSDT > "$LOG/xrp_live.log" 2>&1 &
    XRP_PID=$!

    echo "[start_services]   BTC=$BTC_PID ETH=$ETH_PID SOL=$SOL_PID XRP=$XRP_PID"
fi

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

# --- Save PIDs ---
# In consolidated mode, a single "bots" entry replaces btc/eth/sol/xrp.
# watchdog.sh is data-driven (just kill -0's every PID in this file), so the
# new shape is automatically picked up without code changes there.
if [ "$CONSOLIDATED_BOTS" = "1" ]; then
cat > "$PIDS" << JSON
{
  "bots":         {"pid": $BOTS_PID,         "log": "logs/bots_live.log",      "mode": "consolidated", "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"]},
  "alerter":      {"pid": $ALERTER_PID,      "log": "logs/alerter.log"},
  "api":          {"pid": $API_PID,          "log": "logs/api_server.log"},
  "ui":           {"pid": $UI_PID,           "log": "logs/dashboard.log"},
  "news_sentinel":{"pid": $NEWS_SENTINEL_PID,"log": "logs/news_sentinel.log"},
  "news_alerter": {"pid": $NEWS_ALERTER_PID, "log": "logs/news_alerter.log"}
}
JSON
else
cat > "$PIDS" << JSON
{
  "btc":          {"pid": $BTC_PID,          "log": "logs/btc_live.log",       "symbol": "BTCUSDT", "sharpe": 7.92},
  "eth":          {"pid": $ETH_PID,          "log": "logs/eth_live.log",       "symbol": "ETHUSDT", "sharpe": 9.90},
  "sol":          {"pid": $SOL_PID,          "log": "logs/sol_live.log",       "symbol": "SOLUSDT", "sharpe": 6.79},
  "xrp":          {"pid": $XRP_PID,          "log": "logs/xrp_live.log",       "symbol": "XRPUSDT", "sharpe": 12.42},
  "alerter":      {"pid": $ALERTER_PID,      "log": "logs/alerter.log"},
  "api":          {"pid": $API_PID,          "log": "logs/api_server.log"},
  "ui":           {"pid": $UI_PID,           "log": "logs/dashboard.log"},
  "news_sentinel":{"pid": $NEWS_SENTINEL_PID,"log": "logs/news_sentinel.log"},
  "news_alerter": {"pid": $NEWS_ALERTER_PID, "log": "logs/news_alerter.log"}
}
JSON
fi

echo ""
echo "[start_services] ✅ All services started"
echo "[start_services] Dashboard: http://116.203.196.107"
echo "[start_services] PIDs saved to: $PIDS"

# --- Notify Telegram ---
if [ -n "$TELEGRAM_ALERT_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_IDS" ]; then
    _FIRST_CHAT=$(echo "$TELEGRAM_CHAT_IDS" | cut -d',' -f1)
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_ALERT_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${_FIRST_CHAT}&text=🚀 Services restarted%0ADashboard: http://116.203.196.107" > /dev/null 2>&1 || true
    echo "[start_services] Notification sent to Telegram"
fi
