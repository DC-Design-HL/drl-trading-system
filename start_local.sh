#!/bin/bash
# Local startup script — starts all services and exposes dashboard publicly
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH=$PATH:/home/claude/.local/bin

# Load environment
if [ -f .env ]; then
    set -a && source .env && set +a
    echo "✅ Environment loaded from .env"
fi

# Create required directories
mkdir -p logs data/trading

# --- Start trade alerter ---
echo "🔔 Starting trade alerter..."
python3.12 trade_alerter.py > logs/alerter.log 2>&1 &
ALERTER_PID=$!
echo "   PID: $ALERTER_PID"

# --- Start trading bot ---
echo "🤖 Starting trading bot (BTC, ETH, SOL, XRP)..."
python3.12 live_trading_multi.py \
    --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --balance 5000 \
    2>&1 | tee logs/trading_bot.log &
BOT_PID=$!
echo "   PID: $BOT_PID"

# --- Start API server ---
echo "🌐 Starting Flask API server on :5000..."
python3.12 src/ui/api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "   PID: $API_PID"

# --- Start Streamlit dashboard ---
echo "📊 Starting Streamlit dashboard on :7860..."
python3.12 -m streamlit run src/ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > logs/dashboard.log 2>&1 &
DASH_PID=$!
echo "   PID: $DASH_PID"

# Give Streamlit a moment to start
sleep 5

# --- Expose via localtunnel ---
echo ""
echo "🌍 Exposing dashboard publicly via localtunnel..."
npx localtunnel --port 7860 --subdomain drl-trading 2>&1 | tee logs/tunnel.log &
TUNNEL_PID=$!

sleep 3

# Print public URL
PUBLIC_URL=$(grep -m1 "your url is:" logs/tunnel.log 2>/dev/null | awk '{print $NF}' || echo "Check logs/tunnel.log for URL")
echo ""
echo "============================================"
echo "✅ ALL SERVICES RUNNING"
echo "============================================"
echo "📊 Dashboard (local):  http://localhost:7860"
echo "🌍 Dashboard (public): $PUBLIC_URL"
echo "🔌 API server:         http://localhost:5000"
echo ""
echo "Logs:"
echo "  Trading bot:   logs/trading_bot.log"
echo "  API server:    logs/api_server.log"
echo "  Dashboard:     logs/dashboard.log"
echo "  Trade alerts:  logs/alerter.log"
echo "  Tunnel:        logs/tunnel.log"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Save PIDs for stop script
echo "$BOT_PID $ALERTER_PID $API_PID $DASH_PID $TUNNEL_PID" > .pids

# Wait and handle shutdown
cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $BOT_PID $ALERTER_PID $API_PID $DASH_PID $TUNNEL_PID 2>/dev/null
    rm -f .pids
    echo "Done."
}
trap cleanup INT TERM

wait $DASH_PID
