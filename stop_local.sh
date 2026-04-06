#!/bin/bash
# Stop all running services
if [ -f .pids ]; then
    read -r PIDS < .pids
    kill $PIDS 2>/dev/null && echo "✅ All services stopped" || echo "Some services were already stopped"
    rm -f .pids
else
    # Fallback: kill by process name
    pkill -f "live_trading_multi.py" 2>/dev/null
    pkill -f "api_server.py" 2>/dev/null
    pkill -f "streamlit run" 2>/dev/null
    pkill -f "trade_alerter.py" 2>/dev/null
    pkill -f "localtunnel" 2>/dev/null
    echo "✅ Services stopped"
fi
