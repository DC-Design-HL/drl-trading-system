#!/bin/bash
# restart_ui.sh — Restart only the Streamlit dashboard (UI-only changes).
# Does NOT touch bots, alerter, API server, or news services.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$REPO/logs"
PIDS="$REPO/logs/running_services.json"

set -a
source "$REPO/.env"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set +a

echo "[restart_ui] Stopping Streamlit..."
pkill -f "streamlit run" 2>/dev/null || true
sleep 2

echo "[restart_ui] Starting Streamlit dashboard..."
setsid python3 -m streamlit run "$REPO/src/ui/app.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    > "$LOG/dashboard.log" 2>&1 &
UI_PID=$!
echo "[restart_ui] Streamlit PID: $UI_PID"

# Update PID in running_services.json
if [ -f "$PIDS" ]; then
    python3 -c "
import json, sys
with open('$PIDS') as f:
    d = json.load(f)
d['ui']['pid'] = $UI_PID
with open('$PIDS', 'w') as f:
    json.dump(d, f, indent=2)
print('[restart_ui] PID updated in running_services.json')
"
fi

echo "[restart_ui] ✅ UI restarted (bots and services untouched)"
