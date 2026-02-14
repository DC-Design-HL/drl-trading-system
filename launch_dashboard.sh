#!/bin/bash
# Launch Dashboard with Live Updates
# Starts both Streamlit and Flask API server for real-time updates

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting DRL Trading Dashboard with Live Updates..."
echo "=================================================="

# Activate virtual environment
source venv/bin/activate

# Kill any existing processes
echo "Stopping any existing processes..."
pkill -f "api_server.py" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# Start Flask API server in background (for live updates)
echo "📡 Starting API server on port 5001..."
nohup python src/ui/api_server.py > logs/api_server.log 2>&1 &
API_PID=$!
echo "   API server PID: $API_PID"

# Wait for API to start
sleep 2

# Check if API is running
if lsof -i :5001 > /dev/null 2>&1; then
    echo "   ✅ API server running on http://localhost:5001"
else
    echo "   ❌ API server failed to start!"
    exit 1
fi

# Start Streamlit app
echo "📊 Starting Streamlit dashboard on port 8501..."
echo ""
echo "=================================================="
echo "Dashboard URL: http://localhost:8501"
echo "API URL:       http://localhost:5001"
echo "=================================================="
echo ""
echo "Live updates enabled! Press Ctrl+C to stop all services."
echo ""

# Run Streamlit in foreground
streamlit run src/ui/app.py --server.port 8501

# Cleanup on exit
trap "echo 'Stopping services...'; kill $API_PID 2>/dev/null; exit 0" INT TERM
