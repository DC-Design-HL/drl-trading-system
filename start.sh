#!/bin/bash

# Start the trading bot in the background
# Using unbuffered output to ensure logs are visible immediately
python -u live_trading_multi.py --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT --balance 5000 > process.log 2>&1 &

# Start the API server in the background (provides Market Analysis data)
python -u src/ui/api_server.py > api_server.log 2>&1 &

# Start the Streamlit dashboard in the foreground
# Streamlit runs on port 7860 by default in HF Spaces
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0
