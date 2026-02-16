#!/bin/bash

# Start the trading bot in the background
# Using unbuffered output to ensure logs are visible immediately
# Start the trading bot (pipe to file for UI access)
python -u live_trading_multi.py --assets BTCUSDT ETHUSDT --balance 5000 > process.log 2>&1 &

# Start the API serverplease
python -u src/ui/api_server.py > api_server.log 2>&1 &

# Start the Streamlit dashboard
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0
