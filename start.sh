#!/bin/bash

# Start the trading bot in the background
# Using unbuffered output to ensure logs are visible immediately
# Start the trading bot (pipe to file for UI access)
# Start the trading bot (pipe to file for UI access and stdout for debugging)
python -u live_trading.py --model ./data/models/ultimate_agent.zip --symbol BTC/USDT 2>&1 | tee process.log &

# Start the API serverplease
python -u src/ui/api_server.py 2>&1 | tee api_server.log &

# Start the Streamlit dashboard
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0
