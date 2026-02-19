#!/bin/bash
git add live_trading_multi.py start.sh
git commit -m "Optimize: Add aggressive GC and enable 3 assets (BTC, ETH, SOL)"
git push origin main
echo "Deployment initiated. Monitor the Hugging Face space for build progress."
