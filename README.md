---
title: DRL Trading Bot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DRL Trading System

An autonomous Deep Reinforcement Learning trading system using PPO-LSTM for Binance Testnet trading with real-time Streamlit monitoring.

## Features

- 🧠 **PPO-LSTM Agent**: Captures time-series dependencies for smarter trading decisions
- 🔄 **Self-Improvement Loop**: Automatically fine-tunes on successful trades every 24 hours
- 📊 **Real-time Dashboard**: TradingView-style charts with live Buy/Sell signals
- 🛡️ **Risk Management**: Circuit breaker stops trading at 5% daily loss
- 📈 **Backtesting**: Validated on 2024-2025 historical data

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your Binance Testnet credentials

# 4. Run backtest first
python -m src.backtest.engine

# 5. Launch UI
streamlit run src/ui/app.py
```

## Project Structure

```
drl-trading-system/
├── config/
│   └── config.yaml          # All configuration
├── src/
│   ├── env/                  # Gymnasium environment
│   ├── brain/                # PPO-LSTM agent
│   ├── api/                  # Binance connector
│   ├── backtest/             # Backtesting engine
│   └── ui/                   # Streamlit dashboard
├── data/
│   ├── historical/           # Cached OHLCV data
│   └── models/               # Saved checkpoints
└── tests/
```

## Architecture

The system uses a Sharpe/Sortino ratio reward function to prioritize risk-adjusted returns over raw profit.

## License

MIT
