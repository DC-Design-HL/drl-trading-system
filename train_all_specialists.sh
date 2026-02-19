#!/bin/bash
set -e

echo "Creating specialists directory..."
mkdir -p data/models/specialists

echo "Training BULL_TREND specialist..."
./venv/bin/python3 -m src.models.train_specialist --asset BTCUSDT --regime BULL_TREND --timesteps 150000 > data/models/specialists/bull_train.log 2>&1 &
PID1=$!

echo "Training BEAR_TREND specialist..."
./venv/bin/python3 -m src.models.train_specialist --asset BTCUSDT --regime BEAR_TREND --timesteps 150000 > data/models/specialists/bear_train.log 2>&1 &
PID2=$!

echo "Training RANGE_CHOP specialist..."
./venv/bin/python3 -m src.models.train_specialist --asset BTCUSDT --regime RANGE_CHOP --timesteps 150000 > data/models/specialists/range_train.log 2>&1 &
PID3=$!

echo "Training HIGH_VOL_BREAKOUT specialist..."
./venv/bin/python3 -m src.models.train_specialist --asset BTCUSDT --regime HIGH_VOL_BREAKOUT --timesteps 150000 > data/models/specialists/breakout_train.log 2>&1 &
PID4=$!

echo "All 4 training processes launched. Waiting for them to finish..."
wait $PID1 $PID2 $PID3 $PID4

echo "All specialists trained successfully!"
