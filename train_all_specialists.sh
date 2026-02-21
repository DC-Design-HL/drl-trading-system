#!/bin/bash
set -e

ASSETS=("BTCUSDT" "ETHUSDT" "SOLUSDT" "XRPUSDT")
REGIMES=("BULL_TREND" "BEAR_TREND" "RANGE_CHOP" "HIGH_VOL_BREAKOUT")

echo "Creating specialists directory..."
mkdir -p data/models/specialists

PIDS=""

for ASSET in "${ASSETS[@]}"; do
    echo "======================================"
    echo "Launching training for $ASSET..."
    echo "======================================"
    
    for REGIME in "${REGIMES[@]}"; do
        echo "Training $REGIME specialist for $ASSET..."
        # Lowercase the regime for the log filename using tr for Mac compatibility
        REGIME_LOWER=$(echo "$REGIME" | tr '[:upper:]' '[:lower:]')
        LOG_FILE="data/models/specialists/${ASSET}_${REGIME_LOWER}_train.log"
        
        ./venv/bin/python3 -m src.models.train_specialist --asset "$ASSET" --regime "$REGIME" --timesteps 150000 > "$LOG_FILE" 2>&1 &
        PIDS="$PIDS $!"
    done
done

echo
echo "All specialist training processes launched!"
echo "Waiting for them to finish in the background..."
wait $PIDS

echo "All specialists trained successfully for all assets!"
