#!/bin/bash
# Start V3 Model Retraining
# Incorporates Quick Wins insights: No XRP, regime-aware, time-based learning

echo "=========================================="
echo "STARTING ULTIMATE V3 MODEL TRAINING"
echo "=========================================="
echo "Quick Wins Integrated: YES"
echo "XRP Excluded: YES"
echo "Regime-Aware: YES"
echo "Time-Based Learning: YES"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs/training

# Training parameters
TIMESTEPS=2000000  # 2M timesteps (~2-3 hours)
LEARNING_RATE=3e-4

echo ""
echo "Training Parameters:"
echo "  Timesteps: $TIMESTEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Assets: BTC, ETH, SOL (NO XRP)"
echo ""
echo "Output will be saved to: logs/training/v3_training_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Run training with timestamp log
python retrain_ultimate_v3.py \
    --timesteps $TIMESTEPS \
    --lr $LEARNING_RATE \
    2>&1 | tee logs/training/v3_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo "Check results in:"
echo "  - data/models/ultimate_v3.zip"
echo "  - data/models/ultimate_v3_best.zip"
echo "  - data/models/ultimate_v3_training_report.json"
echo "=========================================="
