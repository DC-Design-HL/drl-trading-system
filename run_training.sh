#!/bin/bash
# Wrapper script to run Ultimate Training with proper logging

cd /Users/chenluigi/WebstormProjects/drl-trading-system

LOG_FILE="logs/ultimate_training.log"
mkdir -p logs

echo "Starting Ultimate Training at $(date)" > $LOG_FILE
echo "=====================================" >> $LOG_FILE

./venv/bin/python train_ultimate.py --timesteps 1000000 --n-envs 4 >> $LOG_FILE 2>&1

echo "" >> $LOG_FILE
echo "Training completed at $(date)" >> $LOG_FILE
