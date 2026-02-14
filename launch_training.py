#!/usr/bin/env python3
"""
Training Launcher - Runs in background with proper logging
"""
import subprocess
import sys
import os
from datetime import datetime

# Ensure we're in the right directory
os.chdir('/Users/chenluigi/WebstormProjects/drl-trading-system')

# Create logs directory
os.makedirs('logs', exist_ok=True)

log_file = 'logs/ultimate_training.log'

with open(log_file, 'w') as f:
    f.write(f"Training started at: {datetime.now()}\n")
    f.write("=" * 50 + "\n\n")

# Run the training script
cmd = [
    './venv/bin/python',
    'train_ultimate.py',
    '--timesteps', '1000000',
    '--n-envs', '4'
]

with open(log_file, 'a') as f:
    process = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd='/Users/chenluigi/WebstormProjects/drl-trading-system',
    )

print(f"Training started with PID: {process.pid}")
print(f"Log file: {log_file}")
print(f"Check progress with: tail -f {log_file}")
